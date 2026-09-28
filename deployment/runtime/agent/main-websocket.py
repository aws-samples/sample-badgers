"""WebSocket streaming version of the agent for real-time event visibility.

This version uses async streaming to yield events (thinking, tool_use, tool_result, text)
as they happen, enabling the frontend to display progress in real-time.

Key differences from main.py:
- Uses agent.stream_async() instead of agent()
- Yields events via async generator for SSE streaming
- Frontend receives events as they occur (tool calls, results, thinking)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional

from bedrock_agentcore.runtime import BedrockAgentCoreApp, PingStatus
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

# The foundation module is copied into the container build context by
# build_and_push_websocket.sh (the same way build_container_lambdas.sh supplies it
# to the container Lambdas). Guarded so the agent still imports outside that build
# — job tracking then degrades to a no-op rather than breaking the runtime.
#
# The reason is retained and logged rather than assumed. A missing directory and a
# missing transitive dependency both land here, and reporting only the first sent
# an investigation to the build script when the actual cause was foundation's
# package __init__ importing Pillow, which this image does not install.
_JOB_STATE_IMPORT_ERROR: str = ""
try:
    from foundation import job_state
except ImportError as e:  # pragma: no cover - foundation is present in the container
    job_state = None  # type: ignore[assignment]
    _JOB_STATE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [AGENT-WS] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def log(message: str, level: str = "info") -> None:
    """Log with flush to ensure CloudWatch captures output immediately."""
    getattr(logger, level)(message)
    sys.stdout.flush()


# =============================================================================
# AGENTCORE APP INITIALIZATION
# =============================================================================

app = BedrockAgentCoreApp()


# Processing state tracking for ping handler
@dataclass
class ProcessingState:
    """Tracks current processing state for ping handler."""

    processing: bool = False
    session_id: Optional[str] = None
    started_at: Optional[str] = None


_processing_state = ProcessingState()


@app.ping
def ping_handler() -> PingStatus:
    """Custom ping handler to signal HEALTHY_BUSY during long-running operations."""
    if _processing_state.processing:
        return PingStatus.HEALTHY_BUSY
    return PingStatus.HEALTHY


@contextmanager
def processing_context(session_id: str) -> Iterator[None]:
    """Context manager to track processing state for ping handler."""
    _processing_state.processing = True
    _processing_state.session_id = session_id
    _processing_state.started_at = datetime.utcnow().isoformat()
    log(f"Processing started for session: {session_id}")
    try:
        yield
    finally:
        _processing_state.processing = False
        _processing_state.session_id = None
        _processing_state.started_at = None
        log(f"Processing completed for session: {session_id}")


# =============================================================================
# WARM STATE
# =============================================================================
#
# One AgentCore runtime session is one microVM, and every WebSocket the UI server
# opens with the same session id lands on it. Module-level state therefore lives
# for the whole chat session even though the UI server opens a fresh socket per
# turn. The UI sends a {"warmup": true} frame when the chat mounts so this state is
# filled before the first prompt; each turn then reuses it instead of rebuilding
# the token, Gateway connection, tool list and model.
#
# What is deliberately NOT cached:
# - The system prompt and model config are re-validated every turn with an ETag
#   conditional GET, so editing them in the config bucket still takes effect on
#   the next message with no redeploy.
# - The Agent and its memory session manager are built per turn. The session
#   manager reloads and sanitizes history on construction, which is what repairs
#   a turn the browser abandoned mid-tool-call.

# Refresh the Cognito token this long before it expires, so a turn that starts
# just inside the window cannot outlive it.
_TOKEN_REFRESH_MARGIN_S = 300


@dataclass
class _WarmState:
    """Per-session (per-microVM) cache of everything a turn needs before the model."""

    credentials: Optional[dict[str, str]] = None
    access_token: str = ""
    token_expires_at: float = 0.0
    s3_client: Any = None
    # key -> (etag, body)
    s3_cache: dict[str, tuple[str, str]] = field(default_factory=dict)
    mcp_client: Any = None
    # The token the live MCP client was opened with; a new token means a new client.
    mcp_token: str = ""
    tools: list[Any] = field(default_factory=list)
    model: Any = None
    model_key: str = ""


@dataclass(frozen=True)
class WarmSnapshot:
    """What one turn reads out of the warm state."""

    system_prompt: str
    model_config: dict[str, Any]
    model: Any
    tools: list[Any]


_warm = _WarmState()
# The warmup frame and the first prompt arrive on separate sockets and can overlap
# if the user sends quickly. Serializing here makes the prompt wait for the warmup
# it raced instead of starting a second, duplicate initialization.
_warm_lock = threading.Lock()


# =============================================================================
# AUTHENTICATION HELPERS
# =============================================================================


def get_cognito_credentials() -> dict[str, str]:
    """Fetch Cognito credentials from AWS Secrets Manager."""
    import boto3

    secret_arn = os.environ.get("COGNITO_CREDENTIALS_SECRET_ARN")
    if not secret_arn:
        raise ValueError("COGNITO_CREDENTIALS_SECRET_ARN not set")

    client = boto3.client(
        "secretsmanager", region_name=os.environ.get("AWS_REGION", "us-west-2")
    )
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


def get_cognito_token() -> str:
    """Get OAuth token from Cognito for Gateway authentication.

    Returns the cached token until it is within _TOKEN_REFRESH_MARGIN_S of expiry.
    The client secret is cached too; a rejected token request re-reads it once in
    case it was rotated since.
    """
    import httpx

    now = time.time()
    if _warm.access_token and now < _warm.token_expires_at - _TOKEN_REFRESH_MARGIN_S:
        return _warm.access_token

    status_code = 0
    for _attempt in range(2):
        if _warm.credentials is None:
            _warm.credentials = get_cognito_credentials()
        credentials = _warm.credentials
        response = httpx.post(
            credentials["token_endpoint"],
            data={
                "grant_type": "client_credentials",
                "client_id": credentials["client_id"],
                "client_secret": credentials["client_secret"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        status_code = response.status_code
        if status_code == 200:
            body = response.json()
            _warm.access_token = body["access_token"]
            _warm.token_expires_at = now + float(body.get("expires_in", 3600))
            return _warm.access_token
        _warm.credentials = None

    raise ValueError(f"Failed to get Cognito token: {status_code}")


# =============================================================================
# MCP TRANSPORT
# =============================================================================


def create_mcp_transport(gateway_url: str, access_token: str) -> Any:
    """Create MCP transport for AgentCore Gateway connection."""
    from mcp.client.streamable_http import streamablehttp_client

    return streamablehttp_client(
        gateway_url, headers={"Authorization": f"Bearer {access_token}"}
    )


def _mcp_client_alive(client: Any) -> bool:
    """Whether the MCP client's background session is still running.

    Strands exposes this only privately. If a future release drops it, assume
    alive; a dead session then surfaces as tool errors and is rebuilt when the
    token next rotates.
    """
    check = getattr(client, "_is_session_active", None)
    return bool(check()) if callable(check) else True


def _ensure_mcp_tools(gateway_url: str, access_token: str) -> list[Any]:
    """Return the Gateway tool list, keeping one MCP session open across turns.

    The session is rebuilt when the token changes (the token is baked into the
    transport's headers) or when its background thread has died. The tool list is
    fetched once per session, so a Gateway target added mid-session appears on the
    next token rotation or in a new chat session.
    """
    from strands.tools.mcp.mcp_client import MCPClient

    client = _warm.mcp_client
    if (
        client is not None
        and _warm.mcp_token == access_token
        and _mcp_client_alive(client)
    ):
        return _warm.tools

    if client is not None:
        try:
            client.stop(None, None, None)
        except Exception as e:  # a dead session can fail to stop cleanly
            log(f"Ignoring error stopping stale MCP client: {e}", level="warning")
        _warm.mcp_client = None
        _warm.tools = []

    client = MCPClient(lambda: create_mcp_transport(gateway_url, access_token))
    client.start()
    try:
        tools: list[Any] = []
        pagination_token = None
        while True:
            result = client.list_tools_sync(pagination_token=pagination_token)
            tools.extend(result)
            if hasattr(result, "pagination_token") and result.pagination_token:
                pagination_token = result.pagination_token
            else:
                break
    except Exception:
        client.stop(None, None, None)
        raise

    _warm.mcp_client = client
    _warm.mcp_token = access_token
    _warm.tools = tools
    log(f"Opened MCP session and fetched {len(tools)} tools")
    return tools


# =============================================================================
# CONVERSATION HISTORY SANITIZATION
# =============================================================================


def sanitize_conversation_history(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sanitize conversation history to fix Bedrock Converse API violations."""
    if not messages:
        return messages

    # First pass: Remove orphaned toolResults
    sanitized: list[dict[str, Any]] = []
    pending_tool_use_ids: set[str] = set()

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "assistant":
            pending_tool_use_ids.clear()
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "toolUse" in block:
                        tool_use_id = block["toolUse"].get("toolUseId")
                        if tool_use_id:
                            pending_tool_use_ids.add(tool_use_id)
            sanitized.append(msg)

        elif role == "user":
            if isinstance(content, list):
                filtered_content = [
                    block
                    for block in content
                    if not (isinstance(block, dict) and "toolResult" in block)
                    or block.get("toolResult", {}).get("toolUseId")
                    in pending_tool_use_ids
                ]
                if filtered_content:
                    sanitized.append({**msg, "content": filtered_content})
            else:
                sanitized.append(msg)
            pending_tool_use_ids.clear()
        else:
            sanitized.append(msg)

    # Second pass: Fix consecutive messages of same role
    merged: list[dict[str, Any]] = []
    for msg in sanitized:
        if not merged:
            merged.append(msg)
            continue

        role = msg.get("role", "")
        prev_role = merged[-1].get("role", "")

        if role == prev_role:
            prev_content = merged[-1].get("content", [])
            content = msg.get("content", [])
            if isinstance(prev_content, list) and isinstance(content, list):
                merged[-1] = {**merged[-1], "content": prev_content + content}
            else:
                merged.append(msg)
        else:
            merged.append(msg)

    return merged


class SanitizingSessionManager:
    """Wrapper that sanitizes loaded history."""

    def __init__(self, inner_manager: AgentCoreMemorySessionManager) -> None:
        self._inner = inner_manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_messages(self) -> list[dict[str, Any]]:
        messages = self._inner.get_messages()
        return sanitize_conversation_history(messages) if messages else messages

    def save_messages(self, messages: list[dict[str, Any]]) -> Any:
        return self._inner.save_messages(messages)


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

DEFAULT_MODEL_CONFIG = {
    "model_id": "us.anthropic.claude-opus-4-6-v1",
    "temperature": 1.0,
    "max_tokens": 16000,
    "thinking": {"type": "adaptive"},
    "effort": "high",
}

# There is deliberately no "fallback_models" key. One used to be defined here and was read
# by nothing — the string appeared exactly once in the whole repo, at its own definition —
# so it advertised resilience the agent does not have. Its two entries also named models
# being retired, which made it look like something that needed updating rather than
# deleting.
#
# The agent is not without protection: Strands retries ModelThrottledException with
# exponential backoff (6 attempts, 4s doubling to 240s, reset after each success), which
# covers the dominant transient failure. What it has no answer for is a non-retryable error
# — AccessDenied, ValidationException, a model reaching EOL — where a different model would
# be the only escape. The chosen fix for that is resumable runs rather than a fallback
# model, because resume also covers retry-budget exhaustion and avoids swapping models
# mid-conversation through accumulated thinking blocks and tool results. Tracked separately;
# do not reintroduce a fallback list here without that decision being revisited.

DEFAULT_SYSTEM_PROMPT = """You are an intelligent BADGERS assistant with access to specialized tools via AgentCore Gateway."""

# Effort levels Claude accepts. "xhigh" and "max" exist but only on specific Opus models,
# and an unsupported value is an error rather than a downgrade, so they are not offered.
VALID_EFFORT = ("low", "medium", "high")


def _build_additional_request_fields(model_config: dict[str, Any]) -> dict[str, Any]:
    """Build BedrockModel's additional_request_fields from the agent's model config.

    This previously forwarded only ``{"thinking": ...}``, so the ``effort`` and
    ``adaptive_thinking`` keys declared in agent_config/agent_model_config.json were read by
    nothing. The value happened to match the model's default, which is why it went unnoticed
    — anyone lowering effort to trim cost would have seen no change and no error.

    ``effort`` must travel in its own ``output_config`` object. Putting it inside
    ``thinking`` raises a ValidationException, so the two are siblings here, not nested.
    """
    thinking = model_config.get("thinking") or {}
    fields: dict[str, Any] = {}

    if thinking:
        fields["thinking"] = thinking

    # effort only means anything when the model is actually thinking. "adaptive_thinking" is
    # accepted as an alias for thinking.type == "adaptive", since the config file carries
    # both spellings.
    is_adaptive = thinking.get("type") == "adaptive" or bool(
        model_config.get("adaptive_thinking")
    )
    if is_adaptive and not thinking:
        fields["thinking"] = {"type": "adaptive"}

    effort = model_config.get("effort")
    if effort and (is_adaptive or thinking):
        if effort not in VALID_EFFORT:
            log(
                f"Ignoring invalid effort {effort!r}; expected one of {VALID_EFFORT}",
                level="warning",
            )
        else:
            fields["output_config"] = {"effort": effort}

    return fields


# =============================================================================
# JOB TRACKING
# =============================================================================


class JobTrackingHook:
    """Mints the job row and stamps job/doc identifiers onto specialist tool calls.

    Job tracking uses a three-level hierarchy (see
    deployment/stacks/dynamodb_stack.py):

        doc_id  ->  job_id  ->  subtask_id

    ``doc_id`` is minted by the UI server at upload time and arrives on the
    request payload. ``subtask_id`` is derived by each specialist Lambda from its
    own name plus the page it analysed. This hook supplies the middle level.

    The agent is the only component that observes a tool invocation — the UI
    server merely proxies a WebSocket — so ``job_id`` is minted here, lazily, on
    the first specialist tool call of a turn. A conversational turn that calls no
    specialist therefore creates no job record at all.

    Identifiers are written directly into the tool input rather than requested of
    the model in the system prompt. The model would otherwise have to invent and
    then remember a UUID across turns, which is not something to depend on for
    the integrity of a tracking record.

    Stamping is limited to tools whose ``inputSchema`` declares ``job_id``, so
    non-specialist tools never receive parameters they do not accept.
    """

    # Warn once rather than per turn if the foundation module never made it into
    # the image. Tracking degrading silently is the failure mode worth shouting
    # about, since nothing else in the request path changes when it happens.
    _warned_unavailable = False

    def __init__(
        self,
        *,
        doc_id: str = "",
        session_id: str = "",
        actor_id: str = "local",
        user_name: str = "local",
    ) -> None:
        self.doc_id = doc_id
        self.session_id = session_id
        self.actor_id = actor_id
        self.user_name = user_name
        self.job_id = ""

        if job_state is None and not JobTrackingHook._warned_unavailable:
            JobTrackingHook._warned_unavailable = True
            log(
                "foundation.job_state is not importable — job tracking is DISABLED, "
                "so no job_id is minted and specialists that require it (such as "
                "html_report_specialist) will fail. Cause: "
                f"{_JOB_STATE_IMPORT_ERROR or 'unknown'}",
                level="warning",
            )

    def register_hooks(self, registry: Any, **_kwargs: Any) -> None:
        """Subscribe to the turn and tool-call lifecycle events."""
        from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent

        registry.add_callback(BeforeInvocationEvent, self._on_turn_start)
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)

    def _on_turn_start(self, _event: Any) -> None:
        """Clear the current job so each turn mints at most one new job."""
        self.job_id = ""

    @staticmethod
    def _declared_properties(tool: Any) -> dict[str, Any]:
        """Return declared input properties for Gateway or native tool schemas."""
        try:
            schema = tool.tool_spec.get("inputSchema") or {}
        except Exception:  # tool_spec is a property and may raise
            return {}
        return (schema.get("json") or schema).get("properties") or {}

    def _on_before_tool_call(self, event: Any) -> None:
        """Stamp job, document, and verified user identity into tool input."""
        if job_state is None:
            return
        properties = self._declared_properties(event.selected_tool)
        if "job_id" not in properties:
            return

        tool_name = event.tool_use.get("name", "unknown")

        if not self.job_id:
            self.job_id = uuid.uuid4().hex
            log(
                f"Minted job_id={self.job_id} "
                f"(doc_id={self.doc_id or 'none'}, first tool: {tool_name})"
            )
            # Writes are no-ops when JOBS_TABLE_NAME is unset and never raise, so
            # an untracked deployment still stamps identifiers harmlessly.
            job_state.create_job(
                self.job_id,
                doc_id=self.doc_id,
                session_id=self.session_id,
                reason=f"first specialist tool call: {tool_name}",
                owner_sub=self.actor_id,
                user_name=self.user_name,
            )

        # The executor reads tool_use back off the event after callbacks run, so
        # mutating the input in place is what reaches the Lambda.
        tool_input = event.tool_use.setdefault("input", {})
        tool_input["job_id"] = self.job_id
        if self.doc_id:
            tool_input["doc_id"] = self.doc_id
        if "user_id" in properties:
            tool_input["user_id"] = self.actor_id
        if "user_name" in properties:
            tool_input["user_name"] = self.user_name


def _s3_read_text(bucket: str, key: str) -> str:
    """Read an S3 object, revalidating a cached copy by ETag.

    An unchanged object costs one 304 round trip instead of a download, while an
    edit is still picked up on the very next call.
    """
    import boto3
    from botocore.exceptions import ClientError

    if _warm.s3_client is None:
        _warm.s3_client = boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION", "us-west-2")
        )

    cached = _warm.s3_cache.get(key)
    request: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if cached:
        request["IfNoneMatch"] = cached[0]
    try:
        response = _warm.s3_client.get_object(**request)
    except ClientError as e:
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if cached and status == 304:
            return cached[1]
        # Deleted or unreadable: drop the copy so a stale body is never served.
        _warm.s3_cache.pop(key, None)
        raise

    text = str(response["Body"].read().decode("utf-8"))
    _warm.s3_cache[key] = (str(response["ETag"]), text)
    return text


def _ensure_model(model_config: dict[str, Any]) -> Any:
    """Return a BedrockModel for this config, rebuilding only when the config changes."""
    from strands.models import BedrockModel

    key = json.dumps(model_config, sort_keys=True, default=str)
    if _warm.model is None or key != _warm.model_key:
        _warm.model = BedrockModel(
            model_id=model_config.get("model_id", DEFAULT_MODEL_CONFIG["model_id"]),
            region_name=os.environ.get("AWS_REGION", "us-west-2"),
            temperature=model_config.get("temperature", 1.0),
            max_tokens=model_config.get("max_tokens", 8000),
            additional_request_fields=_build_additional_request_fields(model_config),
        )
        _warm.model_key = key
    return _warm.model


def ensure_warm(gateway_url: str) -> WarmSnapshot:
    """Bring the warm state up to date and return what a turn needs.

    Blocking; call through asyncio.to_thread so the event loop (and the platform
    health check it serves) is not stalled while the Gateway session opens.
    """
    with _warm_lock:
        access_token = get_cognito_token()
        tools = _ensure_mcp_tools(gateway_url, access_token)
        system_prompt, model_config = load_config_from_s3()
        model = _ensure_model(model_config)
        return WarmSnapshot(
            system_prompt=system_prompt,
            model_config=model_config,
            model=model,
            tools=list(tools),
        )


def load_config_from_s3() -> tuple[str, dict[str, Any]]:
    """Load system prompt and model config from S3."""
    try:
        # Injected by the runtime stack. Previously read from the global SSM path
        # /badgers/config-bucket-name, which two deployments would fight over.
        bucket_name = os.environ.get("CONFIG_BUCKET_NAME", "")
        if not bucket_name:
            raise RuntimeError(
                "CONFIG_BUCKET_NAME is not set; cannot locate the config bucket."
            )

        system_prompt = _s3_read_text(
            bucket_name, "agent_system_prompt/agent_system_prompt.xml"
        )

        # Load operating environment config and inject into system prompt
        try:
            env_config = json.loads(
                _s3_read_text(
                    bucket_name, "agent_config/agent_operating_environment_config.json"
                )
            )
            env_value = env_config.get("operating_environment", "")
            if env_value:
                env_block = (
                    f"<operating_environment>{env_value}</operating_environment>\n\n"
                )
                system_prompt = env_block + system_prompt
                log("Loaded operating environment context")
        except Exception:
            logger.debug("No operating environment config found, continuing without it")

        model_config = DEFAULT_MODEL_CONFIG.copy()
        try:
            model_config.update(
                json.loads(
                    _s3_read_text(bucket_name, "agent_config/agent_model_config.json")
                )
            )
        except Exception:
            # Optional config file - use defaults if not found
            logger.debug("agent_model_config.json not found, using defaults")

        return system_prompt, model_config
    except Exception as e:
        log(f"Could not load config from S3: {e}")
        return DEFAULT_SYSTEM_PROMPT, DEFAULT_MODEL_CONFIG.copy()


# =============================================================================
# JSON SERIALIZATION HELPERS
# =============================================================================


def is_json_serializable(obj: Any) -> bool:
    """Check if an object is JSON serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def sanitize_event_for_json(obj: Any, max_depth: int = 10) -> Any:
    """Recursively sanitize an object to be JSON serializable."""
    if max_depth <= 0:
        return str(obj)

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, dict):
        return {
            k: sanitize_event_for_json(v, max_depth - 1)
            for k, v in obj.items()
            if isinstance(k, str) and not k.startswith("_")
        }

    if isinstance(obj, (list, tuple)):
        return [sanitize_event_for_json(item, max_depth - 1) for item in obj]

    # For objects with __dict__, extract serializable attributes
    if hasattr(obj, "__dict__"):
        return {
            k: sanitize_event_for_json(v, max_depth - 1)
            for k, v in obj.__dict__.items()
            if not k.startswith("_") and is_json_serializable(v)
        }

    # Fallback to string representation
    return str(obj)


# =============================================================================
# STREAMING AGENT INVOCATION
# =============================================================================


async def stream_agent_events(
    gateway_url: str,
    query: str,
    session_id: str,
    actor_id: str,
    user_name: str,
    runtime_session_id: str,
    doc_id: str = "",
) -> AsyncIterator[dict[str, Any]]:
    """Stream agent events as they occur.

    Yields events like:
    - {"type": "thinking", "text": "..."}
    - {"type": "tool_use", "name": "...", "input": {...}}
    - {"type": "tool_result", "name": "...", "result": "..."}
    - {"type": "text", "text": "..."}
    - {"type": "complete", "response": "..."}
    - {"type": "error", "message": "..."}
    """
    from strands import Agent

    log("Creating streaming agent...")

    # Token, Gateway session, tool list and model come from the warm state; after
    # a warmup frame this is a token check plus three ETag revalidations.
    warm = await asyncio.to_thread(ensure_warm, gateway_url)
    tools = warm.tools

    # Enhance system prompt with runtime session ID
    enhanced_system_prompt = f"""{warm.system_prompt}

RUNTIME SESSION ID: {runtime_session_id}
Include session_id: "{runtime_session_id}" in ALL tool calls."""

    log(f"Using {len(tools)} tools")
    yield {"type": "status", "message": f"Loaded {len(tools)} tools from Gateway"}

    # Configure session manager
    session_manager = None
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if memory_id:
        memory_config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
        )
        inner_manager = AgentCoreMemorySessionManager(
            agentcore_memory_config=memory_config,
            region_name=os.environ.get("AWS_REGION", "us-west-2"),
        )
        session_manager = SanitizingSessionManager(inner_manager)

    # Mints job_id on the first specialist tool call and stamps job_id/doc_id
    # into the tool input for every specialist invocation in this turn.
    job_hook = JobTrackingHook(
        doc_id=doc_id,
        session_id=session_id,
        actor_id=actor_id,
        user_name=user_name,
    )

    # Create agent
    agent = Agent(
        system_prompt=enhanced_system_prompt,
        name="PDFAnalysisAgent",
        tools=tools,
        model=warm.model,
        session_manager=session_manager,
        hooks=[job_hook],
        callback_handler=None,  # We handle events ourselves
    )

    log(f"Streaming agent response for query: {query[:100]}...")
    yield {"type": "status", "message": "Agent processing started"}

    # Stream the agent response
    final_response = ""
    announced_job_id = ""
    async for event in agent.stream_async(query):
        # Surface the job id once it exists so the client can correlate this
        # turn with its job record without polling for it.
        if job_hook.job_id and job_hook.job_id != announced_job_id:
            announced_job_id = job_hook.job_id
            yield {
                "type": "job",
                "job_id": job_hook.job_id,
                "doc_id": doc_id,
            }

        # Convert event to serializable dict
        if isinstance(event, dict):
            event_data = event
        elif hasattr(event, "__dict__"):
            event_data = {
                k: v
                for k, v in event.__dict__.items()
                if not k.startswith("_") and is_json_serializable(v)
            }
        else:
            event_data = {"raw": str(event)}

        # Handle Strands lifecycle events
        if event_data.get("init_event_loop"):
            yield {"init_event_loop": True}
            continue
        if event_data.get("start_event_loop"):
            yield {"start_event_loop": True}
            continue
        if event_data.get("start"):
            yield {"start": True}
            continue
        if event_data.get("complete"):
            yield {"complete": True, "response": final_response}
            continue
        if event_data.get("force_stop"):
            yield {
                "force_stop": True,
                "force_stop_reason": event_data.get("force_stop_reason", ""),
            }
            continue

        # Handle result event - extract only serializable parts
        if "result" in event_data:
            result = event_data["result"]
            if hasattr(result, "message"):
                final_response = (
                    str(result.message) if result.message else final_response
                )
            yield {"result": {"message": final_response}}
            continue

        # Handle text data
        if "data" in event_data:
            data = event_data["data"]
            if isinstance(data, str):
                final_response += data
                yield {"data": data}
            continue

        # Handle message events
        if "message" in event_data:
            msg = event_data["message"]
            if isinstance(msg, dict):
                yield {"message": msg}
            continue

        # Handle tool events
        if "current_tool_use" in event_data:
            tool_use = event_data["current_tool_use"]
            if isinstance(tool_use, dict):
                yield {
                    "current_tool_use": {
                        "name": tool_use.get("name"),
                        "toolUseId": tool_use.get("toolUseId"),
                        "input": tool_use.get("input", {}),
                    }
                }
            continue

        # Handle reasoning events
        if event_data.get("reasoning") or "reasoningText" in event_data:
            yield {
                "reasoning": True,
                "reasoningText": event_data.get("reasoningText", {}),
            }
            continue

        # Handle raw model events (nested in "event" key)
        if "event" in event_data:
            raw_event = event_data["event"]
            if isinstance(raw_event, dict):
                # Only pass through serializable model events
                yield {"event": sanitize_event_for_json(raw_event)}
            continue

        # Legacy event types
        if "reasoningContent" in event_data or "thinking" in str(event_data).lower():
            yield {"type": "thinking", "data": sanitize_event_for_json(event_data)}
        elif "toolUse" in event_data:
            tool_use = event_data.get("toolUse", {})
            yield {
                "type": "tool_use",
                "name": tool_use.get("name", "unknown"),
                "toolUseId": tool_use.get("toolUseId"),
                "input": tool_use.get("input", {}),
            }
        elif "toolResult" in event_data:
            tool_result = event_data.get("toolResult", {})
            yield {
                "type": "tool_result",
                "toolUseId": tool_result.get("toolUseId"),
                "content": tool_result.get("content", []),
            }
        elif "text" in event_data:
            text = event_data.get("text", "")
            final_response += text
            yield {"type": "text", "text": text}

    yield {"type": "complete", "response": final_response}


# =============================================================================
# MAIN ENTRYPOINT - STREAMING VERSION
# =============================================================================


@app.entrypoint
async def invoke(payload: dict[str, Any], context) -> AsyncIterator[dict[str, Any]]:
    """Async streaming entrypoint for AgentCore Runtime.

    Yields events as the agent processes, enabling real-time visibility
    of thinking, tool calls, and results in the frontend.
    """
    log("=" * 70)
    log("STREAMING INVOKE STARTED")
    log("=" * 70)

    runtime_session_id = context.session_id
    log(f"Runtime Session ID: {runtime_session_id}")

    # Extract request parameters
    query = "Hello!"
    session_id = f"session_{uuid.uuid4().hex}"
    actor_id = "default_user"
    user_name = "local"

    doc_id = ""

    if isinstance(payload, dict):
        query = str(payload.get("prompt", "Hello!"))
        session_id = str(payload.get("session_id") or f"session_{uuid.uuid4().hex}")
        actor_id = str(payload.get("actor_id", "default_user"))
        user_name = str(payload.get("user_name", "local"))
        # Top level of the job hierarchy, minted by the UI server at upload time.
        doc_id = str(payload.get("doc_id") or "")

    log(f"Session: {session_id}, Query: {query[:100]}...")

    with processing_context(session_id):
        try:
            gateway_url = os.environ.get("GATEWAY_URL")
            if not gateway_url:
                yield {"type": "error", "message": "GATEWAY_URL not set"}
                return

            yield {"type": "status", "message": "Connecting to Gateway..."}

            # Stream events from agent
            async for event in stream_agent_events(
                gateway_url=gateway_url,
                query=query,
                session_id=session_id,
                actor_id=actor_id,
                user_name=user_name,
                runtime_session_id=runtime_session_id,
                doc_id=doc_id,
            ):
                yield event

            log("STREAMING INVOKE COMPLETED")

        except Exception as e:
            log(f"Error: {e}", level="error")
            log(traceback.format_exc(), level="error")
            yield {
                "type": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }


@app.websocket
async def websocket_handler(websocket, context) -> None:
    """WebSocket handler for real-time streaming.

    Handles bidirectional WebSocket communication for streaming agent responses.
    """
    from starlette.websockets import WebSocket

    log("=" * 70)
    log("WEBSOCKET CONNECTION STARTED")
    log("=" * 70)

    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            log(f"Received WebSocket message: {json.dumps(data)[:200]}...")

            # Sent by the UI server when the chat mounts, on the same runtime session
            # id the prompts will use. It allocates this microVM and fills the warm
            # state; it never reaches the model.
            if data.get("warmup"):
                started = time.monotonic()
                try:
                    gateway_url = os.environ.get("GATEWAY_URL")
                    if not gateway_url:
                        raise RuntimeError("GATEWAY_URL not set")
                    warm = await asyncio.to_thread(ensure_warm, gateway_url)
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    log(f"Warmup complete: {len(warm.tools)} tools in {elapsed_ms}ms")
                    await websocket.send_json(
                        {
                            "type": "warm",
                            "tools": len(warm.tools),
                            "elapsed_ms": elapsed_ms,
                        }
                    )
                except Exception as e:
                    log(f"Warmup failed: {e}", level="error")
                    await websocket.send_json({"type": "warm_error", "message": str(e)})
                continue

            # Extract request parameters
            query = data.get("prompt", "Hello!")
            session_id = data.get("session_id") or f"session_{uuid.uuid4().hex}"
            actor_id = data.get("actor_id", "default_user")
            user_name = data.get("user_name", "local")
            runtime_session_id = context.session_id or f"ws-{uuid.uuid4().hex}"
            # Top level of the job hierarchy, minted by the UI server at upload time.
            doc_id = str(data.get("doc_id") or "")

            log(f"Session: {session_id}, Query: {query[:100]}...")

            with processing_context(session_id):
                try:
                    gateway_url = os.environ.get("GATEWAY_URL")
                    if not gateway_url:
                        await websocket.send_json(
                            {"type": "error", "message": "GATEWAY_URL not set"}
                        )
                        continue

                    await websocket.send_json(
                        {"type": "status", "message": "Connecting to Gateway..."}
                    )

                    # Stream events from agent
                    async for event in stream_agent_events(
                        gateway_url=gateway_url,
                        query=query,
                        session_id=session_id,
                        actor_id=actor_id,
                        user_name=user_name,
                        runtime_session_id=runtime_session_id,
                        doc_id=doc_id,
                    ):
                        await websocket.send_json(event)

                    log("WEBSOCKET STREAMING COMPLETED")

                except Exception as e:
                    log(f"Error: {e}", level="error")
                    log(traceback.format_exc(), level="error")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )

    except Exception as e:
        log(f"WebSocket connection closed: {e}", level="info")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    app.run()
