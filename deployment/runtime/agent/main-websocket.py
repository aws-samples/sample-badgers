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
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional

from bedrock_agentcore.runtime import BedrockAgentCoreApp, PingStatus
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

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
    """Get OAuth token from Cognito for Gateway authentication."""
    import httpx

    credentials = get_cognito_credentials()
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
    if response.status_code != 200:
        raise ValueError(f"Failed to get Cognito token: {response.status_code}")
    return response.json()["access_token"]


# =============================================================================
# MCP TRANSPORT
# =============================================================================


def create_mcp_transport(gateway_url: str, access_token: str) -> Any:
    """Create MCP transport for AgentCore Gateway connection."""
    from mcp.client.streamable_http import streamablehttp_client

    return streamablehttp_client(
        gateway_url, headers={"Authorization": f"Bearer {access_token}"}
    )


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
    "model_id": "global.anthropic.claude-opus-4-6-v1",
    "temperature": 1.0,
    "max_tokens": 16000,
    "thinking": {"type": "adaptive"},
    "fallback_models": [
        {
            "model_id": "global.anthropic.claude-opus-4-5-20251101-v1:0",
            "thinking": {"type": "enabled", "budget_tokens": 8192},
        },
        {
            "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "thinking": {"type": "enabled", "budget_tokens": 8192},
        },
    ],
}

DEFAULT_SYSTEM_PROMPT = """You are an intelligent PDF analysis assistant with access to specialized tools via AgentCore Gateway."""


def load_config_from_s3() -> tuple[str, dict[str, Any]]:
    """Load system prompt and model config from S3."""
    import boto3

    try:
        region = os.environ.get("AWS_REGION", "us-west-2")
        ssm = boto3.client("ssm", region_name=region)
        bucket_name = ssm.get_parameter(Name="/badgers/config-bucket-name")[
            "Parameter"
        ]["Value"]

        s3 = boto3.client("s3", region_name=region)
        response = s3.get_object(
            Bucket=bucket_name, Key="agent_system_prompt/agent_system_prompt.xml"
        )
        system_prompt = response["Body"].read().decode("utf-8")

        model_config = DEFAULT_MODEL_CONFIG.copy()
        try:
            response = s3.get_object(
                Bucket=bucket_name, Key="agent_config/agent_model_config.json"
            )
            model_config.update(json.loads(response["Body"].read().decode("utf-8")))
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
    access_token: str,
    query: str,
    session_id: str,
    actor_id: str,
    runtime_session_id: str,
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
    from strands.models import BedrockModel
    from strands.tools.mcp.mcp_client import MCPClient

    log("Creating streaming agent...")

    system_prompt, model_config = load_config_from_s3()

    # Enhance system prompt with runtime session ID
    enhanced_system_prompt = f"""{system_prompt}

RUNTIME SESSION ID: {runtime_session_id}
Include session_id: "{runtime_session_id}" in ALL tool calls."""

    model = BedrockModel(
        model_id=model_config.get("model_id", DEFAULT_MODEL_CONFIG["model_id"]),
        region_name=os.environ.get("AWS_REGION", "us-west-2"),
        temperature=model_config.get("temperature", 1.0),
        max_tokens=model_config.get("max_tokens", 8000),
        additional_request_fields={"thinking": model_config.get("thinking", {})},
    )

    mcp_client = MCPClient(lambda: create_mcp_transport(gateway_url, access_token))

    with mcp_client:
        # Fetch tools
        tools = []
        pagination_token = None
        while True:
            result = mcp_client.list_tools_sync(pagination_token=pagination_token)
            tools.extend(result)
            if hasattr(result, "pagination_token") and result.pagination_token:
                pagination_token = result.pagination_token
            else:
                break

        log(f"Fetched {len(tools)} tools")
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

        # Create agent
        agent = Agent(
            system_prompt=enhanced_system_prompt,
            name="PDFAnalysisAgent",
            tools=tools,
            model=model,
            session_manager=session_manager,
            callback_handler=None,  # We handle events ourselves
        )

        log(f"Streaming agent response for query: {query[:100]}...")
        yield {"type": "status", "message": "Agent processing started"}

        # Stream the agent response
        final_response = ""
        async for event in agent.stream_async(query):
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
            if (
                "reasoningContent" in event_data
                or "thinking" in str(event_data).lower()
            ):
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

    if isinstance(payload, dict):
        query = str(payload.get("prompt", "Hello!"))
        session_id = str(payload.get("session_id") or f"session_{uuid.uuid4().hex}")
        actor_id = str(payload.get("actor_id", "default_user"))

    log(f"Session: {session_id}, Query: {query[:100]}...")

    with processing_context(session_id):
        try:
            gateway_url = os.environ.get("GATEWAY_URL")
            if not gateway_url:
                yield {"type": "error", "message": "GATEWAY_URL not set"}
                return

            yield {"type": "status", "message": "Obtaining authentication token..."}
            access_token = get_cognito_token()

            yield {"type": "status", "message": "Connecting to Gateway..."}

            # Stream events from agent
            async for event in stream_agent_events(
                gateway_url=gateway_url,
                access_token=access_token,
                query=query,
                session_id=session_id,
                actor_id=actor_id,
                runtime_session_id=runtime_session_id,
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

            # Extract request parameters
            query = data.get("prompt", "Hello!")
            session_id = data.get("session_id") or f"session_{uuid.uuid4().hex}"
            actor_id = data.get("actor_id", "default_user")
            runtime_session_id = context.session_id or f"ws-{uuid.uuid4().hex}"

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
                        {
                            "type": "status",
                            "message": "Obtaining authentication token...",
                        }
                    )
                    access_token = get_cognito_token()

                    await websocket.send_json(
                        {"type": "status", "message": "Connecting to Gateway..."}
                    )

                    # Stream events from agent
                    async for event in stream_agent_events(
                        gateway_url=gateway_url,
                        access_token=access_token,
                        query=query,
                        session_id=session_id,
                        actor_id=actor_id,
                        runtime_session_id=runtime_session_id,
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
