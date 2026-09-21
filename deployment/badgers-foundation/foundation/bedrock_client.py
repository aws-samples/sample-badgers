"""
AWS Bedrock client management for specialist system.

Handles Bedrock client creation, invocation, and error handling for different specialist types.
"""

import base64
import binascii
import json
import logging
import os
import time
from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import boto3
from botocore.exceptions import ClientError

print("BEDROCK_CLIENT MODULE LOADING - TOP OF FILE")


# Model ID -> application inference profile ARN, read once per container from the SSM
# parameter that InferenceProfilesStack writes in the same loop that creates the profiles.
#
# This replaces MODEL_TO_PROFILE_ENV_MAP, a hand-maintained dict of eight model-ID spellings
# pointing at five per-model environment variables. That map covered only four models, so it
# had two failure modes at once: a model absent from it silently lost cost attribution even
# when its profile existed (Claude Sonnet 4.6 was in exactly that state — profile created,
# environment variable set by CDK, and no entry here to read it), and adding a model meant
# editing CDK wiring in four stacks plus this dict.
#
# One parameter replaces all of it. Because the parameter is written in the same loop that
# creates the profiles, it cannot name a model that has no profile.
_MODEL_PROFILES_PARAM_ENV = "MODEL_PROFILES_PARAM"

# Module-level cache. Populated on first lookup and reused for the life of the container, so
# a warm Lambda makes no further SSM calls.
_profile_map_cache: Optional[Dict[str, str]] = None


def _load_profile_map() -> Dict[str, str]:
    """Read and cache the model ID -> profile ARN map from SSM.

    Returns an empty dict on any failure and caches that too, so a missing parameter or a
    denied read costs one call rather than one per invocation. Failure is non-fatal by
    design: without the map, `_invoke_single_model` falls through to the model ID itself,
    which is a valid geo inference ID. That loses cost attribution but still works — a hard
    failure here would be worse than the problem being solved.
    """
    global _profile_map_cache
    if _profile_map_cache is not None:
        return _profile_map_cache

    logger = logging.getLogger(__name__)
    param_name = os.environ.get(_MODEL_PROFILES_PARAM_ENV)
    if not param_name:
        logger.warning(
            "%s is not set; inference profile ARNs unavailable and cost attribution "
            "will be lost",
            _MODEL_PROFILES_PARAM_ENV,
        )
        _profile_map_cache = {}
        return _profile_map_cache

    try:
        ssm = boto3.client("ssm")
        raw = ssm.get_parameter(Name=param_name)["Parameter"]["Value"]
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError(f"expected a JSON object, got {type(loaded).__name__}")
        _profile_map_cache = loaded
        logger.info("Loaded %d inference profile ARNs from %s", len(loaded), param_name)
    except Exception as e:  # noqa: BLE001 - any failure degrades to no attribution
        logger.warning(
            "Could not read inference profile map from %s: %s. Cost attribution will be "
            "lost; invocations continue using geo model IDs.",
            param_name,
            e,
        )
        _profile_map_cache = {}

    return _profile_map_cache


def clear_profile_map_cache() -> None:
    """Drop the cached profile map. For tests."""
    global _profile_map_cache
    _profile_map_cache = None


def get_inference_profile_arn(model_id: str) -> Optional[str]:
    """
    Get the application inference profile ARN for a model ID, if one is provisioned.

    Args:
        model_id: The Bedrock model ID, as named in the manifest (a `us.`-prefixed geo ID)

    Returns:
        Inference profile ARN if the deployment provisioned one, None otherwise.

    Note:
        Reads the SSM map on first call and caches it for the container's lifetime. Returning
        None is not an error — the caller invokes the geo model ID directly, which works but
        is not attributed to a profile.
    """
    return _load_profile_map().get(model_id)


def get_default_aws_profile() -> Optional[str]:
    """
    Get the default AWS profile from server config.

    Returns:
        AWS profile name from config, or None if not available
    """
    try:
        from config.config import ServerConfig

        server_config = ServerConfig.from_env()
        return str(server_config.aws_profile)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Could not load AWS profile from config: %s", e
        )
        return None


class BedrockError(Exception):
    """Raised when Bedrock operations fail."""


def get_model_family(model_id: str) -> str:
    """
    Detect model family from model ID string.

    Args:
        model_id: The Bedrock model ID

    Returns:
        'claude', 'nova', 'openai', 'kimi', or 'mistral'

    Raises:
        BedrockError: If model family cannot be determined

    Note:
        Under Converse the family no longer selects a request *shape* — Converse normalises
        that. It selects only which provider-specific fields go into
        ``additionalModelRequestFields``, which is why 'openai' — and, like it, 'kimi' and
        'mistral' — can be families that add nothing at all.
    """
    model_lower = model_id.lower()

    if "anthropic" in model_lower or "claude" in model_lower:
        return "claude"
    elif "nova" in model_lower or "amazon.nova" in model_lower:
        return "nova"
    elif "openai" in model_lower or "gpt-" in model_lower:
        return "openai"
    elif "moonshotai" in model_lower or "kimi" in model_lower:
        return "kimi"
    elif "mistral" in model_lower or "pixtral" in model_lower:
        return "mistral"
    else:
        raise BedrockError(f"Unknown model family for model ID: {model_id}")


def thinking_default_on(model_id: str) -> bool:
    """Whether the model reasons unless the request explicitly turns it off.

    Mirrors the ``thinking_default_on`` flag in ``s3_files/config/model_registry.json``.
    The registry is read by CDK at synth and by the UI at request time, but is not shipped
    in the Lambda layer, so this is the one place its content is repeated in runtime code.
    It exists for a single consequence: Claude requires ``temperature`` to be 1 whenever
    thinking is on, and on a default-on model a request that asks for no thinking is still
    a thinking request. Without this, a wizard-created specialist on such a model inherits
    the specialist default temperature of 0.1 and every call is rejected.

    Claude Opus 5 is the one such model in the current set -- its model card reads
    "adaptive thinking is on by default; can be disabled". Add here whenever the registry
    flag is set on a new entry.
    """
    return "claude-opus-5" in model_id.lower()


class BedrockClient:
    """Manages Bedrock client creation and invocation."""

    def __init__(self, throttle_delay: float = 1.0, aws_region: Optional[str] = None):
        """
        Initialize the Bedrock client manager.

        Args:
            throttle_delay: Delay in seconds for throttling retry
            aws_region: AWS region for Bedrock client
        """
        self.throttle_delay = throttle_delay
        self.aws_region = aws_region
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=4)
    def get_client(self, profile_name: Optional[str] = None):
        """
        Get or create boto3 bedrock client with caching.

        Args:
            profile_name: Optional AWS profile name

        Returns:
            boto3 bedrock-runtime client

        Raises:
            BedrockError: If client creation fails
        """
        try:
            # If no profile provided, try to get from config
            if not profile_name:
                profile_name = get_default_aws_profile()

            # Check if we're in Lambda (AWS_EXECUTION_ENV is set)
            import os

            in_lambda = "AWS_EXECUTION_ENV" in os.environ

            if profile_name and not in_lambda:
                session = boto3.Session(profile_name=profile_name)
                self.logger.info("Using AWS profile: %s", profile_name)
            else:
                session = boto3.Session()
                if in_lambda:
                    self.logger.info("Using Lambda execution role credentials")
                else:
                    self.logger.info("Using default AWS credentials")

            # Use configured region or default
            region = self.aws_region or "us-west-2"

            # Configure timeouts for large response streaming
            read_timeout = int(os.environ.get("BEDROCK_READ_TIMEOUT", "900"))
            connect_timeout = int(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "30"))

            from botocore.config import Config

            bedrock_config = Config(
                read_timeout=read_timeout,
                connect_timeout=connect_timeout,
                retries={"max_attempts": 0},
            )

            client = session.client(
                "bedrock-runtime", region_name=region, config=bedrock_config
            )

            return client

        except Exception as e:
            raise BedrockError(f"Failed to create Bedrock client: {e}") from e

    def invoke_model(
        self,
        model_id: str,
        payload: Dict[str, Any],
        profile_name: Optional[str] = None,
        fallback_list: Optional[list] = None,
        max_retries: int = 3,
        extended_thinking: bool = False,
        budget_tokens: Optional[int] = None,
        adaptive_thinking: bool = False,
        adaptive_effort: str = "high",
    ) -> Dict[str, Any]:
        """
        Invoke a Bedrock model with the given payload, with fallback support.

        Args:
            model_id: The primary model ID to invoke
            payload: Request payload for the model (will be auto-converted for model family)
            profile_name: Optional AWS profile name
            fallback_list: Optional list of model configs to try on failure
                          Each item can be a string (model_id) or dict with model_id, extended_thinking, budget_tokens, adaptive_thinking, effort
            max_retries: Maximum retry attempts for throttling (default: 3)
            extended_thinking: Whether to enable extended thinking for primary model
            budget_tokens: Optional budget tokens for extended thinking on primary model
            adaptive_thinking: Whether to enable adaptive thinking for primary model (Claude only)
            adaptive_effort: Effort level for adaptive thinking ("low", "medium", "high")

        Returns:
            Response dictionary from Bedrock (includes 'thinking' key if extended/adaptive thinking enabled)

        Raises:
            BedrockError: If model invocation fails (including all fallbacks)
        """
        client = self.get_client(profile_name)

        # Build the full model chain with thinking settings
        # Each entry is (model_id, extended_thinking, budget_tokens, adaptive_thinking, effort)
        model_chain = [
            (
                model_id,
                extended_thinking,
                budget_tokens,
                adaptive_thinking,
                adaptive_effort,
            )
        ]
        if fallback_list:
            for fallback in fallback_list:
                if isinstance(fallback, dict):
                    fb_model_id = fallback.get("model_id")
                    if not isinstance(fb_model_id, str):
                        self.logger.warning(
                            "Skipping fallback with invalid model_id: %s", fb_model_id
                        )
                        continue
                    fb_extended_thinking = bool(
                        fallback.get("extended_thinking", False)
                    )
                    fb_budget_tokens: Optional[int] = fallback.get("budget_tokens")
                    fb_adaptive_thinking = bool(
                        fallback.get("adaptive_thinking", False)
                    )
                    fb_effort = fallback.get("effort", "high")
                    model_chain.append(
                        (
                            fb_model_id,
                            fb_extended_thinking,
                            fb_budget_tokens,
                            fb_adaptive_thinking,
                            fb_effort,
                        )
                    )
                else:
                    # Legacy format: just model_id string
                    model_chain.append((fallback, False, None, False, "high"))

        last_error = None

        last_index = len(model_chain) - 1

        for chain_index, (
            current_model_id,
            current_extended_thinking,
            current_budget_tokens,
            current_adaptive_thinking,
            current_effort,
        ) in enumerate(model_chain):
            try:
                self.logger.info("current_model_id: %s", current_model_id)
                self.logger.info("payload: %s", payload)
                self.logger.info(
                    "current_extended_thinking: %s", current_extended_thinking
                )
                self.logger.info("current_budget_tokens: %s", current_budget_tokens)
                self.logger.info(
                    "current_adaptive_thinking: %s", current_adaptive_thinking
                )

                result = self._invoke_single_model(
                    client,
                    current_model_id,
                    payload,
                    max_retries,
                    current_extended_thinking,
                    current_budget_tokens,
                    current_adaptive_thinking,
                    current_effort,
                )
                return result
            except BedrockError as e:
                last_error = e
                # Compare by position, not by value. This previously tested the current
                # 5-tuple against model_chain[-1], so any earlier entry that happened to
                # equal the last one — a repeated model ID with the same thinking settings —
                # reported itself as last, and the loop raised instead of continuing. The
                # remaining fallbacks were silently skipped.
                is_last = chain_index == last_index
                self.logger.info("last error is: %s", last_error)
                # Only fallback for specific transient errors (throttling, service unavailable)
                # For other errors (access denied, validation, etc.), fail immediately
                if self._should_fallback(e) and not is_last:
                    self.logger.warning(
                        "Model %s failed with fallback-eligible error: %s. Trying next model...",
                        current_model_id,
                        str(e),
                    )
                    continue
                else:
                    # Non-fallback-eligible error OR last model - raise immediately
                    self.logger.error(
                        "Model %s failed with non-recoverable error: %s",
                        current_model_id,
                        str(e),
                    )
                    raise

        raise BedrockError(
            f"All models in chain failed. Last error: {last_error}"
        ) from last_error

    def _invoke_single_model(
        self,
        client,
        model_id: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
        extended_thinking: bool = False,
        budget_tokens: Optional[int] = None,
        adaptive_thinking: bool = False,
        adaptive_effort: str = "high",
    ) -> Dict[str, Any]:
        """
        Invoke a single Bedrock model.

        Args:
            client: Bedrock client
            model_id: Model ID to invoke
            payload: Base payload (will be converted to model-specific format)
            max_retries: Maximum retry attempts for throttling
            extended_thinking: Whether to enable extended thinking (Claude only)
            budget_tokens: Optional budget tokens for extended thinking
            adaptive_thinking: Whether to enable adaptive thinking (Claude only)
            adaptive_effort: Effort level for adaptive thinking

        Returns:
            Normalized response dictionary (includes 'thinking' key if extended/adaptive thinking enabled)

        Raises:
            BedrockError: If invocation fails
        """
        try:
            model_family = get_model_family(model_id)

            # Check for inference profile ARN - use it instead of model_id for cost tracking
            invoke_model_id = model_id
            profile_arn = get_inference_profile_arn(model_id)
            if profile_arn:
                invoke_model_id = profile_arn
                self.logger.info(
                    "Using inference profile ARN for model %s: %s",
                    model_id,
                    profile_arn,
                )
            else:
                self.logger.info(
                    "No inference profile configured for %s, using model ID directly",
                    model_id,
                )

            self.logger.info(
                "Invoking model: %s (family: %s, extended_thinking: %s, adaptive_thinking: %s)",
                invoke_model_id,
                model_family,
                extended_thinking,
                adaptive_thinking,
            )

            # Build the Converse request. Converse is used for every model, not InvokeModel:
            # it is the only API all eight supported models share — the four OpenAI models
            # support no Invoke API at all — and application inference profiles work with
            # Converse only, which is what preserves cost attribution.
            request = self._build_converse_request(
                payload,
                model_family,
                extended_thinking,
                budget_tokens,
                adaptive_thinking,
                adaptive_effort,
                default_on_thinking=thinking_default_on(model_id),
            )

            # Add throttling delay to prevent rate limiting
            time.sleep(self.throttle_delay)

            response = self.handle_throttling(
                client.converse,
                modelId=invoke_model_id,
                max_retries=max_retries,
                **request,
            )

            # Converse returns a parsed dict — no streaming body to read and no JSON to
            # decode, which is why _read_streaming_body is no longer on this path.
            normalized = self._normalize_response(response, model_family)

            self.logger.info("Model invocation successful")
            return normalized

        except BedrockError:
            raise
        except Exception as e:
            raise BedrockError(f"Model invocation failed: {e}") from e

    # Anthropic media types -> Converse image formats. Converse names the format
    # separately from the bytes, where the Anthropic body carried a MIME type.
    _IMAGE_FORMATS = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }

    def _to_content_blocks(self, content: Any) -> list:
        """Convert Anthropic-style message content into Converse ContentBlocks.

        Accepts either a bare string (correlation_specialist passes one) or a list of
        Anthropic blocks, and returns Converse blocks:

            {"type": "text",  "text": t}   ->  {"text": t}
            {"type": "image", "source": {"media_type": m, "data": b64}}
                                           ->  {"image": {"format": f,
                                                          "source": {"bytes": raw}}}

        The image case is the one that matters. ``ImageSource.bytes`` is a blob, and the
        API reference is explicit: *"If you use an AWS SDK, you don't need to encode the
        image bytes in base64."* boto3 base64-encodes whatever it is handed, so forwarding
        the base64 **string** that message_chain_builder.py produces would encode it twice
        and the model would receive garbage. That string is correct for invoke_model, where
        the body is hand-built JSON — which is exactly why this bug is invisible until the
        transport changes.
        """
        if isinstance(content, str):
            return [{"text": content}]

        if not isinstance(content, list):
            raise BedrockError(
                f"Message content must be a string or a list, got {type(content).__name__}"
            )

        blocks = []
        for index, item in enumerate(content):
            if isinstance(item, str):
                blocks.append({"text": item})
                continue

            if not isinstance(item, dict):
                raise BedrockError(
                    f"Content block {index} must be a string or an object, "
                    f"got {type(item).__name__}"
                )

            # Already a Converse block — pass through untouched.
            if "text" in item and "type" not in item:
                blocks.append(item)
                continue
            if "image" in item and "type" not in item:
                blocks.append(item)
                continue

            block_type = item.get("type")

            if block_type == "text":
                blocks.append({"text": item.get("text", "")})

            elif block_type == "image":
                source = item.get("source", {}) or {}
                media_type = source.get("media_type", "image/png")
                image_format = self._IMAGE_FORMATS.get(media_type)
                if image_format is None:
                    raise BedrockError(
                        f"Content block {index}: unsupported image media type "
                        f"{media_type!r}. Supported: {sorted(self._IMAGE_FORMATS)}"
                    )

                data = source.get("data", "")
                if isinstance(data, bytes):
                    raw = data
                else:
                    try:
                        raw = base64.b64decode(data, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise BedrockError(
                            f"Content block {index}: image data is not valid base64 — "
                            f"{exc}"
                        ) from exc
                if not raw:
                    raise BedrockError(f"Content block {index}: image data is empty")

                blocks.append(
                    {"image": {"format": image_format, "source": {"bytes": raw}}}
                )

            else:
                raise BedrockError(
                    f"Content block {index}: unsupported block type {block_type!r}"
                )

        if not blocks:
            raise BedrockError("Message content produced no Converse content blocks")

        return blocks

    def _thinking_fields(
        self,
        model_family: str,
        extended_thinking: bool,
        budget_tokens: Optional[int],
        adaptive_thinking: bool,
        adaptive_effort: str,
    ) -> Dict[str, Any]:
        """Build the provider-specific reasoning fields for additionalModelRequestFields.

        Converse normalises the request shape but not reasoning configuration, so this is
        the one place that still branches on provider.

        Claude: ``thinking``, plus ``effort`` inside a **separate** ``output_config``
        object. Putting ``effort`` inside ``thinking`` raises ValidationException.

        Nova 2: ``reasoningConfig`` with ``type`` and ``maxReasoningEffort``. There is no
        ``budget_tokens`` analogue — Nova expresses depth as three named levels — so a
        budget-based request is mapped to a level and the loss is logged rather than
        silently absorbed.

        OpenAI: nothing. The four GPT model cards document no reasoning parameter at all
        (unlike every Anthropic card, which has an explicit ``Reasoning:`` field), and an
        unrecognised key in additionalModelRequestFields earns a ValidationException.
        """
        if not (extended_thinking or adaptive_thinking):
            return {}

        if model_family == "claude":
            if adaptive_thinking:
                return {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": adaptive_effort},
                }
            fields: Dict[str, Any] = {"thinking": {"type": "enabled"}}
            if budget_tokens:
                fields["thinking"]["budget_tokens"] = budget_tokens
            return fields

        if model_family == "nova":
            # R8: the previous code logged "not supported for Nova models" and dropped the
            # request. That was true of Nova Premier and is false of Nova 2 Lite, which is
            # the fallback floor beneath thinking-enabled specialists.
            if adaptive_thinking:
                effort = adaptive_effort
            else:
                effort = "medium"
                self.logger.info(
                    "Nova has no budget_tokens equivalent (requested %s); using "
                    "maxReasoningEffort=medium",
                    budget_tokens,
                )
            if effort not in ("low", "medium", "high"):
                self.logger.warning(
                    "Nova maxReasoningEffort must be low/medium/high, got %r; using medium",
                    effort,
                )
                effort = "medium"
            return {
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}
            }

        if model_family in ("openai", "kimi", "mistral"):
            # None of these expose a reasoning parameter through Converse: the OpenAI,
            # Kimi K3, and Pixtral Large model cards document none for bedrock-runtime, and
            # an unrecognised key in additionalModelRequestFields earns a
            # ValidationException. Their registry entries set thinking=null, so this is
            # only reached if a specialist config asks for thinking anyway; drop it.
            self.logger.info(
                "Thinking requested for a %s model; its model card documents no Converse "
                "reasoning parameter, so none is sent",
                model_family,
            )
            return {}

        raise BedrockError(f"Unknown model family: {model_family}")

    def _build_converse_request(
        self,
        payload: Dict[str, Any],
        model_family: str,
        extended_thinking: bool = False,
        budget_tokens: Optional[int] = None,
        adaptive_thinking: bool = False,
        adaptive_effort: str = "high",
        default_on_thinking: bool = False,
    ) -> Dict[str, Any]:
        """Turn the internal payload into keyword arguments for ``client.converse``.

        ``payload`` is the Anthropic-shaped dict that ``create_anthropic_payload`` still
        produces. Under Converse it is no longer a wire format — it is an internal
        intermediate representation, so its callers did not have to change.

        ``default_on_thinking`` is ``thinking_default_on(model_id)``: the model reasons
        unless explicitly told not to, which matters only for the temperature rule below.

        Returns a dict suitable for ``client.converse(modelId=..., **request)``.
        """
        messages = []
        for index, message in enumerate(payload.get("messages") or []):
            role = message.get("role")
            if role not in ("user", "assistant"):
                raise BedrockError(f"Message {index} has invalid role: {role!r}")
            messages.append(
                {
                    "role": role,
                    "content": self._to_content_blocks(message.get("content")),
                }
            )

        if not messages:
            raise BedrockError("Converse requires at least one message")

        request: Dict[str, Any] = {"messages": messages}

        system_prompt = payload.get("system")
        if system_prompt:
            request["system"] = [{"text": system_prompt}]

        inference_config: Dict[str, Any] = {}
        max_tokens = payload.get("max_tokens")
        temperature = payload.get("temperature")

        # Claude requires temperature 1 whenever thinking is on -- extended (type
        # "enabled") and adaptive alike. The pre-Converse code forced this in both
        # _add_extended_thinking_to_payload and _add_adaptive_thinking_to_payload; the first
        # Converse draft carried over only the adaptive case, which would have failed the
        # correlation specialist's extended-thinking fallback with a ValidationException
        # that _should_fallback correctly refuses to retry.
        #
        # A default-on model (Opus 5) thinks unless told not to, so it is a thinking request
        # even when the manifest asked for nothing -- hence default_on_thinking.
        if model_family == "claude" and (
            extended_thinking or adaptive_thinking or default_on_thinking
        ):
            temperature = 1

        nova_high_effort = (
            model_family == "nova"
            and (extended_thinking or adaptive_thinking)
            and adaptive_thinking
            and adaptive_effort == "high"
        )
        if nova_high_effort:
            # Nova 2 rejects temperature, topP, topK and maxTokens together with
            # maxReasoningEffort="high". Omitting maxTokens removes the output bound, and
            # the docs warn output can reach 128K tokens, so this is logged loudly rather
            # than applied quietly.
            self.logger.warning(
                "Nova maxReasoningEffort=high requires temperature and maxTokens to be "
                "unset; omitting both. Output is unbounded and may reach 128K tokens."
            )
        else:
            if max_tokens:
                inference_config["maxTokens"] = max_tokens
            if temperature is not None:
                inference_config["temperature"] = temperature

        if inference_config:
            request["inferenceConfig"] = inference_config

        extra = self._thinking_fields(
            model_family,
            extended_thinking,
            budget_tokens,
            adaptive_thinking,
            adaptive_effort,
        )
        if extra:
            request["additionalModelRequestFields"] = extra

        return request

    def _normalize_response(
        self, response: Dict[str, Any], model_family: str
    ) -> Dict[str, Any]:
        """
        Normalize a Converse response to the internal format callers already expect.

        Converse returns one response shape for every provider, so this no longer branches
        on family — the parameter is kept only for logging. The previous version had a
        Claude branch that read a native ``content`` array and a Nova branch that walked
        ``output.message.content``; both collapse into the single path below.

        Args:
            response: Raw Converse response
            model_family: 'claude', 'nova', or 'openai' — logging only

        Returns:
            ``{"content": [{"type": "text", "text": ...}], ...}`` with an optional
            ``thinking`` key, matching what response_processor and the specialists parse.
        """
        try:
            blocks = response["output"]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BedrockError(
                f"Invalid Converse response structure ({model_family}): {exc}"
            ) from exc

        text_blocks = []
        reasoning_parts = []
        for item in blocks:
            if not isinstance(item, dict):
                continue
            if "text" in item:
                text_blocks.append({"type": "text", "text": item["text"]})
            elif "reasoningContent" in item:
                # Claude and Nova 2 both return reasoning here. Nova 2 redacts the text to
                # "[REDACTED]" while still billing for the tokens, so an empty or redacted
                # value is expected rather than an error.
                reasoning_text = (
                    (item.get("reasoningContent") or {}).get("reasoningText") or {}
                ).get("text")
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)

        if not text_blocks:
            raise BedrockError(
                f"Converse returned no text content ({model_family}); "
                f"stopReason={response.get('stopReason')!r}"
            )

        result: Dict[str, Any] = {"content": text_blocks}

        if reasoning_parts:
            result["thinking"] = "\n\n".join(reasoning_parts)
            self.logger.info(
                "Extracted reasoning content: %d characters", len(result["thinking"])
            )

        # Carry through the metadata callers may want, under the names Converse uses.
        if "stopReason" in response:
            result["stopReason"] = response["stopReason"]
        if "usage" in response:
            result["usage"] = response["usage"]

        return result

    def _should_fallback(self, error: BedrockError) -> bool:
        """
        Determine if an error should trigger fallback to alternate model.

        Only transient/availability errors should trigger fallback.
        Errors like AccessDenied, ValidationException should NOT fallback
        as they indicate configuration problems, not transient issues.

        Args:
            error: The BedrockError that occurred

        Returns:
            True if fallback should be attempted
        """
        error_str = str(error).lower()

        # Only fallback for transient/availability errors
        fallback_triggers = [
            "serviceunavailable",
            "service unavailable",
            "throttlingexception",
            "throttling",
            "modelnotreadyexception",
            "model not ready",
            "resourcenotfoundexception",  # Model doesn't exist in region
        ]
        return any(trigger in error_str for trigger in fallback_triggers)

    def handle_throttling(
        self, func: Callable, *args, max_retries: int = 3, **kwargs
    ) -> Any:
        """
        Handle throttling exceptions with retry logic.

        Args:
            func: Function to call with throttling handling
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            BedrockError: If function fails after retry
        """
        base_delay = self.throttle_delay

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                if error_code == "ThrottlingException" and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    self.logger.warning(
                        "Throttling detected (attempt %d/%d), retrying in %.1fs...",
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Re-raise if not throttling or max retries reached
                    raise BedrockError(f"Bedrock API error: {e}") from e

            except Exception as e:
                # For non-ClientError exceptions, don't retry
                raise BedrockError(
                    f"Unexpected error during model invocation: {e}"
                ) from e

        # This should never be reached, but just in case
        raise BedrockError("Max retries exceeded")

    def validate_model_id(self, model_id: str) -> bool:
        """
        Validate that a model ID is in the expected format.

        Args:
            model_id: Model ID to validate

        Returns:
            True if model ID appears valid

        Raises:
            BedrockError: If model ID is invalid
        """
        if not model_id or not isinstance(model_id, str):
            raise BedrockError("Model ID must be a non-empty string")

        # Valid prefixes for supported model families
        valid_prefixes = [
            # Claude/Anthropic (including global inference profiles)
            "global.anthropic.claude-",
            "us.anthropic.claude-",
            "anthropic.claude-",
            # Nova/Amazon
            "amazon.nova-",
            "us.amazon.nova-",
            "global.amazon.nova-",
            # OpenAI. Reachable only through Converse — these models support no Invoke API
            # at all — and only via a geo or global inference profile, since they offer no
            # in-Region inference on bedrock-runtime.
            "us.openai.gpt-",
            "global.openai.gpt-",
            # Other supported models
            "amazon.titan-",
            "ai21.j2-",
            "cohere.command-",
            "meta.llama2-",
            "meta.llama3-",
            # Application inference profile ARNs
            "arn:aws:bedrock:",
        ]

        if not any(model_id.startswith(prefix) for prefix in valid_prefixes):
            self.logger.warning("Model ID may not be valid: %s", model_id)

        return True

    def create_anthropic_payload(
        self,
        system_prompt: str,
        messages: list,
        max_tokens: int = 8000,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Create a properly formatted payload for Anthropic Claude models.

        Args:
            system_prompt: System prompt string
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Formatted payload dictionary

        Raises:
            BedrockError: If payload creation fails
        """
        try:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": messages,
            }

            # Validate payload structure
            if not system_prompt.strip():
                raise BedrockError("System prompt cannot be empty")

            if not messages or not isinstance(messages, list):
                raise BedrockError("Messages must be a non-empty list")

            # Validate message structure
            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise BedrockError(f"Message {i} must be a dictionary")

                if "role" not in message or "content" not in message:
                    raise BedrockError(
                        f"Message {i} missing required 'role' or 'content' fields"
                    )

                if message["role"] not in ["user", "assistant"]:
                    raise BedrockError(
                        f"Message {i} has invalid role: {message['role']}"
                    )

            self.logger.debug(
                "Created Anthropic payload with %d messages", len(messages)
            )
            return payload

        except Exception as e:
            if isinstance(e, BedrockError):
                raise
            raise BedrockError(f"Failed to create payload: {e}") from e

    def clear_client_cache(self) -> None:
        """Clear the cached Bedrock clients."""
        self.get_client.cache_clear()
        self.logger.info("Cleared Bedrock client cache")
