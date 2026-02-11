"""
AWS Bedrock client management for analyzer system.

Handles Bedrock client creation, invocation, and error handling for different analyzer types.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import boto3
from botocore.exceptions import ClientError

print("BEDROCK_CLIENT MODULE LOADING - TOP OF FILE")


# Model ID to inference profile ARN mapping via environment variables
MODEL_TO_PROFILE_ENV_MAP = {
    # Claude Sonnet 4.5 variants
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": "CLAUDE_SONNET_PROFILE_ARN",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "CLAUDE_SONNET_PROFILE_ARN",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "CLAUDE_SONNET_PROFILE_ARN",
    # Claude Haiku 4.5 variants (note: date is 20251001)
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": "CLAUDE_HAIKU_PROFILE_ARN",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "CLAUDE_HAIKU_PROFILE_ARN",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "CLAUDE_HAIKU_PROFILE_ARN",
    # Claude Opus 4.6 variants (with and without :0 suffix)
    "global.anthropic.claude-opus-4-6-v1:0": "CLAUDE_OPUS_46_PROFILE_ARN",
    "us.anthropic.claude-opus-4-6-v1:0": "CLAUDE_OPUS_46_PROFILE_ARN",
    "anthropic.claude-opus-4-6-v1:0": "CLAUDE_OPUS_46_PROFILE_ARN",
    "global.anthropic.claude-opus-4-6-v1": "CLAUDE_OPUS_46_PROFILE_ARN",
    "us.anthropic.claude-opus-4-6-v1": "CLAUDE_OPUS_46_PROFILE_ARN",
    "anthropic.claude-opus-4-6-v1": "CLAUDE_OPUS_46_PROFILE_ARN",
    # Nova Premier variants
    "us.amazon.nova-premier-v1:0": "NOVA_PREMIER_PROFILE_ARN",
    "amazon.nova-premier-v1:0": "NOVA_PREMIER_PROFILE_ARN",
}


def get_inference_profile_arn(model_id: str) -> Optional[str]:
    """
    Get inference profile ARN for a model ID if configured via environment variable.

    Args:
        model_id: The Bedrock model ID

    Returns:
        Inference profile ARN if available, None otherwise
    """
    import os

    env_var = MODEL_TO_PROFILE_ENV_MAP.get(model_id)
    if env_var:
        return os.environ.get(env_var)
    return None


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
        'claude' or 'nova'

    Raises:
        BedrockError: If model family cannot be determined
    """
    model_lower = model_id.lower()

    if "anthropic" in model_lower or "claude" in model_lower:
        return "claude"
    elif "nova" in model_lower or "amazon.nova" in model_lower:
        return "nova"
    else:
        raise BedrockError(f"Unknown model family for model ID: {model_id}")


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
            read_timeout = int(os.environ.get("BEDROCK_READ_TIMEOUT", "300"))
            connect_timeout = int(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "10"))

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

        for (
            current_model_id,
            current_extended_thinking,
            current_budget_tokens,
            current_adaptive_thinking,
            current_effort,
        ) in model_chain:
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
                is_last = (
                    current_model_id,
                    current_extended_thinking,
                    current_budget_tokens,
                    current_adaptive_thinking,
                    current_effort,
                ) == model_chain[-1]
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

            # Convert payload to model-specific format
            model_payload = self._convert_payload_for_model(
                payload,
                model_family,
                extended_thinking,
                budget_tokens,
                adaptive_thinking,
                adaptive_effort,
            )
            self.logger.debug(
                "Payload size: %d characters", len(json.dumps(model_payload))
            )

            # Add throttling delay to prevent rate limiting
            time.sleep(self.throttle_delay)

            response = self.handle_throttling(
                client.invoke_model,
                modelId=invoke_model_id,
                body=json.dumps(model_payload),
                max_retries=max_retries,
            )

            # Parse response - read streaming body in chunks for reliability
            self.logger.info("Reading response body from Bedrock...")
            raw_body = self._read_streaming_body(response["body"])
            self.logger.info("Response body received: %d bytes", len(raw_body))
            response_body = json.loads(raw_body)

            # Normalize response to common format
            normalized = self._normalize_response(response_body, model_family)

            self.logger.info("Model invocation successful")
            return normalized

        except BedrockError:
            raise
        except Exception as e:
            raise BedrockError(f"Model invocation failed: {e}") from e

    def _convert_payload_for_model(
        self,
        payload: Dict[str, Any],
        model_family: str,
        extended_thinking: bool = False,
        budget_tokens: Optional[int] = None,
        adaptive_thinking: bool = False,
        adaptive_effort: str = "high",
    ) -> Dict[str, Any]:
        """
        Convert a base payload to model-specific format.

        Args:
            payload: Base payload with system, messages, max_tokens, temperature
            model_family: 'claude' or 'nova'
            extended_thinking: Whether to enable extended thinking (Claude only)
            budget_tokens: Optional budget tokens for extended thinking
            adaptive_thinking: Whether to enable adaptive thinking (Claude only)
            adaptive_effort: Effort level for adaptive thinking

        Returns:
            Model-specific payload
        """
        if model_family == "claude":
            # Payload is already in Claude format from create_anthropic_payload
            if adaptive_thinking:
                return self._add_adaptive_thinking_to_payload(payload, adaptive_effort)
            if extended_thinking:
                return self._add_extended_thinking_to_payload(payload, budget_tokens)
            return payload
        elif model_family == "nova":
            # Nova doesn't support extended/adaptive thinking
            if extended_thinking or adaptive_thinking:
                self.logger.warning(
                    "Extended/adaptive thinking not supported for Nova models, ignoring"
                )
            return self._convert_to_nova_payload(payload)
        else:
            raise BedrockError(f"Unknown model family: {model_family}")

    def _add_extended_thinking_to_payload(
        self, payload: Dict[str, Any], budget_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add extended thinking configuration to Claude payload.

        Args:
            payload: Claude-format payload
            budget_tokens: Optional budget tokens (defaults to 80% of max_tokens)

        Returns:
            Payload with extended thinking enabled
        """
        max_tokens = payload.get("max_tokens", 8000)

        # Use provided budget or default to 80% of max_tokens
        if budget_tokens is None:
            budget_tokens = int(max_tokens * 0.8)

        thinking_payload = dict(payload)
        thinking_payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
        # Extended thinking requires temperature = 1
        thinking_payload["temperature"] = 1

        self.logger.info(
            "Extended thinking enabled with budget_tokens=%d", budget_tokens
        )
        return thinking_payload

    def _add_adaptive_thinking_to_payload(
        self, payload: Dict[str, Any], effort: str = "high"
    ) -> Dict[str, Any]:
        """
        Add adaptive thinking configuration to Claude payload.

        Args:
            payload: Claude-format payload
            effort: Effort level ("low", "medium", "high")

        Returns:
            Payload with adaptive thinking enabled
        """
        thinking_payload = dict(payload)
        thinking_payload["thinking"] = {
            "type": "adaptive",
            "effort": effort,
        }
        # Adaptive thinking requires temperature = 1
        thinking_payload["temperature"] = 1

        self.logger.info("Adaptive thinking enabled with effort=%s", effort)
        return thinking_payload

    def _convert_to_nova_payload(
        self, claude_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert Claude-format payload to Nova format.

        Args:
            claude_payload: Payload in Claude/Anthropic format

        Returns:
            Payload in Nova format
        """
        # Convert messages content format
        nova_messages: list[Dict[str, Any]] = []
        for msg in claude_payload.get("messages", []):
            nova_content: list[Dict[str, Any]] = []
            content = msg.get("content", [])

            # Handle both list and string content
            if isinstance(content, str):
                nova_content.append({"text": content})
            else:
                for item in content:
                    if item.get("type") == "image":
                        # Convert Claude image format to Nova format
                        source = item.get("source", {})
                        media_type = source.get("media_type", "image/png")
                        image_format = (
                            media_type.split("/")[1] if "/" in media_type else "png"
                        )
                        nova_content.append(
                            {
                                "image": {
                                    "format": image_format,
                                    "source": {"bytes": source.get("data", "")},
                                }
                            }
                        )
                    elif item.get("type") == "text":
                        nova_content.append({"text": item.get("text", "")})
                    elif "text" in item:
                        nova_content.append({"text": item["text"]})

            nova_messages.append(
                {"role": msg.get("role", "user"), "content": nova_content}
            )

        # Build Nova payload
        system_text = claude_payload.get("system", "")
        nova_payload = {
            "schemaVersion": "messages-v1",
            "system": [{"text": system_text}] if system_text else [],
            "messages": nova_messages,
            "inferenceConfig": {
                "maxTokens": claude_payload.get("max_tokens", 4096),
                "temperature": claude_payload.get("temperature", 0.7),
            },
        }

        return nova_payload

    def _normalize_response(
        self, response_body: Dict[str, Any], model_family: str
    ) -> Dict[str, Any]:
        """
        Normalize model response to common format (Claude format).

        Args:
            response_body: Raw response from model
            model_family: 'claude' or 'nova'

        Returns:
            Normalized response with 'content' key and optional 'thinking' key
        """
        if model_family == "claude":
            # Validate Claude response structure
            if "content" not in response_body or not response_body["content"]:
                raise BedrockError("Empty response from Claude model")

            result = dict(response_body)

            # Extract thinking content if present
            thinking_content = self._extract_thinking_from_claude_response(
                response_body
            )
            if thinking_content:
                result["thinking"] = thinking_content
                self.logger.info(
                    "Extracted thinking content: %d characters", len(thinking_content)
                )

            return result

        elif model_family == "nova":
            # Convert Nova response to Claude format
            try:
                nova_content = response_body["output"]["message"]["content"]
                # Convert Nova content format to Claude format
                claude_content = []
                for item in nova_content:
                    if "text" in item:
                        claude_content.append({"type": "text", "text": item["text"]})

                if not claude_content:
                    raise BedrockError("Empty response from Nova model")

                return {"content": claude_content}
            except KeyError as e:
                raise BedrockError(f"Invalid Nova response structure: {e}") from e

        else:
            raise BedrockError(f"Unknown model family: {model_family}")

    def _extract_thinking_from_claude_response(
        self, response_body: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract thinking content from Claude response with extended thinking.

        Args:
            response_body: Raw Claude response

        Returns:
            Thinking content as string, or None if not present
        """
        content = response_body.get("content", [])
        thinking_parts = []

        for item in content:
            if item.get("type") == "thinking":
                thinking_text = item.get("thinking", "")
                if thinking_text:
                    thinking_parts.append(thinking_text)

        if thinking_parts:
            return "\n\n".join(thinking_parts)
        return None

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

    def _read_streaming_body(self, streaming_body, chunk_size: int = 65536) -> bytes:
        """
        Read streaming body in chunks for better reliability with large responses.

        Args:
            streaming_body: botocore StreamingBody object
            chunk_size: Size of chunks to read (default 64KB)

        Returns:
            Complete response body as bytes

        Raises:
            BedrockError: If reading fails
        """
        chunks = []
        total_bytes = 0
        chunk_count = 0

        try:
            while True:
                chunk = streaming_body.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                total_bytes += len(chunk)
                chunk_count += 1

                if chunk_count % 10 == 0:
                    self.logger.debug(
                        "Read %d chunks, %d bytes so far...", chunk_count, total_bytes
                    )

            self.logger.debug(
                "Finished reading: %d chunks, %d total bytes", chunk_count, total_bytes
            )
            return b"".join(chunks)

        except Exception as e:
            self.logger.error(
                "Failed reading streaming body after %d bytes: %s", total_bytes, e
            )
            raise BedrockError(
                f"Failed to read response body after {total_bytes} bytes: {e}"
            ) from e
