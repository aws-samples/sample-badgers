#!/usr/bin/env python3
"""WebSocket streaming Gradio UI for AgentCore Runtime.

This version uses WebSocket connections to stream events in real-time,
showing thinking, tool calls, and results as they happen.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import boto3
import gradio as gr
from dotenv import load_dotenv

# Load environment
env_file = Path(__file__).parent.parent / "config" / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "agent_chat_websocket.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Separate logger for chat history (human-readable format)
chat_log_dir = log_dir / "chat_sessions"
chat_log_dir.mkdir(exist_ok=True)


def get_chat_logger(session_id: str) -> logging.Logger:
    """Get or create a logger for a specific session."""
    logger_name = f"chat_{session_id}"
    session_logger = logging.getLogger(logger_name)

    # Only add handler if not already configured
    if not session_logger.handlers:
        session_logger.setLevel(logging.INFO)
        log_file = chat_log_dir / f"{session_id}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s\n%(message)s\n"))
        session_logger.addHandler(handler)
        session_logger.propagate = False

    return session_logger


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_UPLOAD_BUCKET = os.getenv("S3_UPLOAD_BUCKET", "")


# =============================================================================
# S3 UPLOAD
# =============================================================================


def upload_file_to_s3(file_path: str) -> str:
    """Upload a file to S3 and return the S3 URI."""
    if not S3_UPLOAD_BUCKET:
        return "❌ S3_UPLOAD_BUCKET not configured in .env"

    if not file_path:
        return ""

    try:
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-west-2"))
        filename = Path(file_path).name
        s3_key = f"uploads/{filename}"

        logger.info("Uploading %s to s3://%s/%s", filename, S3_UPLOAD_BUCKET, s3_key)
        s3.upload_file(file_path, S3_UPLOAD_BUCKET, s3_key)

        s3_uri = f"s3://{S3_UPLOAD_BUCKET}/{s3_key}"
        logger.info("Upload complete: %s", s3_uri)
        return s3_uri
    except Exception as e:
        logger.error("Upload failed: %s", e, exc_info=True)
        return f"❌ Upload failed: {e}"


AGENTCORE_READ_TIMEOUT = int(os.getenv("AGENTCORE_READ_TIMEOUT", "600"))

custom_css = """
gradio-app { background-color: #eef2ff !important; }
.thought-group .title { background-color: #e0e7ff !important; }
.my-markdown-background { background-color: #fff; padding: 1rem; }
.tool-call { background-color: #fef3c7; padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px; }
.tool-result { background-color: #d1fae5; padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px; }
.status-msg { color: #6b7280; font-style: italic; }
"""


# =============================================================================
# WEBSOCKET STREAMING CLIENT
# =============================================================================


class WebSocketStreamingClient:
    """Client for streaming AgentCore Runtime via WebSocket."""

    def __init__(self):
        self.agentcore_runtime_arn = os.getenv("AGENTCORE_RUNTIME_WEBSOCKET_ARN")
        self.region = os.getenv("AWS_REGION", "us-west-2")
        self.agentcore_gateway_id = os.getenv("AGENTCORE_GATEWAY_ID")

        if (
            not self.agentcore_runtime_arn
            or self.agentcore_runtime_arn == "NOT_DEPLOYED"
        ):
            raise ValueError("AGENTCORE_RUNTIME_WEBSOCKET_ARN not set or not deployed")

        # Create gateway client for fetching tools
        session = boto3.Session(region_name=self.region)
        self.gateway_client = session.client("bedrock-agentcore-control")

        logger.info("WebSocket client initialized for region: %s", self.region)

    def get_available_tools(self):
        """Get list of tools from Gateway."""
        try:
            logger.info("Fetching tools from gateway: %s", self.agentcore_gateway_id)
            all_targets = []
            next_token = None

            while True:
                params = {"gatewayIdentifier": self.agentcore_gateway_id}
                if next_token:
                    params["nextToken"] = next_token

                response = self.gateway_client.list_gateway_targets(**params)
                all_targets.extend(response.get("items", []))

                next_token = response.get("nextToken")
                if not next_token:
                    break

            tools = []
            for target in all_targets:
                tools.append(
                    {
                        "name": target.get("name", "unknown"),
                        "id": target.get("targetId", "unknown"),
                    }
                )

            logger.info("Found %d tools", len(tools))
            return tools
        except Exception as e:
            logger.error("Error fetching tools: %s", e, exc_info=True)
            return [{"name": f"Error: {e}", "id": "error"}]

    def _get_presigned_ws_url(self, session_id: str) -> str:
        """Generate presigned WebSocket URL for AgentCore Runtime."""
        from bedrock_agentcore.runtime import AgentCoreRuntimeClient

        client = AgentCoreRuntimeClient(
            region=self.region,
            session=boto3.Session(region_name=self.region),
        )

        presigned_url = client.generate_presigned_url(
            runtime_arn=self.agentcore_runtime_arn,
            session_id=session_id,
            expires=300,
        )

        logger.info("Generated presigned WebSocket URL for session: %s", session_id)
        return str(presigned_url)

    async def stream_invoke(
        self, prompt: str, session_id: str, audit_mode: bool = False
    ):
        """Stream events from AgentCore Runtime via WebSocket.

        Yields events as they arrive from the agent.
        """
        import websockets

        ws_url = self._get_presigned_ws_url(session_id)
        logger.info("Connecting to WebSocket...")

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
            ) as websocket:
                # Send the invocation request
                payload = json.dumps(
                    {
                        "prompt": prompt,
                        "session_id": session_id,
                        "actor_id": "gradio_user",
                        "audit_mode": audit_mode,
                    }
                )
                await websocket.send(payload)
                logger.info("Sent prompt to WebSocket")

                # Stream responses
                async for raw_message in websocket:
                    try:
                        # Handle bytes vs string
                        if isinstance(raw_message, bytes):
                            message_str = raw_message.decode("utf-8")
                        else:
                            message_str = str(raw_message)

                        # Parse SSE format: "data: {...}\n\n"
                        if message_str.startswith("data: "):
                            data = json.loads(message_str[6:])
                        else:
                            data = json.loads(message_str)

                        logger.debug(
                            "Received event type: %s, keys: %s",
                            data.get("type", "n/a"),
                            list(data.keys())[:5],
                        )
                        yield data

                        # Check for completion - be specific about what signals end of stream
                        # Strands uses: complete=True, force_stop=True, or result with AgentResult
                        is_complete = data.get("complete") is True
                        is_force_stop = data.get("force_stop") is True
                        is_final_result = (
                            "result" in data and data.get("result") is not None
                        )
                        is_error = data.get("type") == "error"

                        if is_error:
                            logger.error(
                                "Error event from backend: %s",
                                safe_json_serialize(data, 500),
                            )

                        if is_complete or is_force_stop or is_final_result or is_error:
                            logger.info(
                                "Stream complete (complete=%s, force_stop=%s, result=%s, error=%s)",
                                is_complete,
                                is_force_stop,
                                is_final_result,
                                is_error,
                            )
                            break

                    except json.JSONDecodeError:
                        logger.warning(
                            "Non-JSON message (len=%d): %s",
                            len(message_str),
                            message_str[:100],
                        )
                        yield {"type": "raw", "data": message_str}

        except Exception as e:
            logger.error("WebSocket error: %s", e, exc_info=True)
            yield {"type": "error", "message": str(e)}


# =============================================================================
# EVENT FORMATTING - Strands Agent Event Types
# =============================================================================
# Strands streams these event types:
# Lifecycle: init_event_loop, start_event_loop, start, complete, force_stop, result
# Text: data (text chunks)
# Tools: current_tool_use, tool_stream_event
# Messages: message (complete assistant/user messages)
# Reasoning: reasoning, reasoningText
# Raw model: event (contains nested messageStart, contentBlockDelta, etc.)
# =============================================================================


def safe_json_serialize(obj: Any, max_len: int = 200) -> str:
    """Safely serialize an object to JSON string, handling non-serializable types."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)[:max_len]
    except (TypeError, ValueError, RecursionError) as e:
        logger.warning("JSON serialize failed: %s", e)
        return str(obj)[:max_len]


def format_event_for_display(event: dict[str, Any]) -> Optional[str]:
    """Format a Strands agent event for display in the chat.

    See: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/streaming/
    """
    # Log all events for debugging
    event_keys = list(event.keys())
    logger.debug("Processing event with keys: %s", event_keys)

    # --- Lifecycle Events ---
    if event.get("init_event_loop"):
        logger.info("🔄 Event loop initialized")
        return "🔄 *Agent initialized*"

    if event.get("start_event_loop") or event.get("start"):
        logger.info("▶️ Event loop cycle starting")
        return "▶️ *Processing...*"

    if event.get("complete"):
        logger.info("✅ Agent cycle completed")
        return None  # Final response handled separately

    if event.get("force_stop"):
        reason = event.get("force_stop_reason", "unknown")
        logger.warning("🛑 Force stopped: %s", reason)
        return f"🛑 *Stopped: {reason}*"

    if "result" in event:
        logger.info("📋 Final result received")
        return None  # Result handled in build_streaming_response

    # --- Text Output ---
    if "data" in event:
        text = event["data"]
        if text:
            logger.debug("📝 Text chunk: %s...", text[:50] if len(text) > 50 else text)
            return str(text)  # Return raw text for accumulation
        return None

    # --- Tool Events ---
    if "current_tool_use" in event:
        tool_use = event["current_tool_use"]
        if isinstance(tool_use, dict) and tool_use.get("name"):
            name = tool_use["name"]
            tool_input = tool_use.get("input", {})
            logger.info("🔧 Tool call: %s", name)
            input_preview = safe_json_serialize(tool_input, 200)
            return f"🔧 **Tool:** `{name}`\n```json\n{input_preview}\n```"
        return None

    if "tool_stream_event" in event:
        tool_event = event["tool_stream_event"]
        logger.debug("🔧 Tool stream event: %s", safe_json_serialize(tool_event, 100))
        return None  # Usually internal, don't display

    # --- Message Events ---
    if "message" in event:
        msg = event["message"]
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            logger.info("📬 Message from: %s", role)
            # Don't display - these are complete messages handled elsewhere
        return None

    # --- Reasoning/Thinking Events ---
    if event.get("reasoning") or "reasoningText" in event:
        reasoning_text = event.get("reasoningText", {})
        if isinstance(reasoning_text, dict):
            text = reasoning_text.get("text", "")
        else:
            text = str(reasoning_text)
        if text:
            logger.debug("🧠 Reasoning: %s...", text[:50])
            return f"🧠 *{text[:200]}*"
        return None

    # --- Raw Model Events (nested in "event" key) ---
    if "event" in event:
        raw_event = event["event"]
        if isinstance(raw_event, dict):
            # Handle specific nested event types
            if "messageStart" in raw_event:
                role = raw_event["messageStart"].get("role", "unknown")
                logger.debug("📨 Message start: %s", role)
                return None

            if "contentBlockStart" in raw_event:
                logger.debug("📦 Content block start")
                return None

            if "contentBlockDelta" in raw_event:
                delta = raw_event["contentBlockDelta"].get("delta", {})
                # Text delta
                if "text" in delta:
                    return str(delta["text"])
                # Reasoning delta
                if "reasoningContent" in delta:
                    reasoning = delta["reasoningContent"].get("text", "")
                    if reasoning:
                        logger.debug("🧠 Reasoning delta: %s...", reasoning[:30])
                        return f"🧠 *{reasoning}*"
                # Tool use delta
                if "toolUse" in delta:
                    logger.debug("🔧 Tool use delta")
                return None

            if "contentBlockStop" in raw_event:
                logger.debug("📦 Content block stop")
                return None

            if "messageStop" in raw_event:
                stop_reason = raw_event["messageStop"].get("stopReason", "")
                logger.debug("📨 Message stop: %s", stop_reason)
                return None

            if "metadata" in raw_event:
                logger.debug("📊 Metadata received")
                return None

        # Generic fallback for unknown event structures
        logger.debug("📡 Raw event: %s", safe_json_serialize(raw_event, 100))
        return None

    # --- Legacy/Custom Event Types ---
    event_type = event.get("type", "")

    if event_type == "status":
        return f"*{event.get('message', '')}*"

    if event_type == "error":
        error_msg = event.get("message", "Unknown error")
        logger.error("❌ Error event: %s", error_msg)
        return f"❌ **Error:** {error_msg}"

    if event_type == "text":
        return str(event.get("text", ""))

    # Log unhandled events for debugging
    logger.debug("⚠️ Unhandled event: %s", safe_json_serialize(event, 150))
    return None


def build_streaming_response(events: list[dict[str, Any]]) -> str:
    """Build the final response from accumulated Strands agent events."""
    text_parts: list[str] = []
    final_response = ""

    for event in events:
        # Check for final result
        if "result" in event:
            result = event["result"]
            if isinstance(result, dict):
                # AgentResult structure
                final_response = str(
                    result.get("message", "") or result.get("response", "")
                )
            elif hasattr(result, "message"):
                final_response = str(result.message)
            else:
                final_response = str(result)
            continue

        # Check for complete event with response
        if event.get("complete") and "response" in event:
            final_response = str(event["response"])
            continue

        # Accumulate text from data events
        if "data" in event and event["data"]:
            data = event["data"]
            if isinstance(data, str):
                text_parts.append(data)
            else:
                # Convert non-string data to string
                text_parts.append(str(data))
            continue

        # Accumulate text from contentBlockDelta
        if "event" in event:
            raw_event = event.get("event", {})
            if isinstance(raw_event, dict):
                delta = raw_event.get("contentBlockDelta", {}).get("delta", {})
                if isinstance(delta, dict) and "text" in delta:
                    text = delta["text"]
                    if isinstance(text, str):
                        text_parts.append(text)
                    else:
                        text_parts.append(str(text))

    # Prefer final response, fall back to accumulated text
    if final_response:
        return final_response
    if text_parts:
        return "".join(text_parts)
    return "No response received"


def _fix_incomplete_markdown(text: str) -> str:
    """Fix incomplete markdown formatting that can occur during streaming.

    When text streams in chunks, markdown delimiters like ** or * can be split
    across chunks, causing rendering issues. This function ensures paired
    delimiters are complete.

    Args:
        text: The accumulated text that may have incomplete markdown

    Returns:
        Text with incomplete markdown delimiters closed
    """
    if not text:
        return text

    # Count unpaired bold markers (**)
    # We need to be careful: *** is italic+bold, so count ** specifically
    # Find all ** that aren't part of ***
    # Replace *** temporarily, count **, then restore
    temp_text = text.replace("***", "\x00\x00\x00")
    bold_count = temp_text.count("**")

    # If odd number of **, add closing **
    if bold_count % 2 == 1:
        text = text + "**"

    # Count unpaired italic markers (single *)
    # After handling bold, check for unpaired single *
    # This is trickier - need to count * that aren't part of **
    temp_text = text.replace("**", "")
    italic_count = temp_text.count("*")

    if italic_count % 2 == 1:
        text = text + "*"

    return text


def _sanitize_reasoning_markdown(text: str) -> str:
    """Sanitize markdown in reasoning/thinking content for cleaner display.

    Converts markdown formatting to plain text with structural breaks:
    - **bold** → newline + text + newline (treat as section headers)
    - *italic* → plain text
    - Removes other common markdown artifacts

    Args:
        text: Raw reasoning text that may contain markdown

    Returns:
        Sanitized text with markdown converted to plain formatting
    """
    import re

    if not text:
        return text

    # Handle ***bold italic*** first (convert to just the text with breaks)
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\n\1\n", text)

    # Handle **bold** - treat as section breaks/headers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\n\1\n", text)

    # Handle *italic* - just remove the markers
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    # Handle incomplete/dangling markers at end of text
    # Remove trailing ** or * that aren't closed
    text = re.sub(r"\*+$", "", text)

    # Remove leading ** or * that aren't part of a pair
    text = re.sub(r"^\*+", "", text)

    # Clean up excessive newlines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def _get_complete_words(text: str) -> tuple[str, str]:
    """Split text into complete words and trailing partial word.

    During streaming, tokens can arrive mid-word causing display issues.
    This function returns only complete words (up to last whitespace)
    and the remaining partial content to buffer.

    Args:
        text: The accumulated text that may end mid-word

    Returns:
        Tuple of (complete_words, remaining_partial)
    """
    if not text:
        return "", ""

    # Find the last whitespace character
    last_space = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] in " \t\n\r":
            last_space = i
            break

    if last_space == -1:
        # No whitespace found - entire text might be partial
        # Return it anyway if it's reasonably long (likely complete)
        if len(text) > 20:
            return text, ""
        return "", text

    # Return complete words and the partial remainder
    return text[: last_space + 1], text[last_space + 1 :]


def _build_display_content(
    accumulated_reasoning: list, accumulated_text: list, is_final: bool = False
) -> str:
    """Build the display content with thinking section and response.

    Args:
        accumulated_reasoning: List of reasoning text chunks
        accumulated_text: List of response text chunks
        is_final: If True, include all content; if False, buffer partial words

    Returns:
        Formatted display string with thinking tags
    """
    parts = []

    # Add thinking section using Gradio's native reasoning tags
    if accumulated_reasoning:
        reasoning_content = "".join(str(r) for r in accumulated_reasoning if r)
        # Sanitize markdown in reasoning for cleaner display
        reasoning_content = _sanitize_reasoning_markdown(reasoning_content)

        # During streaming, only show complete words to prevent duplicates
        if not is_final:
            reasoning_content, _ = _get_complete_words(reasoning_content)

        if reasoning_content:
            parts.append(f"<think>{reasoning_content}</think>")

    # Add response text
    if accumulated_text:
        text_content = "".join(str(t) for t in accumulated_text if t)

        # During streaming, only show complete words
        if not is_final:
            text_content, _ = _get_complete_words(text_content)

        # Fix incomplete markdown in response text
        text_content = _fix_incomplete_markdown(text_content)
        if text_content:
            parts.append(text_content)

    return "".join(parts) if parts else "🔄 *Processing...*"


# =============================================================================
# GRADIO UI
# =============================================================================


def create_ui():
    """Create the streaming Gradio interface."""
    logger.info("Creating WebSocket streaming UI")

    try:
        client = WebSocketStreamingClient()
        tools = client.get_available_tools()

        if tools and tools[0].get("id") == "error":
            initial_status = f"❌ {tools[0]['name']}\n\nPlease configure .env file"
            client = None
            logger.warning("Client initialization failed - configuration issue")
        else:
            sorted_tools = sorted(tools, key=lambda t: t["name"].lower())
            tools_text = "\n".join([f"- {t['name']}" for t in sorted_tools])
            initial_status = f"**Available Tools ({len(tools)}):**\n{tools_text}"
            logger.info("Client initialized successfully with %d tools", len(tools))
    except Exception as e:
        initial_status = f"❌ Error: {e}\n\nPlease configure .env file"
        client = None
        logger.error("Client init error: %s", e, exc_info=True)

    initial_session_id = f"ws-session-{uuid.uuid4()}"

    def get_session_info(sid):
        """Generate session info text."""
        return f"Session: {sid}\nMode: WebSocket Streaming"

    with gr.Blocks(title="AgentCore Runtime UI (Streaming)") as blocks_demo:
        # State for session ID
        session_id = gr.State(value=initial_session_id)
        gr.Markdown("# 🤖 AgentCore Runtime UI (WebSocket Streaming)")
        gr.Markdown("Real-time visibility into agent thinking, tool calls, and results")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    buttons=["copy"],
                    allow_tags=True,
                    autoscroll=True,
                    render_markdown=True,
                    group_consecutive_messages=True,
                    # Auto-extract <think>...</think> into collapsible sections
                    reasoning_tags=[("<think>", "</think>")],
                    avatar_images=(
                        "./images/user.png",
                        "./images/assistant.png",
                    ),
                    elem_classes="chat-window",
                )

                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask your agent something... (Press Enter to send)",
                )
                with gr.Row(elem_classes="chat-buttons-row"):
                    gr.HTML("<div style=''></div>")
                    clear_btn = gr.Button(
                        "Clear", variant="secondary", scale=0, min_width=100
                    )
                    send_btn = gr.Button(
                        "Send", variant="primary", scale=0, min_width=100
                    )

                with gr.Row(elem_classes="download-buttons-row"):
                    download_last_btn = gr.DownloadButton(
                        "📥 Download Last Response",
                        variant="secondary",
                        size="sm",
                        scale=1,
                    )
                    download_full_btn = gr.DownloadButton(
                        "📥 Download Full Chat",
                        variant="secondary",
                        size="sm",
                        scale=1,
                    )

            with gr.Column(scale=1):
                gr.Text("Enable confidence scoring and human review flags")
                audit_mode_checkbox = gr.Checkbox(
                    label="🔍 Audit Mode",
                    value=False,
                    info="",
                )
                with gr.Accordion(
                    "🔧 Available Tools", open=False, elem_classes="badgers_expander"
                ):
                    tools_display = gr.Markdown(initial_status)
                    refresh_tools_btn = gr.Button("🔄 Refresh Tools", size="sm")
                session_display = gr.Textbox(
                    label="Session Info",
                    value=get_session_info(initial_session_id),
                    interactive=False,
                )
                new_session_btn = gr.Button("🔄 New Session", variant="secondary")

        async def send_message_streaming(message, history, audit_mode, sid):
            """Send message and stream events."""
            if not client:
                history.append({"role": "user", "content": message})
                history.append(
                    {
                        "role": "assistant",
                        "content": "❌ Client not configured. Check .env file.",
                    }
                )
                yield history, "*Error: Client not configured*"
                return

            if not message.strip():
                yield history, ""
                return

            # Add user message
            history.append({"role": "user", "content": message})
            get_chat_logger(sid).info("USER: %s", message)

            # Add placeholder for assistant
            history.append(
                {
                    "role": "assistant",
                    "content": "🔄 *Connecting to agent...*",
                }
            )
            yield history, "*Connecting...*"

            # Collect events and stream updates - separate reasoning from text
            events = []
            event_display_parts = []
            accumulated_reasoning = []
            accumulated_text = []

            try:
                async for event in client.stream_invoke(message, sid, audit_mode):
                    events.append(event)
                    logger.debug("Received event keys: %s", list(event.keys()))

                    # Extract reasoning and text from events
                    try:
                        # Check for reasoning in various formats
                        reasoning_text = None
                        response_text = None

                        # Priority 1: nested in event.contentBlockDelta.delta (streaming chunks)
                        if "event" in event:
                            raw_event = event.get("event", {})
                            if isinstance(raw_event, dict):
                                delta = raw_event.get("contentBlockDelta", {}).get(
                                    "delta", {}
                                )
                                if isinstance(delta, dict):
                                    # Reasoning content
                                    if "reasoningContent" in delta:
                                        rc = delta["reasoningContent"]
                                        if isinstance(rc, dict):
                                            reasoning_text = rc.get("text", "")
                                        else:
                                            reasoning_text = str(rc)
                                    # Text content
                                    elif "text" in delta:
                                        response_text = str(delta["text"])

                        # SKIP data events - they duplicate contentBlockDelta
                        # Priority 3: reasoningText at top level (only if not already captured)
                        elif "reasoningText" in event and not reasoning_text:
                            rt = event["reasoningText"]
                            if isinstance(rt, dict):
                                reasoning_text = rt.get("text", "")
                            else:
                                reasoning_text = str(rt)

                        # Accumulate reasoning
                        if reasoning_text:
                            accumulated_reasoning.append(reasoning_text)
                            logger.debug(
                                "Accumulated reasoning (len=%d)", len(reasoning_text)
                            )

                        # Accumulate text (but filter out reasoning markers)
                        if response_text and not response_text.startswith("🧠"):
                            accumulated_text.append(response_text)
                            logger.debug(
                                "Accumulated text (len=%d): %s...",
                                len(response_text),
                                response_text[:30],
                            )

                        # Format for event log
                        formatted = format_event_for_display(event)
                        if formatted:
                            event_display_parts.append(str(formatted))

                    except (TypeError, ValueError) as fmt_err:
                        logger.warning("Event format error: %s", fmt_err)
                        event_display_parts.append("📡 (unserializable event)")

                    # Update the assistant message with separated content
                    if accumulated_reasoning or accumulated_text:
                        history[-1] = {
                            "role": "assistant",
                            "content": _build_display_content(
                                accumulated_reasoning, accumulated_text, is_final=False
                            ),
                        }
                    elif event.get("complete") or "result" in event:
                        # Final response
                        final = build_streaming_response(events)
                        history[-1] = {
                            "role": "assistant",
                            "content": final,
                        }
                    elif event.get("type") == "error" or "error" in event:
                        error_msg = event.get("message") or event.get(
                            "error", "Unknown error"
                        )
                        history[-1] = {
                            "role": "assistant",
                            "content": f"❌ {error_msg}",
                        }
                    else:
                        # Show progress indicator
                        status = "Processing"
                        if event.get("init_event_loop"):
                            status = "Initializing"
                        elif event.get("start_event_loop") or event.get("start"):
                            status = "Thinking"
                        elif "current_tool_use" in event:
                            tool_name = event["current_tool_use"].get("name", "tool")
                            status = f"Using {tool_name}"
                        elif event.get("reasoning") or "reasoningText" in event:
                            status = "Reasoning"

                        history[-1] = {
                            "role": "assistant",
                            "content": f"🔄 *{status}...*",
                        }

                    yield history, ""

                # Final update - use Gradio's native reasoning tags
                final_parts = []
                if accumulated_reasoning:
                    reasoning_content = "".join(
                        str(r) for r in accumulated_reasoning if r
                    )
                    # Sanitize markdown in final reasoning
                    reasoning_content = _sanitize_reasoning_markdown(reasoning_content)
                    final_parts.append(f"<think>{reasoning_content}</think>")

                # Get final response text
                if accumulated_text:
                    final_parts.append("".join(str(t) for t in accumulated_text if t))
                else:
                    # Fall back to build_streaming_response
                    final_response = build_streaming_response(events)
                    if final_response and final_response != "No response received":
                        final_parts.append(final_response)

                final_content = (
                    "".join(final_parts) if final_parts else "No response received"
                )
                history[-1] = {
                    "role": "assistant",
                    "content": final_content,
                }
                # Log the final assistant response (strip think tags for readability)
                import re

                clean_response = re.sub(
                    r"<think>.*?</think>", "", final_content, flags=re.DOTALL
                ).strip()
                get_chat_logger(sid).info(
                    "ASSISTANT: %s",
                    (
                        clean_response[:2000]
                        if len(clean_response) > 2000
                        else clean_response
                    ),
                )
                yield history, ""

            except Exception as e:
                logger.error("Streaming error: %s", e, exc_info=True)
                history[-1] = {"role": "assistant", "content": f"❌ Error: {e}"}
                yield history, ""

        def clear_chat():
            return [], ""

        def start_new_session():
            """Generate a new session ID and update display."""
            new_sid = f"ws-session-{uuid.uuid4()}"
            logger.info("Started new session: %s", new_sid)
            return new_sid, get_session_info(new_sid), []

        def refresh_tools():
            """Refresh the tools list from the gateway."""
            if not client:
                return "❌ Client not configured. Check .env file."
            try:
                tools = client.get_available_tools()
                if tools and tools[0].get("id") == "error":
                    return f"❌ {tools[0]['name']}"
                sorted_tools = sorted(tools, key=lambda t: t["name"].lower())
                tools_text = "\n".join([f"- {t['name']}" for t in sorted_tools])
                logger.info("Refreshed tools list: %d tools", len(tools))
                return f"**Available Tools ({len(tools)}):**\n{tools_text}"
            except Exception as e:
                logger.error("Error refreshing tools: %s", e)
                return f"❌ Error refreshing tools: {e}"

        def download_last_response(history, sid):
            """Download the last agent response as a markdown file."""
            import re
            from datetime import datetime

            # Create responses directory
            responses_dir = Path(__file__).parent.parent / "logs" / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)

            if not history:
                # Return a temporary file with error message
                temp_path = responses_dir / "no_response.txt"
                temp_path.write_text("No messages in chat history.")
                return str(temp_path)

            # Find the last assistant message
            last_assistant_msg = None
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg
                    break

            if not last_assistant_msg:
                temp_path = responses_dir / "no_response.txt"
                temp_path.write_text("No agent responses found in chat history.")
                return str(temp_path)

            content = last_assistant_msg.get("content", "")
            # Handle content as list or string
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            else:
                content = str(content)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_response_{sid[:8]}_{timestamp}.md"
            filepath = responses_dir / filename

            # Format content for download
            output = f"# Agent Response\n\n"
            output += f"**Session:** {sid}\n"
            output += (
                f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            output += "---\n\n"
            output += content

            filepath.write_text(output, encoding="utf-8")
            logger.info("Downloaded last response to: %s", filepath)
            return str(filepath)

        def download_full_chat(history, sid):
            """Download the full chat history as a markdown file."""
            from datetime import datetime

            # Create responses directory
            responses_dir = Path(__file__).parent.parent / "logs" / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)

            if not history:
                temp_path = responses_dir / "empty_chat.txt"
                temp_path.write_text("No messages in chat history.")
                return str(temp_path)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"full_chat_{sid[:8]}_{timestamp}.md"
            filepath = responses_dir / filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{sid[:8]}_{timestamp}.md"
            filepath = Path("/tmp") / filename

            # Format full conversation
            output = f"# Chat History\n\n"
            output += f"**Session:** {sid}\n"
            output += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            output += f"**Messages:** {len(history)}\n\n"
            output += "---\n\n"

            for i, msg in enumerate(history, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Handle content as list or string
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content)
                else:
                    content = str(content)

                if role == "user":
                    output += f"## 👤 User (Message {i})\n\n"
                elif role == "assistant":
                    output += f"## 🤖 Assistant (Message {i})\n\n"
                else:
                    output += f"## {role.title()} (Message {i})\n\n"

                output += content + "\n\n"
                output += "---\n\n"

            filepath.write_text(output, encoding="utf-8")
            logger.info("Downloaded full chat history to: %s", filepath)
            return str(filepath)

        refresh_tools_btn.click(fn=refresh_tools, outputs=[tools_display])

        send_btn.click(
            fn=send_message_streaming,
            inputs=[msg_input, chatbot, audit_mode_checkbox, session_id],
            outputs=[chatbot, msg_input],
        )

        msg_input.submit(
            fn=send_message_streaming,
            inputs=[msg_input, chatbot, audit_mode_checkbox, session_id],
            outputs=[chatbot, msg_input],
        )

        clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])

        new_session_btn.click(
            fn=start_new_session,
            outputs=[session_id, session_display, chatbot],
        )

        download_last_btn.click(
            fn=download_last_response,
            inputs=[chatbot, session_id],
            outputs=[download_last_btn],
        )

        download_full_btn.click(
            fn=download_full_chat,
            inputs=[chatbot, session_id],
            outputs=[download_full_btn],
        )

    return blocks_demo


demo = create_ui()

if __name__ == "__main__":
    logger.info("Starting WebSocket streaming UI on http://localhost:7861")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(),
        app_kwargs={"title": "AgentCore Runtime UI (Streaming)"},
    )
