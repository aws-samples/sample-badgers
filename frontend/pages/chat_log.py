"""Chat Log tab - View chat session history."""

from pathlib import Path

import gradio as gr

CHAT_LOG_DIR = Path(__file__).parent.parent / "logs" / "chat_sessions"


def get_session_list():
    """Get list of available session log files."""
    if CHAT_LOG_DIR.exists():
        sessions = sorted(
            CHAT_LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return [s.stem for s in sessions] if sessions else ["No sessions yet"]
    return ["No sessions yet"]


def read_chat_log(session_id: str):
    """Read the chat history log file for a specific session."""
    if not session_id or session_id == "No sessions yet":
        return "No chat history yet."

    log_file = CHAT_LOG_DIR / f"{session_id}.log"
    if log_file.exists():
        try:
            content = log_file.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            formatted = []
            prev_was_assistant = False

            for line in lines:
                if line.startswith("USER:"):
                    if prev_was_assistant:
                        formatted.append("=" * 60)
                    formatted.append(line)
                    prev_was_assistant = False
                elif line.startswith("ASSISTANT:"):
                    formatted.append("-" * 40)
                    formatted.append(line)
                    prev_was_assistant = True
                elif line.strip():
                    formatted.append(line)

            return "\n".join(formatted)
        except Exception as e:
            return f"Error reading log: {e}"
    return "No chat history for this session."


def create_chat_log_page():
    """Create the chat log viewer page."""
    with gr.Blocks() as demo_app:
        with gr.Column():
            with gr.Row():
                session_dropdown = gr.Dropdown(
                    label="Session",
                    choices=get_session_list(),
                    value=get_session_list()[0] if get_session_list() else None,
                    interactive=True,
                )
                refresh_sessions_btn = gr.Button(
                    "🔄 Refresh", variant="secondary", scale=0
                )
            chat_log_display = gr.Textbox(
                label="Chat History",
                value="Select a session to view logs",
                lines=30,
                max_lines=50,
                interactive=False,
                autoscroll=True,
            )
            session_dropdown.change(
                fn=read_chat_log,
                inputs=[session_dropdown],
                outputs=[chat_log_display],
            )
            refresh_sessions_btn.click(
                fn=lambda: gr.update(choices=get_session_list()),
                outputs=[session_dropdown],
            )
    return demo_app


# Module-level demo for import
demo = create_chat_log_page()

if __name__ == "__main__":
    demo.launch(share=False)
