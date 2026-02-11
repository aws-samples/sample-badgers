"""Main entry point for the Gradio frontend application."""

import os
from pathlib import Path

# Disable Gradio Analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from pages import (
    agent_chat_websocket,
    agent_observability,
    analyzer_editor,
    analyzer_wizard,
    chat_log,
    home,
    pricing_calculator,
    result_evaluator,
)

CSS_PATH = Path(__file__).parent / "css" / "custom_style.css"
with open(CSS_PATH, "r", encoding="utf-8") as file:
    custom_css = file.read()


FAVICON_PATH = Path(__file__).parent / "images" / "badgers-favicon.png"

with gr.Blocks(title="BADGERS - SAMPLE") as demo:
    with gr.Tabs():
        with gr.Tab("Home"):
            home.demo.render()
        with gr.Tab("Chat"):
            agent_chat_websocket.demo.render()
        with gr.Tab("Edit Analyzer"):
            analyzer_editor.demo.render()
        with gr.Tab("Create Analyzer"):
            analyzer_wizard.demo.render()
        with gr.Tab("Evaluations"):
            result_evaluator.demo.render()
        with gr.Tab("Pricing Calculator"):
            pricing_calculator.demo.render()
        with gr.Tab("Observability"):
            agent_observability.demo.render()
        with gr.Tab("Conversation Log"):
            chat_log.demo.render()

theme = gr.themes.Soft()

if __name__ == "__main__":
    demo.launch(
        css=custom_css,
        theme=theme,
        favicon_path=str(FAVICON_PATH),
    )
