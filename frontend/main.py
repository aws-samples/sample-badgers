"""Main entry point for the Gradio frontend application.
Run with hot-reload:
    gradio main.py --watch-dirs pages css
"""

import os
from pathlib import Path
import time

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
FAVICON_PATH = Path(__file__).parent / "images" / "badgers-favicon.png"
theme = gr.themes.Soft()


def load_css():
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        return file.read()


with gr.Blocks(title="BADGERS - SAMPLE") as demo:
    with gr.Tabs():
        print(
            f"build start {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        with gr.Tab("Home"):
            home.build()
            print(
                f"build home {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        with gr.Tab("Chat"):
            agent_chat_websocket.build()
            print(
                f"build chat {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        with gr.Tab("Edit Analyzer"):
            analyzer_editor.build()
            print(
                f"build edit analyzer {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        with gr.Tab("Create Analyzer"):
            analyzer_wizard.build()
            print(
                f"build create analyzer{__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        with gr.Tab("Evaluations"):
            result_evaluator.build()
            print(
                f"build evaluations {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        with gr.Tab("Pricing Calculator"):
            pricing_calculator.build()
            print(
                f"build pricing calculator {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        with gr.Tab("Observability"):
            agent_observability.build()
            print(
                f"build observability {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        with gr.Tab("Conversation Log"):
            chat_log.build()
            print(
                f"build conversation log {__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            )

if __name__ == "__main__":
    demo.launch(
        css=load_css(),
        favicon_path=str(FAVICON_PATH),
    )
