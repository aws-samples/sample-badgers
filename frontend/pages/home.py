import base64
import gradio as gr


def get_base64_image(path: str) -> str:
    """Convert image file to base64 string for HTML embedding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def create_home_page():
    """Create the BADGERS home page with overview content."""
    from pathlib import Path

    logo_path = Path(__file__).parent.parent / "images" / "badgers-logo.png"
    logo_base64 = get_base64_image(str(logo_path))

    with gr.Blocks() as demo_app:
        # Load logo as base64 for HTML embedding
        gr.HTML(
            f"""
           <div class="home-page">
    <div class="overview">
        <img src="data:image/png;base64,{logo_base64}" alt="BADGERS Logo" class="logo" />
        <h1>BADGERS Test Interface</h1>
        <p>This is a test harness for <b>BADGERS</b> (Broad Agentic Document Generative Extraction & Recognition System),
            a vision-enabled AI system that processes documents using specialized analyzers.</p>
        <hr />
        <h2>Available Tabs</h2>
        <ul>
            <li><b>Streaming Experience</b> - Chat with the BADGERS agent to process documents in real-time</li>
            <li><b>Agent Observability</b> - Monitor agent execution traces and performance metrics</li>
            <li><b>Edit Analyzer</b> - Modify existing analyzer prompts and configurations</li>
            <li><b>Create Analyzer</b> - Build new specialized analyzers using the wizard</li>
            <li><b>Evaluate Results</b> - Review and compare analysis outputs from previous sessions</li>
            <li><b>Pricing Calculator</b> - Estimate costs for document processing workloads</li>
            <li><b>Chat Log</b> - View historical chat conversations and agent interactions</li>
        </ul>
        <hr />
    </p>

    </div>
                    <div class="warning">
                <strong>⚠️ Notices</strong><br><br>
                Customers are responsible for making their own independent assessment of the information in this Guidance. This Guidance: (a) is for informational purposes only, (b) represents AWS current product offerings and practices, which are subject to change without notice, and (c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided "as is" without warranties, representations, or conditions of any kind, whether express or implied. AWS responsibilities and liabilities to its customers are controlled by AWS agreements, and this Guidance is not part of, nor does it modify, any agreement between AWS and its customers.
            </div>
</div>
        """
        )

    return demo_app


# Module-level demo for import
demo = create_home_page()

if __name__ == "__main__":
    demo.launch(share=False)
