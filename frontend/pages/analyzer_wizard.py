"""
Analyzer Creation Wizard - Gradio-based tool for creating new analyzers.

This wizard guides users through creating a new analyzer by:
1. Collecting basic info and description
2. Using LLM to generate roles, rules, and prompts
3. Allowing example image uploads
4. Generating all required files and deploying to S3

Custom analyzers are stored in S3 for deployment via the custom-analyzers CDK stack.
"""

from typing import Any
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, replace
import boto3
from botocore.exceptions import ClientError
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths relative to deployment folder (for local reference files)
DEPLOYMENT_ROOT = Path(__file__).parent.parent.parent / "deployment"
LAMBDAS_CODE = DEPLOYMENT_ROOT / "lambdas" / "code"
S3_FILES = DEPLOYMENT_ROOT / "s3_files"
MANIFESTS_DIR = S3_FILES / "manifests"
SCHEMAS_DIR = S3_FILES / "schemas"
PROMPTS_DIR = S3_FILES / "prompts"

# Reference analyzer to copy lambda handler from
REFERENCE_ANALYZER = "charts_analyzer"

# System prompt for LLM generation
SYSTEM_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts_wizard/analyzer_wizard_system_prompt.xml"
)
USER_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts_wizard/analyzer_wizard_user_prompt.xml"
)

# S3 configuration - loaded from environment
CONFIG_BUCKET = os.environ.get("S3_CONFIG_BUCKET", "")
CUSTOM_ANALYZERS_PREFIX = "custom-analyzers"


def get_s3_client():
    """Get S3 client."""
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))


def get_config_bucket() -> str:
    """Get the config bucket name, with fallback to CloudFormation lookup."""
    global CONFIG_BUCKET
    if CONFIG_BUCKET:
        return CONFIG_BUCKET

    # Try to get from CloudFormation stack outputs
    try:
        cfn = boto3.client(
            "cloudformation", region_name=os.environ.get("AWS_REGION", "us-west-2")
        )
        response = cfn.describe_stacks(StackName="badgers-s3")
        for output in response["Stacks"][0].get("Outputs", []):
            if output["OutputKey"] == "ConfigBucketName":
                CONFIG_BUCKET = output["OutputValue"]
                logger.info(
                    "Found config bucket from CloudFormation: %s", CONFIG_BUCKET
                )
                return CONFIG_BUCKET
    except Exception as e:
        logger.warning("Could not get config bucket from CloudFormation: %s", e)

    return CONFIG_BUCKET


# Available models
AVAILABLE_MODELS = {
    "Claude Sonnet 4.5": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Amazon Nova Premier": "us.amazon.nova-premier-v1:0",
}
MODEL_CHOICES = list(AVAILABLE_MODELS.keys())


def load_analyzer_registry() -> dict:
    """Load the custom analyzer registry from S3."""
    bucket = get_config_bucket()
    if not bucket:
        return {"analyzers": []}

    try:
        s3 = get_s3_client()
        response = s3.get_object(
            Bucket=bucket, Key=f"{CUSTOM_ANALYZERS_PREFIX}/analyzer_registry.json"
        )
        data = json.loads(response["Body"].read().decode("utf-8"))
        return dict(data) if isinstance(data, dict) else {"analyzers": []}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return {"analyzers": []}
        logger.error("Error loading registry: %s", e)
        return {"analyzers": []}


def save_analyzer_registry(registry: dict) -> bool:
    """Save the custom analyzer registry to S3."""
    bucket = get_config_bucket()
    if not bucket:
        logger.error("No config bucket configured")
        return False

    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=f"{CUSTOM_ANALYZERS_PREFIX}/analyzer_registry.json",
            Body=json.dumps(registry, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except Exception as e:
        logger.error("Error saving registry: %s", e)
        return False


def get_existing_analyzer_names() -> set[str]:
    """Get set of existing analyzer names to prevent duplicates."""
    existing: set[str] = set()

    # Check local base analyzers
    if LAMBDAS_CODE.exists():
        existing.update(
            d.name.replace("_analyzer", "")
            for d in LAMBDAS_CODE.iterdir()
            if d.is_dir()
        )
    if MANIFESTS_DIR.exists():
        existing.update(
            f.stem.replace("_analyzer", "") for f in MANIFESTS_DIR.glob("*.json")
        )

    # Check custom analyzers in S3 registry
    registry = load_analyzer_registry()
    for analyzer in registry.get("analyzers", []):
        name = analyzer.get("name", "").replace("_analyzer", "")
        if name:
            existing.add(name)

    return existing


def validate_analyzer_name(display_name: str) -> tuple[bool, str]:
    """Validate analyzer name doesn't already exist and meets length constraints."""
    if not display_name or not display_name.strip():
        return False, "Name is required"

    sanitized = sanitize_name(display_name)

    # Length constraints: tool name is "analyze_{name}_tool" (64 char limit)
    # 8 (analyze_) + name + 5 (_tool) = 13 chars overhead, max name = 51
    # Using 40 as practical limit for readability
    if len(sanitized) < 3:
        return False, "Name must be at least 3 characters."

    if len(sanitized) > 40:
        return (
            False,
            f"Name too long ({len(sanitized)} chars). Maximum is 40 characters.",
        )

    existing = get_existing_analyzer_names()

    if sanitized in existing:
        return False, f"Analyzer '{sanitized}' already exists. Choose a different name."

    if not sanitized.replace("_", "").isalnum():
        return False, "Name must contain only letters, numbers, spaces, or hyphens."

    return True, sanitized


@dataclass
class WizardState:
    """Tracks wizard progress and collected data."""

    analyzer_name: str = ""
    display_name: str = ""
    description: str = ""
    user_paragraph: str = ""

    # LLM-generated content (editable by user)
    generated_gestalt: str = ""
    generated_role: str = ""
    generated_rules: str = ""
    generated_context: str = ""
    generated_tasks: str = ""
    generated_format: str = ""

    # Example images
    example_images: list = field(default_factory=list)
    max_examples: int = 6

    # Model config
    primary_model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    fallback_models: list = field(
        default_factory=lambda: [
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.amazon.nova-premier-v1:0",
        ]
    )

    # Enhancement config
    enhancement_eligible: bool = False

    # Generated files preview
    manifest_json: dict = field(default_factory=dict)
    schema_json: list = field(default_factory=list)


def sanitize_name(name: str) -> str:
    """Convert display name to valid analyzer name."""
    return name.lower().replace(" ", "_").replace("-", "_")


def load_system_prompt() -> str:
    """Load system prompt from external XML file."""
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_PATH}")

    content = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    # Strip XML declaration if present
    if content.startswith("<?xml"):
        content = content.split("?>", 1)[1]
    return content.strip()


def load_user_prompt(analyzer_name: str, display_name: str, description: str) -> str:
    """Load user prompt template and substitute values."""
    if not USER_PROMPT_PATH.exists():
        raise FileNotFoundError(f"User prompt not found: {USER_PROMPT_PATH}")

    content = USER_PROMPT_PATH.read_text(encoding="utf-8")
    # Strip XML declaration if present
    if content.startswith("<?xml"):
        content = content.split("?>", 1)[1]

    # Substitute placeholders
    content = content.replace("{analyzer_name}", analyzer_name)
    content = content.replace("{display_name}", display_name)
    content = content.replace("{description}", description)
    return content.strip()


def generate_prompts_with_llm(
    description: str, analyzer_name: str, display_name: str
) -> dict[str, Any]:
    """Use Bedrock to generate analyzer prompts from user description."""
    logger.info("Generating prompts for analyzer: %s", analyzer_name)
    logger.info("Description: %s...", description[:100])

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    system_prompt = load_system_prompt()
    user_prompt = load_user_prompt(analyzer_name, display_name, description)

    response = bedrock.converse(
        modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": system_prompt}],
        inferenceConfig={"maxTokens": 4000, "temperature": 0.3},
    )

    result_text = response["output"]["message"]["content"][0]["text"]
    logger.info("LLM response received, length: %d", len(result_text))

    # Parse JSON from response
    try:
        import re

        text = result_text.strip()

        # Strip XML tags that might wrap the JSON (e.g., <response_format>...</response_format>)
        xml_tag_pattern = r"<(\w+)>\s*([\s\S]*?)\s*</\1>"
        xml_match = re.search(xml_tag_pattern, text)
        if xml_match:
            text = xml_match.group(2).strip()

        # Handle markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Try to find JSON object if there's extra text
        if not text.startswith("{"):
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                text = json_match.group(0)

        parsed: dict[str, Any] = json.loads(text)
        logger.info(
            "Successfully parsed LLM response with keys: %s", list(parsed.keys())
        )
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM response as JSON: %s", e)
        logger.warning("Raw response: %s...", result_text[:500])
        # Return placeholder if parsing fails
        return {
            "job_role": f"<job_role><role>You are a '{display_name} Specialist'.</role></job_role>",
            "rules": f"<{analyzer_name}_rules><rule>Analyze the provided content carefully.</rule></{analyzer_name}_rules>",
            "context": f"<context>This analyzer processes {description}</context>",
            "tasks": "<tasks><task>Extract and analyze relevant content.</task></tasks>",
            "format": "<format>Return structured analysis results.</format>",
        }


def generate_manifest(state: WizardState) -> dict:
    """Generate the manifest JSON for the analyzer."""
    from datetime import datetime, timezone

    # Avoid double _analyzer suffix in tool name
    base_name = state.analyzer_name
    if base_name.endswith("_analyzer"):
        base_name = base_name[:-9]  # Remove "_analyzer" suffix

    analyzer_name = f"{base_name}_analyzer"

    return {
        "tool": {
            "name": f"analyze_{base_name}_tool",
            "description": state.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute file path to the image to analyze.",
                    },
                    "aws_profile": {
                        "type": "string",
                        "description": "Optional AWS profile name for Bedrock authentication.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Runtime session ID",
                    },
                    "audit_mode": {
                        "type": "boolean",
                        "description": "When true, includes confidence scoring and human review flags in the analysis output",
                    },
                },
                "required": ["image_path", "session_id"],
            },
        },
        "analyzer": {
            "name": analyzer_name,
            "description": state.description,
            "enhancement_eligible": state.enhancement_eligible,
            "prompt_analyzer_prompt_base_path": "prompts",
            "prompt_files": [
                f"{base_name}_gestalt.xml",
                f"{base_name}_job_role.xml",
                f"{base_name}_context.xml",
                f"{base_name}_rules.xml",
                f"{base_name}_tasks.xml",
                f"{base_name}_format.xml",
            ],
            "examples_path": "prompts",
            "max_examples": len(state.example_images),
            "analysis_text": state.display_name.lower(),
            "expected_output_tokens": 4000,
            "model_selections": {
                "primary": state.primary_model,
                "fallback_list": state.fallback_models,
            },
        },
        "metadata": {
            "version": "1.0.0",
            "dependencies": ["boto3"],
            "wizard_managed": True,
            "last_modified": datetime.now(timezone.utc).isoformat(),
            "analyzer_type": "standard",
        },
    }


def generate_schema(state: WizardState) -> list:
    """Generate the schema JSON for the analyzer."""
    # Avoid double _analyzer suffix in tool name
    base_name = state.analyzer_name
    if base_name.endswith("_analyzer"):
        base_name = base_name[:-9]  # Remove "_analyzer" suffix

    return [
        {
            "name": f"analyze_{base_name}_tool",
            "description": state.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "S3 URL (s3://bucket/key) or absolute file path to the image.",
                    },
                    "image_data": {
                        "type": "string",
                        "description": "Base64-encoded image data. Use this OR image_path, not both.",
                    },
                    "aws_profile": {
                        "type": "string",
                        "description": "Optional AWS profile name for Bedrock authentication.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Runtime session ID",
                    },
                    "audit_mode": {
                        "type": "boolean",
                        "description": "When true, includes confidence scoring and human review flags in the analysis output",
                    },
                },
                "required": ["session_id"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Analysis result"},
                    "success": {
                        "type": "boolean",
                        "description": "Whether analysis completed successfully",
                    },
                },
            },
        }
    ]


def deploy_custom_analyzers_cdk() -> str:
    """Run the CDK deployment script for custom analyzers."""
    import subprocess

    deploy_script = DEPLOYMENT_ROOT / "deploy_custom_analyzers.sh"
    if not deploy_script.exists():
        return f"❌ Deploy script not found: {deploy_script}"

    try:
        logger.info("Running CDK deployment script: %s", deploy_script)
        result = subprocess.run(
            [str(deploy_script)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(DEPLOYMENT_ROOT),
            check=False,
        )

        output_lines = []
        if result.stdout:
            output_lines.append(result.stdout)
        if result.stderr:
            output_lines.append("\n--- STDERR ---\n")
            output_lines.append(result.stderr)

        if result.returncode == 0:
            output_lines.insert(0, "✅ CDK DEPLOYMENT SUCCESSFUL\n\n")
        else:
            output_lines.insert(
                0, f"❌ CDK DEPLOYMENT FAILED (exit code {result.returncode})\n\n"
            )

        return "".join(output_lines)

    except subprocess.TimeoutExpired:
        return "❌ Deployment timed out after 10 minutes"
    except Exception as e:
        logger.error("CDK deployment failed: %s", str(e), exc_info=True)
        return f"❌ Deployment error: {str(e)}"


def create_analyzer(state: WizardState) -> str:
    """Save all analyzer files locally to deployment/custom_analyzers/."""
    logger.info("Saving analyzer locally: %s", state.analyzer_name)
    results = []

    if not state.analyzer_name:
        return "❌ No analyzer configured. Complete the wizard steps first."

    # Avoid double _analyzer suffix
    if state.analyzer_name.endswith("_analyzer"):
        analyzer_name = state.analyzer_name
    else:
        analyzer_name = f"{state.analyzer_name}_analyzer"

    base_name = state.analyzer_name
    if base_name.endswith("_analyzer"):
        base_name = base_name[:-9]

    # Save to deployment/custom_analyzers/
    custom_dir = DEPLOYMENT_ROOT / "custom_analyzers"
    custom_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Save manifest
        manifests_dir = custom_dir / "manifests"
        manifests_dir.mkdir(exist_ok=True)
        manifest_path = manifests_dir / f"{analyzer_name}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(state.manifest_json, f, indent=4)
        results.append(f"✓ Saved manifest: {manifest_path}")

        # 2. Save schema
        schemas_dir = custom_dir / "schemas"
        schemas_dir.mkdir(exist_ok=True)
        schema_path = schemas_dir / f"{analyzer_name}.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(state.schema_json, f, indent=4)
        results.append(f"✓ Saved schema: {schema_path}")

        # 3. Save XML prompt files
        prompts_dir = custom_dir / "prompts" / analyzer_name
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prompt_files = {
            f"{base_name}_gestalt.xml": state.generated_gestalt,
            f"{base_name}_job_role.xml": state.generated_role,
            f"{base_name}_rules.xml": state.generated_rules,
            f"{base_name}_context.xml": state.generated_context,
            f"{base_name}_tasks.xml": state.generated_tasks,
            f"{base_name}_format.xml": state.generated_format,
        }

        for filename, content in prompt_files.items():
            prompt_path = prompts_dir / filename
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(content)
        results.append(f"✓ Saved {len(prompt_files)} prompt files to {prompts_dir}/")

        # 4. Copy example images
        if state.example_images:
            import shutil

            examples_dir = prompts_dir / "few-shot-images"
            examples_dir.mkdir(exist_ok=True)
            for i, img_path in enumerate(state.example_images):
                ext = Path(img_path).suffix
                dest = examples_dir / f"example_{i+1}{ext}"
                shutil.copy2(img_path, dest)
            results.append(f"✓ Copied {len(state.example_images)} example images")

        # 5. Update local analyzer registry
        registry_path = custom_dir / "analyzer_registry.json"
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
        else:
            registry = {"analyzers": []}

        # Remove existing entry if present (for updates)
        registry["analyzers"] = [
            a for a in registry.get("analyzers", []) if a.get("name") != analyzer_name
        ]

        # Add new entry
        from datetime import datetime, timezone

        registry["analyzers"].append(
            {
                "name": analyzer_name,
                "display_name": state.display_name,
                "description": state.description,
                "enabled": True,
                "max_tokens": 8000,
                "temperature": 0.1,
                "memory_size": 2048,
                "concurrency": 5,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "wizard_managed": True,
            }
        )

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        results.append(f"✓ Updated registry ({len(registry['analyzers'])} analyzers)")

        results.append("\n" + "=" * 50)
        results.append("ANALYZER SAVED LOCALLY!")
        results.append("=" * 50)
        results.append(f"\nFiles saved to: {custom_dir}")
        results.append("\nClick 'Deploy Custom Analyzers Stack' to deploy to AWS.")
        logger.info("Local save complete for %s", analyzer_name)

    except Exception as e:
        logger.error("Save failed: %s", str(e), exc_info=True)
        results.append(f"❌ Error: {str(e)}")

    return "\n".join(results)


# ============================================================================
# GRADIO UI
# ============================================================================


def create_wizard():
    """Create the Gradio wizard interface."""

    with gr.Blocks(title="Analyzer Creation Wizard") as wizard_demo:
        # Shared state - must be inside Blocks context
        state = gr.State(WizardState())

        gr.Markdown("# 🧙 Analyzer Creation Wizard")
        gr.Markdown("Create a new document analyzer in 4 easy steps.")

        # Progress/status indicator - visible at top
        wizard_status = gr.Markdown("", elem_classes=["wizard-status"])

        with gr.Tabs() as tabs:

            # ================================================================
            # STEP 1: Basic Info
            # ================================================================
            with gr.Tab("1️⃣ Basic Info", id=0) as _tab1:
                gr.Markdown("### Tell us about your analyzer")

                display_name = gr.Textbox(
                    label="Analyzer Display Name",
                    placeholder="e.g., Medical Form, Invoice, Blueprint",
                    info="Human-readable name for your analyzer",
                )

                description = gr.Textbox(
                    label="Short Description",
                    placeholder="e.g., Analyzes medical intake forms to extract patient information and diagnoses",
                    lines=2,
                    info="One-line description of what this analyzer does",
                )

                user_paragraph = gr.Textbox(
                    label="Detailed Description",
                    placeholder="Describe in detail what this analyzer should look for, what kind of documents it will process, what information should be extracted, and any special considerations...",
                    lines=6,
                    info="The more detail you provide, the better the generated prompts will be",
                )

                gr.Markdown("### Model Selection")
                with gr.Row():
                    primary_model = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="Claude Sonnet 4.5",
                        label="Primary Model",
                        info="Main model for analysis",
                    )
                    fallback_model_1 = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="Claude Haiku 4.5",
                        label="Fallback Model 1",
                        info="First fallback if primary fails",
                    )
                    fallback_model_2 = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="Amazon Nova Premier",
                        label="Fallback Model 2",
                        info="Second fallback option",
                    )

                gr.Markdown("### Enhancement Options")
                enhancement_eligible = gr.Checkbox(
                    label="Enhancement Eligible",
                    value=False,
                    info="Enable image enhancement for degraded/historical documents. Recommended for handwriting, historical manuscripts, or annotated content.",
                )

                with gr.Row():
                    clear_wizard_btn = gr.Button(
                        "🗑️ Clear Form", variant="secondary", scale=1
                    )
                    gr.Markdown("")  # spacer
                    next_btn_1 = gr.Button(
                        "Generate Prompts →", variant="primary", scale=1
                    )

            # ================================================================
            # STEP 2: Review Generated Prompts
            # ================================================================
            with gr.Tab("2️⃣ Review Prompts", id=1) as _tab2:
                gr.Markdown("### Review and edit the generated prompts")
                gr.Markdown(
                    "*The LLM generated these based on your description. Feel free to edit.*"
                )

                with gr.Accordion("Gestalt Perception", open=True):
                    gr.Markdown(
                        "*Visual perception guidance - how to see before extracting*"
                    )
                    gestalt_editor = gr.Code(
                        label="gestalt.xml", language="html", lines=15
                    )

                with gr.Accordion("Job Role", open=True):
                    role_editor = gr.Code(
                        label="job_role.xml", language="html", lines=15
                    )

                with gr.Accordion("Rules", open=True):
                    rules_editor = gr.Code(label="rules.xml", language="html", lines=12)

                with gr.Accordion("Context", open=True):
                    context_editor = gr.Code(
                        label="context.xml", language="html", lines=10
                    )

                with gr.Accordion("Tasks", open=True):
                    tasks_editor = gr.Code(label="tasks.xml", language="html", lines=10)

                with gr.Accordion("Format", open=True):
                    format_editor = gr.Code(
                        label="format.xml", language="html", lines=10
                    )

                with gr.Row():
                    back_btn_2 = gr.Button("← Back", scale=1)
                    gr.Markdown("")  # spacer
                    next_btn_2 = gr.Button("Continue →", variant="primary", scale=1)

            # ================================================================
            # STEP 3: Example Images
            # ================================================================
            with gr.Tab("3️⃣ Examples", id=2) as _tab3:
                gr.Markdown("### Upload example images (optional)")
                gr.Markdown(
                    "*Few-shot examples help the analyzer understand what to look for. Max 6 images.*"
                )

                example_upload = gr.File(
                    label="Example Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )

                example_gallery = gr.Gallery(
                    label="Uploaded Examples", columns=3, height=200
                )

                with gr.Row():
                    back_btn_3 = gr.Button("← Back", scale=1)
                    gr.Markdown("")
                    next_btn_3 = gr.Button(
                        "Preview Config →", variant="primary", scale=1
                    )

            # ================================================================
            # STEP 4: Preview & Deploy
            # ================================================================
            with gr.Tab("4️⃣ Deploy", id=3) as _tab4:
                gr.Markdown("### Review configuration and deploy")

                gr.Markdown("**Files that will be created:**")
                files_summary = gr.Markdown("")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Manifest (manifest.json)**")
                        manifest_preview = gr.JSON(label="")

                    with gr.Column():
                        gr.Markdown("**Schema (schema.json)**")
                        schema_preview = gr.JSON(label="")

                gr.Markdown("---")

                with gr.Row():
                    back_btn_4 = gr.Button("← Back", scale=1)
                    create_btn = gr.Button(
                        "💾 Save Analyzer", variant="primary", scale=1
                    )

                deploy_output = gr.Textbox(
                    label="Save Status", lines=15, interactive=False
                )

                gr.Markdown("---")
                gr.Markdown("### Deploy to AWS")
                gr.Markdown("After saving, deploy the Lambda and Gateway target:")

                deploy_cdk_btn = gr.Button(
                    "🚀 Deploy Custom Analyzers Stack", variant="primary"
                )

                cdk_output = gr.Textbox(
                    label="CDK Deployment Output", lines=20, interactive=False
                )

        # ====================================================================
        # EVENT HANDLERS
        # ====================================================================

        def step1_next(
            display_name,
            description,
            user_paragraph,
            primary_model,
            fallback_1,
            fallback_2,
            enhancement_eligible_val,
            current_state,
        ):
            """Process step 1 and generate prompts with progress updates."""

            # Validation
            if not display_name or not description:
                yield (
                    current_state,
                    "❌ Please fill in the analyzer name and description.",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(selected=0),
                )
                return

            is_valid, result = validate_analyzer_name(display_name)
            if not is_valid:
                yield (
                    current_state,
                    f"❌ {result}",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(selected=0),
                )
                return

            # Show loading state FIRST
            yield (
                current_state,
                "⏳ Generating prompts with AI... (this may take 10-20 seconds)",
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(selected=0),  # Stay on current tab while loading
            )

            # Create new state
            new_state = WizardState(
                display_name=display_name,
                analyzer_name=result,
                description=description,
                user_paragraph=user_paragraph,
                primary_model=AVAILABLE_MODELS[primary_model],
                fallback_models=[
                    AVAILABLE_MODELS[fallback_1],
                    AVAILABLE_MODELS[fallback_2],
                ],
                enhancement_eligible=enhancement_eligible_val,
            )

            # Generate prompts
            full_description = (
                f"{description}\n\nDetails: {user_paragraph}"
                if user_paragraph
                else description
            )

            try:
                generated = generate_prompts_with_llm(
                    full_description, new_state.analyzer_name, display_name
                )

                new_state.generated_gestalt = generated.get("gestalt", "")
                new_state.generated_role = generated.get("job_role", "")
                new_state.generated_rules = generated.get("rules", "")
                new_state.generated_context = generated.get("context", "")
                new_state.generated_tasks = generated.get("tasks", "")
                new_state.generated_format = generated.get("format", "")

                logger.info("Generated role length: %d", len(new_state.generated_role))

            except Exception as e:
                logger.error("Error generating prompts: %s", e, exc_info=True)
                yield (
                    current_state,
                    f"❌ Error generating prompts: {str(e)}",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(selected=0),
                )
                return

            # Final yield with all the data
            yield (
                new_state,
                "✓ Prompts generated! Review and edit below.",
                gr.update(value=new_state.generated_gestalt),
                gr.update(value=new_state.generated_role),
                gr.update(value=new_state.generated_rules),
                gr.update(value=new_state.generated_context),
                gr.update(value=new_state.generated_tasks),
                gr.update(value=new_state.generated_format),
                gr.update(selected=1),
            )

        def step2_next(gestalt, role, rules, context, tasks, format_xml, current_state):
            """Save edited prompts and continue."""
            logger.info(
                "step2_next: saving prompts for %s", current_state.analyzer_name
            )

            new_state = replace(
                current_state,
                generated_gestalt=gestalt,
                generated_role=role,
                generated_rules=rules,
                generated_context=context,
                generated_tasks=tasks,
                generated_format=format_xml,
            )
            return new_state, gr.update(selected=2)

        def step3_next(files, current_state):
            """Process uploaded examples and generate config preview."""

            # First yield - show loading state
            yield (
                current_state,
                "⏳ Generating configuration preview...",
                gr.update(),
                gr.update(),
                gr.update(),  # Stay on current tab
            )

            logger.info("step3_next: analyzer_name=%s", current_state.analyzer_name)
            logger.info("step3_next: display_name=%s", current_state.display_name)

            example_images = files[:6] if files else []

            # Create new state with example images first
            new_state = replace(current_state, example_images=example_images)

            # Generate manifest and schema
            try:
                manifest = generate_manifest(new_state)
                schema = generate_schema(new_state)
                logger.info(
                    "Generated manifest: %s", json.dumps(manifest, indent=2)[:500]
                )
                logger.info("Generated schema: %s", json.dumps(schema, indent=2)[:500])
            except Exception as e:
                logger.error("Error generating config: %s", e, exc_info=True)
                yield (
                    current_state,
                    f"❌ Error generating configuration: {str(e)}",
                    gr.update(),
                    gr.update(),
                    gr.update(selected=2),  # Stay on examples tab
                )
                return

            new_state = replace(
                new_state,
                manifest_json=manifest,
                schema_json=schema,
            )

            # Generate file summary
            base_name = new_state.analyzer_name
            if base_name.endswith("_analyzer"):
                analyzer_name = base_name
                base_name = base_name[:-9]
            else:
                analyzer_name = f"{base_name}_analyzer"

            num_examples = len(new_state.example_images)

            # Get bucket for display
            bucket = get_config_bucket() or "<CONFIG_BUCKET>"

            summary_lines = [
                "📁 **Lambda Code:**",
                f"  - `s3://{bucket}/{CUSTOM_ANALYZERS_PREFIX}/code/{analyzer_name}/`",
                "",
                "📄 **Configuration:**",
                f"  - `s3://{bucket}/{CUSTOM_ANALYZERS_PREFIX}/manifests/{analyzer_name}.json`",
                f"  - `s3://{bucket}/{CUSTOM_ANALYZERS_PREFIX}/schemas/{analyzer_name}.json`",
                "",
                f"📝 **Prompts** (`s3://{bucket}/{CUSTOM_ANALYZERS_PREFIX}/prompts/{analyzer_name}/`):",
                f"  - `{base_name}_gestalt.xml`",
                f"  - `{base_name}_job_role.xml`",
                f"  - `{base_name}_rules.xml`",
                f"  - `{base_name}_context.xml`",
                f"  - `{base_name}_tasks.xml`",
                f"  - `{base_name}_format.xml`",
            ]

            if num_examples > 0:
                summary_lines.extend(
                    [
                        "",
                        f"🖼️ **Example Images:** {num_examples} file(s) in `few-shot-images/`",
                    ]
                )

            file_summary = "\n".join(summary_lines)

            logger.info("step3_next: returning manifest with %d keys", len(manifest))

            # Final yield with all the data
            yield (
                new_state,
                file_summary,
                gr.update(value=manifest),
                gr.update(value=schema),
                gr.update(selected=3),
            )

        def handle_deploy(current_state):
            """Deploy the analyzer."""
            return create_analyzer(current_state)

        def update_gallery(files):
            """Update the gallery with uploaded files."""
            if files:
                return files[:6]
            return []

        def clear_wizard():
            """Clear all wizard form fields and reset to initial state."""
            return (
                WizardState(),  # state
                "",  # wizard_status
                "",  # display_name
                "",  # description
                "",  # user_paragraph
                "Claude Sonnet 4.5",  # primary_model
                "Claude Haiku 4.5",  # fallback_model_1
                "Amazon Nova Premier",  # fallback_model_2
                False,  # enhancement_eligible
                "",  # gestalt_editor
                "",  # role_editor
                "",  # rules_editor
                "",  # context_editor
                "",  # tasks_editor
                "",  # format_editor
                None,  # example_upload
                [],  # example_gallery
                "",  # files_summary
                None,  # manifest_preview
                None,  # schema_preview
                "",  # deploy_output
                "",  # cdk_output
                gr.update(selected=0),  # tabs - go back to step 1
            )

        clear_wizard_btn.click(
            clear_wizard,
            outputs=[
                state,
                wizard_status,
                display_name,
                description,
                user_paragraph,
                primary_model,
                fallback_model_1,
                fallback_model_2,
                enhancement_eligible,
                gestalt_editor,
                role_editor,
                rules_editor,
                context_editor,
                tasks_editor,
                format_editor,
                example_upload,
                example_gallery,
                files_summary,
                manifest_preview,
                schema_preview,
                deploy_output,
                cdk_output,
                tabs,
            ],
        )

        # Wire up events with loading indicators
        next_btn_1.click(
            step1_next,
            inputs=[
                display_name,
                description,
                user_paragraph,
                primary_model,
                fallback_model_1,
                fallback_model_2,
                enhancement_eligible,
                state,
            ],
            outputs=[
                state,
                wizard_status,
                gestalt_editor,
                role_editor,
                rules_editor,
                context_editor,
                tasks_editor,
                format_editor,
                tabs,
            ],
        )

        next_btn_2.click(
            step2_next,
            inputs=[
                gestalt_editor,
                role_editor,
                rules_editor,
                context_editor,
                tasks_editor,
                format_editor,
                state,
            ],
            outputs=[state, tabs],
        )

        back_btn_2.click(lambda: gr.update(selected=0), outputs=[tabs])

        example_upload.change(
            update_gallery, inputs=[example_upload], outputs=[example_gallery]
        )

        next_btn_3.click(
            step3_next,
            inputs=[example_upload, state],
            outputs=[state, files_summary, manifest_preview, schema_preview, tabs],
        )

        back_btn_3.click(lambda: gr.update(selected=1), outputs=[tabs])
        back_btn_4.click(lambda: gr.update(selected=2), outputs=[tabs])

        create_btn.click(
            lambda: "⏳ Saving analyzer locally...",
            outputs=[deploy_output],
        ).then(handle_deploy, inputs=[state], outputs=[deploy_output])

        deploy_cdk_btn.click(
            lambda: "⏳ Running CDK deployment... this may take several minutes...",
            outputs=[cdk_output],
        ).then(deploy_custom_analyzers_cdk, outputs=[cdk_output])

    return wizard_demo


# Module-level demo for import
demo = create_wizard()

if __name__ == "__main__":
    demo.launch(share=False)
