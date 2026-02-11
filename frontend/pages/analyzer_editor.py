"""
Analyzer Editor - Gradio-based tool for editing wizard-managed analyzers.

Only shows analyzers that have "wizard_managed": true in their metadata.
Allows editing prompts, description, and model selections.

Custom analyzers are stored in S3 and loaded/saved from there.
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
import gradio as gr
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths relative to deployment folder (for local base analyzers)
DEPLOYMENT_ROOT = Path(__file__).parent.parent.parent / "deployment"
S3_FILES = DEPLOYMENT_ROOT / "s3_files"
MANIFESTS_DIR = S3_FILES / "manifests"
SCHEMAS_DIR = S3_FILES / "schemas"
PROMPTS_DIR = S3_FILES / "prompts"

# S3 configuration
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
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude Sonnet 4.5",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
    "us.amazon.nova-premier-v1:0": "Amazon Nova Premier",
}
MODEL_IDS = list(AVAILABLE_MODELS.keys())
MODEL_NAMES = list(AVAILABLE_MODELS.values())


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
        return json.loads(response["Body"].read().decode("utf-8"))
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


def get_wizard_managed_analyzers() -> dict:
    """Find all analyzers from S3 - both base and custom."""
    analyzers: dict[str, dict] = {}
    bucket = get_config_bucket()

    if not bucket:
        logger.warning("No config bucket configured")
        return analyzers

    s3 = get_s3_client()

    # Load CUSTOM analyzers from registry
    registry = load_analyzer_registry()
    for analyzer_entry in registry.get("analyzers", []):
        analyzer_name = analyzer_entry.get("name", "")
        if not analyzer_name:
            continue

        try:
            manifest_key = f"{CUSTOM_ANALYZERS_PREFIX}/manifests/{analyzer_name}.json"
            response = s3.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read().decode("utf-8"))

            analyzers[analyzer_name] = {
                "manifest": manifest,
                "registry_entry": analyzer_entry,
                "last_modified": analyzer_entry.get("created_at", "Unknown"),
                "source": "custom",
                "editable": True,
            }
            logger.info("Found custom analyzer: %s", analyzer_name)
        except ClientError as e:
            logger.warning(
                "Could not load custom manifest for %s: %s", analyzer_name, e
            )

    # Load BASE analyzers from manifests/ prefix
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix="manifests/")
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            analyzer_name = key.split("/")[-1].replace(".json", "")

            # Skip if already loaded as custom
            if analyzer_name in analyzers:
                continue

            try:
                manifest_response = s3.get_object(Bucket=bucket, Key=key)
                manifest = json.loads(manifest_response["Body"].read().decode("utf-8"))

                # Check if it's a valid analyzer manifest
                if "analyzer" not in manifest:
                    continue

                analyzers[analyzer_name] = {
                    "manifest": manifest,
                    "last_modified": obj.get("LastModified", "Unknown"),
                    "source": "base",
                    "editable": False,  # Base analyzers read-only by default
                }
                logger.info("Found base analyzer: %s", analyzer_name)
            except ClientError as e:
                logger.warning("Could not load base manifest %s: %s", key, e)
    except ClientError as e:
        logger.warning("Could not list base manifests: %s", e)

    return analyzers


def load_analyzer_prompts_from_s3(analyzer_name: str, source: str = "custom") -> dict:
    """Load all XML prompt files for an analyzer from S3."""
    prompts: dict[str, str] = {}
    bucket = get_config_bucket()

    if not bucket:
        return prompts

    s3 = get_s3_client()

    # Determine prefix based on source
    if source == "custom":
        prefix = f"{CUSTOM_ANALYZERS_PREFIX}/prompts/{analyzer_name}/"
    else:
        prefix = f"prompts/{analyzer_name}/"

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".xml"):
                filename = key.split("/")[-1]
                content_response = s3.get_object(Bucket=bucket, Key=key)
                prompts[filename] = content_response["Body"].read().decode("utf-8")
    except Exception as e:
        logger.error("Error loading prompts from S3: %s", e)

    return prompts


def save_analyzer(
    analyzer_name: str,
    description: str,
    primary_model: str,
    fallback_1: str,
    fallback_2: str,
    enhancement_eligible_val: bool,
    gestalt_xml: str,
    role_xml: str,
    rules_xml: str,
    context_xml: str,
    tasks_xml: str,
    format_xml: str,
    source: str = "custom",
) -> str:
    """Save changes to an analyzer in S3."""
    logger.info("Saving changes to analyzer: %s (source: %s)", analyzer_name, source)
    results = []

    bucket = get_config_bucket()
    if not bucket:
        return "❌ Error: CONFIG_BUCKET not configured"

    s3 = get_s3_client()

    # Determine S3 prefix based on source
    if source == "custom":
        prefix = f"{CUSTOM_ANALYZERS_PREFIX}/"
    else:
        prefix = ""  # Base analyzers use root prefixes

    try:
        # Load current manifest from S3
        manifest_key = f"{prefix}manifests/{analyzer_name}.json"
        response = s3.get_object(Bucket=bucket, Key=manifest_key)
        manifest = json.loads(response["Body"].read().decode("utf-8"))

        # Update manifest
        manifest["tool"]["description"] = description
        manifest["analyzer"]["description"] = description
        manifest["analyzer"]["enhancement_eligible"] = enhancement_eligible_val
        manifest["analyzer"]["model_selections"]["primary"] = MODEL_IDS[
            MODEL_NAMES.index(primary_model)
        ]
        manifest["analyzer"]["model_selections"]["fallback_list"] = [
            MODEL_IDS[MODEL_NAMES.index(fallback_1)],
            MODEL_IDS[MODEL_NAMES.index(fallback_2)],
        ]
        manifest["metadata"]["last_modified"] = datetime.now(timezone.utc).isoformat()

        # Save manifest to S3
        s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=4).encode("utf-8"),
            ContentType="application/json",
        )
        results.append("✓ Updated manifest in S3")

        # Update schema in S3
        schema_key = f"{prefix}schemas/{analyzer_name}.json"
        try:
            schema_response = s3.get_object(Bucket=bucket, Key=schema_key)
            schema = json.loads(schema_response["Body"].read().decode("utf-8"))
            if schema and len(schema) > 0:
                schema[0]["description"] = description
                s3.put_object(
                    Bucket=bucket,
                    Key=schema_key,
                    Body=json.dumps(schema, indent=4).encode("utf-8"),
                    ContentType="application/json",
                )
                results.append("✓ Updated schema in S3")
        except ClientError:
            logger.warning("Schema not found, skipping")

        # Save prompt files to S3
        base_name = analyzer_name.replace("_analyzer", "")

        prompt_files = {
            f"{base_name}_gestalt.xml": gestalt_xml,
            f"{base_name}_job_role.xml": role_xml,
            f"{base_name}_rules.xml": rules_xml,
            f"{base_name}_context.xml": context_xml,
            f"{base_name}_tasks.xml": tasks_xml,
            f"{base_name}_format.xml": format_xml,
        }

        for filename, content in prompt_files.items():
            if content:
                prompt_key = f"{prefix}prompts/{analyzer_name}/{filename}"
                s3.put_object(
                    Bucket=bucket,
                    Key=prompt_key,
                    Body=content.encode("utf-8"),
                    ContentType="application/xml",
                )
                results.append(f"✓ Updated prompt: {filename}")

        # Update registry timestamp (only for custom analyzers)
        if source == "custom":
            registry = load_analyzer_registry()
            for analyzer in registry.get("analyzers", []):
                if analyzer.get("name") == analyzer_name:
                    analyzer["description"] = description
                    analyzer["last_modified"] = datetime.now(timezone.utc).isoformat()
                    break
            save_analyzer_registry(registry)

        results.append("\n" + "=" * 50)
        results.append("SAVE COMPLETE!")
        results.append("=" * 50)
        results.append(f"\nLast modified: {manifest['metadata']['last_modified']}")

        if source == "custom":
            results.append("\nTo deploy changes:")
            results.append("  cd deployment")
            results.append("  ./sync_custom_analyzers.sh")
            results.append("  cdk deploy badgers-custom-analyzers")
        else:
            results.append("\n⚠️ BASE ANALYZER MODIFIED")
            results.append(
                "Changes will be overwritten on next base deployment update."
            )
            results.append("Consider creating a custom analyzer instead.")

        logger.info("Successfully saved analyzer: %s", analyzer_name)

    except Exception as e:
        logger.error("Error saving analyzer: %s", e, exc_info=True)
        results.append(f"✗ Error: {str(e)}")

    return "\n".join(results)


def create_editor():
    """Create the Gradio editor interface."""

    # Get available analyzers
    analyzers = get_wizard_managed_analyzers()

    # Build dropdown choices with type labels
    analyzer_choices = []
    for name, data in analyzers.items():
        source = data.get("source", "custom")
        label = f"{name} (Base)" if source == "base" else f"{name} (Custom)"
        analyzer_choices.append(label)

    with gr.Blocks(title="Analyzer Editor") as demo:
        # State for current analyzer
        current_analyzer = gr.State("")
        current_source = gr.State("custom")

        gr.Markdown("# ✏️ Analyzer Editor")
        gr.Markdown(
            "Edit analyzers. Base analyzers are read-only by default. Custom analyzers are always editable."
        )

        if not analyzer_choices:
            gr.Markdown(
                "⚠️ **No analyzers found.** Create one using the Analyzer Wizard first."
            )
            return demo

        with gr.Row():
            analyzer_dropdown = gr.Dropdown(
                choices=analyzer_choices,
                label="Select Analyzer",
                info="Base = shipped with deployment, Custom = created via wizard",
            )
            load_btn = gr.Button("Load", variant="primary")

        analyzer_info = gr.Markdown("")

        # Edit toggle for base analyzers (hidden by default)
        with gr.Row(visible=False) as edit_toggle_row:
            enable_edit_toggle = gr.Checkbox(
                label="Enable editing (Base analyzer)",
                value=False,
                info="⚠️ Warning: Changes to base analyzers may be overwritten on deployment updates",
            )

        with gr.Tabs() as tabs:
            # Settings tab
            with gr.Tab("⚙️ Settings"):
                description_input = gr.Textbox(
                    label="Description",
                    lines=3,
                    info="Tool description shown to users",
                )

                gr.Markdown("### Model Selection")
                with gr.Row():
                    primary_model = gr.Dropdown(
                        choices=MODEL_NAMES,
                        label="Primary Model",
                    )
                    fallback_1 = gr.Dropdown(
                        choices=MODEL_NAMES,
                        label="Fallback Model 1",
                    )
                    fallback_2 = gr.Dropdown(
                        choices=MODEL_NAMES,
                        label="Fallback Model 2",
                    )

                gr.Markdown("### Enhancement Options")
                enhancement_eligible = gr.Checkbox(
                    label="Enhancement Eligible",
                    value=False,
                    info="Enable image enhancement for degraded/historical documents before analysis",
                )

            # Prompts tab
            with gr.Tab("📝 Prompts"):
                with gr.Accordion("Gestalt Perception", open=True):
                    gr.Markdown(
                        "*Visual perception guidance - how to see before extracting*"
                    )
                    gestalt_editor = gr.Code(
                        label="gestalt.xml", language="html", lines=15
                    )

                with gr.Accordion("Job Role", open=True):
                    role_editor = gr.Code(
                        label="job_role.xml", language="html", lines=12
                    )

                with gr.Accordion("Rules", open=True):
                    rules_editor = gr.Code(label="rules.xml", language="html", lines=10)

                with gr.Accordion("Context", open=False):
                    context_editor = gr.Code(
                        label="context.xml", language="html", lines=8
                    )

                with gr.Accordion("Tasks", open=False):
                    tasks_editor = gr.Code(label="tasks.xml", language="html", lines=8)

                with gr.Accordion("Format", open=False):
                    format_editor = gr.Code(
                        label="format.xml", language="html", lines=8
                    )

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("")
            clear_btn = gr.Button("🗑️ Clear Form", variant="secondary", scale=1)
            save_btn = gr.Button("💾 Save Changes", variant="primary", scale=1)

        save_output = gr.Textbox(label="Status", lines=10, interactive=False)

        # Event handlers
        def load_analyzer(dropdown_value):
            """Load analyzer data from S3 into the form."""
            if not dropdown_value:
                return (
                    "",  # current_analyzer
                    "custom",  # current_source
                    "",  # analyzer_info
                    gr.update(visible=False),  # edit_toggle_row
                    False,  # enable_edit_toggle
                    gr.update(interactive=True),  # save_btn
                    "",  # description
                    None,
                    None,
                    None,  # models
                    False,  # enhancement_eligible
                    "",  # gestalt
                    "",
                    "",
                    "",
                    "",
                    "",  # prompts
                )

            # Parse dropdown value to get name and source
            # Format: "analyzer_name (Base)" or "analyzer_name (Custom)"
            if dropdown_value.endswith(" (Base)"):
                name = dropdown_value[:-7]
                source = "base"
            elif dropdown_value.endswith(" (Custom)"):
                name = dropdown_value[:-9]
                source = "custom"
            else:
                name = dropdown_value
                source = "custom"

            logger.info("Loading analyzer: %s (source: %s)", name, source)

            bucket = get_config_bucket()
            if not bucket:
                return (
                    "",
                    "custom",
                    "❌ CONFIG_BUCKET not configured",
                    gr.update(visible=False),
                    False,
                    gr.update(interactive=False),
                    "",
                    None,
                    None,
                    None,
                    False,
                    "",  # gestalt
                    "",
                    "",
                    "",
                    "",
                    "",
                )

            s3 = get_s3_client()

            # Determine manifest path based on source
            if source == "custom":
                manifest_key = f"{CUSTOM_ANALYZERS_PREFIX}/manifests/{name}.json"
            else:
                manifest_key = f"manifests/{name}.json"

            # Load manifest from S3
            try:
                response = s3.get_object(Bucket=bucket, Key=manifest_key)
                manifest = json.loads(response["Body"].read().decode("utf-8"))
            except ClientError as e:
                logger.error("Could not load manifest: %s", e)
                return (
                    "",
                    "custom",
                    f"❌ Could not load manifest: {e}",
                    gr.update(visible=False),
                    False,
                    gr.update(interactive=False),
                    "",
                    None,
                    None,
                    None,
                    False,
                    "",  # gestalt
                    "",
                    "",
                    "",
                    "",
                    "",
                )

            # Extract data
            description = manifest.get("tool", {}).get("description", "")
            model_sel = manifest.get("analyzer", {}).get("model_selections", {})
            primary = model_sel.get("primary", MODEL_IDS[0])
            fallbacks = model_sel.get("fallback_list", MODEL_IDS[1:3])
            enhancement_eligible_val = manifest.get("analyzer", {}).get(
                "enhancement_eligible", False
            )

            # Convert model IDs to names
            primary_name = AVAILABLE_MODELS.get(primary, MODEL_NAMES[0])
            fallback_1_name = AVAILABLE_MODELS.get(
                fallbacks[0] if fallbacks else MODEL_IDS[1], MODEL_NAMES[1]
            )
            fallback_2_name = AVAILABLE_MODELS.get(
                fallbacks[1] if len(fallbacks) > 1 else MODEL_IDS[2], MODEL_NAMES[2]
            )

            # Load prompts from S3
            prompts = load_analyzer_prompts_from_s3(name, source)
            base_name = name.replace("_analyzer", "")

            role_xml = prompts.get(f"{base_name}_job_role.xml", "")
            rules_xml = prompts.get(f"{base_name}_rules.xml", "")
            context_xml = prompts.get(f"{base_name}_context.xml", "")
            tasks_xml = prompts.get(f"{base_name}_tasks.xml", "")
            format_xml = prompts.get(f"{base_name}_format.xml", "")
            gestalt_xml = prompts.get(f"{base_name}_gestalt.xml", "")

            # Info text
            last_mod = manifest.get("metadata", {}).get("last_modified", "Unknown")
            source_label = "Base (read-only)" if source == "base" else "Custom"
            info = f"**Loaded:** {name} | **Last modified:** {last_mod} | **Type:** {source_label}"

            # For base analyzers: show toggle, disable save by default
            # For custom analyzers: hide toggle, enable save
            show_toggle = source == "base"
            save_enabled = source == "custom"

            return (
                name,  # current_analyzer state
                source,  # current_source state
                info,  # analyzer_info
                gr.update(visible=show_toggle),  # edit_toggle_row
                False,  # enable_edit_toggle (reset to unchecked)
                gr.update(interactive=save_enabled),  # save_btn
                description,
                primary_name,
                fallback_1_name,
                fallback_2_name,
                enhancement_eligible_val,
                gestalt_xml,
                role_xml,
                rules_xml,
                context_xml,
                tasks_xml,
                format_xml,
            )

        def handle_save(
            analyzer_name,
            source,
            description,
            primary,
            fallback_1,
            fallback_2,
            enhancement_eligible_val,
            gestalt_xml,
            role_xml,
            rules_xml,
            context_xml,
            tasks_xml,
            format_xml,
        ):
            if not analyzer_name:
                return "❌ No analyzer selected"

            return save_analyzer(
                analyzer_name,
                description,
                primary,
                fallback_1,
                fallback_2,
                enhancement_eligible_val,
                gestalt_xml,
                role_xml,
                rules_xml,
                context_xml,
                tasks_xml,
                format_xml,
                source,
            )

        def toggle_edit_mode(enabled):
            """Enable/disable save button based on edit toggle."""
            return gr.update(interactive=enabled)

        def clear_form():
            """Clear all form fields."""
            return (
                "",  # current_analyzer
                "custom",  # current_source
                "",  # analyzer_info
                gr.update(visible=False),  # edit_toggle_row
                False,  # enable_edit_toggle
                gr.update(interactive=True),  # save_btn
                "",  # description
                MODEL_NAMES[0],  # primary_model
                MODEL_NAMES[1],  # fallback_1
                MODEL_NAMES[2],  # fallback_2
                False,  # enhancement_eligible
                "",  # gestalt_editor
                "",  # role_editor
                "",  # rules_editor
                "",  # context_editor
                "",  # tasks_editor
                "",  # format_editor
                "",  # save_output
                None,  # analyzer_dropdown
            )

        # Wire events
        clear_btn.click(
            clear_form,
            outputs=[
                current_analyzer,
                current_source,
                analyzer_info,
                edit_toggle_row,
                enable_edit_toggle,
                save_btn,
                description_input,
                primary_model,
                fallback_1,
                fallback_2,
                enhancement_eligible,
                gestalt_editor,
                role_editor,
                rules_editor,
                context_editor,
                tasks_editor,
                format_editor,
                save_output,
                analyzer_dropdown,
            ],
        )

        load_btn.click(
            load_analyzer,
            inputs=[analyzer_dropdown],
            outputs=[
                current_analyzer,
                current_source,
                analyzer_info,
                edit_toggle_row,
                enable_edit_toggle,
                save_btn,
                description_input,
                primary_model,
                fallback_1,
                fallback_2,
                enhancement_eligible,
                gestalt_editor,
                role_editor,
                rules_editor,
                context_editor,
                tasks_editor,
                format_editor,
            ],
        )

        enable_edit_toggle.change(
            toggle_edit_mode,
            inputs=[enable_edit_toggle],
            outputs=[save_btn],
        )

        save_btn.click(
            lambda: "⏳ Saving changes...",
            outputs=[save_output],
        ).then(
            handle_save,
            inputs=[
                current_analyzer,
                current_source,
                description_input,
                primary_model,
                fallback_1,
                fallback_2,
                enhancement_eligible,
                gestalt_editor,
                role_editor,
                rules_editor,
                context_editor,
                tasks_editor,
                format_editor,
            ],
            outputs=[save_output],
        )

    return demo


# Module-level demo for import
demo = create_editor()

if __name__ == "__main__":
    demo.launch(share=False)
