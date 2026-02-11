"""
Pricing Calculator - Estimate costs for document/image analysis.
"""

import json
import os
from pathlib import Path

import boto3
import gradio as gr

CONFIG_PATH = Path(__file__).parent.parent / "config" / "pricing_config.json"

# Fixed image tokens - all images normalized to max 2048px by ImageProcessor
FIXED_IMAGE_TOKENS = 1600
CHARS_PER_TOKEN = 4.5


def load_config() -> dict:
    """Load pricing configuration."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "ingestion": {
            "characters_per_token": 4.5,
            "avg_characters_per_word": 5,
            "avg_words_per_page": 500,
            "avg_pages_per_document": 15,
            "avg_tokens_per_image": 1600,
        },
        "models": {
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0": {
                "name": "Claude Sonnet 4.5",
                "input_cost_per_million": 3.00,
                "output_cost_per_million": 15.00,
            },
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": {
                "name": "Claude Haiku 4.5",
                "input_cost_per_million": 1.00,
                "output_cost_per_million": 5.00,
            },
            "us.amazon.nova-premier-v1:0": {
                "name": "Amazon Nova Premier",
                "input_cost_per_million": 2.50,
                "output_cost_per_million": 12.50,
            },
        },
    }


def load_analyzer_manifests() -> dict:
    """Load all analyzer manifests from S3 and calculate prompt token counts."""
    analyzers = {}

    config_bucket = os.getenv("S3_CONFIG_BUCKET")
    if not config_bucket:
        return analyzers

    aws_profile = os.getenv("AWS_PROFILE")
    session = (
        boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    )
    s3 = session.client("s3")

    try:
        # List all manifest files
        response = s3.list_objects_v2(Bucket=config_bucket, Prefix="manifests/")
        if "Contents" not in response:
            return analyzers

        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            try:
                manifest_response = s3.get_object(Bucket=config_bucket, Key=key)
                manifest = json.loads(manifest_response["Body"].read().decode("utf-8"))

                # Skip non-analyzer manifests
                if "analyzer" not in manifest:
                    continue

                analyzer = manifest["analyzer"]
                name = analyzer.get("name", Path(key).stem)

                # Calculate prompt tokens from actual prompt files in S3
                prompt_tokens = 0
                prompt_files = analyzer.get("prompt_files", [])

                for pf in prompt_files:
                    prompt_key = f"prompts/{name}/{pf}"
                    try:
                        prompt_response = s3.get_object(
                            Bucket=config_bucket, Key=prompt_key
                        )
                        content = prompt_response["Body"].read().decode("utf-8")
                        prompt_tokens += int(len(content) / CHARS_PER_TOKEN)
                    except s3.exceptions.NoSuchKey:
                        continue

                analyzers[name] = {
                    "name": name,
                    "description": analyzer.get("description", ""),
                    "analysis_text": analyzer.get("analysis_text", name),
                    "prompt_tokens": prompt_tokens,
                    "expected_output_tokens": analyzer.get(
                        "expected_output_tokens", 2000
                    ),
                    "prompt_files": prompt_files,
                }
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception:
        return analyzers

    return analyzers


def get_model_choices(config: dict) -> list:
    return [m["name"] for m in config["models"].values()]


def get_preset_choices(config: dict) -> list:
    presets = config.get("presets", {})
    return [p["name"] for p in presets.values()]


def get_preset_by_name(config: dict, preset_name: str) -> dict | None:
    for preset in config.get("presets", {}).values():
        if preset["name"] == preset_name:
            return preset
    return None


def get_model_pricing(config: dict, model_name: str) -> tuple:
    for model_info in config["models"].values():
        if model_info["name"] == model_name:
            return (
                model_info["input_cost_per_million"],
                model_info["output_cost_per_million"],
            )
    first = list(config["models"].values())[0]
    return first["input_cost_per_million"], first["output_cost_per_million"]


def calculate_derived_values(cpt, cpw, wpp, ppd):
    chars_per_page = wpp * cpw
    tokens_per_page = chars_per_page / cpt
    words_per_doc = wpp * ppd
    chars_per_doc = chars_per_page * ppd
    tokens_per_doc = chars_per_doc / cpt
    return {
        "chars_per_page": chars_per_page,
        "tokens_per_page": round(tokens_per_page, 2),
        "words_per_doc": words_per_doc,
        "chars_per_doc": chars_per_doc,
        "tokens_per_doc": round(tokens_per_doc, 2),
    }


def calculate_costs(
    num_docs,
    num_imgs,
    tokens_per_doc,
    tokens_per_img,
    input_cost_m,
    output_cost_m,
    output_ratio,
):
    doc_tokens = num_docs * tokens_per_doc
    img_tokens = num_imgs * tokens_per_img
    total_input = doc_tokens + img_tokens
    total_output = total_input * output_ratio

    input_cost = (total_input / 1_000_000) * input_cost_m
    output_cost = (total_output / 1_000_000) * output_cost_m

    return {
        "total_input_tokens": round(total_input),
        "total_output_tokens": round(total_output),
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(input_cost + output_cost, 4),
    }


def calculate_advanced_costs(
    selected_analyzers: list,
    num_pages: int,
    input_cost_m: float,
    output_cost_m: float,
    analyzers_data: dict,
) -> dict:
    """Calculate costs based on selected analyzers and actual prompt sizes."""
    if not selected_analyzers:
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
            "breakdown": [],
        }

    breakdown = []
    total_input = 0
    total_output = 0

    for analyzer_name in selected_analyzers:
        if analyzer_name not in analyzers_data:
            continue

        analyzer = analyzers_data[analyzer_name]
        # Per page: prompt_tokens + fixed_image_tokens (input) + expected_output_tokens (output)
        prompt_tokens = analyzer["prompt_tokens"]
        output_tokens = analyzer["expected_output_tokens"]

        # Input = prompt + image tokens per page
        input_per_page = prompt_tokens + FIXED_IMAGE_TOKENS
        output_per_page = output_tokens

        analyzer_input = input_per_page * num_pages
        analyzer_output = output_per_page * num_pages

        total_input += analyzer_input
        total_output += analyzer_output

        breakdown.append(
            {
                "name": analyzer_name,
                "prompt_tokens": prompt_tokens,
                "image_tokens": FIXED_IMAGE_TOKENS,
                "output_tokens": output_tokens,
                "total_input": analyzer_input,
                "total_output": analyzer_output,
            }
        )

    input_cost = (total_input / 1_000_000) * input_cost_m
    output_cost = (total_output / 1_000_000) * output_cost_m

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(input_cost + output_cost, 4),
        "breakdown": breakdown,
    }


def create_calculator():
    config = load_config()
    ingestion = config["ingestion"]
    model_choices = get_model_choices(config)
    preset_choices = get_preset_choices(config)
    analyzers_data = load_analyzer_manifests()
    analyzer_choices = sorted(analyzers_data.keys())

    with gr.Blocks(title="Pricing Calculator") as demo:
        gr.Markdown("# 💰 Pricing Calculator")

        with gr.Tabs():
            # ==================== BASIC CALCULATOR TAB ====================
            with gr.TabItem("📊 Basic Calculator"):
                gr.Markdown(
                    "Estimate Bedrock costs for document/image analysis. "
                    "Select an industry preset to load recommended values, or adjust manually."
                )

                with gr.Row(elem_classes=["row-padding", "row-mustard"]):
                    with gr.Column(scale=1):
                        gr.Markdown("### 🏭 Industry Preset")
                        preset_dropdown = gr.Dropdown(
                            choices=preset_choices,
                            value=preset_choices[0] if preset_choices else None,
                            label="Select Preset",
                            info="Load industry-specific defaults",
                        )
                    with gr.Column(scale=2):
                        preset_description = gr.Markdown(
                            "*Select a preset to see its description*"
                        )

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 Model Selection")
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0],
                            label="Select Model",
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 💵 Model Pricing")
                        with gr.Row():
                            input_cost_display = gr.Number(
                                label="Input ($/M tokens)",
                                interactive=False,
                                precision=2,
                            )
                            output_cost_display = gr.Number(
                                label="Output ($/M tokens)",
                                interactive=False,
                                precision=2,
                            )
                            input_per_token = gr.Number(
                                label="Input ($/token)", interactive=False, precision=10
                            )
                            output_per_token = gr.Number(
                                label="Output ($/token)",
                                interactive=False,
                                precision=10,
                            )

                gr.Markdown("---")

                with gr.Row():
                    gr.Markdown("### 📊 Ingestion Values")
                    chars_per_token = gr.Number(
                        label="Characters per Token",
                        value=ingestion["characters_per_token"],
                        precision=2,
                    )
                    chars_per_word = gr.Number(
                        label="Avg Characters per Word",
                        value=ingestion["avg_characters_per_word"],
                        precision=0,
                    )
                    words_per_page = gr.Number(
                        label="Avg Words per Page",
                        value=ingestion["avg_words_per_page"],
                        precision=0,
                    )
                    pages_per_doc = gr.Number(
                        label="Avg Pages per Document",
                        value=ingestion["avg_pages_per_document"],
                        precision=0,
                    )
                    tokens_per_image = gr.Number(
                        label="Avg Tokens per Image",
                        value=ingestion["avg_tokens_per_image"],
                        precision=0,
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📈 Derived Values")
                        with gr.Accordion("> Click to show calculations <", open=False):
                            chars_per_page_display = gr.Number(
                                label="Characters per Page", interactive=False
                            )
                            tokens_per_page_display = gr.Number(
                                label="Tokens per Page", interactive=False
                            )
                            words_per_doc_display = gr.Number(
                                label="Words per Document", interactive=False
                            )
                            chars_per_doc_display = gr.Number(
                                label="Characters per Document", interactive=False
                            )
                            tokens_per_doc_display = gr.Number(
                                label="Tokens per Document", interactive=False
                            )
                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🧮 Cost Estimation Inputs")
                        num_documents = gr.Number(
                            label="Number of Documents", value=100, precision=0
                        )
                        num_images = gr.Number(
                            label="Number of Images", value=50, precision=0
                        )
                        output_ratio = gr.Slider(
                            label="Output/Input Token Ratio",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.5,
                            step=0.1,
                            info="Output tokens as ratio of input",
                        )
                    with gr.Column(scale=1):
                        with gr.Column(
                            scale=2, elem_classes=["row-padding", "background-blue"]
                        ):
                            gr.Markdown("### 📋 Cost Outputs")
                            with gr.Row():
                                total_input_tokens = gr.Number(
                                    label="Total Input Tokens", interactive=False
                                )
                                total_output_tokens = gr.Number(
                                    label="Total Output Tokens", interactive=False
                                )
                            with gr.Row():
                                input_cost_result = gr.Number(
                                    label="Input Cost ($)",
                                    interactive=False,
                                    precision=4,
                                )
                                output_cost_result = gr.Number(
                                    label="Output Cost ($)",
                                    interactive=False,
                                    precision=4,
                                )
                                total_cost_result = gr.Number(
                                    label="Total Cost ($)",
                                    interactive=False,
                                    precision=4,
                                )
                            cost_summary = gr.Markdown("")

                            with gr.Row():
                                calculate_btn = gr.Button(
                                    "🔢 Calculate Cost", variant="primary", size="lg"
                                )
                                reset_basic_btn = gr.Button(
                                    "🔄 Reset", variant="secondary", size="lg"
                                )
                gr.Markdown("---")

            # ==================== ADVANCED CALCULATOR TAB ====================
            with gr.TabItem("🔬 Advanced Calculator"):
                gr.Markdown(
                    "Calculate costs based on actual deployed analyzer prompts. "
                    "Select which analyzers to run per page and see precise token counts."
                )
                gr.Markdown(
                    f"*Image tokens fixed at {FIXED_IMAGE_TOKENS} (all images normalized to max 2048px)*"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 Model Selection")
                        adv_model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0],
                            label="Select Model",
                        )
                        gr.Markdown("### 📄 Pages to Process")
                        adv_num_pages = gr.Number(
                            label="Number of Pages", value=100, precision=0
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 🔧 Select Analyzers")
                        gr.Markdown("*Each selected analyzer runs once per page*")
                        adv_analyzer_select = gr.CheckboxGroup(
                            choices=analyzer_choices,
                            value=["full_text_analyzer"],
                            label="Analyzers to Include",
                        )

                gr.Markdown("---")

                with gr.Row():

                    with gr.Column(
                        scale=1, elem_classes=["row-padding", "background-blue"]
                    ):
                        gr.Markdown("### 📋 Cost Outputs")
                        with gr.Row():
                            adv_total_input = gr.Number(
                                label="Total Input Tokens", interactive=False
                            )
                            adv_total_output = gr.Number(
                                label="Total Output Tokens", interactive=False
                            )
                        with gr.Row():
                            adv_input_cost = gr.Number(
                                label="Input Cost ($)", interactive=False, precision=4
                            )
                            adv_output_cost = gr.Number(
                                label="Output Cost ($)", interactive=False, precision=4
                            )
                        with gr.Row():
                            adv_total_cost = gr.Number(
                                label="Total Cost ($)", interactive=False, precision=4
                            )
                        with gr.Row():
                            adv_calculate_btn = gr.Button(
                                "🔢 Calculate Cost", variant="primary", size="lg"
                            )
                            reset_adv_btn = gr.Button(
                                "🔄 Reset", variant="secondary", size="lg"
                            )

                gr.Markdown("### 📊 Analyzer Breakdown")
                adv_breakdown = gr.Dataframe(
                    headers=[
                        "Analyzer",
                        "Prompt Tokens",
                        "Image Tokens",
                        "Output Tokens",
                        "Total Input",
                        "Total Output",
                    ],
                    datatype=["str", "number", "number", "number", "number", "number"],
                    interactive=False,
                )
                adv_summary = gr.Markdown("")

        # ==================== EVENT HANDLERS ====================
        def apply_preset(preset_name):
            cfg = load_config()
            preset = get_preset_by_name(cfg, preset_name)
            if not preset:
                return [gr.update()] * 6

            desc = f"**{preset['name']}:** {preset['description']}"
            return [
                desc,
                preset["recommended_model"],
                preset["words_per_page"],
                preset["pages_per_document"],
                preset["tokens_per_image"],
                preset["output_ratio"],
            ]

        def update_model_pricing(model_name):
            cfg = load_config()
            inp, out = get_model_pricing(cfg, model_name)
            return [inp, out, inp / 1_000_000, out / 1_000_000]

        def update_derived(cpt, cpw, wpp, ppd):
            d = calculate_derived_values(cpt, cpw, wpp, ppd)
            return [
                d["chars_per_page"],
                d["tokens_per_page"],
                d["words_per_doc"],
                d["chars_per_doc"],
                d["tokens_per_doc"],
            ]

        def run_calculation(cpt, cpw, wpp, ppd, tpi, model_name, nd, ni, ratio):
            cfg = load_config()
            derived = calculate_derived_values(cpt, cpw, wpp, ppd)
            inp_m, out_m = get_model_pricing(cfg, model_name)
            costs = calculate_costs(
                int(nd), int(ni), derived["tokens_per_doc"], tpi, inp_m, out_m, ratio
            )

            total_tokens = costs["total_input_tokens"] + costs["total_output_tokens"]
            cost_per_doc = costs["total_cost"] / int(nd) if int(nd) > 0 else 0
            cost_per_page = (
                costs["total_cost"] / (int(nd) * ppd) if int(nd) > 0 and ppd > 0 else 0
            )

            summary = (
                f"**Summary:** {int(nd):,} documents at {int(ppd)} pages each = "
                f"{total_tokens:,} total tokens. "
                f"**Cost per document:** ${cost_per_doc:.4f} | **Cost per page:** ${cost_per_page:.6f}"
            )

            return [
                costs["total_input_tokens"],
                costs["total_output_tokens"],
                costs["input_cost"],
                costs["output_cost"],
                costs["total_cost"],
                summary,
            ]

        def run_advanced_calculation(selected_analyzers, num_pages, model_name):
            cfg = load_config()
            inp_m, out_m = get_model_pricing(cfg, model_name)

            costs = calculate_advanced_costs(
                selected_analyzers,
                int(num_pages),
                inp_m,
                out_m,
                analyzers_data,
            )

            # Build breakdown table
            breakdown_data = [
                [
                    b["name"],
                    b["prompt_tokens"],
                    b["image_tokens"],
                    b["output_tokens"],
                    b["total_input"],
                    b["total_output"],
                ]
                for b in costs["breakdown"]
            ]

            cost_per_page = (
                costs["total_cost"] / int(num_pages) if int(num_pages) > 0 else 0
            )
            total_tokens = costs["total_input_tokens"] + costs["total_output_tokens"]

            summary = (
                f"**Summary:** {len(selected_analyzers)} analyzers × {int(num_pages):,} pages = "
                f"{total_tokens:,} total tokens. "
                f"**Cost per page:** ${cost_per_page:.6f}"
            )

            return [
                costs["total_input_tokens"],
                costs["total_output_tokens"],
                costs["input_cost"],
                costs["output_cost"],
                costs["total_cost"],
                breakdown_data,
                summary,
            ]

        # Wire basic calculator events
        def reset_basic():
            """Reset basic calculator to defaults."""
            cfg = load_config()
            ing = cfg["ingestion"]
            return [
                ing["characters_per_token"],
                ing["avg_characters_per_word"],
                ing["avg_words_per_page"],
                ing["avg_pages_per_document"],
                ing["avg_tokens_per_image"],
                100,  # num_documents
                50,  # num_images
                0.5,  # output_ratio
                0,  # total_input_tokens
                0,  # total_output_tokens
                0,  # input_cost
                0,  # output_cost
                0,  # total_cost
                "",  # cost_summary
            ]

        reset_basic_btn.click(
            reset_basic,
            outputs=[
                chars_per_token,
                chars_per_word,
                words_per_page,
                pages_per_doc,
                tokens_per_image,
                num_documents,
                num_images,
                output_ratio,
                total_input_tokens,
                total_output_tokens,
                input_cost_result,
                output_cost_result,
                total_cost_result,
                cost_summary,
            ],
        )

        preset_dropdown.change(
            apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                preset_description,
                model_dropdown,
                words_per_page,
                pages_per_doc,
                tokens_per_image,
                output_ratio,
            ],
        )

        model_dropdown.change(
            update_model_pricing,
            inputs=[model_dropdown],
            outputs=[
                input_cost_display,
                output_cost_display,
                input_per_token,
                output_per_token,
            ],
        )

        ingestion_inputs = [
            chars_per_token,
            chars_per_word,
            words_per_page,
            pages_per_doc,
        ]
        derived_outputs = [
            chars_per_page_display,
            tokens_per_page_display,
            words_per_doc_display,
            chars_per_doc_display,
            tokens_per_doc_display,
        ]
        for inp in ingestion_inputs:
            inp.change(update_derived, inputs=ingestion_inputs, outputs=derived_outputs)

        calculate_btn.click(
            run_calculation,
            inputs=[
                chars_per_token,
                chars_per_word,
                words_per_page,
                pages_per_doc,
                tokens_per_image,
                model_dropdown,
                num_documents,
                num_images,
                output_ratio,
            ],
            outputs=[
                total_input_tokens,
                total_output_tokens,
                input_cost_result,
                output_cost_result,
                total_cost_result,
                cost_summary,
            ],
        )

        # Wire advanced calculator events
        def reset_advanced():
            """Reset advanced calculator to defaults."""
            return [
                ["full_text_analyzer"],  # default analyzer selection
                100,  # num_pages
                0,  # total_input
                0,  # total_output
                0,  # input_cost
                0,  # output_cost
                0,  # total_cost
                [],  # breakdown
                "",  # summary
            ]

        reset_adv_btn.click(
            reset_advanced,
            outputs=[
                adv_analyzer_select,
                adv_num_pages,
                adv_total_input,
                adv_total_output,
                adv_input_cost,
                adv_output_cost,
                adv_total_cost,
                adv_breakdown,
                adv_summary,
            ],
        )

        adv_calculate_btn.click(
            run_advanced_calculation,
            inputs=[adv_analyzer_select, adv_num_pages, adv_model_dropdown],
            outputs=[
                adv_total_input,
                adv_total_output,
                adv_input_cost,
                adv_output_cost,
                adv_total_cost,
                adv_breakdown,
                adv_summary,
            ],
        )

        # Initialize on load
        demo.load(
            apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                preset_description,
                model_dropdown,
                words_per_page,
                pages_per_doc,
                tokens_per_image,
                output_ratio,
            ],
        )
        demo.load(
            update_model_pricing,
            inputs=[model_dropdown],
            outputs=[
                input_cost_display,
                output_cost_display,
                input_per_token,
                output_per_token,
            ],
        )
        demo.load(update_derived, inputs=ingestion_inputs, outputs=derived_outputs)

    return demo


demo = create_calculator()

if __name__ == "__main__":
    demo.launch(share=False)
