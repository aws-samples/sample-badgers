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


def _fetch_exact_prompt_tokens(analyzer_name: str, prompt_files: list[str]) -> int:
    """Fetch actual prompt content from S3 and count tokens exactly."""
    config_bucket = os.getenv("S3_CONFIG_BUCKET")
    if not config_bucket or not prompt_files:
        return 0

    aws_profile = os.getenv("AWS_PROFILE")
    session = (
        boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    )
    s3 = session.client("s3")

    total_tokens = 0
    for pf in prompt_files:
        prompt_key = f"prompts/{analyzer_name}/{pf}"
        try:
            resp = s3.get_object(Bucket=config_bucket, Key=prompt_key)
            content = resp["Body"].read().decode("utf-8")
            total_tokens += int(len(content) / CHARS_PER_TOKEN)
        except Exception:
            continue
    return total_tokens


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
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
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
    """Load all analyzer manifests from S3 and calculate prompt token counts.
    Falls back to analyzer_defaults from config when S3 is unavailable.

    Prompt token counts are estimated from manifest metadata to avoid
    fetching each prompt file individually (which caused ~30s startup delay).
    """
    analyzers = {}

    config_bucket = os.getenv("S3_CONFIG_BUCKET")
    if config_bucket:
        aws_profile = os.getenv("AWS_PROFILE")
        session = (
            boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        )
        s3 = session.client("s3")

        try:
            response = s3.list_objects_v2(Bucket=config_bucket, Prefix="manifests/")
            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    if not key.endswith(".json"):
                        continue

                    try:
                        manifest_response = s3.get_object(Bucket=config_bucket, Key=key)
                        manifest = json.loads(
                            manifest_response["Body"].read().decode("utf-8")
                        )

                        if "analyzer" not in manifest:
                            continue

                        analyzer = manifest["analyzer"]
                        name = analyzer.get("name", Path(key).stem)

                        # Use pre-calculated token count from manifest if available,
                        # otherwise estimate from prompt file count.
                        # Avoids N extra S3 get_object calls per analyzer.
                        prompt_tokens = analyzer.get("prompt_tokens", 0)
                        if not prompt_tokens:
                            prompt_files = analyzer.get("prompt_files", [])
                            # Rough estimate: ~500 tokens per prompt file
                            prompt_tokens = len(prompt_files) * 500

                        # Resolve default model from manifest model_selections
                        default_model = "Claude Sonnet 4.5"
                        model_sel = analyzer.get("model_selections", {})
                        primary = model_sel.get("primary")
                        if primary:
                            model_id = (
                                primary.get("model_id")
                                if isinstance(primary, dict)
                                else primary
                            )
                            if model_id:
                                default_model = _resolve_model_display_name(model_id)

                        analyzers[name] = {
                            "name": name,
                            "description": analyzer.get("description", ""),
                            "analysis_text": analyzer.get("analysis_text", name),
                            "prompt_tokens": prompt_tokens,
                            "expected_output_tokens": analyzer.get(
                                "expected_output_tokens", 2000
                            ),
                            "prompt_files": analyzer.get("prompt_files", []),
                            "default_model": default_model,
                        }
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

    # Fall back to config defaults for any analyzers not loaded from S3
    if not analyzers:
        config = load_config()
        for name, info in config.get("analyzer_defaults", {}).items():
            analyzers[name] = {
                "name": name,
                "description": "",
                "analysis_text": name,
                "prompt_tokens": info["prompt_tokens"],
                "expected_output_tokens": info["expected_output_tokens"],
                "prompt_files": [],
                "default_model": info.get("default_model", "Claude Sonnet 4.5"),
            }

    return analyzers


def _resolve_model_display_name(model_id: str) -> str:
    """Map a Bedrock model ID to its display name from config."""
    config = load_config()
    for mid, info in config.get("models", {}).items():
        if mid == model_id or model_id in mid or mid in model_id:
            return info["name"]
    # Fallback heuristics
    ml = model_id.lower()
    if "opus" in ml and "4-6" in ml:
        return "Claude Opus 4.6"
    if "sonnet" in ml and "4-5" in ml:
        return "Claude Sonnet 4.5"
    if "haiku" in ml:
        return "Claude Haiku 4.5"
    if "nova-premier" in ml:
        return "Nova Premier"
    if "nova-pro" in ml:
        return "Nova Pro"
    if "nova-lite" in ml:
        return "Nova Lite"
    if "nova-micro" in ml:
        return "Nova Micro"
    return "Claude Sonnet 4.5"


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
    model_overrides: dict,
    config: dict,
    analyzers_data: dict,
) -> dict:
    """Calculate costs based on selected analyzers with per-analyzer model pricing."""
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
    total_input_cost = 0.0
    total_output_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for analyzer_name in selected_analyzers:
        if analyzer_name not in analyzers_data:
            continue

        analyzer = analyzers_data[analyzer_name]
        prompt_tokens = analyzer["prompt_tokens"]
        output_tokens = analyzer["expected_output_tokens"]

        input_per_page = prompt_tokens + FIXED_IMAGE_TOKENS
        output_per_page = output_tokens

        analyzer_input = input_per_page * num_pages
        analyzer_output = output_per_page * num_pages

        # Get per-analyzer model pricing
        model_name = model_overrides.get(
            analyzer_name, analyzer.get("default_model", "Claude Sonnet 4.5")
        )
        inp_m, out_m = get_model_pricing(config, model_name)

        a_input_cost = (analyzer_input / 1_000_000) * inp_m
        a_output_cost = (analyzer_output / 1_000_000) * out_m

        total_input_tokens += analyzer_input
        total_output_tokens += analyzer_output
        total_input_cost += a_input_cost
        total_output_cost += a_output_cost

        breakdown.append(
            {
                "name": analyzer_name,
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "image_tokens": FIXED_IMAGE_TOKENS,
                "output_tokens": output_tokens,
                "total_input": analyzer_input,
                "total_output": analyzer_output,
                "input_cost": round(a_input_cost, 6),
                "output_cost": round(a_output_cost, 6),
                "total_cost": round(a_input_cost + a_output_cost, 6),
            }
        )

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": round(total_input_cost, 4),
        "output_cost": round(total_output_cost, 4),
        "total_cost": round(total_input_cost + total_output_cost, 4),
        "breakdown": breakdown,
    }


def build():
    """Create the Pricing Calculator UI (no Blocks wrapper)."""
    config = load_config()
    ingestion = config["ingestion"]
    model_choices = get_model_choices(config)
    preset_choices = get_preset_choices(config)
    analyzers_data = load_analyzer_manifests()
    analyzer_choices = sorted(analyzers_data.keys())

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
                "Toggle analyzers on/off and set per-analyzer models, matching the Excel pricing sheet."
            )
            gr.Markdown(
                f"*Image tokens fixed at {FIXED_IMAGE_TOKENS} (all images normalized to max 2048px)*"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Pages to Process")
                    adv_num_pages = gr.Number(
                        label="Number of Pages", value=100, precision=0
                    )
                    adv_num_docs = gr.Number(
                        label="Documents (for per-doc cost)",
                        value=1,
                        precision=0,
                        info="Pages ÷ avg pages/doc for per-document cost",
                    )

            gr.Markdown("---")
            gr.Markdown("### 🔧 Analyzer Configuration")
            gr.Markdown(
                "*Toggle Include to add/remove analyzers. "
                "Change Model to override the default for each analyzer.*"
            )

            # Build per-analyzer rows with include toggle + model dropdown
            analyzer_include_checks = {}
            analyzer_model_dropdowns = {}

            # Default included analyzers (matching Excel "Yes" selections)
            default_included = {
                "charts_analyzer",
                "classify_pdf_content",
                "correlation_analyzer",
                "diagram_analyzer",
                "elements_analyzer",
                "general_visual_analysis",
                "handwriting_analyzer",
                "keyword_topic_analyzer",
                "pdf_processor",
                "robust_elements_analyzer",
                "table_analyzer",
            }

            # Table header
            with gr.Row():
                gr.Markdown("**Include?**", elem_classes=["col-header"])
                gr.Markdown("**Analyzer**", elem_classes=["col-header"])
                gr.Markdown("**Model**", elem_classes=["col-header"])
                gr.Markdown("**Prompt Tokens**", elem_classes=["col-header"])
                gr.Markdown("**Max Output Tokens**", elem_classes=["col-header"])

            for aname in analyzer_choices:
                adata = analyzers_data.get(aname, {})
                default_model = adata.get("default_model", "Claude Sonnet 4.5")
                prompt_tok = adata.get("prompt_tokens", 0)
                output_tok = adata.get("expected_output_tokens", 0)

                with gr.Row():
                    chk = gr.Checkbox(
                        value=aname in default_included,
                        label="",
                        show_label=False,
                        scale=1,
                    )
                    with gr.Column(scale=2, min_width=0):
                        gr.Markdown(f"`{aname}`")
                    mdl = gr.Dropdown(
                        choices=model_choices,
                        value=default_model,
                        label="",
                        show_label=False,
                        scale=2,
                    )
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown(f"{prompt_tok:,}")
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown(f"{output_tok:,}")

                analyzer_include_checks[aname] = chk
                analyzer_model_dropdowns[aname] = mdl

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
                    adv_analyzers_selected = gr.Number(
                        label="Analyzers Selected", interactive=False, precision=0
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
                    "Model",
                    "Prompt Tokens",
                    "Image Tokens",
                    "Output Tokens",
                    "Input Cost ($)",
                    "Output Cost ($)",
                    "Total Cost ($)",
                ],
                datatype=[
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                ],
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

    def run_advanced_calculation(*args):
        # args = [num_pages, num_docs, check1, model1, check2, model2, ...]
        num_pages = int(args[0])
        num_docs = max(int(args[1]), 1)

        cfg = load_config()

        selected = []
        model_overrides = {}
        idx = 2
        for aname in analyzer_choices:
            included = args[idx]
            model_name = args[idx + 1]
            if included:
                selected.append(aname)
                model_overrides[aname] = model_name
            idx += 2

        # Fetch exact prompt token counts from S3 for selected analyzers
        for aname in selected:
            adata = analyzers_data.get(aname, {})
            prompt_files = adata.get("prompt_files", [])
            if prompt_files:
                exact_tokens = _fetch_exact_prompt_tokens(aname, prompt_files)
                if exact_tokens > 0:
                    adata["prompt_tokens"] = exact_tokens

        costs = calculate_advanced_costs(
            selected,
            num_pages,
            model_overrides,
            cfg,
            analyzers_data,
        )

        breakdown_data = [
            [
                b["name"],
                b["model"],
                b["prompt_tokens"],
                b["image_tokens"],
                b["output_tokens"],
                b["input_cost"],
                b["output_cost"],
                b["total_cost"],
            ]
            for b in costs["breakdown"]
        ]

        cost_per_page = costs["total_cost"] / num_pages if num_pages > 0 else 0
        cost_per_doc = costs["total_cost"] / num_docs if num_docs > 0 else 0
        total_tokens = costs["total_input_tokens"] + costs["total_output_tokens"]

        summary = (
            f"**Summary:** {len(selected)} analyzers × {num_pages:,} pages = "
            f"{total_tokens:,} total tokens. "
            f"**Cost per page:** ${cost_per_page:.6f} | "
            f"**Cost per document:** ${cost_per_doc:.4f} | "
            f"**Total:** ${costs['total_cost']:.4f}"
        )

        return [
            costs["total_input_tokens"],
            costs["total_output_tokens"],
            costs["input_cost"],
            costs["output_cost"],
            costs["total_cost"],
            len(selected),
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
        results = [
            100,  # num_pages
            1,  # num_docs
        ]
        for aname in analyzer_choices:
            adata = analyzers_data.get(aname, {})
            results.append(aname in default_included)  # checkbox
            results.append(adata.get("default_model", "Claude Sonnet 4.5"))  # model
        results.extend(
            [
                0,  # total_input
                0,  # total_output
                0,  # input_cost
                0,  # output_cost
                0,  # total_cost
                0,  # analyzers_selected
                [],  # breakdown
                "",  # summary
            ]
        )
        return results

    adv_reset_outputs = [adv_num_pages, adv_num_docs]
    for aname in analyzer_choices:
        adv_reset_outputs.append(analyzer_include_checks[aname])
        adv_reset_outputs.append(analyzer_model_dropdowns[aname])
    adv_reset_outputs.extend(
        [
            adv_total_input,
            adv_total_output,
            adv_input_cost,
            adv_output_cost,
            adv_total_cost,
            adv_analyzers_selected,
            adv_breakdown,
            adv_summary,
        ]
    )

    reset_adv_btn.click(reset_advanced, outputs=adv_reset_outputs)

    # Build inputs list for advanced calculate: [num_pages, num_docs, chk1, mdl1, chk2, mdl2, ...]
    adv_calc_inputs = [adv_num_pages, adv_num_docs]
    for aname in analyzer_choices:
        adv_calc_inputs.append(analyzer_include_checks[aname])
        adv_calc_inputs.append(analyzer_model_dropdowns[aname])

    adv_calculate_btn.click(
        run_advanced_calculation,
        inputs=adv_calc_inputs,
        outputs=[
            adv_total_input,
            adv_total_output,
            adv_input_cost,
            adv_output_cost,
            adv_total_cost,
            adv_analyzers_selected,
            adv_breakdown,
            adv_summary,
        ],
    )

    # One-shot timer to initialize on first render (replaces 3x demo.load calls)
    def _init_on_load():
        preset_results = apply_preset(preset_choices[0] if preset_choices else None)
        model_results = update_model_pricing(model_choices[0])
        derived_results = update_derived(
            ingestion["characters_per_token"],
            ingestion["avg_characters_per_word"],
            ingestion["avg_words_per_page"],
            ingestion["avg_pages_per_document"],
        )
        return (
            *preset_results,
            *model_results,
            *derived_results,
            gr.update(active=False),
        )

    timer = gr.Timer(value=0.5, active=True)
    timer.tick(
        fn=_init_on_load,
        outputs=[
            # apply_preset outputs
            preset_description,
            model_dropdown,
            words_per_page,
            pages_per_doc,
            tokens_per_image,
            output_ratio,
            # update_model_pricing outputs
            input_cost_display,
            output_cost_display,
            input_per_token,
            output_per_token,
            # update_derived outputs
            chars_per_page_display,
            tokens_per_page_display,
            words_per_doc_display,
            chars_per_doc_display,
            tokens_per_doc_display,
            # timer self-deactivate
            timer,
        ],
    )
