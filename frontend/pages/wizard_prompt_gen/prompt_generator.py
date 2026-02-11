"""Per-prompt-type LLM generation with few-shot examples.

Each of the 6 prompt types gets its own focused Bedrock call with a real
example loaded from deployment/s3_files/prompts/.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_TYPES = ["gestalt", "job_role", "rules", "context", "tasks", "format"]

# Maps prompt type -> (analyzer_dir, filename) inside deployment/s3_files/prompts/
EXAMPLE_MAP: dict[str, tuple[str, str]] = {
    "gestalt": ("elements_analyzer", "elements_gestalt.xml"),
    "job_role": ("elements_analyzer", "elements_job_role.xml"),
    "rules": ("correlation_analyzer", "correlation_rules.xml"),
    "context": ("correlation_analyzer", "correlation_context.xml"),
    "tasks": ("elements_analyzer", "elements_tasks_extraction.xml"),
    "format": ("elements_analyzer", "elements_format.xml"),
}

# Resolve base path relative to the repo root (3 levels up from this file)
EXAMPLES_BASE_PATH = (
    Path(__file__).resolve().parents[3] / "deployment" / "s3_files" / "prompts"
)

PROMPT_TYPE_INSTRUCTIONS: dict[str, str] = {
    "gestalt": (
        "Generate a gestalt_perception XML prompt that applies Gestalt perception principles "
        "to the target document/content type. Include:\n"
        "- A <prescan> section with 5-6 steps for holistic visual scanning before extraction\n"
        "- A <principles_emphasis> section with 2-3 primary/secondary Gestalt principles "
        "(proximity, similarity, continuity, closure, figure_ground, common_fate) with specific "
        "guidance on how each applies to this analyzer's domain\n"
        "- An <element_detection_cues> section with 5-8 visual cues specific to the content type\n"
        "- A <validation> section with 4-5 checks to verify perception accuracy after extraction\n"
        "The prompt must guide the Vision LLM to 'see' the document holistically before extracting."
    ),
    "job_role": (
        "Generate a job_role XML prompt that establishes the LLM's expert persona. Include:\n"
        "- A <role> tag with a specialist title and one-sentence expertise statement\n"
        "- A <job_description> with title, 2-3 sentence summary, 5-7 responsibilities, "
        "and 5-7 skills\n"
        "All responsibilities and skills must be specific to the analysis domain, not generic."
    ),
    "rules": (
        "Generate a rules XML prompt with 8-12 rules. Order them as:\n"
        "1. Anti-hallucination rules first (do not invent, do not create elements that don't exist)\n"
        "2. Two-step verification rule (PASS 1 identify, PASS 2 verify)\n"
        "3. Format compliance rules (use only the response_format provided)\n"
        "4. Honest negatives rule (finding nothing is always valid)\n"
        "5. Domain-specific rules last\n"
        "Rules may include a priority attribute (critical, high, medium, low)."
    ),
    "context": (
        "Generate a context XML prompt with 10-12 <item> elements. Include:\n"
        "- What the analyzer searches for and extracts\n"
        "- Clear definition of what IS the target element type\n"
        "- Clear definition of what is NOT the target element type (common misidentifications)\n"
        "- Variations and forms the target element may take\n"
        "- Two-step verification reminder\n"
        "- Guidance on prioritizing accuracy over comprehensiveness\n"
        "- What to do when nothing is found (not_found response)"
    ),
    "tasks": (
        "Generate a tasks XML prompt with 12-16 <task> elements. Tasks may contain <sub_task> "
        "children for multi-step instructions. Structure as:\n"
        "- Initial review/deep-breath task\n"
        "- PASS 1 tasks: scan and identify potential elements by visual/textual indicators\n"
        "- PASS 2 tasks: verify each candidate against strict criteria, reject false positives\n"
        "- Extraction tasks: for verified elements, extract specific data points\n"
        "- Organization task: structure findings per the response_format\n"
        "- Final review task: verify accuracy, completeness, correct ordering\n"
        "Tasks should be detailed and actionable, not one-liners."
    ),
    "format": (
        "Generate a response_format XML prompt that defines the exact output structure. Include:\n"
        "- A <response> wrapper with extraction_type attribute\n"
        "- A <metadata> section with page_number, examples_count, element_count\n"
        "- An <elements> section with one <element> template showing all fields to extract, "
        "using {SCREAMING_SNAKE_CASE} placeholder tokens for values the LLM fills in\n"
        "- A <not_found> section with a message template for when no elements are detected\n"
        "- A comment noting to omit <elements> when element_count is 0"
    ),
}


def load_few_shot_example(prompt_type: str) -> str | None:
    """Load the few-shot example file for a prompt type.

    Returns file content as a string, or None if the file is missing.
    """
    mapping = EXAMPLE_MAP.get(prompt_type)
    if mapping is None:
        logger.warning("No example mapping for prompt type: %s", prompt_type)
        return None

    analyzer_dir, filename = mapping
    path = EXAMPLES_BASE_PATH / analyzer_dir / filename

    if not path.exists():
        logger.warning("Few-shot example not found: %s", path)
        return None

    content = path.read_text(encoding="utf-8")
    # Strip XML declaration if present
    if content.startswith("<?xml"):
        content = content.split("?>", 1)[1]
    return content.strip()


def extract_xml_from_response(response_text: str) -> str:
    """Strip markdown code fences and preamble, return raw XML content."""
    text = response_text.strip()
    if not text:
        return text

    # Strip markdown code fences: ```xml ... ``` or ``` ... ```
    if "```xml" in text:
        text = text.split("```xml", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]

    text = text.strip()

    # Strip any preamble text before the first XML tag
    if not text.startswith("<"):
        match = re.search(r"<", text)
        if match:
            text = text[match.start() :]

    return text.strip()


def build_system_prompt(prompt_type: str, few_shot_example: str | None) -> str:
    """Build a focused system prompt for one prompt type."""
    instructions = PROMPT_TYPE_INSTRUCTIONS.get(prompt_type, "")

    parts = [
        "You are an expert at creating XML prompts for document analysis systems.",
        "You are generating ONLY the {type} prompt for a new analyzer.".format(
            type=prompt_type
        ),
        "",
        "## Instructions",
        instructions,
        "",
        "## Critical Rules",
        "- Gestalt-First Perception: guide the model to perceive holistically before extracting.",
        "- Two-Pass Verification: PASS 1 identifies potential elements, PASS 2 verifies against strict criteria.",
        "- Honest Negatives: finding nothing is ALWAYS valid. Never force identification.",
        "- Anti-Hallucination: never create, invent, or infer elements that don't exist in the source.",
        "- Use {SCREAMING_SNAKE_CASE} placeholder tokens for values the LLM fills in during analysis.",
        "",
    ]

    if few_shot_example:
        parts.extend(
            [
                "## Reference Example",
                "Here is a high-quality example of a {type} prompt from an existing analyzer. "
                "Use it as a reference for structure, depth, and quality — but adapt the content "
                "to the new analyzer's domain:".format(type=prompt_type),
                "",
                few_shot_example,
                "",
            ]
        )

    parts.extend(
        [
            "## Output",
            "Return ONLY the raw XML content. No markdown fencing, no preamble, no commentary.",
            "Start your response with the opening XML tag.",
        ]
    )

    return "\n".join(parts)


def generate_single_prompt(
    prompt_type: str,
    analyzer_name: str,
    display_name: str,
    description: str,
    bedrock_client: Any,
) -> str:
    """Generate one prompt type via Bedrock. Returns XML string or 'ERROR: ...'."""
    logger.info("Generating %s prompt for %s", prompt_type, analyzer_name)

    few_shot = load_few_shot_example(prompt_type)
    system_prompt = build_system_prompt(prompt_type, few_shot)

    user_prompt = (
        f"Create the {prompt_type} XML prompt for a new analyzer with these details:\n\n"
        f"Analyzer name: {analyzer_name}\n"
        f"Display name: {display_name}\n"
        f"Description: {description}\n"
    )

    try:
        response = bedrock_client.converse(
            modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": 8000, "temperature": 0.3},
        )

        result_text = response["output"]["message"]["content"][0]["text"]
        xml_content = extract_xml_from_response(result_text)
        logger.info("Generated %s prompt, length: %d", prompt_type, len(xml_content))
        return xml_content

    except Exception as e:
        logger.error("Failed to generate %s prompt: %s", prompt_type, e, exc_info=True)
        return f"ERROR: Failed to generate {prompt_type} prompt: {e}"


def generate_all_prompts(
    description: str,
    analyzer_name: str,
    display_name: str,
) -> dict[str, str]:
    """Generate all 6 prompt types via individual Bedrock calls.

    Returns dict with keys: gestalt, job_role, rules, context, tasks, format.
    Values are XML strings, or 'ERROR: ...' on failure.
    """
    logger.info("Generating all prompts for analyzer: %s", analyzer_name)

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        config=BotoConfig(read_timeout=300),
    )

    results: dict[str, str] = {}
    for prompt_type in PROMPT_TYPES:
        try:
            results[prompt_type] = generate_single_prompt(
                prompt_type=prompt_type,
                analyzer_name=analyzer_name,
                display_name=display_name,
                description=description,
                bedrock_client=bedrock,
            )
        except Exception as e:
            logger.error(
                "Unexpected error generating %s: %s", prompt_type, e, exc_info=True
            )
            results[prompt_type] = f"ERROR: {e}"

    logger.info(
        "Prompt generation complete. Results: %s",
        {
            k: "OK" if not v.startswith("ERROR:") else "FAILED"
            for k, v in results.items()
        },
    )
    return results
