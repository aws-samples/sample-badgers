"""Blind reader — one vision-model call per region (KNOWLEDGE.md §4.2 item 5, step 3).

The read is *blind*: the model receives only the crop and a transcription instruction. The
specialist's ``concern``, the candidate readings, and the specialist name are withheld and
used only later, in ``comparison.py``. This is correctness property 5 (§4.5): the reading
prompt contains no substring of ``concern``. It is enforced structurally — the functions here
take only the crop bytes and the task, so there is no path for ``concern`` to enter the
prompt.

The prompt text is loaded from S3 (``prompts/region_inspector/*.txt`` in CONFIG_BUCKET) so it
can be edited without a code deploy, exactly like the other tools' prompts. Loading is
fail-soft: a missing or unreadable object falls back to the in-code default below, and the
result is cached in a module global for the container's lifetime (same pattern as
image_enhancer). Because it is cached, an edited prompt reaches new containers, not warm ones.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_CONFIDENCE = {"low", "medium", "high"}

SPECIALIST_NAME = "region_inspector"

# S3 object basenames under prompts/region_inspector/, one per prompt slot.
PROMPT_FILES = {
    "system": "blind_read_system.txt",
    "transcribe": "blind_read_transcribe.txt",
    "assign_rows": "blind_read_assign_rows.txt",
}

# In-code fallbacks, identical to the shipped S3 files. Used only when the S3 object is
# missing or unreadable, so the tool still works if the prompts were not synced. The reading
# is deliberately literal: UNREADABLE and [?] are the escape hatches so the model never
# guesses to fill a word — a guess is exactly what inflates confidence downstream.
_DEFAULTS = {
    "system": (
        "You are reading one small region cropped from a scanned document page. Transcribe "
        "exactly what is written in the region. Do not guess to complete words. If a "
        "character cannot be read, write [?] in its place. If the region contains nothing "
        "legible, answer UNREADABLE.\n\n"
        "Return JSON only, with no surrounding prose or code fences:\n"
        '{"reading": "...", '
        '"per_character": [{"char": "1", "confidence": "high"}], '
        '"marks": [{"text": "...", "page_y": 0.00}], '
        '"notes": "..."}\n\n'
        'Each per_character confidence is one of "high", "medium", "low".'
    ),
    "transcribe": "Transcribe the characters in this region. Leave the marks array empty.",
    "assign_rows": (
        "This region has red ticks along its left edge, each labelled with a page "
        "y-coordinate. For every mark you see, report its text and the tick value level with "
        "the mark's baseline in the marks array. Still transcribe any characters in the region."
    ),
}

# Module-level cache, populated on first load and reused for the container's lifetime.
_prompt_cache: Optional[Dict[str, str]] = None


def default_prompts() -> Dict[str, str]:
    """The in-code fallback prompts. Used when no S3 prompts are provided (e.g. tests)."""
    return dict(_DEFAULTS)


def clear_prompt_cache() -> None:
    """Drop the cached prompts. For tests."""
    global _prompt_cache
    _prompt_cache = None


def load_prompts(
    config_bucket: str, specialist_name: str = SPECIALIST_NAME
) -> Dict[str, str]:
    """Load the three prompt slots from S3, fail-soft to the in-code defaults per slot, and
    cache the result. The foundation import is local so this module stays importable without
    the foundation layer (the crop core and comparison tests do not need AWS)."""
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache

    prompts = dict(_DEFAULTS)
    if config_bucket:
        try:
            from foundation.s3_config_loader import load_prompt_from_s3

            for slot, filename in PROMPT_FILES.items():
                try:
                    text = load_prompt_from_s3(config_bucket, specialist_name, filename)
                    if text and text.strip():
                        prompts[slot] = text.strip()
                except Exception as exc:  # noqa: BLE001 - any load failure falls back
                    logger.warning(
                        "region_inspector: could not load prompt %s (%s); using in-code default",
                        filename,
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001 - foundation loader unavailable
            logger.warning(
                "region_inspector: prompt loader unavailable (%s); using in-code defaults",
                exc,
            )

    _prompt_cache = prompts
    return prompts


def build_user_instruction(task: str, prompts: Optional[Dict[str, str]] = None) -> str:
    """Task-specific user instruction. Contains no specialist metadata."""
    prompts = prompts or _DEFAULTS
    slot = "assign_rows" if task == "assign_rows" else "transcribe"
    return prompts.get(slot, _DEFAULTS[slot])


def build_prompt_text(task: str, prompts: Optional[Dict[str, str]] = None) -> str:
    """The full reading prompt (system + user instruction) as plain text. For property-5
    verification and logging; carries no concern, candidates, or specialist name."""
    prompts = prompts or _DEFAULTS
    system = prompts.get("system", _DEFAULTS["system"])
    return f"{system}\n\n{build_user_instruction(task, prompts)}"


def build_messages(
    png_base64: str, task: str, prompts: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """One user message: the crop image followed by the task instruction."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"media_type": "image/png", "data": png_base64},
                },
                {"type": "text", "text": build_user_instruction(task, prompts)},
            ],
        }
    ]


def _extract_text(response: Dict[str, Any]) -> str:
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text and text.strip():
                return text
    return ""


def _parse_reading(text: str) -> Dict[str, Any]:
    """Parse the model's JSON reply defensively into the reading fields."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # The model did not return JSON. Treat the raw text as the reading with unknown
        # per-character confidence rather than fabricating one.
        logger.warning("Blind read did not return JSON; using raw text as reading")
        return {"reading": text.strip(), "per_character": [], "marks": [], "notes": ""}

    per_character = []
    for entry in data.get("per_character") or []:
        if not isinstance(entry, dict):
            continue
        confidence = str(entry.get("confidence", "")).lower()
        if confidence not in _VALID_CONFIDENCE:
            confidence = "low"
        per_character.append(
            {"char": str(entry.get("char", "")), "confidence": confidence}
        )

    marks = data.get("marks") if isinstance(data.get("marks"), list) else []
    return {
        "reading": str(data.get("reading", "")),
        "per_character": per_character,
        "marks": marks,
        "notes": str(data.get("notes", "")),
    }


def read_region(
    bedrock_client: Any,
    model_selection: Dict[str, Any],
    png_base64: str,
    task: str,
    prompts: Optional[Dict[str, str]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Make the single blind vision call for one crop and return the parsed reading.

    ``model_selection`` is the dict from ``foundation.model_selection.parse_model_selection``.
    ``prompts`` is the dict from ``load_prompts``; the in-code defaults are used if omitted.
    """
    prompts = prompts or _DEFAULTS
    payload = bedrock_client.create_anthropic_payload(
        system_prompt=prompts.get("system", _DEFAULTS["system"]),
        messages=build_messages(png_base64, task, prompts),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response = bedrock_client.invoke_model(
        model_id=model_selection["model_id"],
        payload=payload,
        fallback_list=model_selection.get("fallback_list"),
        extended_thinking=model_selection.get("extended_thinking", False),
        budget_tokens=model_selection.get("budget_tokens"),
        adaptive_thinking=model_selection.get("adaptive_thinking", False),
        adaptive_effort=model_selection.get("effort", "high"),
    )
    text = _extract_text(response)
    if not text:
        raise ValueError("Blind read returned no text content")
    return _parse_reading(text)
