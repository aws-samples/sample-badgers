"""Validation and canonicalisation for release-pinned prompt object keys."""

from __future__ import annotations

CORE_PROMPT_FILES = {
    "prompt_system_wrapper": "wrappers/prompt_system_wrapper.xml",
    "core_rules_rules": "core_system_prompts/core_rules/rules.xml",
    "error_handling_error_handler": "core_system_prompts/error_handling/error_handler.xml",
    "error_handling_not_found_handler": "core_system_prompts/error_handling/not_found_handler.xml",
    "audit_confidence_assessment": "core_system_prompts/audit/confidence_assessment.xml",
}


def prompt_key(specialist: str, entry: str) -> str:
    """Resolve a manifest prompt entry to a safe S3-relative key."""
    if not isinstance(entry, str) or not entry:
        raise ValueError("Prompt entry must be a non-empty string")
    if entry.startswith("/") or "\\" in entry:
        raise ValueError(f"Invalid prompt key: {entry!r}")
    parts = entry.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid prompt key: {entry!r}")
    if entry.startswith("shared/"):
        return f"prompts/{entry}"
    return f"prompts/{specialist}/{entry}"
