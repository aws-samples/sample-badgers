"""One parser for a manifest's ``model_selections`` block.

There used to be two, and they had drifted in opposite directions:

* ``specialist_foundation._get_model_selection`` returned a 4-tuple carrying only
  ``extended_thinking`` and ``budget_tokens``. It never read ``adaptive_thinking`` or
  ``effort``, so those settings were dropped before ``invoke_model`` was ever called.
* ``correlation_specialist.lambda_handler._get_model_config`` did read ``adaptive_thinking``
  and ``effort`` for the primary model, but flattened ``fallback_list`` to bare model-ID
  strings, discarding every per-fallback thinking setting.

Between them, no caller could express "fall back to a different model *and* keep thinking
on". `correlation_specialist` is the only built-in manifest that uses adaptive thinking, and
it worked only because it carried its own parser — any other specialist declaring the same
setting would silently lose it.

This module is the single source of that parsing. It returns a dict rather than a tuple
specifically so that adding a field later cannot silently change what an existing caller
unpacks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Effort levels accepted by both Claude (`output_config.effort`) and Nova 2
#: (`reasoningConfig.maxReasoningEffort`). Claude also documents `xhigh` and `max`, but only
#: for specific Opus models, so they are not offered here — see the migration plan.
VALID_EFFORT = ("low", "medium", "high")

DEFAULT_EFFORT = "high"


class ModelSelectionError(ValueError):
    """Raised when a manifest's model_selections block cannot be parsed."""


def _thinking_from(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the four thinking fields off one primary or fallback entry."""
    effort = entry.get("effort", DEFAULT_EFFORT)
    if effort not in VALID_EFFORT:
        raise ModelSelectionError(
            f"effort must be one of {list(VALID_EFFORT)}, got {effort!r}"
        )
    return {
        "extended_thinking": bool(entry.get("extended_thinking", False)),
        "budget_tokens": entry.get("budget_tokens"),
        "adaptive_thinking": bool(entry.get("adaptive_thinking", False)),
        "effort": effort,
    }


def _normalise_entry(entry: Any, where: str) -> Dict[str, Any]:
    """Normalise a primary or fallback entry to a full dict.

    Accepts the dict shape and the legacy bare-string shape. A string carries no thinking
    settings, which is a property of the shape rather than an omission — 26 of the 27
    built-in manifests use it.
    """
    if isinstance(entry, str):
        if not entry:
            raise ModelSelectionError(f"{where}: model ID is empty")
        return {
            "model_id": entry,
            "extended_thinking": False,
            "budget_tokens": None,
            "adaptive_thinking": False,
            "effort": DEFAULT_EFFORT,
        }

    if not isinstance(entry, dict):
        raise ModelSelectionError(
            f"{where}: expected an object or a model ID string, "
            f"got {type(entry).__name__}"
        )

    model_id = entry.get("model_id")
    if not model_id:
        raise ModelSelectionError(f"{where}: model_id is required")

    result = {"model_id": model_id}
    result.update(_thinking_from(entry))
    return result


def parse_model_selection(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a specialist manifest into a complete model-selection structure.

    Args:
        config: the manifest's ``specialist`` block (or the manifest itself).

    Returns:
        ``{model_id, extended_thinking, budget_tokens, adaptive_thinking, effort,
        fallback_list}`` where every ``fallback_list`` entry has that same shape minus
        ``fallback_list``. Nothing is dropped.

    Raises:
        ModelSelectionError: on a malformed block. Raising beats defaulting here — a
        silently defaulted model ID is how a specialist ends up invoking something nobody
        chose.
    """
    selections = config.get("model_selections")

    if selections:
        primary = selections.get("primary")
        if not primary:
            raise ModelSelectionError("model_selections.primary is required")

        result = _normalise_entry(primary, "model_selections.primary")

        fallbacks: List[Dict[str, Any]] = []
        for index, entry in enumerate(selections.get("fallback_list") or []):
            fallbacks.append(
                _normalise_entry(entry, f"model_selections.fallback_list[{index}]")
            )
        result["fallback_list"] = fallbacks
        return result

    # Legacy format: model_id plus an optional single fallback_model_id.
    model_id = config.get("model_id")
    if not model_id:
        raise ModelSelectionError("model_id or model_selections.primary is required")

    result = _normalise_entry(model_id, "model_id")
    fallback_model_id = config.get("fallback_model_id")
    result["fallback_list"] = (
        [_normalise_entry(fallback_model_id, "fallback_model_id")]
        if fallback_model_id
        else []
    )
    return result


def max_tokens_from(config: Dict[str, Any], default: Optional[int] = None) -> Optional[int]:
    """Read the manifest's output-token budget.

    Separate from ``parse_model_selection`` because it lives outside the
    ``model_selections`` block — ``correlation_specialist`` reads it from
    ``expected_output_tokens`` at the top level.
    """
    value = config.get("expected_output_tokens", default)
    return int(value) if value is not None else None
