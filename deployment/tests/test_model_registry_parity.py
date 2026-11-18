from __future__ import annotations

from deployment.stacks.model_registry import active_models, load_registry, pricing_notice


def test_registry_records_price_provenance_and_active_claude_five_defaults() -> None:
    registry = load_registry()
    pricing = pricing_notice()
    assert pricing["url"].startswith("https://")
    assert pricing["not_verified"] == ["google.gemma-4-31b"]
    for model_id, spec in active_models(registry).items():
        if ".claude-" in model_id and "-5" in model_id:
            assert spec["thinking_default_on"] is True
