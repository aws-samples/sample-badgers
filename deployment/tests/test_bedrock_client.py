"""Security-focused unit tests for the shared Bedrock client."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "badgers-foundation"
    / "foundation"
    / "bedrock_client.py"
)
SPEC = importlib.util.spec_from_file_location("bedrock_client_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
bedrock_client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bedrock_client)


def test_mantle_endpoint_allows_aws_regions_only() -> None:
    client = bedrock_client.BedrockClient()

    assert client._mantle_chat_completions_url("us-west-2") == (
        "https://bedrock-mantle.us-west-2.api.aws/openai/v1/chat/completions"
    )

    with pytest.raises(bedrock_client.BedrockError, match="Invalid AWS Region"):
        client._mantle_chat_completions_url("us-west-2.attacker.example")
