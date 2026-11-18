"""Contract tests for text-only keyword/topic extraction."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT / "badgers-foundation"))
_SPEC = importlib.util.spec_from_file_location(
    "keyword_topic_text_handler",
    DEPLOYMENT / "lambdas" / "code" / "keyword_topic_specialist" / "lambda_handler.py",
)
if not (_SPEC and _SPEC.loader):
    raise AssertionError
handler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = handler
_SPEC.loader.exec_module(handler)


def test_keyword_input_reads_bounded_text_from_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SimpleNamespace(
        get_object=lambda **_kwargs: {"Body": io.BytesIO(b"document text")}
    )
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda _name: client))

    if not (handler._load_text_input(
            {"full_text_uri": "s3://output/page/full-text.xml"},
            "full_text_uri",
            required=True,
        )
        == "document text"):
        raise AssertionError


def test_keyword_input_requires_a_valid_full_text_uri() -> None:
    with pytest.raises(Exception, match="full_text_uri"):
        handler._load_text_input({}, "full_text_uri", required=True)
    with pytest.raises(Exception, match="must be an S3 URI"):
        handler._load_text_input(
            {"full_text_uri": "relative.xml"}, "full_text_uri", required=True
        )


def test_keyword_manifest_and_gateway_schema_accept_text_not_images() -> None:
    manifest = json.loads(
        (DEPLOYMENT / "s3_files" / "manifests" / "keyword_topic_specialist.json").read_text()
    )
    schema = json.loads(
        (DEPLOYMENT / "s3_files" / "schemas" / "keyword_topic_specialist.json").read_text()
    )
    manifest_properties = manifest["tool"]["inputSchema"]["properties"]
    schema_properties = schema[0]["inputSchema"]["properties"]

    if not ("full_text_uri" in manifest_properties):
        raise AssertionError
    if not ("correlation_specialist_uri" in manifest_properties):
        raise AssertionError
    if not ("image_path" not in manifest_properties):
        raise AssertionError
    if not ("image_data" not in schema_properties):
        raise AssertionError
    if not ("full_text_uri" in schema[0]["inputSchema"]["required"]):
        raise AssertionError


def test_keyword_agent_prompt_names_the_text_only_contract() -> None:
    prompt = (
        DEPLOYMENT / "s3_files" / "agent_system_prompt" / "agent_system_prompt.xml"
    ).read_text()

    if not ("It accepts no image_path or image_data." in prompt):
        raise AssertionError
    if not ("page_number, and source_image_path" in prompt):
        raise AssertionError
