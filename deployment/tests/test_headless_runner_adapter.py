"""Focused tests for the authenticated Headless-to-runner adapter."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
FOUNDATION_DIR = DEPLOYMENT_DIR / "badgers-foundation"
RUNTIME_PATH = DEPLOYMENT_DIR / "runtime" / "agent" / "main-websocket.py"
sys.path.insert(0, str(FOUNDATION_DIR))

SPEC = importlib.util.spec_from_file_location("badgers_headless_runtime", RUNTIME_PATH)
assert SPEC and SPEC.loader
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def _bearer_token(client_id: str, *, scope: str = "badgers-headless/invoke") -> str:
    claims = {
        "token_use": "access",
        "client_id": client_id,
        "scope": scope,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(claims, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"Bearer ignored.{encoded}.ignored"


def _context(token: str = "") -> SimpleNamespace:
    headers = {"Authorization": token} if token else {}
    return SimpleNamespace(
        session_id="agentcore-session-104",
        request_headers=headers,
    )


def _request() -> dict[str, Any]:
    return {
        "schema_version": "badgers-agent-request/v1",
        "session_id": "contract-review-104",
        "document_uri": (
            "s3://badgers-source/tenants/headless-client-abc123/contracts/msa.pdf"
        ),
        "instruction": "Identify renewal terms.",
        "routing_mode": "auto",
    }


def test_versioned_headless_request_starts_runner_with_verified_client_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    started: dict[str, Any] = {}
    monkeypatch.setenv("HEADLESS_AUTH_MODE", "cognito_jwt")
    monkeypatch.setenv("HEADLESS_AUTHORIZED_CLIENT_ID", "abc123")
    monkeypatch.setenv("DEPLOYMENT_ID", "headless-test")
    monkeypatch.setattr(
        runtime.operation_contract,
        "new_operation_id",
        lambda: "op_" + ("a" * 32),
    )
    monkeypatch.setattr(
        runtime.operation_state,
        "create_operation",
        lambda **kwargs: created.update(kwargs)
        or {
            "config_release_id": "b" * 64,
            "plan_uri": "s3://output/operations/headless-abc123/op/plan/plan.json",
            "status_manifest_uri": "s3://output/operations/headless-abc123/op/status.json",
        },
    )
    monkeypatch.setattr(
        runtime,
        "start_headless_workflow_execution",
        lambda **kwargs: started.update(kwargs),
    )

    events = asyncio.run(
        _collect(runtime.invoke(_request(), _context(_bearer_token("abc123"))))
    )

    assert created["owner_sub"] == "headless-abc123"
    assert created["principal_id"] == "abc123"
    assert created["identity_provider"] == "cognito-jwt"
    assert created["executor"] == "workflow_runner"
    assert started == {
        "operation_id": "op_" + ("a" * 32),
        "owner_sub": "headless-abc123",
        "requested_capabilities": (),
        "audit_mode": False,
        "runtime_session_id": "agentcore-session-104",
    }
    assert events == [
        {
            "type": "status",
            "status": "accepted",
            "stage": "accepted",
            "operation_id": "op_" + ("a" * 32),
            "session_id": "contract-review-104",
            "workflow_id": "standard-document-analysis",
            "routing_mode": "auto",
            "config_release_id": "b" * 64,
            "plan_uri": "s3://output/operations/headless-abc123/op/plan/plan.json",
            "status_manifest_uri": "s3://output/operations/headless-abc123/op/status.json",
        }
    ]


def test_omitted_workflow_uses_deployment_configured_docx_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    monkeypatch.setenv("HEADLESS_AUTH_MODE", "cognito_jwt")
    monkeypatch.setenv("HEADLESS_AUTHORIZED_CLIENT_ID", "abc123")
    monkeypatch.setenv("DEPLOYMENT_ID", "headless-test")
    monkeypatch.setenv(
        "ROUTING_WORKFLOWS_BY_TYPE_JSON",
        json.dumps(
            {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                "native-docx-analysis"
            }
        ),
    )
    monkeypatch.setattr(
        runtime.operation_contract,
        "new_operation_id",
        lambda: "op_" + ("d" * 32),
    )
    monkeypatch.setattr(
        runtime.operation_state,
        "create_operation",
        lambda **kwargs: created.update(kwargs)
        or {
            "config_release_id": "b" * 64,
            "plan_uri": "s3://output/operations/headless-abc123/op/plan/plan.json",
            "status_manifest_uri": "s3://output/operations/headless-abc123/op/status.json",
        },
    )
    monkeypatch.setattr(runtime, "start_headless_workflow_execution", lambda **_k: None)
    request = _request()
    request["document_uri"] = (
        "s3://badgers-source/tenants/headless-client-abc123/contracts/brief.docx"
    )

    asyncio.run(_collect(runtime.invoke(request, _context(_bearer_token("abc123")))))

    assert created["workflow_id"] == "native-docx-analysis"


def test_versioned_headless_request_rejects_missing_verified_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADLESS_AUTH_MODE", "cognito_jwt")
    monkeypatch.setenv("HEADLESS_AUTHORIZED_CLIENT_ID", "abc123")
    monkeypatch.setattr(
        runtime.operation_state,
        "create_operation",
        lambda **_kwargs: pytest.fail("operation must not be created"),
    )
    monkeypatch.setattr(
        runtime,
        "start_headless_workflow_execution",
        lambda **_kwargs: pytest.fail("workflow must not start"),
    )

    events = asyncio.run(_collect(runtime.invoke(_request(), _context())))

    assert events == [
        {
            "type": "error",
            "code": "UNAUTHENTICATED",
            "message": "A validated bearer token is required",
        }
    ]


async def _collect(stream: Any) -> list[dict[str, Any]]:
    return [event async for event in stream]
