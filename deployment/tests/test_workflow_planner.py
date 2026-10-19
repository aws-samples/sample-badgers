"""Tests for immutable workflow configuration and deterministic plan resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from deployment.profile_resolver import resolve_profile


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
FOUNDATION_DIR = DEPLOYMENT_DIR / "badgers-foundation"
sys.path.insert(0, str(FOUNDATION_DIR))

from foundation import operation_state, workflow_planner  # noqa: E402


SOURCE_ROOT = DEPLOYMENT_DIR / "s3_files"
HEADLESS = resolve_profile("headless")
FULL = resolve_profile("full")


class FakeTable:
    """Small DynamoDB table double sufficient for operation plan persistence."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, Any]] = {}

    def put_item(self, *, Item: dict[str, Any], **_kwargs: Any) -> None:
        key = (Item["job_id"], Item["subtask_id"])
        if key in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[key] = dict(Item)

    def update_item(
        self,
        *,
        Key: dict[str, str],
        ExpressionAttributeValues: dict[str, Any],
        **_kwargs: Any,
    ) -> None:
        item = self.items[(Key["job_id"], Key["subtask_id"])]
        values = ExpressionAttributeValues
        for field in (
            "plan_uri",
            "workflow_version",
            "routing_policy_id",
            "routing_policy_version",
            "updated_at",
            "ttl",
        ):
            placeholder = f":{field}"
            if placeholder in values:
                item[field] = values[placeholder]

    def get_item(self, *, Key: dict[str, str], **_kwargs: Any) -> dict[str, Any]:
        item = self.items.get((Key["job_id"], Key["subtask_id"]))
        return {"Item": dict(item)} if item else {}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(self, **kwargs: Any) -> None:
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs


def _catalog() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    workflows, policies, capability_index = workflow_planner.load_workflow_catalog(
        SOURCE_ROOT, HEADLESS.specialists
    )
    workflow = workflows["standard-document-analysis"]
    return workflow, policies[workflow["routing_policy_id"]], capability_index


def _auto_plan() -> dict[str, Any]:
    workflow, policy, capability_index = _catalog()
    return workflow_planner.resolve_plan(
        operation_id="op_" + ("a" * 32),
        session_id="review-104",
        config_release_id="b" * 64,
        workflow=workflow,
        policy=policy,
        capability_index=capability_index,
        routing_mode="auto",
        classifier_facts=[{"content_type": "table_dominant", "confidence": 0.91}],
        classification_artifacts=[
            "s3://badgers-output/tenant-a/op/classification/page-001.json"
        ],
    )


def test_fixed_release_and_classifier_facts_resolve_identical_plans() -> None:
    first = _auto_plan()
    second = _auto_plan()

    assert first == second
    assert first["recommended_capabilities"] == [
        "source_conversion",
        "classification",
        "full_text",
        "elements",
        "table",
        "correlation",
    ]
    assert first["resolved_specialists"] == [
        "pdf_to_images_converter",
        "classify_pdf_content",
        "full_text_specialist",
        "elements_specialist",
        "table_specialist",
        "correlation_specialist",
    ]


def test_workflow_catalog_validates_against_full_and_headless_profiles() -> None:
    for profile in (HEADLESS, FULL):
        workflow_planner.validate_release_configuration(SOURCE_ROOT, profile.specialists)


def test_missing_pack_capability_fails_before_execution() -> None:
    workflow, policy, capability_index = _catalog()
    capability_index = dict(capability_index)
    capability_index.pop("table")

    with pytest.raises(
        workflow_planner.WorkflowPlanningError,
        match="Pinned specialist pack lacks required capabilities: table",
    ):
        workflow_planner.resolve_plan(
            operation_id="op_" + ("b" * 32),
            session_id="review-104",
            config_release_id="c" * 64,
            workflow=workflow,
            policy=policy,
            capability_index=capability_index,
            routing_mode="auto",
            classifier_facts=[{"content_type": "text", "confidence": 0.99}],
            classification_artifacts=["s3://badgers-output/tenant-a/classification.json"],
        )


def test_manual_capability_selection_cannot_omit_an_invariant() -> None:
    workflow, policy, capability_index = _catalog()

    with pytest.raises(
        workflow_planner.WorkflowPlanningError,
        match="Manual selection cannot omit required invariants",
    ):
        workflow_planner.resolve_plan(
            operation_id="op_" + ("c" * 32),
            session_id="review-104",
            config_release_id="d" * 64,
            workflow=workflow,
            policy=policy,
            capability_index=capability_index,
            routing_mode="manual",
            requested_capabilities=["full_text", "elements", "correlation"],
        )


def test_resolved_plan_replaces_initial_private_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeTable()
    s3 = FakeS3()
    operation_id = "op_" + ("d" * 32)
    monkeypatch.setenv("JOBS_TABLE_NAME", "badgers-jobs-test")
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output-test")
    monkeypatch.setenv("CONFIG_PREFIX", "releases/" + ("e" * 64))
    monkeypatch.setattr(operation_state, "_table", lambda: table)
    monkeypatch.setattr(operation_state, "_s3_client", lambda: s3)
    monkeypatch.setattr(operation_state.job_state, "create_job", lambda *_a, **_k: None)

    operation_state.create_operation(
        operation_id=operation_id,
        session_id="review-104",
        owner_sub="tenant-a-workload",
        workflow_id="standard-document-analysis",
        routing_mode="auto",
        source_document_s3_uri="s3://customer-source/tenant-a/msa.pdf",
    )
    plan = _auto_plan()
    plan["operation_id"] = operation_id
    plan["config_release_id"] = "e" * 64
    plan_uri = operation_state.persist_resolved_plan(
        operation_id, owner_sub="tenant-a-workload", plan=plan
    )

    assert plan_uri.endswith("/plan/plan.json")
    stored = s3.objects[(
        "badgers-output-test",
        "operations/tenant-a-workload/" + operation_id + "/plan/plan.json",
    )]
    assert json.loads(stored["Body"]) == plan
    record = operation_state.get_operation(operation_id, owner_sub="tenant-a-workload")
    assert record["routing_policy_id"] == "standard-document-routing"
