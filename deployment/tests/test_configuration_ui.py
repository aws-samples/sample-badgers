"""Tests for the private local configuration planner."""

from __future__ import annotations

import http.client
import io
import json
import stat
import threading
import time
from pathlib import Path

from deployment.configuration_ui import (
    _HTML,
    ConfigurationServer,
    DeploymentRunner,
    UIError,
    _deployment_command,
    configuration,
    configurations,
    delete_configuration,
    plan,
    save_observability,
    save_tags,
)


def test_configuration_creates_local_desired_state_with_private_permissions(
    tmp_path: Path,
) -> None:
    result = configuration(tmp_path, "beta", "headless")
    state_path = tmp_path / ".deploy-state/beta.json"

    if not (result["selection"]["profile"] == "headless"):
        raise AssertionError
    if not (result["routing"]["templates"] == ["standard-document-analysis"]):
        raise AssertionError
    if not (result["observability"]["mode"] == "links"):
        raise AssertionError
    if not (result["observability"]["candidate_plan"]["resources"]["standard_alarms"] == 0):
        raise AssertionError
    if not (result["tags"]["tags"] == {}):
        raise AssertionError
    if not (stat.S_IMODE(state_path.stat().st_mode) == 0o600):
        raise AssertionError


def test_deployment_runner_records_missing_process_as_a_failure(tmp_path: Path) -> None:
    runner = DeploymentRunner(tmp_path)

    runner._consume("configuration-fingerprint")

    status = runner.status()
    if not (status["status"] == "failed"):
        raise AssertionError
    if not ("did not start a process" in status["output"]):
        raise AssertionError


def test_plan_describes_the_local_wizard_and_cli_fallback(tmp_path: Path) -> None:
    result = plan(tmp_path, "beta", "headless")

    if not (result["cli_next_step"] == "./deploy.sh headless resume"):
        raise AssertionError
    if not ("preflight" in result["apply_note"].lower()):
        raise AssertionError
    if not ("local wizard deploys the selected observability posture" in result[
        "observability_handoff"
    ]):
        raise AssertionError
    if not ("publish configuration" in result["required_stages"]):
        raise AssertionError
    if not ("deploy selected observability posture" in result["required_stages"]):
        raise AssertionError


def test_delete_configuration_requires_exact_confirmation(tmp_path: Path) -> None:
    configuration(tmp_path, "beta", "headless")
    state_path = tmp_path / ".deploy-state/beta.json"

    try:
        delete_configuration(tmp_path, {"deployment_id": "beta", "confirmation": "wrong"})
    except UIError as error:
        if not ("exact deployment ID" in str(error)):
            raise AssertionError
    else:
        raise AssertionError("Deleting without exact confirmation must fail.")
    if not (state_path.is_file()):
        raise AssertionError

    if not (delete_configuration(tmp_path, {"deployment_id": "beta", "confirmation": "beta"}) == {
        "deleted": "beta"
    }):
        raise AssertionError
    if not (not state_path.exists()):
        raise AssertionError


def test_configurations_lists_valid_local_deployment_records(tmp_path: Path) -> None:
    configuration(tmp_path, "beta", "headless")
    configuration(tmp_path, "demo", "specialists")
    (tmp_path / ".deploy-state" / "invalid.json").write_text("not json", encoding="utf-8")

    result = configurations(tmp_path)

    if not ([(item["deployment_id"], item["profile"]) for item in result] == [
        ("beta", "headless"),
        ("demo", "specialists"),
    ]):
        raise AssertionError
    if not (all(item["started_at"] for item in result)):
        raise AssertionError


def test_local_planner_offers_xlsx_as_one_processor_workflow() -> None:
    if not ("native-xlsx-analysis" in _HTML):
        raise AssertionError
    if not ("XLSX',['xlsx_processor'],'headless" in _HTML):
        raise AssertionError
    if not ("Workbook Analysis pack first" in _HTML):
        raise AssertionError


def test_specialist_bulk_selection_is_independent_and_reactive() -> None:
    if not ('id="select-all-packs"' in _HTML):
        raise AssertionError
    if not ('id="select-all-direct-specialists"' in _HTML):
        raise AssertionError
    if not ('id="select-all-specialists"' in _HTML):
        raise AssertionError
    if not ('id="deselect-all-specialists"' in _HTML):
        raise AssertionError
    if not ("Select all capabilities" in _HTML):
        raise AssertionError
    if not ("Deselect all capabilities" in _HTML):
        raise AssertionError
    if not ("syncSpecialistSectionToggles" in _HTML):
        raise AssertionError
    if not ("toggleSelectionSection('[data-pack],[data-specialist]',true)" in _HTML):
        raise AssertionError
    if not ("toggleSelectionSection('[data-pack],[data-specialist]',false)" in _HTML):
        raise AssertionError
    if not ("control.indeterminate=false;control.checked=allSelected" in _HTML):
        raise AssertionError
    if not (".selection-section + .selection-section .selection-head{margin-top:22px" in _HTML):
        raise AssertionError
    if not ("--cg:#addfc4" in _HTML):
        raise AssertionError
    if not (".on .check{border-color:var(--cg-border);background:var(--cg)}" in _HTML):
        raise AssertionError
    if not (".row:not(.on) .check,.selection-toggle input:not(:checked):not(:indeterminate)" in _HTML):
        raise AssertionError
    if not (".selection-toggle input:checked{border-color:var(--cg-border);background-color:var(--cg)" in _HTML):
        raise AssertionError
    if not (".selection-toggle input{-webkit-appearance:none;appearance:none" in _HTML):
        raise AssertionError
    if not ('id="delete-configuration"' in _HTML):
        raise AssertionError
    if not ("/api/configuration/delete" in _HTML):
        raise AssertionError
    if not ("Open a configuration from Deployment details" in _HTML):
        raise AssertionError


def test_configuration_ui_starts_with_a_welcome_screen_and_tutorial() -> None:
    if not ('<section id="welcome" class="view on">' in _HTML):
        raise AssertionError
    if not ("Use an existing local configuration" in _HTML):
        raise AssertionError
    if not ("I’d prefer a visual guide to configuring my deployment" in _HTML):
        raise AssertionError
    if not ('data-config-intent="existing"' in _HTML):
        raise AssertionError
    if not ('data-config-intent="guided"' in _HTML):
        raise AssertionError
    if not ('data-to="tutorial">Preview the visual guide</button>' in _HTML):
        raise AssertionError
    if not ('id="tutorial"' in _HTML):
        raise AssertionError
    if not ('data-v="tutorial">How it works</button>' in _HTML):
        raise AssertionError
    if not ('data-v="overview">1. Deployment details</button>' in _HTML):
        raise AssertionError
    if not ('data-v="tags">4. Resource tags</button>' in _HTML):
        raise AssertionError
    if not ('data-v="observability">5. Observability' in _HTML):
        raise AssertionError
    if not ('id="active-config">No configuration loaded</span>' in _HTML):
        raise AssertionError
    if not ("/api/configurations" in _HTML):
        raise AssertionError
    if not ("<b>Open or create a configuration</b>" in _HTML):
        raise AssertionError
    if not ("<b>1. Open" not in _HTML):
        raise AssertionError


def test_configuration_ui_requires_an_explicit_profile_and_clear_cli_handoff() -> None:
    if not ('placeholder="e.g. poc-demo"' in _HTML):
        raise AssertionError
    if not ('Select a product profile' in _HTML):
        raise AssertionError
    if not ('Full experience — recommended POC start' in _HTML):
        raise AssertionError
    if not ("Choose a product profile first." in _HTML):
        raise AssertionError
    if not ("Preflight and deploy" in _HTML):
        raise AssertionError
    if not ("Type the exact deployment ID" in _HTML):
        raise AssertionError
    if not ("Saving changed tags marks every enabled stack for reconciliation" in _HTML):
        raise AssertionError
    if not ("New configurations begin empty" in _HTML):
        raise AssertionError
    if not ('id="run-preflight"' in _HTML):
        raise AssertionError
    if not ('id="start-deployment"' in _HTML):
        raise AssertionError
    if not ("Technical details" in _HTML):
        raise AssertionError
    if not ("/api/deployment/preflight" in _HTML):
        raise AssertionError
    if not ("/api/deployment/apply" in _HTML):
        raise AssertionError
    if not ("function saveProgress(selector)" in _HTML):
        raise AssertionError
    if not ("Validate, save & continue" in _HTML):
        raise AssertionError
    if not ("Step 1 of 7" in _HTML):
        raise AssertionError
    if not ("Validate & continue to specialists" in _HTML):
        raise AssertionError
    if not ("function saveWizardStep(view)" in _HTML):
        raise AssertionError
    if not ("function beginConfiguration(intent)" in _HTML):
        raise AssertionError


def test_local_deployment_runner_uses_fixed_commands_and_fresh_preflight(
    tmp_path: Path,
) -> None:
    configuration(tmp_path, "beta", "headless")
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FinishedProcess:
        stdout = io.StringIO("preflight output\n")

        def wait(self) -> int:
            return 0

    def start_process(command: list[str], **kwargs: object) -> FinishedProcess:
        calls.append((command, kwargs))
        return FinishedProcess()

    runner = DeploymentRunner(tmp_path, process_factory=start_process)
    runner.start_preflight({"deployment_id": "beta", "profile": "headless"})
    for _ in range(20):
        if runner.status()["status"] != "running":
            break
        time.sleep(0.01)

    status = runner.status()
    if not (status["status"] == "succeeded"):
        raise AssertionError
    if not (status["preflight"]["matches_current_configuration"] is True):
        raise AssertionError
    if not (calls[0][0] == _deployment_command(tmp_path, "headless", "beta", "preflight")):
        raise AssertionError
    if not (calls[0][1]["stdin"] is not None):
        raise AssertionError
    if not ("shell" not in calls[0][1]):
        raise AssertionError

    runner.start_deployment(
        {
            "deployment_id": "beta",
            "profile": "headless",
            "confirmation": "beta",
            "ui_public_access": False,
        }
    )
    if not (calls[1][0] == _deployment_command(tmp_path, "headless", "beta", "resume")):
        raise AssertionError
    if not (calls[1][1]["env"]["BADGERS_ASSUME_YES"] == "1"):
        raise AssertionError
    if not (calls[1][1]["env"]["UI_PUBLIC_ACCESS"] == "false"):
        raise AssertionError


def test_local_deployment_requires_typed_confirmation_and_current_preflight(
    tmp_path: Path,
) -> None:
    configuration(tmp_path, "beta", "specialists")
    runner = DeploymentRunner(tmp_path)

    try:
        runner.start_deployment(
            {
                "deployment_id": "beta",
                "profile": "specialists",
                "confirmation": "beta",
            }
        )
    except UIError as error:
        if not ("preflight" in str(error).lower()):
            raise AssertionError
    else:
        raise AssertionError("Deployment must require fresh preflight.")

    try:
        runner.start_deployment(
            {
                "deployment_id": "beta",
                "profile": "specialists",
                "confirmation": "wrong",
            }
        )
    except UIError as error:
        if not ("exact deployment ID" in str(error)):
            raise AssertionError
    else:
        raise AssertionError("Deployment must require typed confirmation.")


def test_observability_plan_is_local_and_validates_notification_handoff(tmp_path: Path) -> None:
    initial = configuration(tmp_path, "beta", "headless")
    result = save_observability(
        tmp_path,
        {
            "deployment_id": "beta",
            "profile": "headless",
            "mode": "baseline",
            "notification_kind": "sns",
            "notification_target": "arn:aws:sns:us-east-1:123456789012:badgers-alerts",
            "expected_revision": initial["state"]["observability_revision"],
        },
    )

    observability = result["observability"]
    if not (observability["mode"] == "baseline"):
        raise AssertionError
    if not (observability["notification"]["kind"] == "sns"):
        raise AssertionError
    if not (observability["candidate_plan"]["resources"]["cloudwatch_dashboards"] == 1):
        raise AssertionError
    if not (observability["candidate_plan"]["resources"]["notification_routes"] == 1):
        raise AssertionError

    try:
        save_observability(
            tmp_path,
            {
                "deployment_id": "beta",
                "profile": "headless",
                "mode": "links",
                "notification_kind": "sns",
                "notification_target": "arn:aws:sns:us-east-1:123456789012:badgers-alerts",
                "expected_revision": observability["revision"],
            },
        )
    except UIError as error:
        if not ("requires baseline or enhanced" in str(error)):
            raise AssertionError
    else:
        raise AssertionError("Links mode must not allow a notification handoff.")


def test_observability_screen_explains_local_deployment_and_optional_handoff() -> None:
    if not ('data-v="observability"' in _HTML):
        raise AssertionError
    if not ("Applied during deployment" in _HTML):
        raise AssertionError
    if not ("CloudWatch alarms remain visible without a notification destination" in _HTML):
        raise AssertionError
    if not ("/api/observability" in _HTML):
        raise AssertionError
    if not ("Observability add-on" in _HTML):
        raise AssertionError
    if not ("no aws inventory" in _HTML.lower()):
        raise AssertionError
    if not ("No added CloudWatch resources — no dashboard or alarms" in _HTML):
        raise AssertionError
    if not ("Enhanced — requires Full or Headless" in _HTML):
        raise AssertionError
    if not ("Enhanced is unavailable for Specialists" in _HTML):
        raise AssertionError


def test_tags_are_saved_per_deployment_and_require_reconciliation(tmp_path: Path) -> None:
    initial = configuration(tmp_path, "beta", "headless")
    result = save_tags(
        tmp_path,
        {
            "deployment_id": "beta",
            "profile": "headless",
            "tags": {
                "environment": "poc",
                "owner": "customer-team",
                "customer:program": "claims",
            },
            "expected_revision": initial["state"]["tag_revision"],
        },
    )

    if not (result["tags"]["tags"] == {
        "customer:program": "claims",
        "environment": "poc",
        "owner": "customer-team",
    }):
        raise AssertionError
    if not (plan(tmp_path, "beta", "headless")["tags_changed"]):
        raise AssertionError
    if not ("reconcile deployment tags across enabled stacks" in plan(
        tmp_path, "beta", "headless"
    )["required_stages"]):
        raise AssertionError

    try:
        save_tags(
            tmp_path,
            {
                "deployment_id": "beta",
                "profile": "headless",
                "tags": {"aws:reserved": "no"},
                "expected_revision": result["state"]["tag_revision"],
            },
        )
    except UIError as error:
        if not ("reserved by AWS" in str(error)):
            raise AssertionError
    else:
        raise AssertionError("AWS-reserved tag keys must be rejected.")


def test_api_requires_session_capability_and_same_loopback_origin(tmp_path: Path) -> None:
    server = ConfigurationServer(tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    port = server.server_port
    try:
        unauthenticated = http.client.HTTPConnection("127.0.0.1", port)
        unauthenticated.request("GET", "/api/packs")
        response = unauthenticated.getresponse()
        if not (response.status == 403):
            raise AssertionError
        if not (json.loads(response.read())["error"] == "Local session verification failed."):
            raise AssertionError
        unauthenticated.close()

        capability_only = http.client.HTTPConnection("127.0.0.1", port)
        capability_only.request(
            "GET",
            "/api/packs",
            headers={"X-Badgers-Session": server.session_token},
        )
        response = capability_only.getresponse()
        if not (response.status == 200):
            raise AssertionError
        if not (isinstance(json.loads(response.read()), list)):
            raise AssertionError
        capability_only.close()

        read_with_unexpected_origin = http.client.HTTPConnection("127.0.0.1", port)
        read_with_unexpected_origin.request(
            "GET",
            "/api/packs",
            headers={
                "X-Badgers-Session": server.session_token,
                "Origin": "https://example.invalid",
            },
        )
        response = read_with_unexpected_origin.getresponse()
        if not (response.status == 200):
            raise AssertionError
        if not (isinstance(json.loads(response.read()), list)):
            raise AssertionError
        read_with_unexpected_origin.close()

        wrong_origin_write = http.client.HTTPConnection("127.0.0.1", port)
        wrong_origin_write.request(
            "POST",
            "/api/packs",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": "2",
                "X-Badgers-Session": server.session_token,
                "Origin": "https://example.invalid",
            },
        )
        response = wrong_origin_write.getresponse()
        if not (response.status == 403):
            raise AssertionError
        if not (json.loads(response.read())["error"] == "Unexpected request origin."):
            raise AssertionError
        if not (response.getheader("Access-Control-Allow-Origin") is None):
            raise AssertionError
        if not (response.getheader("Cache-Control") == "no-store"):
            raise AssertionError
        response.read()
        wrong_origin_write.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
