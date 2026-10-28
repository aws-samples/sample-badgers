"""Tests for the private local configuration planner."""

from __future__ import annotations

import http.client
import json
import stat
import threading
from pathlib import Path

from deployment.configuration_ui import _HTML, ConfigurationServer, configuration, plan


def test_configuration_creates_local_desired_state_with_private_permissions(
    tmp_path: Path,
) -> None:
    result = configuration(tmp_path, "beta", "headless")
    state_path = tmp_path / ".deploy-state/beta.json"

    assert result["selection"]["profile"] == "headless"
    assert result["routing"]["templates"] == ["standard-document-analysis"]
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600


def test_plan_is_a_cli_handoff_not_a_deployment_action(tmp_path: Path) -> None:
    result = plan(tmp_path, "beta", "headless")

    assert result["cli_next_step"] == "./deploy.sh headless resume"
    assert "CLI re-runs preflight" in result["apply_note"]
    assert "publish configuration" in result["required_stages"]


def test_local_planner_offers_xlsx_as_one_processor_workflow() -> None:
    assert "native-xlsx-analysis" in _HTML
    assert "XLSX',['xlsx_processor'],'headless" in _HTML
    assert "Workbook Analysis pack first" in _HTML


def test_api_requires_session_capability_and_same_loopback_origin(tmp_path: Path) -> None:
    server = ConfigurationServer(tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    port = server.server_port
    try:
        unauthenticated = http.client.HTTPConnection("127.0.0.1", port)
        unauthenticated.request("GET", "/api/packs")
        response = unauthenticated.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["error"] == "Local session verification failed."
        unauthenticated.close()

        capability_only = http.client.HTTPConnection("127.0.0.1", port)
        capability_only.request(
            "GET",
            "/api/packs",
            headers={"X-Badgers-Session": server.session_token},
        )
        response = capability_only.getresponse()
        assert response.status == 200
        assert isinstance(json.loads(response.read()), list)
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
        assert response.status == 200
        assert isinstance(json.loads(response.read()), list)
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
        assert response.status == 403
        assert json.loads(response.read())["error"] == "Unexpected request origin."
        assert response.getheader("Access-Control-Allow-Origin") is None
        assert response.getheader("Cache-Control") == "no-store"
        response.read()
        wrong_origin_write.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
