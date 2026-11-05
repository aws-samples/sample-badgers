"""Regression coverage for deployment-script failures found during beta testing."""

from __future__ import annotations

import subprocess
from pathlib import Path


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOYMENT_DIR.parent
COMMON_SH = DEPLOYMENT_DIR / "scripts" / "common.sh"


def _bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cdk_executable_uses_windows_cmd_shim_only_on_windows() -> None:
    windows = _bash(
        f'source "{COMMON_SH}"; uname() {{ printf "%s\\n" MINGW64_NT-10.0; }}; cdk_executable'
    )
    unix = _bash(
        f'source "{COMMON_SH}"; uname() {{ printf "%s\\n" Darwin; }}; cdk_executable'
    )

    assert windows.stdout.strip() == "cdk.cmd"
    assert unix.stdout.strip() == "cdk"


def test_beta_preflight_contracts_do_not_reintroduce_known_failures() -> None:
    common = COMMON_SH.read_text()
    runtime_build = (
        DEPLOYMENT_DIR / "runtime" / "build_and_push_websocket.sh"
    ).read_text()
    native_pdf_dockerfile = (
        DEPLOYMENT_DIR / "lambdas" / "containers" / "native_pdf_processor" / "Dockerfile"
    ).read_text()
    native_pdf_requirements = (
        DEPLOYMENT_DIR
        / "lambdas"
        / "containers"
        / "native_pdf_processor"
        / "requirements.txt"
    ).read_text()

    assert "--query 'stopReason' --output text" in common
    assert "L-0263D0A3" in common
    assert "describe-addresses" in common
    assert 'REGION="${AWS_REGION}"' in runtime_build
    assert "ensure_region" in runtime_build
    assert "public.ecr.aws/lambda/python:3.12" in native_pdf_dockerfile
    assert "PyMuPDF>=1.26.6" in native_pdf_requirements
    assert not (DEPLOYMENT_DIR / "lambdas" / "build_pdf_processing_layer.sh").exists()


def test_ecs_endpoint_is_not_prefixed_with_a_second_scheme() -> None:
    ecs_stack = (DEPLOYMENT_DIR / "stacks" / "ecs_stack.py").read_text()
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    assert 'CallbackURLs": [f"{service_url}/callback"]' in ecs_stack
    assert 'LogoutURLs": [f"{service_url}/"]' in ecs_stack
    assert 'value=f"https://{self.express_service.attr_endpoint}"' not in ecs_stack
    assert 'echo "  │  https://${endpoint}"' not in deploy_script


def test_deployment_menu_offers_the_local_configuration_planner() -> None:
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    assert 'C  CONFIGURATION PLANNER' in deploy_script
    assert 'c|C|configure)' in deploy_script
    assert 'python3 "${SCRIPT_DIR}/deployment/configuration_ui.py" --repo "${SCRIPT_DIR}"' in deploy_script
