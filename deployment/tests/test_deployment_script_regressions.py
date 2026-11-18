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

    if not (windows.stdout.strip() == "cdk.cmd"):
        raise AssertionError
    if not (unix.stdout.strip() == "cdk"):
        raise AssertionError


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

    if not ("--query 'stopReason' --output text" in common):
        raise AssertionError
    if not ("L-0263D0A3" in common):
        raise AssertionError
    if not ("describe-addresses" in common):
        raise AssertionError
    if not ('REGION="${AWS_REGION}"' in runtime_build):
        raise AssertionError
    if not ("ensure_region" in runtime_build):
        raise AssertionError
    if not ("public.ecr.aws/lambda/python:3.12" in native_pdf_dockerfile):
        raise AssertionError
    if not ("PyMuPDF>=1.26.6" in native_pdf_requirements):
        raise AssertionError
    if not (not (DEPLOYMENT_DIR / "lambdas" / "build_pdf_processing_layer.sh").exists()):
        raise AssertionError


def test_ecs_endpoint_is_not_prefixed_with_a_second_scheme() -> None:
    ecs_stack = (DEPLOYMENT_DIR / "stacks" / "ecs_stack.py").read_text()
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    if not ('CallbackURLs": [f"{service_url}/callback"]' in ecs_stack):
        raise AssertionError
    if not ('LogoutURLs": [f"{service_url}/"]' in ecs_stack):
        raise AssertionError
    if not ('value=f"https://{self.express_service.attr_endpoint}"' not in ecs_stack):
        raise AssertionError
    if not ('echo "  │  https://${endpoint}"' not in deploy_script):
        raise AssertionError


def test_deployment_menu_offers_the_local_configuration_planner() -> None:
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    if not ('C  LOCAL DEPLOYMENT WIZARD' in deploy_script):
        raise AssertionError
    if not ('c|C|configure)' in deploy_script):
        raise AssertionError
    if not ('python3 "${SCRIPT_DIR}/deployment/configuration_ui.py" --repo "${SCRIPT_DIR}"' in deploy_script):
        raise AssertionError
    if not ('if [ "$#" -eq 0 ] && [ "${CLI_MODE}" = "0" ]' in deploy_script):
        raise AssertionError
    if not ('if [ "${1:-}" = "--cli" ]; then' in deploy_script):
        raise AssertionError


def test_local_ui_executor_has_a_constrained_noninteractive_entry_point() -> None:
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    if not ("--local-ui-execute" in deploy_script):
        raise AssertionError
    if not ("Invalid local UI deployment operation." in deploy_script):
        raise AssertionError
    if not ("LOCAL_UI_DEPLOYMENT_ID" in deploy_script):
        raise AssertionError
    if not ("run_preflight all || exit 1" in deploy_script):
        raise AssertionError
    if not ("step_resume" in deploy_script):
        raise AssertionError


def test_runtime_resume_uses_the_image_tag_that_was_pushed() -> None:
    common = COMMON_SH.read_text()
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    if not ("use_pushed_runtime_image_tag()" in common):
        raise AssertionError
    if not ('get_state "runtime_image_tag"' in common):
        raise AssertionError
    if not ("describe-images" in common):
        raise AssertionError
    if not ("Recovered pushed runtime image tag from ECR" in common):
        raise AssertionError
    if not ("use_pushed_runtime_image_tag || return 1" in common):
        raise AssertionError
    if not ('set_state "runtime_image_tag" "\\"${RUNTIME_IMAGE_TAG}\\""' in deploy_script):
        raise AssertionError


def test_cdk_environment_reuses_the_existing_ui_exposure_choice() -> None:
    common = COMMON_SH.read_text()
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    if not ("resolve_ui_public_access()" in common):
        raise AssertionError
    if not ('get_state "ui_public_access"' in common):
        raise AssertionError
    if not ("cloudformation get-template" in common):
        raise AssertionError
    if not ("PublicSubnet" in common):
        raise AssertionError
    if not ("resolve_ui_public_access || return 1" in common):
        raise AssertionError
    if not ('set_state "ui_public_access" "${UI_PUBLIC_ACCESS}"' in deploy_script):
        raise AssertionError


def test_usage_examples_name_the_product_before_the_action() -> None:
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()
    scripts_doc = (DEPLOYMENT_DIR / "DEPLOYMENT_SCRIPTS.md").read_text()

    if not ("./deploy.sh full preflight" in deploy_script):
        raise AssertionError
    if not ("./deploy.sh full resume" in deploy_script):
        raise AssertionError
    if not ("./deploy.sh full 4" in deploy_script):
        raise AssertionError
    if not ("./deploy.sh full resume" in scripts_doc):
        raise AssertionError


def test_ecs_deploy_waits_for_the_service_linked_role_to_propagate() -> None:
    common = COMMON_SH.read_text()
    deploy_script = (REPO_ROOT / "deploy.sh").read_text()

    if not ("aws iam list-roles" in common):
        raise AssertionError
    if not ("--path-prefix /aws-service-role/ecs.amazonaws.com/" in common):
        raise AssertionError
    if not ("MSYS_NO_PATHCONV=1" in common):
        raise AssertionError
    if not ("ECS_SLR_SETTLE_SECONDS" in common):
        raise AssertionError
    if not ("read-only preflight will not create it" in common):
        raise AssertionError
    if not ("PREFLIGHT_READ_ONLY=0 preflight_ecs_slr || return 1" in deploy_script):
        raise AssertionError
