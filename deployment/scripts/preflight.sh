#!/usr/bin/env bash
# Read-only, profile-aware deployment preflight reporting.
#
# This file is sourced by deploy.sh after common.sh.  It deliberately reuses the
# concrete preflight functions there so there is one operational implementation
# of each check.  A report is written for people and automation; deploy.sh
# always runs the applicable checks again rather than accepting an attestation
# that a previous failure has been fixed.

PREFLIGHT_REPORT_VERSION=1

_preflight_init() {
  PREFLIGHT_FAILURES=0
  PREFLIGHT_WARNINGS=0
  PREFLIGHT_ROWS="$(mktemp "${TMPDIR:-/tmp}/badgers-preflight.XXXXXX")"
  PREFLIGHT_REPORT_PATH="${REPO_ROOT}/.deploy-state/${DEPLOYMENT_ID}.preflight.json"
  export PREFLIGHT_READ_ONLY=1
  export PREFLIGHT_REPORT_MODE=1
}

_preflight_cleanup() {
  rm -f "${PREFLIGHT_ROWS:-}"
  unset PREFLIGHT_READ_ONLY PREFLIGHT_REPORT_MODE
}

_preflight_record() {
  local id label status detail
  id="$1"; label="$2"; status="$3"; detail="${4:-}"
  printf '%s\t%s\t%s\t%s\n' "${id}" "${label}" "${status}" "${detail//$'\t'/ }" >> "${PREFLIGHT_ROWS}"
  case "${status}" in
    PASS) log_success "Preflight: ${label}" ;;
    SKIP) log_info "Preflight: ${label} — not applicable" ;;
    DEFERRED) log_info "Preflight: ${label} — deferred until its prerequisite stage completes" ;;
    WARN) log_warn "Preflight: ${label}${detail:+ — ${detail}}" ; PREFLIGHT_WARNINGS=$((PREFLIGHT_WARNINGS + 1)) ;;
    *)    log_error "Preflight: ${label}${detail:+ — ${detail}}" ; PREFLIGHT_FAILURES=$((PREFLIGHT_FAILURES + 1)) ;;
  esac
}

_preflight_run() {
  local id label output rc
  id="$1"; label="$2"; shift 2
  output="$(mktemp "${TMPDIR:-/tmp}/badgers-preflight-output.XXXXXX")"
  rc=0
  "$@" >"${output}" 2>&1 || rc=$?
  sed 's/^/    /' "${output}"
  rm -f "${output}"
  case "${rc}" in
    0) _preflight_record "${id}" "${label}" PASS ;;
    2) _preflight_record "${id}" "${label}" WARN "An explicit decision is required during deployment." ;;
    *) _preflight_record "${id}" "${label}" FAIL "Run the displayed remediation, then run preflight again." ;;
  esac
}

_preflight_skip() {
  _preflight_record "$1" "$2" SKIP
}

_preflight_defer() {
  _preflight_record "$1" "$2" DEFERRED
}

preflight_config_release() {
  local stage_dir release_id rc=0
  stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/badgers-preflight-config.XXXXXX")" || return 1
  release_id="$(python3 "${DEPLOYMENT_DIR}/release_artifact.py" "${BADGERS_PROFILE}" \
    --selection-override "${SPECIALIST_PACK_ARGS[@]}" "${SPECIALIST_DIRECT_ARGS[@]}" \
    --routing-state "$(state_file)" \
    --destination "${stage_dir}")" || rc=$?
  rm -rf -- "${stage_dir}"
  if [ "${rc}" -ne 0 ] || [ -z "${release_id}" ]; then
    log_error "Could not build a deterministic ${BADGERS_PROFILE} configuration release."
    return 1
  fi
  log_info "Deterministic configuration release: ${release_id}"
}

preflight_foundation_layer() {
  local layer="${DEPLOYMENT_DIR}/lambdas/layer.zip"
  local source="${DEPLOYMENT_DIR}/badgers-foundation"
  local newer
  if [ ! -f "${layer}" ]; then
    log_error "Foundation layer is missing: ${layer}"
    log_error "Run deployment option 1 (Build Lambda Layers), then re-run preflight."
    return 1
  fi
  newer="$(_foundation_layer_newer_file "${layer}" "${source}" "${DEPLOYMENT_DIR}/lambdas")" || {
    log_success "Foundation layer is current."
    return 0
  }
  log_error "Foundation layer is stale: ${newer#"${DEPLOYMENT_DIR}/"}"
  log_error "Run deployment option 1 (Build Lambda Layers), then re-run preflight."
  return 1
}

preflight_stack_complete() {
  local logical_name="$1" stack_name status
  stack_name="$(_sn "${logical_name}")"
  status="$(aws cloudformation describe-stacks --stack-name "${stack_name}" \
    --region "${AWS_REGION}" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)"
  case "${status}" in
    CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE)
      log_success "${stack_name} is ${status}."
      return 0
      ;;
    *)
      log_error "${stack_name} is not ready (status: ${status:-not found})."
      return 1
      ;;
  esac
}

preflight_stack_output() {
  local logical_name="$1" output_key="$2" value
  value="$(stack_output "$(_sn "${logical_name}")" "${output_key}" 2>/dev/null || true)"
  if [ -z "${value}" ] || [ "${value}" = "None" ]; then
    log_error "$(_sn "${logical_name}") does not expose ${output_key}."
    return 1
  fi
  log_success "$(_sn "${logical_name}") exposes ${output_key}."
}

preflight_profile_resolution() {
  python3 "${DEPLOYMENT_DIR}/profile_resolver.py" "${BADGERS_PROFILE}" \
    --selection-override "${SPECIALIST_PACK_ARGS[@]}" "${SPECIALIST_DIRECT_ARGS[@]}" \
    >/dev/null
  log_success "${BADGERS_PROFILE} profile resolves safely."
}

_preflight_write_report() {
  local scope="$1" revision
  revision="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "unknown")"
  mkdir -p "$(dirname "${PREFLIGHT_REPORT_PATH}")"
  python3 - "${PREFLIGHT_ROWS}" "${PREFLIGHT_REPORT_PATH}" "${BADGERS_PROFILE}" \
    "${DEPLOYMENT_ID}" "${AWS_REGION}" "${scope}" "${revision}" "${PREFLIGHT_FAILURES}" "${PREFLIGHT_WARNINGS}" <<'PY'
import datetime
import json
import pathlib
import sys

rows_path, report_path, profile, deployment_id, region, scope, revision, failures, warnings = sys.argv[1:]
checks = []
for row in pathlib.Path(rows_path).read_text(encoding="utf-8").splitlines():
    check_id, label, status, detail = (row.split("\t", 3) + [""])[:4]
    checks.append({"id": check_id, "label": label, "status": status, "detail": detail})
report = {
    "version": 1,
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "profile": profile,
    "deployment_id": deployment_id,
    "region": region,
    "scope": scope,
    "source_revision": revision,
    "result": (
        "BLOCKED" if int(failures) else
        "READY_WITH_WARNINGS" if int(warnings) else
        "READY"
    ),
    "checks": checks,
}
path = pathlib.Path(report_path)
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY
}

_preflight_scope_has() {
  local scope="$1" wanted="$2"
  [ "${scope}" = "all" ] || [ "${scope}" = "${wanted}" ]
}

run_preflight() {
  local scope="${1:-all}"
  _preflight_init
  echo ""
  log_info "══════════════════════════════════════════════════════════"
  log_info "  PREFLIGHT — ${BADGERS_PROFILE} / ${DEPLOYMENT_ID} / ${AWS_REGION}"
  log_info "══════════════════════════════════════════════════════════"

  _preflight_run profile-resolution "Profile configuration" preflight_profile_resolution
  _preflight_record target-identity "Target account, Region, and deployment identity" PASS \
    "${ACCOUNT_ID} / ${AWS_REGION} / ${DEPLOYMENT_ID}"

  if _preflight_scope_has "${scope}" foundation; then
    _preflight_run cdk-bootstrap "CDK bootstrap" preflight_bootstrap
    _preflight_run service-quotas "Applicable service quota headroom" preflight_service_quotas
    _preflight_run model-access "Configured model availability" preflight_model_access
    _preflight_run model-invocation "Configured model invocation" preflight_model_invocation
    if [ "${BADGERS_SKIP_XRAY:-}" = "1" ]; then
      _preflight_skip xray-decision "X-Ray tracing capacity"
    else
      _preflight_run xray-decision "X-Ray tracing capacity" ensure_xray_decision
    fi
  fi

  if [ "${scope}" = "config" ]; then
    _preflight_run config-bucket "Configuration bucket availability" preflight_stack_output S3 ConfigBucketName
    _preflight_run configuration-release "Deterministic configuration release" preflight_config_release
  elif [ "${scope}" = "all" ]; then
    _preflight_defer config-bucket "Configuration bucket availability (built by stage 2)"
    _preflight_run configuration-release "Deterministic configuration release" preflight_config_release
  fi

  if [ "${scope}" = "specialists" ]; then
    _preflight_run specialist-foundation "Specialist infrastructure dependencies" preflight_stack_complete IAM
    _preflight_run foundation-layer "Foundation layer freshness" preflight_foundation_layer
  elif [ "${scope}" = "all" ]; then
    _preflight_defer specialist-foundation "Specialist infrastructure dependencies (built by stage 2)"
    _preflight_defer foundation-layer "Foundation layer freshness (built by stage 1 before specialists)"
  fi

  if [ "${scope}" = "gateway" ]; then
    _preflight_run gateway-specialists "Specialist function stack" preflight_stack_complete Lambda
    _preflight_run gateway-identity "Gateway machine identity stack" preflight_stack_complete Cognito
  elif [ "${scope}" = "runtime" ]; then
    _preflight_run runtime-gateway "Gateway endpoint availability" preflight_stack_output Gateway GatewayUrl
  elif [ "${scope}" = "all" ]; then
    _preflight_defer gateway-specialists "Specialist function stack (built by stage 4)"
    _preflight_defer gateway-identity "Gateway machine identity stack (built by stage 2)"
    _preflight_defer runtime-gateway "Gateway endpoint availability (built by stage 5)"
  fi

  if _preflight_scope_has "${scope}" gateway || _preflight_scope_has "${scope}" runtime; then
    if [ "${GATEWAY_ENABLED}" = "1" ]; then
      _preflight_run log-delivery "CloudWatch log-delivery capacity" preflight_log_delivery
    else
      _preflight_skip log-delivery "CloudWatch log-delivery capacity"
    fi
  fi

  if _preflight_scope_has "${scope}" runtime; then
    if [ "${RUNTIME_ENABLED}" = "1" ]; then
      _preflight_run runtime-container-platform "Runtime ARM64 container platform" preflight_docker_cross_platform arm64
    else
      _preflight_skip runtime-container-platform "Runtime ARM64 container platform"
    fi
  fi

  if _preflight_scope_has "${scope}" ui-build; then
    if [ "${UI_ENABLED}" = "1" ]; then
      _preflight_run ui-container-platform "UI AMD64 container platform" preflight_docker_cross_platform amd64
    else
      _preflight_skip ui-container-platform "UI AMD64 container platform"
    fi
  fi

  if [ "${scope}" = "ecs" ]; then
    if [ "${UI_ENABLED}" = "1" ]; then
      _preflight_run ecs-network "UI network stack" preflight_stack_complete Vpc
      _preflight_run ecs-runtime "Agent Runtime stack" preflight_stack_complete RuntimeWebSocket
      _preflight_run ecs-gateway "Gateway stack" preflight_stack_complete Gateway
      _preflight_run ecs-identity "UI identity stack" preflight_stack_complete Cognito
      _preflight_run ecs-service-linked-role "ECS service-linked role" preflight_ecs_slr
    else
      _preflight_skip ecs-network "UI network stack"
      _preflight_skip ecs-runtime "Agent Runtime stack"
      _preflight_skip ecs-gateway "Gateway stack"
      _preflight_skip ecs-identity "UI identity stack"
      _preflight_skip ecs-service-linked-role "ECS service-linked role"
    fi
  elif [ "${scope}" = "all" ]; then
    _preflight_defer ecs-network "UI network stack (built by stage 2)"
    _preflight_defer ecs-runtime "Agent Runtime stack (built by stage 6)"
    _preflight_defer ecs-gateway "Gateway stack (built by stage 5)"
    _preflight_defer ecs-identity "UI identity stack (built by stage 2)"
    _preflight_defer ecs-service-linked-role "ECS service-linked role (verified immediately before stage 8)"
  fi

  _preflight_write_report "${scope}"
  echo ""
  if [ "${PREFLIGHT_FAILURES}" -eq 0 ]; then
    if [ "${PREFLIGHT_WARNINGS}" -gt 0 ]; then
      log_warn "PREFLIGHT READY WITH ${PREFLIGHT_WARNINGS} WARNING(S) — report: ${PREFLIGHT_REPORT_PATH#"${REPO_ROOT}/"}"
    else
      log_success "PREFLIGHT READY — report: ${PREFLIGHT_REPORT_PATH#"${REPO_ROOT}/"}"
    fi
    _preflight_cleanup
    return 0
  fi
  log_error "PREFLIGHT BLOCKED — ${PREFLIGHT_FAILURES} check(s) failed."
  log_error "Fix the reported condition and run the same deployment option again; it re-validates automatically."
  log_info "Report: ${PREFLIGHT_REPORT_PATH#"${REPO_ROOT}/"}"
  _preflight_cleanup
  return 1
}

preflight_scope_for_option() {
  case "$1" in
    1) echo "layers" ;;
    2) echo "foundation" ;;
    3) echo "config" ;;
    4) echo "specialists" ;;
    5) echo "gateway" ;;
    6) echo "runtime" ;;
    7) echo "ui-build" ;;
    8) echo "ecs" ;;
    9|r|R|resume) echo "all" ;;
    *) echo "" ;;
  esac
}
