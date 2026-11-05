#!/bin/bash
set -euo pipefail

# ── BADGERS — Interactive Deployment Menu ──────────────────────────────────
#
# Usage:
#   ./deploy.sh             # Pick a deployment, then the interactive menu
#   ./deploy.sh headless    # Pick a deployment, then deploy the Headless product
#   ./deploy.sh headless 9  # Run every Headless deployment step
#   ./deploy.sh full 9      # Run every Full deployment step
#   ./deploy.sh specialists --pack tables-and-forms 9
#   ./deploy.sh configure --ui # Open the local configuration planner (no AWS actions)
#   ./deploy.sh preflight   # Validate the selected product before deployment
#   ./deploy.sh resume      # Pick a deployment, then run only its remaining steps
#   ./deploy.sh 4           # Pick a deployment, then run option 4 directly
#
# DEPLOYMENT_ID is always chosen interactively and is ignored if exported, so a stale
# value in your shell cannot target the wrong deployment.
#
# This script is fully resumable — re-run after a failure to continue where you
# left off. All operations are idempotent (safe to run multiple times).
#
# Stack names are BADGERS-{Name}-{suffix}. The suffix is generated once per
# DEPLOYMENT_ID and persisted in .deploy-state/{DEPLOYMENT_ID}.json, so several
# deployments can coexist in one account and region.
# ───────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/deployment/scripts/common.sh"
source "${SCRIPT_DIR}/deployment/scripts/preflight.sh"

trap 'echo ""; log_error "deploy.sh failed at line ${LINENO}: ${BASH_COMMAND}"; exit 1' ERR

launch_configuration_ui() {
  log_info "Opening the local configuration planner. Press Ctrl-C there to return to this menu."
  python3 "${SCRIPT_DIR}/deployment/configuration_ui.py" --repo "${SCRIPT_DIR}"
}

# Configuration UI is deliberately a local planning surface. It starts a
# loopback-only server which can edit and validate desired state, but cannot
# deploy or access AWS credentials. The regular CLI remains the executor.
if [ "${1:-}" = "configure" ]; then
  if [ "${2:-}" != "--ui" ] || [ "$#" -ne 2 ]; then
    log_error "Usage: ./deploy.sh configure --ui"
    exit 1
  fi
  exec python3 "${SCRIPT_DIR}/deployment/configuration_ui.py" --repo "${SCRIPT_DIR}"
fi

# Product profile is selected before deployment identity so its safety checks happen
# before any AWS preflight or resource creation. A deployment ID can belong to only
# one product profile; this prevents a Headless run from reusing Full stack state.
#
# An explicit argument (including the three product wrapper scripts) always wins.
# An interactive bare ./deploy.sh exposes the product chooser. Non-interactive
# callers retain the legacy Full default unless BADGERS_PROFILE is set.
BADGERS_PROFILE="${BADGERS_PROFILE:-}"

show_product_chooser() {
  clear
  echo -e "${CYAN}${BOLD}"
  cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                         BADGERS DEPLOY                           ║
║                    SELECT A PRODUCT TO DEPLOY                    ║
╚══════════════════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"
  echo -e "  ${BLUE}${BOLD}C  CONFIGURATION PLANNER${NC}"
  echo -e "     Local UI for specialists, workflow routing, and deployment planning"
  echo ""
  echo -e "  ${CYAN}${BOLD}1  FULL EXPERIENCE${NC}"
  echo -e "     Agent API  •  internal gateway  •  all specialists  •  browser UI"
  echo ""
  echo -e "  ${GREEN}${BOLD}2  HEADLESS AGENT${NC}"
  echo -e "     Authenticated agent API  •  durable operations  •  curated specialist packs"
  echo ""
  echo -e "  ${YELLOW}${BOLD}3  SPECIALIST PACKS${NC}"
  echo -e "     Direct Lambda specialists  •  selected packs  •  no agent, gateway, or UI"
  echo ""
  echo -e "  ${RED}${BOLD}4  QUIT${NC}"
  echo -e "     Leave deployment without selecting a product  ${RED}(or type q, then press Enter)${NC}"
  echo ""
  echo -e "${BLUE}  Your next steps: select product → select deployment → run preflight → deploy${NC}"
  echo ""
}

show_profile_preflight() {
  local profile_label profile_color
  case "${BADGERS_PROFILE}" in
    full)        profile_label="FULL EXPERIENCE"; profile_color="${CYAN}" ;;
    headless)    profile_label="HEADLESS AGENT"; profile_color="${GREEN}" ;;
    specialists) profile_label="SPECIALIST PACKS"; profile_color="${YELLOW}" ;;
  esac

  [ -t 1 ] || return 0
  echo ""
  echo -e "${profile_color}${BOLD}  ┌────────────────────────────────────────────────────────────────┐${NC}"
  echo -e "${profile_color}${BOLD}  │  ${profile_label} — preparing deployment                         │${NC}"
  echo -e "${profile_color}${BOLD}  └────────────────────────────────────────────────────────────────┘${NC}"
  echo -e "  ${BLUE}Next: select deployment identity, validate AWS access, then open the deployment console.${NC}"
  echo ""
}

case "${1:-}" in
  full|headless|specialists)
    BADGERS_PROFILE="$1"
    shift
    ;;
esac

if [ -z "${BADGERS_PROFILE}" ]; then
  if [ -t 0 ]; then
    show_product_chooser
    read -r -p "Product (1-4), c for planner, or q to quit [1]: " profile_choice
    case "${profile_choice:-1}" in
      1|full) BADGERS_PROFILE="full" ;;
      2|headless) BADGERS_PROFILE="headless" ;;
      3|specialists) BADGERS_PROFILE="specialists" ;;
      c|C|configure)
        launch_configuration_ui
        exit $?
        ;;
      4|q|Q)
        log_info "Exiting without selecting a product."
        exit 0
        ;;
      *)
        log_error "Invalid product choice. Select 1–4, c, or q."
        exit 1
        ;;
    esac
  else
    BADGERS_PROFILE="full"
  fi
fi

# Selection flags are an explicit desired-state update for this deployment
# identity.  They compose reusable packs with individual specialists and work
# for every product profile.  The resulting selection is persisted after the
# deployment is chosen, not inferred from transient shell arguments.
SPECIALIST_PACK_ARGS=()
SPECIALIST_DIRECT_ARGS=()
SPECIALIST_SELECTION_FROM_CLI=0
while [ "${1:-}" = "--pack" ] || [ "${1:-}" = "--specialist" ]; do
  if [ -z "${2:-}" ]; then
    log_error "$1 requires a specialist or pack name."
    exit 1
  fi
  case "$1" in
    --pack) SPECIALIST_PACK_ARGS+=(--pack "$2") ;;
    --specialist) SPECIALIST_DIRECT_ARGS+=(--specialist "$2") ;;
  esac
  SPECIALIST_SELECTION_FROM_CLI=1
  shift 2
done
case "${BADGERS_PROFILE}" in
  full)
    UI_ENABLED=1; MEMORY_ENABLED=1; X_RAY_ENABLED=1
    RUNTIME_ENABLED=1; GATEWAY_ENABLED=1; COGNITO_ENABLED=1
    OPERATION_LEDGER_ENABLED=1; ECR_ENABLED=1
    ;;
  headless)
    UI_ENABLED=0; MEMORY_ENABLED=0; X_RAY_ENABLED=0
    RUNTIME_ENABLED=1; GATEWAY_ENABLED=1; COGNITO_ENABLED=1
    OPERATION_LEDGER_ENABLED=1; ECR_ENABLED=1
    ;;
  specialists)
    UI_ENABLED=0; MEMORY_ENABLED=0; X_RAY_ENABLED=0
    RUNTIME_ENABLED=0; GATEWAY_ENABLED=0; COGNITO_ENABLED=0
    OPERATION_LEDGER_ENABLED=0; ECR_ENABLED=0
    ;;
  *)
    log_error "Unknown BADGERS profile: ${BADGERS_PROFILE}. Supported profiles: full, headless, specialists."
    exit 1
    ;;
esac
ECR_REQUIRED_BY_PRODUCT="${ECR_ENABLED}"
export BADGERS_PROFILE

# Keep the resolver command non-empty under `set -u`: expanding an explicitly
# declared-but-empty array is an unbound-variable error in some Bash versions.
PROFILE_RESOLVER_ARGS=("${BADGERS_PROFILE}")
if [ "${SPECIALIST_SELECTION_FROM_CLI}" = "1" ]; then
  PROFILE_RESOLVER_ARGS+=(--selection-override)
  PROFILE_RESOLVER_ARGS+=("${SPECIALIST_PACK_ARGS[@]}" "${SPECIALIST_DIRECT_ARGS[@]}")
fi
if ! python3 "${SCRIPT_DIR}/deployment/profile_resolver.py" "${PROFILE_RESOLVER_ARGS[@]}" >/dev/null; then
  log_error "Profile '${BADGERS_PROFILE}' is invalid. Fix its profile or specialist-pack configuration before deploying."
  exit 1
fi
log_info "Product profile: ${BADGERS_PROFILE}"
show_profile_preflight

# ── Defaults ──
IMAGE_TAG="${IMAGE_TAG:-latest}"
RUNTIME_IMAGE_TAG="${RUNTIME_IMAGE_TAG:-}"
UI_IMAGE_TAG="${UI_IMAGE_TAG:-frontend}"
# Must match CONTAINER_PORT in deployment/stacks/ecs_stack.py — it is sent with the
# forced container rollout in step 8, and a mismatch would repoint the service's target
# port at something the app is not listening on.
UI_CONTAINER_PORT="${UI_CONTAINER_PORT:-7860}"

# _confirm() comes from deployment/scripts/common.sh, so preflight checks there can
# prompt too. Honours BADGERS_ASSUME_YES=1 for non-interactive runs.

# DEPLOYMENT_ID is always chosen interactively here, never inherited from the
# environment. A value left exported in the shell silently targets another
# deployment's stacks and state file, which is how you end up deploying into the
# wrong suffix. destroy.sh still accepts it explicitly.
unset DEPLOYMENT_ID
choose_deployment || exit 1
require_deployment_id
ensure_region || { log_error "ensure_region failed"; exit 1; }
log_info "Region OK: ${AWS_REGION}"

ensure_account || { log_error "ensure_account failed — check AWS credentials"; exit 1; }
log_info "Account OK: ${ACCOUNT_ID}"

had_existing_state=0
[ -f "$(state_file)" ] && had_existing_state=1
ensure_suffix || { log_error "ensure_suffix failed"; exit 1; }
log_info "Suffix OK: ${STACK_SUFFIX}"

# Persist the product profile with the deployment state.  Existing state without this
# marker predates profiles and is treated as Full only, avoiding an unsafe implicit
# conversion of an already-deployed Full stack into Headless.
recorded_profile="$(get_state "product_profile")"
if [ -z "${recorded_profile}" ]; then
  if [ "${BADGERS_PROFILE}" != "full" ] && [ "${had_existing_state}" = "1" ]; then
    log_error "Deployment '${DEPLOYMENT_ID}' has legacy Full state but no profile marker."
    log_error "Choose a new deployment ID for Headless; do not reuse this stack identity."
    exit 1
  fi
  set_state "product_profile" "\"${BADGERS_PROFILE}\""
elif [ "${recorded_profile}" != "${BADGERS_PROFILE}" ]; then
  log_error "Deployment '${DEPLOYMENT_ID}' belongs to the '${recorded_profile}' profile."
  log_error "Choose a different deployment ID for '${BADGERS_PROFILE}' so stacks and state cannot collide."
  exit 1
fi
# Resolve (or create) the deployment's persistent desired selection. Every
# invocation re-resolves the selected pack JSON, so a source-controlled pack
# edit is visible on the next run. A changed fingerprint marks precisely the
# configuration consumers as pending while preserving the same stack suffix.
SPECIALIST_CONTAINER_NAMES=()
SELECTION_CHANGED=0
load_specialist_selection() {
  local use_cli="${1:-0}" selection_command="${2:-ensure}" output line_kind line_value
  local output_file
  output_file="$(mktemp "${TMPDIR:-/tmp}/badgers-specialist-selection.XXXXXX")" || return 1
  local command=(
    python3 "${SCRIPT_DIR}/deployment/specialist_selection.py" "${selection_command}"
    --state "$(state_file)" --profile "${BADGERS_PROFILE}" --lines
  )
  if [ "${use_cli}" = "1" ]; then
    command+=("${SPECIALIST_PACK_ARGS[@]}" "${SPECIALIST_DIRECT_ARGS[@]}")
  fi
  if ! "${command[@]}" >"${output_file}"; then
    sed 's/^/  /' "${output_file}" >&2
    rm -f "${output_file}"
    log_error "Could not resolve the saved specialist selection."
    return 1
  fi

  SPECIALIST_PACK_ARGS=()
  SPECIALIST_DIRECT_ARGS=()
  SPECIALIST_CONTAINER_NAMES=()
  SELECTION_CHANGED=0
  ECR_ENABLED="${ECR_REQUIRED_BY_PRODUCT}"
  while IFS=$'\t' read -r line_kind line_value; do
    case "${line_kind}" in
      pack) SPECIALIST_PACK_ARGS+=(--pack "${line_value}") ;;
      specialist) SPECIALIST_DIRECT_ARGS+=(--specialist "${line_value}") ;;
      container) SPECIALIST_CONTAINER_NAMES+=("${line_value}") ;;
      changed) [ "${line_value}" = "true" ] && SELECTION_CHANGED=1 ;;
    esac
  done <"${output_file}"
  rm -f "${output_file}"

  # The arrays include alternating flags and values. Rebuild their comma lists
  # without relying on temporary state variables or a profile-specific default.
  if [ "${#SPECIALIST_PACK_ARGS[@]}" -gt 0 ]; then
    BADGERS_SPECIALIST_PACKS="$(python3 - "${SPECIALIST_PACK_ARGS[@]}" <<'PY'
import sys
print(",".join(sys.argv[2::2]))
PY
)"
  else
    BADGERS_SPECIALIST_PACKS=""
  fi
  if [ "${#SPECIALIST_DIRECT_ARGS[@]}" -gt 0 ]; then
    BADGERS_DIRECT_SPECIALISTS="$(python3 - "${SPECIALIST_DIRECT_ARGS[@]}" <<'PY'
import sys
print(",".join(sys.argv[2::2]))
PY
)"
  else
    BADGERS_DIRECT_SPECIALISTS=""
  fi
  export BADGERS_SPECIALIST_PACKS BADGERS_DIRECT_SPECIALISTS
  export BADGERS_SELECTION_OVERRIDE=true

  if [ "${#SPECIALIST_CONTAINER_NAMES[@]}" -gt 0 ]; then
    ECR_ENABLED=1
  fi

  if [ "${SELECTION_CHANGED}" = "1" ]; then
    invalidate_state "infra_complete"
    invalidate_state "s3_files_uploaded"
    invalidate_state "specialists_complete"
    invalidate_state "gateway_complete"
    invalidate_state "runtime_image_pushed"
    invalidate_state "runtime_deployed"
    log_warn "Specialist selection changed for deployment '${DEPLOYMENT_ID}'."
    log_info "The same deployment ID will reconcile infrastructure, configuration, Lambdas, and enabled consumers."
  fi
}

load_specialist_selection "${SPECIALIST_SELECTION_FROM_CLI}" || exit 1

ROUTING_CONFIGURATION_CHANGED=0
ROUTING_CONFIGURATION_READY=0
load_routing_configuration() {
  local routing_command="${1:-ensure}" output_file kind value
  output_file="$(mktemp "${TMPDIR:-/tmp}/badgers-routing-configuration.XXXXXX")" || return 1
  if ! python3 "${SCRIPT_DIR}/deployment/routing_configuration.py" "${routing_command}" \
      --state "$(state_file)" --profile "${BADGERS_PROFILE}" --lines >"${output_file}"; then
    sed 's/^/  /' "${output_file}" >&2
    rm -f "${output_file}"
    return 1
  fi
  ROUTING_CONFIGURATION_CHANGED=0
  while IFS=$'\t' read -r kind value; do
    [ "${kind}" = "changed" ] && [ "${value}" = "true" ] && ROUTING_CONFIGURATION_CHANGED=1
  done <"${output_file}"
  rm -f "${output_file}"
  ROUTING_CONFIGURATION_READY=1
  export BADGERS_ROUTING_STATE="$(state_file)"
  if [ "${ROUTING_CONFIGURATION_CHANGED}" = "1" ]; then
    invalidate_state "s3_files_uploaded"
    invalidate_state "specialists_complete"
    invalidate_state "gateway_complete"
    invalidate_state "runtime_image_pushed"
    invalidate_state "runtime_deployed"
    log_warn "Workflow routing changed for deployment '${DEPLOYMENT_ID}'."
    log_info "Publish a new configuration release and redeploy its enabled consumers."
  fi
}

load_routing_configuration || {
  log_error "Saved routing configuration is incompatible with the selected specialists."
  log_error "Use menu option w to choose compatible workflow templates before deploying."
}

# Headless retains CloudWatch application logs but intentionally omits optional
# X-Ray Transaction Search.  This avoids account-level quota consumption for an
# API-only deployment and makes the profile's infrastructure boundary explicit.
if [ "${X_RAY_ENABLED}" = "0" ]; then
  export BADGERS_SKIP_XRAY=1
fi

RUNTIME_IMAGE_TAG="${RUNTIME_IMAGE_TAG:-$(runtime_image_tag 2>/dev/null || echo "websocket")}"
export RUNTIME_IMAGE_TAG
log_info "Runtime image tag: ${RUNTIME_IMAGE_TAG}"

ensure_cdk_deps || { log_error "ensure_cdk_deps failed — check node/cdk/uv installation"; exit 1; }

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
# One repository per deployment, named badgers-{DEPLOYMENT_ID}-{suffix} by the ECR
# stack. Both the runtime and UI images live in it under different tags.
ECR_REPO_NAME="badgers-$(resource_id)"
ECR_REPO="${ECR_BASE}/${ECR_REPO_NAME}"

record_applied_specialist_selection() {
  local config_release_id
  config_release_id="$(get_state "config_release_id")"
  python3 "${SCRIPT_DIR}/deployment/specialist_selection.py" record-applied \
    --state "$(state_file)" --profile "${BADGERS_PROFILE}" \
    --config-release-id "${config_release_id}" >/dev/null \
    || { log_error "Could not record the applied specialist selection."; return 1; }
  python3 "${SCRIPT_DIR}/deployment/routing_configuration.py" record-applied \
    --state "$(state_file)" --profile "${BADGERS_PROFILE}" \
    --config-release-id "${config_release_id}" >/dev/null \
    || { log_error "Could not record the applied routing configuration."; return 1; }
  log_success "Specialist selection and routing reconciled with deployment state."
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Build Lambda Layers
# ══════════════════════════════════════════════════════════════════════════
step_layers() {
  log_step "Step 1: Build Lambda Layers"

  if check_completed "layers_built"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_warn "Already built. Re-running rebuilds the layer archives."
    _confirm || return 0
  fi

  # The foundation layer must be built before container Lambdas: their build
  # scripts stage foundation/config from layer.zip (or a legacy expanded
  # layer/python directory), so a stale archive would ship stale code.
  log_info "Building foundation layer..."
  (cd "${DEPLOYMENT_DIR}/lambdas" && ./build_foundation_layer.sh) \
    || { log_error "Foundation layer build failed."; return 1; }

  if [ -f "${DEPLOYMENT_DIR}/lambdas/build_poppler_qdf_layer.sh" ]; then
    log_info "Building Poppler/qpdf layer..."
    (cd "${DEPLOYMENT_DIR}/lambdas" && ./build_poppler_qdf_layer.sh)
  fi

  mark_complete "layers_built"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Foundational Infrastructure
# ══════════════════════════════════════════════════════════════════════════
step_infra() {
  log_step "Step 2: Foundational Infrastructure"

  if check_completed "infra_complete"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_warn "Already complete. Re-running will update stacks if changes are detected."
    _confirm || return 0
  fi

  # CDK must be bootstrapped before first deploy
  preflight_bootstrap || return 1

  # Service quotas for VPC, S3, ECR, Lambda, etc.
  preflight_service_quotas || return 1
  # Bedrock model access, before anything is created. A model the account cannot reach
  # otherwise surfaces as AccessDenied deep inside a document run, where _should_fallback
  # correctly refuses to retry it — so a chain with a healthy fallback still hard-fails.
  # Order matters: access (is the model there) → invocation (does a real call succeed).
  #
  # An infrastructure-only update (for example, a narrowly scoped IAM grant)
  # may deliberately bypass these account-wide model checks. The opt-out is
  # explicit and logged; normal deployments retain both safety gates.
  if [ "${BADGERS_SKIP_MODEL_PREFLIGHT:-0}" = "1" ]; then
    log_warn "BADGERS_SKIP_MODEL_PREFLIGHT=1 — skipping Bedrock model access and invocation preflights."
  else
    preflight_model_access || return 1
    preflight_model_invocation || return 1
  fi

  # Decide about X-Ray before deploying anything. Enabling Transaction Search needs a
  # CloudWatch Logs resource policy from a hard quota of 10 per region, and finding that
  # out from CloudFormation means failing halfway through with other stacks already up.
  # BADGERS_SKIP_XRAY may also be set by hand to bypass tracing entirely.
  ensure_xray_decision || return 1

  export_cdk_env

  local stacks=(
    "$(_sn S3)"
    "$(_sn IAM)"
    "$(_sn InferenceProfiles)"
  )
  if [ "${COGNITO_ENABLED}" = "1" ]; then
    stacks+=("$(_sn Cognito)")
  fi
  if [ "${OPERATION_LEDGER_ENABLED}" = "1" ]; then
    stacks+=("$(_sn DynamoDB)")
  fi
  if [ "${ECR_ENABLED}" = "1" ]; then
    stacks+=("$(_sn ECR)")
  fi
  if [ "${MEMORY_ENABLED}" = "1" ]; then
    stacks+=("$(_sn Memory)")
  fi
  if [ "${UI_ENABLED}" = "1" ]; then
    stacks+=("$(_sn Vpc)")
  fi
  if [ "${BADGERS_SKIP_XRAY}" != "1" ]; then
    stacks+=("$(_sn XRay)")
  fi

  log_info "Deploying: ${stacks[*]}"
  cdk_deploy "${stacks[@]}" || {
    log_error "Infrastructure deploy failed — not marking this step complete."
    return 1
  }

  mark_complete "infra_complete"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Upload Prompts, Manifests & Schemas
# ══════════════════════════════════════════════════════════════════════════
step_upload() {
  log_step "Step 3: Upload Prompts, Manifests & Schemas"

  if check_completed "s3_files_uploaded"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_info "Already uploaded. Re-uploading to sync any changes..."
  fi

  local config_bucket
  config_bucket="$(stack_output "$(_sn S3)" ConfigBucketName)"
  if [ -z "${config_bucket}" ] || [ "${config_bucket}" = "None" ]; then
    log_error "$(_sn S3) not found. Run Step 2 first."
    return 1
  fi

  # Every product stages and uploads the resolved profile as one release. A
  # partial root-level sync would be invisible to consumers pinned to a release
  # prefix and could leave a schema, manifest, and prompt set inconsistent.
  local choice
  if [ "${DEPLOY_RESUME_MODE:-}" = "1" ]; then
    choice="7"
  else
    echo ""
    log_info "Uploading a complete, immutable ${BADGERS_PROFILE} configuration release."
    choice="7"
  fi
  CURRENT_STEP_SUBMENU="${choice}"

  # Every arm must propagate failure. Arms 1-6 previously did not: a failed `aws s3 sync`
  # inside _sync_config_category fell through to log_success and mark_complete, so the
  # state file said "uploaded" while the bucket held the old files. Option 6 is how the
  # model registry reaches S3 -- a silent miss there leaves GET /api/models serving 503.
  case "$choice" in
    1) _sync_config_category "prompts" "$config_bucket" || return 1 ;;
    2) _sync_config_category "core_system_prompts" "$config_bucket" || return 1 ;;
    3) _sync_config_category "manifests" "$config_bucket" || return 1 ;;
    4) _sync_config_category "schemas" "$config_bucket" || return 1 ;;
    5) _sync_config_category "agent_config" "$config_bucket" || return 1 ;;
    6) _sync_config_category "config" "$config_bucket" || return 1 ;;
    7)
      _sync_profile_config "${config_bucket}" || return 1
      ;;
    *) log_error "Invalid choice: ${choice}"; return 1 ;;
  esac

  log_success "Config files uploaded to ${config_bucket}"
  mark_complete "s3_files_uploaded"
}

# Stage a profile to a fresh temporary directory before S3 upload. This gives
# Headless an exact configuration footprint today and is the source directory a
# later release-artifact command can hash, sign, and publish.
_sync_profile_config() {
  local bucket="$1" stage_dir rc previous_release
  stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/badgers-profile-config.XXXXXX")" || {
    log_error "Could not create a temporary configuration staging directory."
    return 1
  }

  local release_id
  release_id="$(python3 "${DEPLOYMENT_DIR}/release_artifact.py" "${BADGERS_PROFILE}" \
      --selection-override "${SPECIALIST_PACK_ARGS[@]}" "${SPECIALIST_DIRECT_ARGS[@]}" \
      --routing-state "$(state_file)" \
      --destination "${stage_dir}")" || {
    rm -rf -- "${stage_dir}"
    log_error "Could not build a deterministic ${BADGERS_PROFILE} configuration release."
    return 1
  }
  if [ -z "${release_id}" ]; then
    rm -rf -- "${stage_dir}"
    log_error "Configuration release builder returned no release ID."
    return 1
  fi

  log_info "Syncing configuration release ${release_id} → s3://${bucket}/releases/${release_id}/..."
  rc=0
  aws s3 sync "${stage_dir}/" "s3://${bucket}/releases/${release_id}/" \
    --exclude "*.DS_Store" --exclude "*.pyc" --exclude "__pycache__/*" \
    --region "${AWS_REGION}" --quiet || rc=$?
  rm -rf -- "${stage_dir}"

  if [ "${rc}" -ne 0 ]; then
    log_error "Profile configuration upload failed."
    return 1
  fi
  previous_release="$(get_state "config_release_id")"
  set_state "config_release_id" "\"${release_id}\""
  log_success "Pinned configuration release: ${release_id}"
  if [ -n "${previous_release}" ] && [ "${previous_release}" != "${release_id}" ]; then
    invalidate_state "specialists_complete"
    invalidate_state "gateway_complete"
    invalidate_state "runtime_deployed"
    log_info "Configuration changed; specialist, Gateway, and runtime stages must redeploy."
  fi
}

# Sync a single s3_files/ subdirectory to the config bucket. Returns non-zero on a sync
# failure so the caller can refuse to mark the step complete. A missing directory is a
# skip, not a failure -- not every category exists in every checkout.
_sync_config_category() {
  local dir="$1" bucket="$2"
  local full_path="${DEPLOYMENT_DIR}/s3_files/${dir}"
  if [ ! -d "${full_path}" ]; then
    log_warn "  Directory ${dir}/ not found in s3_files/ — skipping"
    return 0
  fi
  log_info "  Syncing ${dir}/ → s3://${bucket}/${dir}/"
  aws s3 sync "${full_path}/" "s3://${bucket}/${dir}/" \
    --exclude "*.DS_Store" --exclude "*.pyc" --exclude "__pycache__/*" \
    --region "${AWS_REGION}" --quiet \
    || { log_error "  Sync of ${dir}/ failed."; return 1; }
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Specialist Lambdas
# ══════════════════════════════════════════════════════════════════════════
# Returns the first foundation-layer source file whose mtime is newer than
# layer.zip, or non-zero if none is. Keep this source set in sync with
# stacks/lambda_stack.py::_assert_layer_not_stale, which is the authoritative
# check at synth — this is the friendly pre-check so step 4 can offer to rebuild
# instead of failing halfway through with a CDK traceback.
_foundation_layer_newer_file() {
  local layer="$1" src="$2" lambdas_dir="$3" f
  if [ -f "${lambdas_dir}/requirements.txt" ] && [ "${lambdas_dir}/requirements.txt" -nt "$layer" ]; then
    echo "${lambdas_dir}/requirements.txt"; return 0
  fi
  f=$(find "${src}/foundation" -type f -name '*.py' -newer "$layer" -print 2>/dev/null | head -n1)
  [ -n "$f" ] && { echo "$f"; return 0; }
  f=$(find "${src}/foundation/core_system_prompts" -type f -newer "$layer" -print 2>/dev/null | head -n1)
  [ -n "$f" ] && { echo "$f"; return 0; }
  if [ -f "${src}/config/config.py" ] && [ "${src}/config/config.py" -nt "$layer" ]; then
    echo "${src}/config/config.py"; return 0
  fi
  return 1
}

# Prompts to rebuild the foundation layer when it is missing or older than its
# sources, so a stale layer never reaches the container build or the CDK deploy.
# Honours BADGERS_ALLOW_STALE_LAYER=1 (same override the synth guard respects)
# and BADGERS_ASSUME_YES=1 (auto-rebuild, for non-interactive runs).
_ensure_foundation_layer_fresh() {
  [ "${BADGERS_ALLOW_STALE_LAYER:-}" = "1" ] && return 0
  local layer="${DEPLOYMENT_DIR}/lambdas/layer.zip"
  local src="${DEPLOYMENT_DIR}/badgers-foundation"
  local newer
  if [ ! -f "$layer" ]; then
    log_warn "Foundation layer not built yet (lambdas/layer.zip is missing)."
  else
    newer=$(_foundation_layer_newer_file "$layer" "$src" "${DEPLOYMENT_DIR}/lambdas") || return 0
    log_warn "Foundation layer is stale — ${newer#"${DEPLOYMENT_DIR}"/} is newer than lambdas/layer.zip."
    log_warn "Deploying without rebuilding would ship the previous layer and its Lambdas would run stale code."
  fi
  if _confirm "  Rebuild the foundation layer now? (y/n): "; then
    log_info "Rebuilding foundation layer..."
    (cd "${DEPLOYMENT_DIR}/lambdas" && ./build_foundation_layer.sh) \
      || { log_error "Foundation layer build failed."; return 1; }
    log_success "Foundation layer rebuilt."
  else
    log_error "Foundation layer is stale. Rebuild it (menu option 1, or"
    log_error "cd deployment/lambdas && ./build_foundation_layer.sh) then re-run step 4."
    return 1
  fi
}

step_specialists() {
  log_step "Step 4: Specialist Lambdas"

  if check_completed "specialists_complete"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_warn "Already complete. Re-running will update if changes are detected."
    _confirm || return 0
  fi

  # The container Lambda build stages foundation/config from layer.zip, and the
  # Lambda stack ships that archive as-is, so both need a current layer. Catch a
  # stale one here and offer to rebuild rather than failing inside CDK synth.
  _ensure_foundation_layer_fresh || return 1

  if [ "${ECR_ENABLED}" = "1" ]; then
    ecr_login
  fi

  if [ "${#SPECIALIST_CONTAINER_NAMES[@]}" -gt 0 ]; then
    log_info "Building selected container Lambda images: ${SPECIALIST_CONTAINER_NAMES[*]}"
    # Lambda container images target linux/amd64. Verify Docker can execute that
    # platform before the build and push consume time or registry credentials.
    preflight_docker_cross_platform amd64 || return 1
    # build_container_lambdas.sh derives its target as badgers-<arg>, which has to match
    # the repository the ECR stack creates (badgers-{DEPLOYMENT_ID}-{suffix}).
    (cd "${DEPLOYMENT_DIR}/lambdas" && ./build_container_lambdas.sh \
      "$(resource_id)" "${SPECIALIST_CONTAINER_NAMES[@]}") \
      || { log_error "Container Lambda build failed."; return 1; }
  else
    log_info "Selected specialists use ZIP Lambdas only; skipping container image build."
  fi

  ensure_xray_decision || return 1
  export_cdk_env

  log_info "Deploying $(_sn Lambda)..."
  cdk_deploy "$(_sn Lambda)" || { log_error "Lambda deploy failed."; return 1; }

  mark_complete "specialists_complete"
  if [ "${RUNTIME_ENABLED}" != "1" ]; then
    record_applied_specialist_selection
  fi
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Gateway
# ══════════════════════════════════════════════════════════════════════════
step_gateway() {
  if [ "${GATEWAY_ENABLED}" != "1" ]; then
    log_error "Gateway deployment is not part of the ${BADGERS_PROFILE} profile."
    return 1
  fi
  log_step "Step 5: Gateway"

  if check_completed "gateway_complete"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_warn "Already complete. Re-running will update if changes are detected."
    _confirm || return 0
  fi

  ensure_xray_decision || return 1
  # This stack creates the Gateway's delivery sources and owns the deployment's
  # log-delivery resource policy, so it is the first place a conflicting source bites.
  preflight_log_delivery || return 1
  export_cdk_env

  log_info "Deploying $(_sn Gateway)..."
  cdk_deploy "$(_sn Gateway)" || { log_error "Gateway deploy failed."; return 1; }

  local gateway_url
  gateway_url="$(stack_output "$(_sn Gateway)" GatewayUrl)"
  if [ -n "${gateway_url}" ] && [ "${gateway_url}" != "None" ]; then
    set_state "gateway_url" "\"${gateway_url}\""
    log_success "Gateway URL: ${gateway_url}"
  else
    log_error "Could not read GatewayUrl from $(_sn Gateway) outputs"
    return 1
  fi

  # The Gateway caches each target's tool schema at registration time, so a schema-only
  # S3 change produces no CFN diff and the deploy above cannot re-read it. Reconcile
  # forces a re-read for any target whose S3 schema is newer than the live target.
  log_info "Reconciling gateway targets (re-reading any schemas newer than their targets)..."
  reconcile_all_gateway_targets

  mark_complete "gateway_complete"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Runtime — Build, Push & Deploy
# ══════════════════════════════════════════════════════════════════════════
step_runtime() {
  if [ "${RUNTIME_ENABLED}" != "1" ]; then
    log_error "Runtime deployment is not part of the ${BADGERS_PROFILE} profile."
    return 1
  fi
  log_step "Step 6: Runtime — Build, Push & Deploy"

  # Both sub-steps, not just the image push. An earlier guard tested only
  # runtime_image_pushed, so a run that pushed the image but never deployed the runtime
  # was skipped on resume; the fix removed the guard entirely, which made every resume
  # rebuild and push the image whether or not step 6 was done. Skip only when the whole
  # step is recorded complete; re-enter otherwise.
  if check_completed "runtime_image_pushed" && check_completed "runtime_deployed"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_warn "Already complete. Re-running rebuilds and pushes the runtime image, then updates the stack."
    _confirm || return 0
  fi

  if [ -z "$(get_state "gateway_url")" ]; then
    log_error "Gateway URL not found. Run Step 5 (Gateway) first."
    return 1
  fi

  ecr_login

  # Checked before the image build, which takes minutes — a conflicting delivery
  # source would otherwise fail the deploy after all that work.
  preflight_log_delivery || return 1

  # Ensure ARM64 cross-compilation works (QEMU on x86 hosts / WSL)
  preflight_docker_cross_platform arm64 || return 1

  # build_and_push_websocket.sh stages deployment/badgers-foundation/foundation into
  # the image so the agent can import foundation.job_state to open a job record.
  log_info "Building and pushing the AgentCore Runtime image (linux/arm64)..."
  (cd "${DEPLOYMENT_DIR}/runtime" && ./build_and_push_websocket.sh) \
    || { log_error "Runtime image build/push failed."; return 1; }

  log_info "Verifying image exists in ECR..."
  if ! aws ecr describe-images \
      --repository-name "${ECR_REPO_NAME}" \
      --image-ids imageTag="${RUNTIME_IMAGE_TAG}" \
      --region "${AWS_REGION}" --no-cli-pager > /dev/null 2>&1; then
    log_error "${ECR_REPO}:${RUNTIME_IMAGE_TAG} not found in ECR after push."
    return 1
  fi
  log_success "Image verified in ECR: ${RUNTIME_IMAGE_TAG}"
  mark_complete "runtime_image_pushed"

  ensure_xray_decision || return 1
  export_cdk_env

  log_info "Deploying $(_sn RuntimeWebSocket)..."
  cdk_deploy "$(_sn RuntimeWebSocket)" || { log_error "Runtime deploy failed."; return 1; }

  mark_complete "runtime_deployed"
  record_applied_specialist_selection
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: UI — Build & Push Image
# ══════════════════════════════════════════════════════════════════════════
step_ui_build() {
  log_step "Step 7: UI — Build & Push Image"

  # This guard is what gives _ui_deploy_failed's invalidate_state("ui_image_pushed") any
  # effect: step 8 clears the flag so that resume re-runs step 7. Without the guard, step
  # 7 re-ran on every resume regardless, and the flag was decorative.
  if check_completed "ui_image_pushed"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_warn "Already pushed. Re-running rebuilds the bundle and image and pushes again."
    _confirm || return 0
  fi

  ecr_login

  # Cognito values are compiled into the Vite bundle, so this must run after the
  # Cognito stack exists and before the image is built.
  log_info "Generating ui/.env from Cognito stack outputs..."
  DEPLOYMENT_ID="${DEPLOYMENT_ID}" \
  STACK_SUFFIX="${STACK_SUFFIX}" \
  AWS_REGION="${AWS_REGION}" \
    bash "${DEPLOYMENT_DIR}/scripts/generate_ui_env.sh"

  log_info "Building React app..."
  (cd "${REPO_ROOT}/ui" && npm ci --silent && npm run build) \
    || { log_error "UI bundle build failed."; return 1; }

  # Ensure AMD64 cross-compilation works (QEMU on ARM hosts / Apple Silicon)
  preflight_docker_cross_platform amd64 || return 1

  log_info "Building UI container image (linux/amd64)..."
  docker build \
    --platform linux/amd64 \
    --file "${REPO_ROOT}/ui/Dockerfile" \
    --tag "${ECR_REPO}:${UI_IMAGE_TAG}" \
    "${REPO_ROOT}/ui"

  log_info "Pushing UI image to ECR..."
  docker push "${ECR_REPO}:${UI_IMAGE_TAG}"

  log_success "UI image pushed: ${ECR_REPO}:${UI_IMAGE_TAG}"
  mark_complete "ui_image_pushed"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: UI — Deploy ECS
# ══════════════════════════════════════════════════════════════════════════
# An ECS rollout almost always fails because of something inside the image — the ALB
# health check path returning non-200, or the server exiting on boot. Redeploying the
# stack on its own will not pick up a code change, so every step 8 failure clears
# ui_image_pushed. Resume then re-runs step 7 (build & push) before retrying step 8,
# rather than redeploying the same broken image and failing identically.
#
# Step 6 needs no equivalent: it builds, pushes and deploys the runtime in one step,
# so any failure there re-enters the whole step already.
# ECS Express Mode derives the load balancer scheme from the subnets of the first
# service in the VPC, so this choice is effectively fixed for the life of the
# deployment — changing it later means tearing the ECS and VPC stacks down. Ask before
# anything is built rather than after the URL turns out not to answer.
#
# Resolved once per run and cached, so a full deploy or resume asks up front instead of
# eight steps in. An explicit UI_PUBLIC_ACCESS in the environment is respected as-is.
_prompt_ui_public_access() {
  [ "${_UI_ACCESS_RESOLVED:-0}" = "1" ] && return 0

  if [ -n "${UI_PUBLIC_ACCESS:-}" ]; then
    log_info "UI_PUBLIC_ACCESS=${UI_PUBLIC_ACCESS} (from the environment)"
    _UI_ACCESS_RESOLVED=1
    return 0
  fi

  if [ "${BADGERS_ASSUME_YES:-}" = "1" ]; then
    export UI_PUBLIC_ACCESS=false
    log_info "BADGERS_ASSUME_YES=1 — defaulting to an internal-only UI."
    _UI_ACCESS_RESOLVED=1
    return 0
  fi

  echo ""
  log_warn "UI network exposure — fixed for the life of this VPC."
  echo ""
  echo "    y) Public    internet-facing load balancer; tasks get public IPs."
  echo "                 The https://<id>.ecs.${AWS_REGION}.on.aws URL opens in a browser."
  echo "    n) Internal  internal load balancer; reachable only from inside the VPC."
  echo "                 The public URL will resolve but never answer."
  echo ""
  # Re-ask on bad input rather than returning non-zero: a typo here would otherwise
  # abort an entire full deployment.
  local ans
  while true; do
    read -rp "Make the UI publicly accessible? (y/n) [n]: " ans
    case "${ans:-n}" in
      y|Y) export UI_PUBLIC_ACCESS=true;  break ;;
      n|N) export UI_PUBLIC_ACCESS=false; break ;;
      *)   log_error "Answer y or n (got '${ans}')." ;;
    esac
  done
  _UI_ACCESS_RESOLVED=1
  log_info "UI_PUBLIC_ACCESS=${UI_PUBLIC_ACCESS}"
}

_ui_deploy_failed() {
  log_error "$1"
  invalidate_state "ui_image_pushed"
  log_warn "Cleared ui_image_pushed — resume will re-run step 7 (build & push) before step 8."
  log_warn "If the cause was outside the image, step 7 is a no-op rebuild and costs only time."
}

step_ui_deploy() {
  log_step "Step 8: UI — Deploy ECS"

  if check_completed "ecs_deployed"; then
    [ "${DEPLOY_RESUME_MODE:-}" = "1" ] && return 0
    log_warn "Already deployed. Re-running will update ECS."
    _confirm || return 0
  fi

  _prompt_ui_public_access || return 1

  ensure_xray_decision || return 1

  local service_name="badgers-ui-$(resource_id)"
  local service_arn="arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:service/default/${service_name}"

  # Submenu: full CDK deploy or force-new-deployment only
  local ecs_choice
  if [ "${DEPLOY_RESUME_MODE:-}" = "1" ]; then
    ecs_choice="1"
  else
    echo ""
    echo "  ECS deployment mode:"
    echo "    1) Full deploy      (CDK — updates task def, ALB config, scaling, env vars)"
    echo "    2) Force new deploy (restarts tasks with latest image — no infra changes)"
    echo ""
    read -rp "  Choice [1]: " ecs_choice
    ecs_choice="${ecs_choice:-1}"
  fi
  CURRENT_STEP_SUBMENU="${ecs_choice}"

  case "$ecs_choice" in
  1)
  export_cdk_env
  log_info "Deploying $(_sn ECS)..."
  if ! cdk_deploy "$(_sn ECS)"; then
    _ui_deploy_failed "ECS deploy failed."
    return 1
  fi

  # The stack references the image by a static tag, so pushing a new image to that same
  # tag leaves the template unchanged — cdk reports "no changes" and the service keeps
  # serving the old image. Updating the primary container explicitly forces a rollout
  # regardless, which is what makes a step 7 rebuild actually take effect.
  log_info "Forcing image rollout for ${service_name} (${ECR_REPO}:${UI_IMAGE_TAG})..."
  if ! aws ecs update-express-gateway-service \
      --service-arn "${service_arn}" \
      --primary-container "{\"image\":\"${ECR_REPO}:${UI_IMAGE_TAG}\",\"containerPort\":${UI_CONTAINER_PORT}}" \
      --region "${AWS_REGION}" --no-cli-pager > /dev/null; then
    log_warn "Could not force rollout — service may be newly created (continuing)."
  fi
  ;;
  2)
  # Force new deployment only (no CDK)
  log_info "Forcing new deployment on ${service_name}..."
  if ! aws ecs update-express-gateway-service \
      --service-arn "${service_arn}" \
      --primary-container "{\"image\":\"${ECR_REPO}:${UI_IMAGE_TAG}\",\"containerPort\":${UI_CONTAINER_PORT}}" \
      --region "${AWS_REGION}" --no-cli-pager > /dev/null 2>&1; then
    _ui_deploy_failed "Could not force rollout. Is the ECS service deployed?"
    return 1
  fi
  ;;
  *)
  log_error "Invalid choice: ${ecs_choice}"
  return 1
  ;;
  esac

  # ── Poll until live ─────────────────────────────────────────────────
  echo ""
  echo "  ┌────────────────────────────────────────────────────────────┐"
  echo "  │  🚀 ECS service is starting up — waiting for healthy...   │"
  echo "  └────────────────────────────────────────────────────────────┘"
  echo ""

  local state="" completed=0 elapsed=0
  local spinner=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
  local spin_idx=0

  for attempt in $(seq 1 120); do
    state="$(aws ecs describe-services \
      --cluster default --services "${service_name}" \
      --region "${AWS_REGION}" \
      --query "services[0].deployments[0].rolloutState" \
      --output text 2>/dev/null || echo "CREATING")"

    local running desired
    running="$(aws ecs describe-services \
      --cluster default --services "${service_name}" \
      --region "${AWS_REGION}" \
      --query "services[0].deployments[0].runningCount" \
      --output text 2>/dev/null || echo "0")"
    desired="$(aws ecs describe-services \
      --cluster default --services "${service_name}" \
      --region "${AWS_REGION}" \
      --query "services[0].deployments[0].desiredCount" \
      --output text 2>/dev/null || echo "1")"

    local status_msg=""
    case "${state}" in
      COMPLETED)    status_msg="✅ Service is live!" ;;
      IN_PROGRESS)  status_msg="🔄 Deploying containers... (${running}/${desired} running)" ;;
      FAILED)       status_msg="❌ Rollout failed" ;;
      *)            status_msg="⏳ Provisioning service..." ;;
    esac

    local s="${spinner[$spin_idx]}"
    spin_idx=$(( (spin_idx + 1) % ${#spinner[@]} ))
    elapsed=$((attempt * 5))
    printf "\r  %s  %s  [%ds elapsed]    " "${s}" "${status_msg}" "${elapsed}"

    if [ "${state}" = "COMPLETED" ]; then
      completed=1
      echo ""
      break
    fi

    if [ "${state}" = "FAILED" ]; then
      echo ""
      _ui_deploy_failed "ECS rollout failed for ${service_name}."
      log_error "Task-level cause: aws logs tail /ecs/badgers-ui-$(resource_id) --region ${AWS_REGION} --since 30m"
      return 1
    fi

    if [ "${attempt}" -eq 60 ]; then
      echo ""
      echo "  ⚠️  Taking longer than usual — trying for 5 more minutes..."
      echo ""
    fi

    sleep 5
  done

  if [ "${completed}" -ne 1 ]; then
    echo ""
    _ui_deploy_failed "Service did not reach healthy state within 10 minutes (last: ${state:-unknown})."
    log_error "Task-level cause: aws logs tail /ecs/badgers-ui-$(resource_id) --region ${AWS_REGION} --since 30m"
    return 1
  fi

  local endpoint
  endpoint="$(stack_output "$(_sn ECS)" ServiceEndpoint)" 2>/dev/null || true
  if [ -n "${endpoint}" ] && [ "${endpoint}" != "None" ]; then
    echo ""
    echo "  ┌────────────────────────────────────────────────────────────┐"
    echo "  │  🌐 UI is live at:                                         │"
    echo "  │  ${endpoint}"
    echo "  └────────────────────────────────────────────────────────────┘"
    echo ""
  else
    log_warn "Could not read ServiceEndpoint from $(_sn ECS)"
  fi

  mark_complete "ecs_deployed"
}

# ══════════════════════════════════════════════════════════════════════════
# Resume — run only the steps this deployment still needs
# ══════════════════════════════════════════════════════════════════════════
ALL_STEPS=(step_layers step_infra step_upload step_specialists step_gateway step_runtime)
if [ "${GATEWAY_ENABLED}" != "1" ]; then
  ALL_STEPS=(step_layers step_infra step_upload step_specialists)
fi
if [ "${UI_ENABLED}" = "1" ]; then
  ALL_STEPS+=(step_ui_build step_ui_deploy)
fi

# Run every step in order, skipping completed ones silently (resume mode).
run_remaining() {
  export DEPLOY_RESUME_MODE=1
  local s
  for s in "${ALL_STEPS[@]}"; do
    if ! "${s}"; then
      echo ""
      log_error "Resume stopped at ${s}."
      log_error "Fix the cause and resume again — completed steps stay recorded."
      unset DEPLOY_RESUME_MODE
      return 1
    fi
  done
  unset DEPLOY_RESUME_MODE
}

step_resume() {
  log_step "Resume — Run Remaining Steps"

  # Only ask if step 8 is actually still pending, so a resume that has nothing left to
  # deploy does not prompt about network exposure it will never apply.
  if [ "${UI_ENABLED}" = "1" ] && ! check_completed "ecs_deployed"; then
    _prompt_ui_public_access || return 1
  fi

  run_remaining || return 1

  echo ""
  log_success "Resume complete for ${DEPLOYMENT_ID}."
  show_status
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: Full Deploy (All Steps)
# ══════════════════════════════════════════════════════════════════════════
step_full() {
  log_step "Full Deployment (All Steps)"
  echo ""
  log_warn "This will run all deployment steps in sequence."
  _confirm || return 0

  # Asked up front rather than at step 8, so the run is unattended after this point.
  if [ "${UI_ENABLED}" = "1" ]; then
    _prompt_ui_public_access || return 1
  fi

  run_remaining || return 1

  echo ""
  log_success "${BADGERS_PROFILE^} deployment complete!"
  show_status
}

# ══════════════════════════════════════════════════════════════════════════
# Show Deployment Status
# ══════════════════════════════════════════════════════════════════════════
_menu_step_complete() {
  local key="$1"
  case "${key}" in
    runtime)
      check_completed "runtime_image_pushed" && check_completed "runtime_deployed"
      ;;
    *)
      check_completed "${key}"
      ;;
  esac
}

_deployment_progress() {
  local steps=(layers_built infra_complete s3_files_uploaded specialists_complete)
  [ "${GATEWAY_ENABLED}" = "1" ] && steps+=(gateway_complete)
  [ "${RUNTIME_ENABLED}" = "1" ] && steps+=(runtime)
  [ "${UI_ENABLED}" = "1" ] && steps+=(ui_image_pushed ecs_deployed)

  local completed=0 step total filled empty bar_done bar_todo
  total="${#steps[@]}"
  for step in "${steps[@]}"; do
    if _menu_step_complete "${step}"; then
      ((completed += 1))
    fi
  done

  filled=$((completed * 18 / total))
  empty=$((18 - filled))
  printf -v bar_done '%*s' "${filled}" ''
  printf -v bar_todo '%*s' "${empty}" ''
  bar_done="${bar_done// /█}"
  bar_todo="${bar_todo// /░}"
  printf '%b' "${GREEN}${bar_done}${NC}${BLUE}${bar_todo}${NC} ${BOLD}${completed}/${total}${NC} stages complete"
  return 0
}

_menu_step_line() {
  local option="$1" key="$2" label="$3" detail="$4" icon
  if _menu_step_complete "${key}"; then
    icon="${GREEN}●${NC}"
  else
    icon="${YELLOW}○${NC}"
  fi
  printf "  %b ${BOLD}%2s${NC}  %-29s ${BLUE}%s${NC}\n" \
    "${icon}" "${option})" "${label}" "${detail}"
}

_profile_summary() {
  case "${BADGERS_PROFILE}" in
    full)
      echo "Agent API  •  internal gateway  •  all specialists  •  browser UI"
      ;;
    headless)
      echo "Authenticated agent API  •  durable operations  •  curated specialist packs"
      ;;
    specialists)
      echo "Direct Lambda specialists  •  selected packs  •  no agent, gateway, or UI"
      ;;
  esac
}

show_specialist_selection() {
  python3 - "$(state_file)" <<'PY'
import json
import sys

try:
    state = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    print("  Specialist selection: not configured")
    raise SystemExit(0)
desired = state.get("desired_specialist_selection", {})
applied = state.get("applied_specialist_selection", {})
if not isinstance(desired, dict):
    print("  Specialist selection: not configured")
    raise SystemExit(0)
packs = ", ".join(desired.get("packs", [])) or "none"
direct = len(desired.get("direct_specialists", []))
members = len(desired.get("resolved_specialists", []))
containers = ", ".join(desired.get("container_specialists", [])) or "none"
sync = isinstance(applied, dict) and applied.get("fingerprint") == desired.get("fingerprint")
print(f"  Specialist selection: {members} resolved · packs: {packs} · individual: {direct}")
print(f"  Container images: {containers} · {'synchronized' if sync else 'pending reconciliation'}")
PY
}

show_routing_configuration() {
  python3 - "$(state_file)" <<'PY'
import json
import sys
try:
    state = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    state = {}
desired = state.get("desired_routing_configuration", {})
applied = state.get("applied_routing_configuration", {})
if not isinstance(desired, dict):
    print("  Routing: not configured")
    raise SystemExit(0)
templates = ", ".join(desired.get("templates", [])) or "direct invocation only"
sync = isinstance(applied, dict) and applied.get("fingerprint") == desired.get("fingerprint")
print(f"  Routing: {templates} · {'synchronized' if sync else 'pending reconciliation'}")
PY
}

configure_specialists() {
  load_specialist_selection 0 configure || return 1
  load_routing_configuration || true
  log_success "Saved desired specialist selection for deployment '${DEPLOYMENT_ID}'."
}

manage_specialist_packs() {
  python3 "${SCRIPT_DIR}/deployment/specialist_selection.py" manage-packs \
    --state-dir "${REPO_ROOT}/.deploy-state" || return 1
  # Re-resolve the current deployment after a pack edit. This catches source
  # controlled/manual JSON edits as well as edits made through the CLI.
  load_specialist_selection 0 ensure || return 1
  load_routing_configuration || true
}

configure_routing() {
  load_routing_configuration configure || return 1
  log_success "Saved desired workflow routing for deployment '${DEPLOYMENT_ID}'."
}

show_status() {
  echo ""
  echo -e "${CYAN}${BOLD}  DEPLOYMENT SNAPSHOT${NC}  $(_deployment_progress)"
  echo -e "${BLUE}  ─────────────────────────────────────────────────────────────────${NC}"
  echo ""

  _step_line() {
    local key=$1 label=$2 mark ts
    mark="$(check_completed "$key" && echo -e "${GREEN}●${NC}" || echo -e "${YELLOW}○${NC}")"
    ts="$(get_state "${key}_at")"
    if [ -n "${ts}" ]; then
      echo -e "  ${mark}  ${label}  ${BLUE}${ts}${NC}"
    else
      echo "  ${mark}  ${label}"
    fi
  }

  echo -e "  ${BOLD}BUILD & CONFIGURE${NC}"
  _step_line "layers_built"         "1. Lambda Layers"
  _step_line "infra_complete"       "2. Foundational Infrastructure"
  _step_line "s3_files_uploaded"    "3. Prompts, Manifests & Schemas"
  echo ""
  echo -e "  ${BOLD}DEPLOY SERVICES${NC}"
  _step_line "specialists_complete" "4. Specialist Lambdas"
  if [ "${GATEWAY_ENABLED}" = "1" ]; then
    _step_line "gateway_complete"     "5. Gateway"
  fi

  # Step 6 has two sub-steps.
  if [ "${RUNTIME_ENABLED}" = "1" ]; then
    local a b mark6 ts6
    a="$(check_completed "runtime_image_pushed" && echo "✓" || echo "○")"
    b="$(check_completed "runtime_deployed" && echo "✓" || echo "○")"
    mark6="○"; [ "$a" = "✓" ] && [ "$b" = "✓" ] && mark6="✓"
    [ "$mark6" = "✓" ] && mark6="${GREEN}●${NC}" || mark6="${YELLOW}○${NC}"
    ts6="$(get_state "runtime_deployed_at")"
    if [ -n "${ts6}" ]; then
      echo -e "  ${mark6}  6. Runtime — Build & Deploy  ${BLUE}${ts6}${NC}"
    else
      echo -e "  ${mark6}  6. Runtime — Build & Deploy"
    fi
  fi

  if [ "${UI_ENABLED}" = "1" ]; then
    _step_line "ui_image_pushed"      "7. UI — Build Image"
    _step_line "ecs_deployed"         "8. UI — Deploy ECS"
  elif [ "${BADGERS_PROFILE}" = "headless" ]; then
    echo -e "  ${BLUE}↳ Headless intentionally omits UI, VPC, session memory, and the UI WebSocket workflow.${NC}"
  else
    echo -e "  ${BLUE}↳ Specialists intentionally omit the agent runtime, gateway, UI, and operation ledger.${NC}"
  fi
  echo ""

  local gateway_url
  gateway_url="$(get_state "gateway_url")"
  [ -n "${gateway_url}" ] && echo "  Gateway URL: ${gateway_url}"
  local config_release
  config_release="$(get_state "config_release_id")"
  [ -n "${config_release}" ] && echo "  Config release: ${config_release}"
  show_specialist_selection
  show_routing_configuration
  echo ""
}

# ══════════════════════════════════════════════════════════════════════════
# Models — what the registry holds, and which steps a change to it requires
# ══════════════════════════════════════════════════════════════════════════
# The registry is the only place models are listed. Everything else -- profiles, IAM,
# the SSM map, the wizard dropdown, prices -- derives from it at synth or at request
# time. But "derives from it" means three different steps have to run for an edit to
# take full effect, and nothing else in this menu says which. This does.
show_models() {
  local registry="${DEPLOYMENT_DIR}/s3_files/config/model_registry.json"
  echo ""
  echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}  Model Registry${NC}"
  echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
  echo ""
  echo "  ${registry#"${REPO_ROOT}/"}"
  echo ""

  if [ ! -f "${registry}" ]; then
    log_error "Registry file not found."
    return 1
  fi

  python3 - "${registry}" <<'PY' || { log_error "Could not parse the registry."; return 1; }
import json, sys
models = json.load(open(sys.argv[1]))["models"]
rows = [(mid, s) for mid, s in models.items()]
w = max(len(m) for m, _ in rows)
print(f"  {'Model ID':<{w}}  {'Status':<9} {'Provider':<10} {'Thinking':<9} {'$/1M in':>8} {'$/1M out':>9}  Flags")
print(f"  {'-'*w}  {'-'*9} {'-'*10} {'-'*9} {'-'*8} {'-'*9}  -----")
for mid, s in rows:
    flags = []
    if s.get("thinking_default_on"):   flags.append("thinking-default-on")
    if s.get("prompt_caching"):        flags.append("caching")
    print(f"  {mid:<{w}}  {s.get('status','?'):<9} {s.get('provider','?'):<10} "
          f"{str(s.get('thinking')):<9} {s.get('price_in',0):>8.2f} {s.get('price_out',0):>9.2f}  "
          f"{', '.join(flags)}")
by = {}
for _, s in rows: by[s.get("status","?")] = by.get(s.get("status","?"), 0) + 1
print()
print("  " + ", ".join(f"{n} {k}" for k, n in sorted(by.items())))
PY

  echo ""
  echo -e "${BOLD}  To change the model set${NC}"
  echo ""
  echo "  Edit the file above. status is one of:"
  echo "    active    provisioned, granted, offered in the wizard"
  echo "    retiring  provisioned and granted so existing manifests keep working; not offered"
  echo "    disabled  nothing provisioned -- for a model this account declines to adopt"
  echo ""
  echo "  Then run these steps, in this order. Nothing else re-reads the registry:"
  echo ""
  echo "    2) Foundational Infra   profiles, IAM grants and the SSM model→profile map are"
  echo "                            regenerated at synth. The model access and invocation"
  echo "                            preflights re-run here."
  echo "    3) Upload Config Files  builds and uploads a complete immutable configuration"
  echo "                            release, including the registry. Then redeploy the"
  echo "                            affected consumers so their pinned release advances."
  echo "    6) Runtime              the agent role's Bedrock grants are generated into the"
  echo "                            Runtime stack, so it must redeploy too. The image is"
  echo "                            unchanged; step 6 will rebuild it anyway."
  echo ""
  echo "  Manifests under s3_files/manifests/ and custom_specialists/manifests/ name models"
  echo "  by ID. A manifest naming a model that is not active fails with AccessDenied on"
  echo "  its first call and does not fall back. grep them before deploying."
  echo ""
}

reset_state() {
  log_warn "This resets deployment state and marks all steps incomplete."
  log_warn "It does NOT delete any AWS resources. The stack suffix is preserved."
  _confirm || return 0

  local keep="${STACK_SUFFIX}"
  rm -f "$(state_file)"
  init_state
  set_state "stack_suffix" "\"${keep}\""
  log_success "Deployment state reset (suffix ${keep} kept)."
}

# ══════════════════════════════════════════════════════════════════════════
# Interactive Menu
# ══════════════════════════════════════════════════════════════════════════
show_menu() {
  clear
  local profile_label profile_color
  case "${BADGERS_PROFILE}" in
    full)        profile_label="FULL EXPERIENCE"; profile_color="${CYAN}" ;;
    headless)    profile_label="HEADLESS AGENT";  profile_color="${GREEN}" ;;
    specialists) profile_label="SPECIALIST PACKS"; profile_color="${YELLOW}" ;;
  esac

  echo -e "${profile_color}${BOLD}"
  cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                         BADGERS DEPLOY                           ║
╚══════════════════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"

  echo -e "  ${profile_color}${BOLD}${profile_label}${NC}  ${BLUE}$(_profile_summary)${NC}"
  echo -e "  ${BLUE}──────────────────────────────────────────────────────────────────${NC}"
  echo -e "  ${BOLD}Deployment${NC} ${DEPLOYMENT_ID}  ${BOLD}Region${NC} ${AWS_REGION}  ${BOLD}Account${NC} ${ACCOUNT_ID}"
  echo -e "  ${BOLD}Stack suffix${NC} ${STACK_SUFFIX}  ${BOLD}Resource ID${NC} $(resource_id)"
  echo -e "  ${BLUE}${STACK_PREFIX}-{Name}-${STACK_SUFFIX}${NC}"
  echo ""

  show_status

  echo -e "${BLUE}  ═════════════════════════════════════════════════════════════════${NC}"
  echo ""
  echo -e "  ${CYAN}${BOLD}BUILD & CONFIGURE${NC}"
  _menu_step_line 1 layers_built "Build Lambda Layers" "required artifacts"
  if [ "${UI_ENABLED}" = "1" ]; then
    _menu_step_line 2 infra_complete "Foundational Infra" "S3, identity, data, network"
  elif [ "${BADGERS_PROFILE}" = "headless" ]; then
    _menu_step_line 2 infra_complete "Foundational Infra" "S3, identity, data, registry"
  else
    _menu_step_line 2 infra_complete "Foundational Infra" "S3, IAM, model profiles"
  fi
  _menu_step_line 3 s3_files_uploaded "Publish Config Release" "immutable, versioned"
  echo ""
  echo -e "  ${CYAN}${BOLD}DEPLOY SERVICES${NC}"
  _menu_step_line 4 specialists_complete "Specialist Lambdas" "functions and containers"
  if [ "${GATEWAY_ENABLED}" = "1" ]; then
    _menu_step_line 5 gateway_complete "Internal Gateway" "private API integration"
  fi
  if [ "${RUNTIME_ENABLED}" = "1" ]; then
    _menu_step_line 6 runtime "Agent Runtime" "build and deploy"
  fi
  if [ "${UI_ENABLED}" = "1" ]; then
    _menu_step_line 7 ui_image_pushed "Browser UI Image" "build and push"
    _menu_step_line 8 ecs_deployed "Browser UI Service" "ECS deployment → submenu"
  fi
  echo ""
  echo -e "  ${GREEN}${BOLD}LAUNCH${NC}"
  echo -e "  ${GREEN}▶${NC} ${BOLD} 9)${NC} Run complete deployment"
  echo -e "  ${GREEN}▶${NC} ${BOLD} r)${NC} Resume remaining stages only"
  echo ""
  echo -e "  ${BOLD}SELECTION${NC}   s) Configure specialists   w) Configure workflow routing   k) Manage reusable packs"
  echo -e "  ${BOLD}TOOLS${NC}       c) Open configuration planner (local UI)   p) Preflight report   10) Status"
  echo -e "               11) Reset local state   m) Model registry   0) Exit"
  echo ""
  echo -e "${BLUE}  ═════════════════════════════════════════════════════════════════${NC}"
  echo ""
}

dispatch() {
  local preflight_scope
  case "$1" in
    p|P|preflight)
      run_preflight all
      return $?
      ;;
  esac

  preflight_scope="$(preflight_scope_for_option "$1")"
  if [ -n "${preflight_scope}" ]; then
    run_preflight "${preflight_scope}" || return 1
  fi

  case "$1" in
    1)  step_layers ;;
    2)  step_infra ;;
    3)  step_upload ;;
    4)  step_specialists ;;
    5)  step_gateway ;;
    6)  step_runtime ;;
    7)
      if [ "${UI_ENABLED}" = "1" ]; then
        step_ui_build
      else
        log_error "UI build is available only in the full profile."
        return 1
      fi
      ;;
    8)
      if [ "${UI_ENABLED}" = "1" ]; then
        step_ui_deploy
      else
        log_error "UI deployment is available only in the full profile."
        return 1
      fi
      ;;
    9)  step_full ;;
    10) show_status ;;
    11) reset_state ;;
    r|R|resume) step_resume ;;
    m|M) show_models ;;
    c|C|configure) launch_configuration_ui ;;
    s|S) configure_specialists ;;
    w|W) configure_routing ;;
    k|K) manage_specialist_packs ;;
    0)  log_info "Exiting..."; exit 0 ;;
    *)  return 1 ;;
  esac
}

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
_valid_option() {
  case "$1" in
    0|1|2|3|4|5|6|7|8|9|10|11|p|P|preflight|r|R|resume|m|M|c|C|configure|s|S|w|W|k|K) return 0 ;;
    *) return 1 ;;
  esac
}

if [ $# -gt 0 ]; then
  _valid_option "$1" || { log_error "Invalid option: $1"; exit 1; }
  # Deliberately not `dispatch "$1" || ...`: a command on the left of || runs with
  # errexit suppressed for its entire call tree, which previously let a failed
  # cdk_deploy fall through to mark_complete.
  CURRENT_STEP_LABEL=""
  CURRENT_STEP_SUBMENU=""
  rc=0
  dispatch "$1" || rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo ""
    # Mirror the interactive loop: name the step that actually failed (set by log_step,
    # including the nested steps run by Full/Resume) rather than just exiting silently.
    _where="${CURRENT_STEP_LABEL:-Step (option $1)}"
    if [ -n "${CURRENT_STEP_SUBMENU}" ]; then
      _where="${_where} — submenu ${CURRENT_STEP_SUBMENU}"
    fi
    log_error "${_where} failed (exit ${rc}). Nothing was marked complete."
  fi
  exit "${rc}"
fi

while true; do
  show_menu
  read -rp "Select option: " choice
  echo ""
  if _valid_option "${choice}"; then
    # Capture dispatch's status directly. Reading $? inside an else branch reports the
    # status of the last command run in that branch, not of the failed step.
    rc=0
    CURRENT_STEP_LABEL=""
    CURRENT_STEP_SUBMENU=""
    dispatch "${choice}" || rc=$?
    if [ "${rc}" -ne 0 ]; then
      echo ""
      # CURRENT_STEP_LABEL is set by log_step at the top of each step (including the
      # nested steps run by Full/Resume), so this names the step that actually failed
      # rather than the menu option chosen. Submenu selection is appended when present.
      _where="${CURRENT_STEP_LABEL:-Step (option ${choice})}"
      if [ -n "${CURRENT_STEP_SUBMENU}" ]; then
        _where="${_where} — submenu ${CURRENT_STEP_SUBMENU}"
      fi
      log_error "${_where} failed (exit ${rc}). Nothing was marked complete."
    fi
  else
    log_error "Invalid option: ${choice}"
  fi
  echo ""
  read -rp "Press Enter to continue..." _
done
