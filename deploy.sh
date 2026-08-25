#!/bin/bash
set -euo pipefail

# ── BADGERS — Interactive Deployment Menu ──────────────────────────────────
#
# Usage:
#   ./deploy.sh             # Pick a deployment, then the interactive menu
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

trap 'echo ""; log_error "deploy.sh failed at line ${LINENO}: ${BASH_COMMAND}"; exit 1' ERR

# ── Defaults ──
IMAGE_TAG="${IMAGE_TAG:-latest}"
RUNTIME_IMAGE_TAG="${RUNTIME_IMAGE_TAG:-websocket}"
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
ensure_account
ensure_suffix

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
# One repository per deployment, named badgers-{DEPLOYMENT_ID}-{suffix} by the ECR
# stack. Both the runtime and UI images live in it under different tags.
ECR_REPO_NAME="badgers-$(resource_id)"
ECR_REPO="${ECR_BASE}/${ECR_REPO_NAME}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Build Lambda Layers
# ══════════════════════════════════════════════════════════════════════════
step_layers() {
  log_step "Step 1: Build Lambda Layers"

  if check_completed "layers_built"; then
    log_warn "Already built. Re-running rebuilds the layer archives."
    _confirm || return 0
  fi

  # The foundation layer must be built before the container Lambdas: their build
  # script copies foundation/ out of layer/python/, so a stale layer ships stale
  # code into the images.
  log_info "Building foundation layer..."
  (cd "${DEPLOYMENT_DIR}/lambdas" && ./build_foundation_layer.sh) \
    || { log_error "Foundation layer build failed."; return 1; }

  log_info "Building PDF processing layer..."
  (cd "${DEPLOYMENT_DIR}/lambdas" && ./build_pdf_processing_layer.sh) \
    || { log_error "PDF processing layer build failed."; return 1; }

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
    log_warn "Already complete. Re-running will update stacks if changes are detected."
    _confirm || return 0
  fi

  ensure_cdk_deps

  # Decide about X-Ray before deploying anything. Enabling Transaction Search needs a
  # CloudWatch Logs resource policy from a hard quota of 10 per region, and finding that
  # out from CloudFormation means failing halfway through with other stacks already up.
  # BADGERS_SKIP_XRAY may also be set by hand to bypass tracing entirely.
  ensure_xray_decision || return 1

  export_cdk_env

  local stacks=(
    "$(_sn S3)"
    "$(_sn Cognito)"
    "$(_sn DynamoDB)"
    "$(_sn IAM)"
    "$(_sn ECR)"
    "$(_sn InferenceProfiles)"
    "$(_sn Memory)"
    "$(_sn Vpc)"
  )
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
    log_info "Already uploaded. Re-uploading to sync any changes..."
  fi

  local config_bucket
  config_bucket="$(stack_output "$(_sn S3)" ConfigBucketName)"
  if [ -z "${config_bucket}" ] || [ "${config_bucket}" = "None" ]; then
    log_error "$(_sn S3) not found. Run Step 2 first."
    exit 1
  fi

  log_info "Syncing s3_files/ to s3://${config_bucket}/..."
  aws s3 sync "${DEPLOYMENT_DIR}/s3_files/" "s3://${config_bucket}/" \
    --exclude "*.DS_Store" --exclude "*.pyc" --exclude "__pycache__/*" \
    --region "${AWS_REGION}" --quiet \
    || { log_error "Config upload failed."; return 1; }

  log_success "Config files uploaded to ${config_bucket}"
  mark_complete "s3_files_uploaded"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Specialist Lambdas
# ══════════════════════════════════════════════════════════════════════════
step_specialists() {
  log_step "Step 4: Specialist Lambdas"

  if check_completed "specialists_complete"; then
    log_warn "Already complete. Re-running will update if changes are detected."
    _confirm || return 0
  fi

  ecr_login

  # Container Lambdas are built before the stack that references their images.
  log_info "Building container Lambda images..."
  # build_container_lambdas.sh derives its target as badgers-<arg>, which has to match
  # the repository the ECR stack creates (badgers-{DEPLOYMENT_ID}-{suffix}).
  (cd "${DEPLOYMENT_DIR}/lambdas" && ./build_container_lambdas.sh "$(resource_id)") \
    || { log_error "Container Lambda build failed."; return 1; }

  ensure_cdk_deps
  ensure_xray_decision || return 1
  export_cdk_env

  log_info "Deploying $(_sn Lambda)..."
  cdk_deploy "$(_sn Lambda)" || { log_error "Lambda deploy failed."; return 1; }

  mark_complete "specialists_complete"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Gateway
# ══════════════════════════════════════════════════════════════════════════
step_gateway() {
  log_step "Step 5: Gateway"

  if check_completed "gateway_complete"; then
    log_warn "Already complete. Re-running will update if changes are detected."
    _confirm || return 0
  fi

  ensure_cdk_deps
  ensure_xray_decision || return 1
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
    exit 1
  fi

  mark_complete "gateway_complete"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Runtime — Build, Push & Deploy
# ══════════════════════════════════════════════════════════════════════════
step_runtime() {
  log_step "Step 6: Runtime — Build, Push & Deploy"

  if [ -z "$(get_state "gateway_url")" ]; then
    log_error "Gateway URL not found. Run Step 5 (Gateway) first."
    exit 1
  fi

  ecr_login

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
    exit 1
  fi
  log_success "Image verified in ECR: ${RUNTIME_IMAGE_TAG}"
  mark_complete "runtime_image_pushed"

  ensure_cdk_deps
  ensure_xray_decision || return 1
  export_cdk_env

  log_info "Deploying $(_sn RuntimeWebSocket)..."
  cdk_deploy "$(_sn RuntimeWebSocket)" || { log_error "Runtime deploy failed."; return 1; }

  mark_complete "runtime_deployed"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: UI — Build & Push Image
# ══════════════════════════════════════════════════════════════════════════
step_ui_build() {
  log_step "Step 7: UI — Build & Push Image"

  ecr_login

  # Cognito values are compiled into the Vite bundle, so this must run after the
  # Cognito stack exists and before the image is built.
  log_info "Generating ui/.env from Cognito stack outputs..."
  DEPLOYMENT_ID="${DEPLOYMENT_ID}" \
  STACK_SUFFIX="${STACK_SUFFIX}" \
  AWS_REGION="${AWS_REGION}" \
    bash "${DEPLOYMENT_DIR}/scripts/generate_ui_env.sh"

  log_info "Building React app..."
  (cd "${REPO_ROOT}/ui" && pnpm install --frozen-lockfile --prod=false --silent && pnpm run build) \
    || { log_error "UI bundle build failed."; return 1; }

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
    export UI_PUBLIC_ACCESS=true
    log_warn "BADGERS_ASSUME_YES=1 — defaulting to a public, internet-facing UI."
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
    read -rp "Make the UI publicly accessible? (y/n) [y]: " ans
    case "${ans:-y}" in
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

  _prompt_ui_public_access || return 1

  ensure_cdk_deps
  ensure_xray_decision || return 1
  export_cdk_env

  log_info "Deploying $(_sn ECS)..."
  if ! cdk_deploy "$(_sn ECS)"; then
    _ui_deploy_failed "ECS deploy failed."
    return 1
  fi

  local service_name="badgers-ui-$(resource_id)"

  # The stack references the image by a static tag, so pushing a new image to that same
  # tag leaves the template unchanged — cdk reports "no changes" and the service keeps
  # serving the old image. Updating the primary container explicitly forces a rollout
  # regardless, which is what makes a step 7 rebuild actually take effect.
  log_info "Forcing image rollout for ${service_name} (${ECR_REPO}:${UI_IMAGE_TAG})..."
  if ! aws ecs update-express-gateway-service \
      --service-arn "arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:service/default/${service_name}" \
      --primary-container "{\"image\":\"${ECR_REPO}:${UI_IMAGE_TAG}\",\"containerPort\":${UI_CONTAINER_PORT}}" \
      --region "${AWS_REGION}" --no-cli-pager > /dev/null; then
    _ui_deploy_failed "Forcing the image rollout failed for ${service_name}."
    return 1
  fi
  log_info "Polling rollout state for ${service_name}..."
  local state completed=0
  for _ in $(seq 1 60); do
    state="$(aws ecs describe-services \
      --cluster default --services "${service_name}" \
      --region "${AWS_REGION}" \
      --query "services[0].deployments[0].rolloutState" \
      --output text 2>/dev/null || echo "UNKNOWN")"
    echo "  $(_ts) — ${state}"
    if [ "${state}" = "COMPLETED" ]; then
      completed=1
      break
    fi
    if [ "${state}" = "FAILED" ]; then
      _ui_deploy_failed "ECS rollout failed for ${service_name}."
      log_error "Task-level cause: aws logs tail /ecs/badgers-ui-$(resource_id) --region ${AWS_REGION} --since 30m"
      return 1
    fi
    sleep 5
  done

  # Falling out of the loop without COMPLETED is a timeout, not a success.
  if [ "${completed}" -ne 1 ]; then
    _ui_deploy_failed "ECS rollout did not reach COMPLETED within 5 minutes (last state: ${state:-unknown})."
    log_error "Task-level cause: aws logs tail /ecs/badgers-ui-$(resource_id) --region ${AWS_REGION} --since 30m"
    return 1
  fi

  mark_complete "ecs_deployed"

  local endpoint
  endpoint="$(stack_output "$(_sn ECS)" ServiceEndpoint)"
  if [ -n "${endpoint}" ] && [ "${endpoint}" != "None" ]; then
    log_success "UI available at https://${endpoint}"
  else
    log_warn "Could not read ServiceEndpoint from $(_sn ECS)"
  fi
}

# ══════════════════════════════════════════════════════════════════════════
# Resume — run only the steps this deployment still needs
# ══════════════════════════════════════════════════════════════════════════
step_resume() {
  log_step "Resume — Run Remaining Steps"

  # Only ask if step 8 is actually still pending, so a resume that has nothing left to
  # deploy does not prompt about network exposure it will never apply.
  if ! check_completed "ecs_deployed"; then
    _prompt_ui_public_access || return 1
  fi

  # A step counts as done only when every state key it writes is set, so a partially
  # finished step (runtime image pushed but never deployed) is re-entered rather than
  # skipped. Completed steps are skipped before being called, so their "already
  # complete, re-run?" prompts never fire — that is the difference from option 9.
  local -a order=(
    "step_layers:layers_built"
    "step_infra:infra_complete"
    "step_upload:s3_files_uploaded"
    "step_specialists:specialists_complete"
    "step_gateway:gateway_complete"
    "step_runtime:runtime_image_pushed,runtime_deployed"
    "step_ui_build:ui_image_pushed"
    "step_ui_deploy:ecs_deployed"
  )

  local entry fn keys k pending ran=0
  for entry in "${order[@]}"; do
    fn="${entry%%:*}"
    keys="${entry#*:}"
    pending=0
    for k in ${keys//,/ }; do
      check_completed "${k}" || { pending=1; break; }
    done
    if [ "${pending}" -eq 0 ]; then
      log_info "skip  ${fn} — already complete"
      continue
    fi
    if ! "${fn}"; then
      echo ""
      log_error "Resume stopped at ${fn}."
      log_error "Fix the cause and resume again — completed steps stay recorded."
      return 1
    fi
    ran=$((ran + 1))
  done

  echo ""
  if [ "${ran}" -eq 0 ]; then
    log_success "Nothing to run — every step is already complete for ${DEPLOYMENT_ID}."
  else
    log_success "Resume complete (${ran} step(s) run)."
  fi
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
  _prompt_ui_public_access || return 1

  # Stop at the first failure. Continuing would deploy on top of a broken
  # foundation and report success at the end.
  local s
  for s in step_layers step_infra step_upload step_specialists \
           step_gateway step_runtime step_ui_build step_ui_deploy; do
    if ! "${s}"; then
      echo ""
      log_error "Full deployment stopped at ${s}."
      log_error "Fix the cause and re-run — completed steps are recorded and will be skipped."
      return 1
    fi
  done

  echo ""
  log_success "Full deployment complete!"
  show_status
}

# ══════════════════════════════════════════════════════════════════════════
# Show Deployment Status
# ══════════════════════════════════════════════════════════════════════════
show_status() {
  echo ""
  echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}  Deployment Status: ${DEPLOYMENT_ID}  (suffix: ${STACK_SUFFIX})${NC}"
  echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
  echo ""

  _step_line() {
    local key=$1 label=$2 mark ts
    mark="$(check_completed "$key" && echo "✓" || echo "○")"
    ts="$(get_state "${key}_at")"
    if [ -n "${ts}" ]; then
      echo -e "  ${mark}  ${label}  ${YELLOW}${ts}${NC}"
    else
      echo "  ${mark}  ${label}"
    fi
  }

  _step_line "layers_built"         "1. Lambda Layers"
  _step_line "infra_complete"       "2. Foundational Infrastructure"
  _step_line "s3_files_uploaded"    "3. Prompts, Manifests & Schemas"
  _step_line "specialists_complete" "4. Specialist Lambdas"
  _step_line "gateway_complete"     "5. Gateway"

  # Step 6 has two sub-steps.
  local a b mark6 ts6
  a="$(check_completed "runtime_image_pushed" && echo "✓" || echo "○")"
  b="$(check_completed "runtime_deployed" && echo "✓" || echo "○")"
  mark6="○"; [ "$a" = "✓" ] && [ "$b" = "✓" ] && mark6="✓"
  ts6="$(get_state "runtime_deployed_at")"
  if [ -n "${ts6}" ]; then
    echo -e "  ${mark6}  6. Runtime — Build & Deploy  ${YELLOW}${ts6}${NC}"
  else
    echo "  ${mark6}  6. Runtime — Build & Deploy"
  fi

  _step_line "ui_image_pushed"      "7. UI — Build Image"
  _step_line "ecs_deployed"         "8. UI — Deploy ECS"
  echo ""

  local gateway_url
  gateway_url="$(get_state "gateway_url")"
  [ -n "${gateway_url}" ] && echo "  Gateway URL: ${gateway_url}"
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
  echo -e "${CYAN}${BOLD}"
  cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║              BADGERS — Deployment Menu                     ║
╚════════════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"

  echo -e "  ${BOLD}Deployment ID:${NC} ${DEPLOYMENT_ID}"
  echo -e "  ${BOLD}Stack Suffix:${NC}  ${STACK_SUFFIX}"
  echo -e "  ${BOLD}Stack Names:${NC}   ${STACK_PREFIX}-{Name}-${STACK_SUFFIX}"
  echo -e "  ${BOLD}Resource ID:${NC}   $(resource_id)"
  echo -e "  ${BOLD}AWS Region:${NC}    ${AWS_REGION}"
  echo -e "  ${BOLD}AWS Account:${NC}   ${ACCOUNT_ID}"
  echo ""

  show_status

  echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
  echo ""
  echo -e "  ${BOLD} 1${NC}) Build Lambda Layers"
  echo -e "  ${BOLD} 2${NC}) Foundational Infra (S3, Cognito, DynamoDB, IAM, ECR, Profiles, XRay, Memory, VPC)"
  echo -e "  ${BOLD} 3${NC}) Upload Prompts, Manifests & Schemas"
  echo -e "  ${BOLD} 4${NC}) Specialist Lambdas (incl. container images)"
  echo -e "  ${BOLD} 5${NC}) Gateway"
  echo -e "  ${BOLD} 6${NC}) Runtime — Build & Deploy"
  echo -e "  ${BOLD} 7${NC}) UI — Build & Push Image"
  echo -e "  ${BOLD} 8${NC}) UI — Deploy ECS"
  echo ""
  echo -e "  ${BOLD} 9${NC}) ${GREEN}Full Deployment (Run All Steps)${NC}"
  echo -e "  ${BOLD}12${NC}) ${GREEN}Resume (Run Remaining Steps Only)${NC}"
  echo ""
  echo -e "  ${BOLD}10${NC}) Show Deployment Status"
  echo -e "  ${BOLD}11${NC}) Reset Deployment State (start fresh)"
  echo -e "  ${BOLD} 0${NC}) Exit"
  echo ""
  echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
  echo ""
}

dispatch() {
  case "$1" in
    1)  step_layers ;;
    2)  step_infra ;;
    3)  step_upload ;;
    4)  step_specialists ;;
    5)  step_gateway ;;
    6)  step_runtime ;;
    7)  step_ui_build ;;
    8)  step_ui_deploy ;;
    9)  step_full ;;
    10) show_status ;;
    11) reset_state ;;
    12|resume) step_resume ;;
    0)  log_info "Exiting..."; exit 0 ;;
    *)  return 1 ;;
  esac
}

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
_valid_option() {
  case "$1" in
    0|1|2|3|4|5|6|7|8|9|10|11|12|resume) return 0 ;;
    *) return 1 ;;
  esac
}

if [ $# -gt 0 ]; then
  _valid_option "$1" || { log_error "Invalid option: $1"; exit 1; }
  # Deliberately not `dispatch "$1" || ...`: a command on the left of || runs with
  # errexit suppressed for its entire call tree, which previously let a failed
  # cdk_deploy fall through to mark_complete.
  dispatch "$1"
  exit $?
fi

while true; do
  show_menu
  read -rp "Select option (0-12): " choice
  echo ""
  if _valid_option "${choice}"; then
    # Capture dispatch's status directly. Reading $? inside an else branch reports the
    # status of the last command run in that branch, not of the failed step.
    rc=0
    dispatch "${choice}" || rc=$?
    if [ "${rc}" -ne 0 ]; then
      echo ""
      log_error "Step failed (exit ${rc}). Nothing was marked complete."
    fi
  else
    log_error "Invalid option: ${choice}"
  fi
  echo ""
  read -rp "Press Enter to continue..." _
done
