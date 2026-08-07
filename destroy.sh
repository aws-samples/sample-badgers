#!/bin/bash
set -euo pipefail

# ── BADGERS — Full Teardown ────────────────────────────────────────────────
#
# Usage:
#   ./destroy.sh                                    # choose the target interactively
#   DEPLOYMENT_ID=dev ./destroy.sh                  # explicit, cross-checked against AWS
#   DEPLOYMENT_ID=dev ./destroy.sh --vpc-cleanup-only
#
# What this does, in order:
#   1. With no DEPLOYMENT_ID, scans CloudFormation for BADGERS-*-{suffix} stacks and
#      lets you pick one; the DEPLOYMENT_ID is read back from the S3 stack rather than
#      typed. With an explicit DEPLOYMENT_ID, verifies it owns that suffix and refuses
#      on a mismatch.
#   2. Requires you to type the DEPLOYMENT_ID to confirm
#   3. Empties the S3 buckets (all versions + delete markers)
#   4. Deletes the ECS Express service and the AgentCore runtime FIRST, then waits
#      for their ENIs to release — CloudFormation cannot delete the VPC while any
#      ENI remains attached
#   5. Destroys every stack in reverse dependency order
#   6. Schedules the KMS key for deletion, which frees its alias for redeployment
#   7. Verifies the stacks are gone, and auto-fixes a stuck VPC stack
#
# Required:
#   DEPLOYMENT_ID   Identifier used when the stacks were deployed
#
# Optional:
#   AWS_REGION      AWS region (default: us-west-2)
#   KMS_WAIT_DAYS   KMS key deletion waiting period, 7-30 (default: 7)
#   STACK_SUFFIX    Override the suffix (default: read from the state file)
# ───────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/deployment/scripts/common.sh"

trap 'echo ""; log_error "destroy.sh failed at line ${LINENO}: ${BASH_COMMAND}"; exit 1' ERR

KMS_WAIT_DAYS="${KMS_WAIT_DAYS:-7}"
VPC_CLEANUP_ONLY=false
[ "${1:-}" = "--vpc-cleanup-only" ] && VPC_CLEANUP_ONLY=true

ensure_account

# With no DEPLOYMENT_ID given, the target is chosen from what is actually deployed in
# CloudFormation. That is the safe default: stack names carry only the suffix, so a
# typed ID can resolve another deployment's stacks, and the local state file may be
# gone while those stacks are still live.
if [ -z "${DEPLOYMENT_ID:-}" ]; then
  if [ "${VPC_CLEANUP_ONLY}" = true ]; then
    log_error "--vpc-cleanup-only needs an explicit DEPLOYMENT_ID and STACK_SUFFIX."
    exit 1
  fi
  choose_deployment_to_destroy || exit 1
else
  # An uppercase character crashes cdk synth on S3 bucket-name validation, which
  # aborts the destroy before any stack is touched.
  if ! _valid_deployment_id "${DEPLOYMENT_ID}"; then
    log_error "Invalid DEPLOYMENT_ID: '${DEPLOYMENT_ID}'"
    log_error "Must match ^[a-z][a-z0-9-]{0,15}\$ — lowercase, starting with a letter."
    log_error "Re-run with no DEPLOYMENT_ID to choose from what is deployed."
    exit 1
  fi

  load_suffix

  # No ownership cross-check is needed any more: stack names carry the DEPLOYMENT_ID, so
  # a wrong id simply matches no stacks rather than resolving someone else's. Confirm
  # something is actually there, so a typo reads as "nothing found" instead of a
  # teardown that quietly does nothing.
  if [ "${VPC_CLEANUP_ONLY}" != true ]; then
    FOUND="$(aws cloudformation list-stacks --region "${AWS_REGION}" \
      --query "length(StackSummaries[?starts_with(StackName, '${STACK_PREFIX}-') \
        && contains(StackName, '-${DEPLOYMENT_ID}-${STACK_SUFFIX}') \
        && StackStatus != 'DELETE_COMPLETE'])" \
      --output text 2>/dev/null || echo "0")"
    if [ "${FOUND}" = "0" ] || [ -z "${FOUND}" ] || [ "${FOUND}" = "None" ]; then
      log_error "No stacks found matching ${STACK_PREFIX}-*-${DEPLOYMENT_ID}-${STACK_SUFFIX} in ${AWS_REGION}."
      log_error "Check the id and suffix, or re-run with no DEPLOYMENT_ID to choose from"
      log_error "what is actually deployed."
      exit 1
    fi
    log_info "Found ${FOUND} stack(s) for ${DEPLOYMENT_ID}-${STACK_SUFFIX}."
  fi
fi

RESOURCE_ID="$(resource_id)"
ECS_SERVICE_NAME="badgers-ui-${RESOURCE_ID}"

# Reverse dependency order. ECS first (it imports VPC exports), VPC last.
STACKS=(
  "$(_sn CustomSpecialists)"
  "$(_sn ECS)"
  "$(_sn RuntimeWebSocket)"
  "$(_sn Gateway)"
  "$(_sn Lambda)"
  "$(_sn Memory)"
  "$(_sn XRay)"
  "$(_sn ECR)"
  "$(_sn InferenceProfiles)"
  "$(_sn IAM)"
  "$(_sn DynamoDB)"
  "$(_sn S3)"
  "$(_sn Cognito)"
  "$(_sn Vpc)"
)

echo "════════════════════════════════════════════════════════"
echo "  BADGERS — Teardown"
echo "  Account:       ${ACCOUNT_ID}"
echo "  Region:        ${AWS_REGION}"
echo "  Deployment ID: ${DEPLOYMENT_ID}"
echo "  Stack Suffix:  ${STACK_SUFFIX}"
echo "  Resource ID:   ${RESOURCE_ID}"
echo "  KMS wait:      ${KMS_WAIT_DAYS} days"
echo "════════════════════════════════════════════════════════"

# ── Shared ENI cleanup ─────────────────────────────────────────────────────
# ENIs are the usual reason a VPC stack will not delete. Interface endpoints own
# ENIs, and ECS/AgentCore release theirs asynchronously well after the service or
# runtime reports gone.
cleanup_vpc_enis() {
  local vpc_id="$1"
  [ -z "${vpc_id}" ] && return 0
  [ "${vpc_id}" = "None" ] && return 0

  log_info "Checking for ENIs in VPC ${vpc_id}..."
  local all
  all="$(aws ec2 describe-network-interfaces \
    --filters Name=vpc-id,Values="${vpc_id}" --region "${AWS_REGION}" \
    --query "NetworkInterfaces[].{ID:NetworkInterfaceId,Type:InterfaceType,Desc:Description,Status:Status}" \
    --output text 2>/dev/null || true)"

  if [ -z "${all}" ] || [ "${all}" = "None" ]; then
    log_success "No ENIs in VPC — clean state"
    return 0
  fi

  log_warn "Found ENIs:"
  echo "${all}" | while IFS=$'\t' read -r id type desc status; do
    echo "      ${id} [${type}] - ${desc} (${status})"
  done

  # Interface endpoints must go first; they own their ENIs.
  local endpoints
  endpoints="$(aws ec2 describe-vpc-endpoints \
    --filters Name=vpc-id,Values="${vpc_id}" Name=vpc-endpoint-type,Values=Interface \
    --region "${AWS_REGION}" --query "VpcEndpoints[].VpcEndpointId" \
    --output text 2>/dev/null || true)"
  if [ -n "${endpoints}" ] && [ "${endpoints}" != "None" ]; then
    for ep in ${endpoints}; do
      aws ec2 delete-vpc-endpoints --vpc-endpoint-ids "${ep}" \
        --region "${AWS_REGION}" 2>/dev/null || true
      log_info "Deleted VPC endpoint ${ep}"
    done
    for i in $(seq 1 12); do
      sleep 10
      local left
      left="$(aws ec2 describe-network-interfaces \
        --filters Name=vpc-id,Values="${vpc_id}" Name=interface-type,Values=vpc_endpoint \
        --region "${AWS_REGION}" --query "NetworkInterfaces[].NetworkInterfaceId" \
        --output text 2>/dev/null || true)"
      if [ -z "${left}" ] || [ "${left}" = "None" ]; then
        log_success "VPC endpoint ENIs released"
        break
      fi
      echo "    $(_ts) — waiting for endpoint ENIs (${i}/12)"
    done
  fi

  # Delete what is available; force-detach what is not.
  for i in $(seq 1 12); do
    local remaining
    remaining="$(aws ec2 describe-network-interfaces \
      --filters Name=vpc-id,Values="${vpc_id}" --region "${AWS_REGION}" \
      --query "NetworkInterfaces[].[NetworkInterfaceId,Status,Attachment.AttachmentId]" \
      --output text 2>/dev/null || true)"
    if [ -z "${remaining}" ] || [ "${remaining}" = "None" ]; then
      log_success "All ENIs cleaned up"
      return 0
    fi
    echo "${remaining}" | while IFS=$'\t' read -r id status attach; do
      if [ "${status}" = "available" ]; then
        aws ec2 delete-network-interface --network-interface-id "${id}" \
          --region "${AWS_REGION}" 2>/dev/null && log_info "Deleted ENI ${id}" || true
      elif [ -n "${attach}" ] && [ "${attach}" != "None" ]; then
        log_warn "Force detaching ENI ${id}..."
        aws ec2 detach-network-interface --attachment-id "${attach}" --force \
          --region "${AWS_REGION}" 2>/dev/null || true
        sleep 2
        aws ec2 delete-network-interface --network-interface-id "${id}" \
          --region "${AWS_REGION}" 2>/dev/null && log_info "Deleted ENI ${id}" || true
      fi
    done
    echo "    $(_ts) — retrying ENI cleanup (${i}/12)"
    sleep 5
  done

  log_warn "Some ENIs could not be cleaned up; VPC deletion may need the auto-fix."
}

resolve_vpc_id() {
  local vpc_id
  vpc_id="$(stack_output "$(_sn Vpc)" VpcId)"
  if [ -z "${vpc_id}" ] || [ "${vpc_id}" = "None" ]; then
    # Stack may already be gone — fall back to tags.
    vpc_id="$(aws ec2 describe-vpcs \
      --filters Name=tag:application_name,Values=badgers \
      --region "${AWS_REGION}" --query "Vpcs[0].VpcId" --output text 2>/dev/null || true)"
  fi
  echo "${vpc_id}"
}

# ── --vpc-cleanup-only ─────────────────────────────────────────────────────
if [ "${VPC_CLEANUP_ONLY}" = true ]; then
  echo ""
  log_info "Running VPC ENI cleanup only"
  VPC_ID="$(resolve_vpc_id)"
  if [ -z "${VPC_ID}" ] || [ "${VPC_ID}" = "None" ]; then
    log_error "No VPC found for deployment ${DEPLOYMENT_ID}"
    exit 1
  fi
  log_info "Found VPC: ${VPC_ID}"
  cleanup_vpc_enis "${VPC_ID}"
  log_success "VPC ENI cleanup complete"
  exit 0
fi

# ── Confirmation ───────────────────────────────────────────────────────────
echo ""
echo "This will permanently delete:"
echo "  - All ${#STACKS[@]} CloudFormation stacks listed above"
echo "  - All objects in the config, source and output buckets, and all versions"
echo "  - The DynamoDB jobs table and every job record in it"
echo "  - The ECR repository and all images"
echo "  - The AgentCore runtime, gateway and memory"
echo "  - The KMS key (scheduled, ${KMS_WAIT_DAYS}-day waiting period)"
echo ""
echo "Type the DEPLOYMENT_ID to confirm destruction:"
read -r CONFIRM
if [ "${CONFIRM}" != "${DEPLOYMENT_ID}" ]; then
  log_error "Confirmation mismatch. Aborted."
  exit 1
fi
echo ""
log_success "Confirmation accepted. Starting teardown."
echo ""

ensure_cdk_deps
export_cdk_env

# ── Empty S3 buckets ───────────────────────────────────────────────────────
# The buckets set auto_delete_objects, so CloudFormation would empty them, but
# doing it here is faster on versioned buckets and avoids the custom resource
# timing out on large object counts.
empty_bucket() {
  local bucket="$1"
  if ! aws s3api head-bucket --bucket "${bucket}" --region "${AWS_REGION}" 2>/dev/null; then
    echo "  (bucket ${bucket} does not exist — skipping)"
    return 0
  fi
  echo "  → Emptying s3://${bucket} (all versions + delete markers)..."
  local key
  for key in Versions DeleteMarkers; do
    aws s3api list-object-versions --bucket "${bucket}" --region "${AWS_REGION}" \
      --output json --query "{Objects: ${key}[].{Key:Key,VersionId:VersionId}}" 2>/dev/null \
      | jq -c '{Objects: (.Objects // [])} | select(.Objects | length > 0)' \
      | while read -r batch; do
          [ -z "${batch}" ] && continue
          aws s3api delete-objects --bucket "${bucket}" --region "${AWS_REGION}" \
            --delete "${batch}" --output text --no-cli-pager > /dev/null || true
        done
  done
  aws s3 rm "s3://${bucket}" --recursive --region "${AWS_REGION}" --only-show-errors || true
  echo "    ✓ ${bucket} emptied"
}

echo "── Emptying S3 buckets ───────────────────────────────────"
empty_bucket "badgers-config-${RESOURCE_ID}"
empty_bucket "badgers-source-${RESOURCE_ID}"
empty_bucket "badgers-output-${RESOURCE_ID}"
echo ""

# ── Capture KMS key and VPC id before the stacks go ────────────────────────
echo "── Capturing KMS key and VPC id ──────────────────────────"
KMS_KEY_ARN="$(stack_output "$(_sn S3)" S3KmsKeyArn)"
[ "${KMS_KEY_ARN}" = "None" ] && KMS_KEY_ARN=""
if [ -n "${KMS_KEY_ARN}" ]; then
  echo "  ✓ KMS key: ${KMS_KEY_ARN}"
else
  log_warn "Could not read S3KmsKeyArn — the key will not be scheduled by this script."
fi
VPC_ID="$(resolve_vpc_id)"
[ -n "${VPC_ID}" ] && [ "${VPC_ID}" != "None" ] && echo "  ✓ VPC: ${VPC_ID}"
echo ""

# ── Delete the ECS Express service (releases its ENIs) ─────────────────────
echo "── Deleting ECS Express service ──────────────────────────"
SERVICE_ARN="$(aws ecs list-services --cluster default --region "${AWS_REGION}" \
  --query "serviceArns[?contains(@, '${ECS_SERVICE_NAME}')] | [0]" \
  --output text 2>/dev/null || true)"
if [ -n "${SERVICE_ARN}" ] && [ "${SERVICE_ARN}" != "None" ]; then
  echo "  → Deleting ${ECS_SERVICE_NAME}..."
  # DeleteService rejects an Express Gateway Service outright. DeleteExpressGatewayService
  # also removes what the service created — notably the load balancer, when no other
  # service is sharing it.
  aws ecs delete-express-gateway-service --service-arn "${SERVICE_ARN}" \
    --region "${AWS_REGION}" --no-cli-pager > /dev/null || true
  DELETED=false
  for i in $(seq 1 30); do
    STATUS="$(express_service_status "${SERVICE_ARN}")"
    if [ "${STATUS}" = "INACTIVE" ] || [ "${STATUS}" = "GONE" ] || [ "${STATUS}" = "None" ] \
       || [ -z "${STATUS}" ]; then
      echo "    ✓ Service deleted"
      DELETED=true
      break
    fi
    echo "    $(_ts) — status: ${STATUS} (${i}/30)"
    sleep 10
  done
  # Say so rather than falling through silently — the VPC stack delete will fail later
  # on ENIs this service still holds.
  if [ "${DELETED}" != true ]; then
    log_warn "Service ${ECS_SERVICE_NAME} did not reach INACTIVE within 5 minutes."
    log_warn "Last status: ${STATUS}. The VPC stack may fail to delete on its ENIs."
  fi
else
  echo "  (no ECS Express service found — skipping)"
fi
echo ""

# ── Delete the AgentCore runtime (releases its ENIs) ───────────────────────
echo "── Deleting AgentCore runtime ────────────────────────────"
RUNTIME_ID="$(stack_output "$(_sn RuntimeWebSocket)" RuntimeId)"
if [ -n "${RUNTIME_ID}" ] && [ "${RUNTIME_ID}" != "None" ]; then
  echo "  → Deleting runtime ${RUNTIME_ID}..."
  ENDPOINTS="$(aws bedrock-agentcore-control list-agent-runtime-endpoints \
    --agent-runtime-id "${RUNTIME_ID}" --region "${AWS_REGION}" \
    --query "agentRuntimeEndpoints[].name" --output text 2>/dev/null || true)"
  for ep in ${ENDPOINTS}; do
    if [ "${ep}" != "DEFAULT" ] && [ "${ep}" != "None" ] && [ -n "${ep}" ]; then
      aws bedrock-agentcore-control delete-agent-runtime-endpoint \
        --agent-runtime-id "${RUNTIME_ID}" --endpoint-name "${ep}" \
        --region "${AWS_REGION}" 2>/dev/null || true
      echo "    ✓ Deleted endpoint ${ep}"
    fi
  done
  aws bedrock-agentcore-control delete-agent-runtime \
    --agent-runtime-id "${RUNTIME_ID}" --region "${AWS_REGION}" 2>/dev/null || true
  for i in $(seq 1 36); do
    RT="$(aws bedrock-agentcore-control get-agent-runtime \
      --agent-runtime-id "${RUNTIME_ID}" --region "${AWS_REGION}" \
      --query "status" --output text 2>/dev/null || echo "DELETED")"
    if [ "${RT}" = "DELETED" ] || [ "${RT}" = "NOT_FOUND" ]; then
      echo "    ✓ Runtime deleted"
      break
    fi
    echo "    $(_ts) — runtime status: ${RT} (${i}/36)"
    sleep 10
  done
else
  echo "  (no AgentCore runtime found — skipping)"
fi
echo ""

# ── Final ENI sweep before destroying stacks ───────────────────────────────
echo "── ENI verification ──────────────────────────────────────"
cleanup_vpc_enis "${VPC_ID}"
echo ""

# ── Destroy stacks ─────────────────────────────────────────────────────────
echo "── Destroying CloudFormation stacks ──────────────────────"
cdk_destroy "${STACKS[@]}" || log_warn "cdk destroy reported errors; verifying below."
echo ""

# ── Verify ─────────────────────────────────────────────────────────────────
# Sets REMAINING. Returns 0 explicitly: the loop body ends in a conditional whose
# failure would otherwise become the function's status under errexit.
_remaining_stacks() {
  local stack state
  REMAINING=()
  for stack in "${STACKS[@]}"; do
    state="$(stack_status "${stack}")"
    [ "${state}" != "DELETED" ] && REMAINING+=("${stack} (${state})")
  done
  return 0
}

echo "── Verifying stack deletion ──────────────────────────────"
_remaining_stacks

if [ ${#REMAINING[@]} -gt 0 ]; then
  log_warn "These stacks still exist:"
  for s in "${REMAINING[@]}"; do echo "      - ${s}"; done

  # A DELETE_FAILED VPC is almost always a lingering ENI. Retain the failed
  # resources so the stack can go, then clean them up directly.
  VPC_STACK="$(_sn Vpc)"
  if [ "$(stack_status "${VPC_STACK}")" = "DELETE_FAILED" ]; then
    echo ""
    echo "── Auto-fixing stuck VPC stack ─────────────────────────"
    FAILED="$(aws cloudformation list-stack-resources --stack-name "${VPC_STACK}" \
      --region "${AWS_REGION}" \
      --query "StackResourceSummaries[?ResourceStatus=='DELETE_FAILED'].LogicalResourceId" \
      --output text 2>/dev/null || true)"
    if [ -n "${FAILED}" ]; then
      echo "  → Retrying delete, retaining: ${FAILED}"
      # shellcheck disable=SC2086
      aws cloudformation delete-stack --stack-name "${VPC_STACK}" \
        --retain-resources ${FAILED} --region "${AWS_REGION}" || true
      aws cloudformation wait stack-delete-complete --stack-name "${VPC_STACK}" \
        --region "${AWS_REGION}" 2>/dev/null || true
      cleanup_vpc_enis "${VPC_ID}"
      log_warn "Retained VPC resources may need manual deletion: ${FAILED}"
    fi
  fi

  # The auto-fix may have cleared it, so re-check before deciding the outcome.
  echo ""
  echo "── Re-verifying after auto-fix ───────────────────────────"
  _remaining_stacks
fi
echo ""

if [ ${#REMAINING[@]} -gt 0 ]; then
  log_error "Teardown INCOMPLETE — ${#REMAINING[@]} stack(s) still exist:"
  for s in "${REMAINING[@]}"; do log_error "      - ${s}"; done
  echo ""
  if [ -n "${KMS_KEY_ARN}" ]; then
    log_warn "KMS key NOT scheduled for deletion, because those stacks are still live:"
    log_warn "      ${KMS_KEY_ARN}"
  fi
  log_error "Re-run once the cause is fixed. Scroll up for the first failure — a cdk"
  log_error "synth or destroy error aborts the run before any stack is touched."
  echo "════════════════════════════════════════════════════════"
  log_error "  ❌ Teardown incomplete  (DEPLOYMENT_ID=${DEPLOYMENT_ID}, STACK_SUFFIX=${STACK_SUFFIX})"
  echo "════════════════════════════════════════════════════════"
  exit 1
fi

log_success "All stacks deleted"
echo ""

# ── Schedule KMS key deletion ──────────────────────────────────────────────
# Only once the stacks are confirmed gone. Scheduling it after a failed destroy marks
# a live deployment's in-use key for deletion, which is unrecoverable after the window.
# The alias stays reserved while the key is pending deletion, so a short window
# matters if you intend to redeploy with the same DEPLOYMENT_ID.
if [ -n "${KMS_KEY_ARN}" ]; then
  echo "── Scheduling KMS key deletion ───────────────────────────"
  STATE="$(aws kms describe-key --key-id "${KMS_KEY_ARN}" --region "${AWS_REGION}" \
    --query 'KeyMetadata.KeyState' --output text 2>/dev/null || echo "MISSING")"
  case "${STATE}" in
    PendingDeletion)
      echo "  ✓ Already scheduled for deletion"
      ;;
    Enabled|Disabled)
      aws kms schedule-key-deletion --key-id "${KMS_KEY_ARN}" \
        --pending-window-in-days "${KMS_WAIT_DAYS}" --region "${AWS_REGION}" \
        --output text --no-cli-pager > /dev/null
      echo "  ✓ Scheduled for deletion in ${KMS_WAIT_DAYS} days"
      echo "    (cancel: aws kms cancel-key-deletion --key-id ${KMS_KEY_ARN})"
      ;;
    MISSING)
      echo "  (key not found — skipping)"
      ;;
    *)
      log_warn "Key is in state ${STATE} — not modifying."
      ;;
  esac
  echo ""
fi

echo "════════════════════════════════════════════════════════"
echo "  ✅ Teardown complete  (DEPLOYMENT_ID=${DEPLOYMENT_ID}, STACK_SUFFIX=${STACK_SUFFIX})"
echo "════════════════════════════════════════════════════════"
