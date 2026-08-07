#!/bin/bash
# Shared helpers for the BADGERS deployment scripts.
#
# Source this, do not execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
#
# Provides:
#   - logging:      log_info / log_success / log_warn / log_error / log_step
#   - identity:     DEPLOYMENT_ID, STACK_SUFFIX, RESOURCE_ID, ACCOUNT_ID, AWS_REGION
#   - stack names:  _sn <Name>            -> BADGERS-<Name>-<suffix>
#   - state:        init_state / get_state / set_state / mark_complete / check_completed
#   - cdk:          ensure_cdk_deps / cdk_deploy / cdk_destroy
#   - outputs:      stack_output <StackName> <OutputKey>
#
# Stack names carry the suffix only (BADGERS-S3-a1b). Resource names carry both
# parts (badgers-config-dev-a1b) via RESOURCE_ID, matching what app.py composes.

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Set BADGERS_ASSUME_YES=1 to answer every confirmation with yes. Required when running
# without a tty: the UI's deploy route streams output with stdin closed, so an
# interactive read would see EOF and the caller would silently skip the step.
_confirm() {
    local prompt="${1:-Continue? (y/n): }"
    if [ "${BADGERS_ASSUME_YES:-}" = "1" ]; then
        echo "${prompt}y  (BADGERS_ASSUME_YES)"
        return 0
    fi
    local reply
    read -rp "${prompt}" reply || return 1
    [ "${reply}" = "y" ]
}
log_info() { echo -e "${BLUE}[$(_ts)]${NC} $1"; }
log_success() { echo -e "${GREEN}[$(_ts) ✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[$(_ts) !]${NC} $1"; }
log_error() { echo -e "${RED}[$(_ts) ✗]${NC} $1"; }
log_step() { echo -e "${CYAN}${BOLD}── [$(_ts)]${NC} ${BOLD}$1${NC}"; }

# ── Paths ──────────────────────────────────────────────────────────────────
# This file lives at <repo>/deployment/scripts/common.sh
BADGERS_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(cd "${BADGERS_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOYMENT_DIR}/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-west-2}"
STACK_PREFIX="${STACK_PREFIX:-BADGERS}"

# ── Deployment identity ────────────────────────────────────────────────────
require_deployment_id() {
  if [ -z "${DEPLOYMENT_ID:-}" ]; then
    log_error "DEPLOYMENT_ID is not set. ./deploy.sh prompts for it; other scripts take it"
    log_error "explicitly, e.g. DEPLOYMENT_ID=dev ./destroy.sh"
    exit 1
  fi
}

state_file() {
  echo "${REPO_ROOT}/.deploy-state/${DEPLOYMENT_ID:-unknown}.json"
}

# ── State management ───────────────────────────────────────────────────────
init_state() {
  mkdir -p "${REPO_ROOT}/.deploy-state"
  local f
  f="$(state_file)"
  if [ ! -f "${f}" ]; then
    cat > "${f}" <<EOF
{
  "deployment_id": "${DEPLOYMENT_ID}",
  "started_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "layers_built": false,
  "infra_complete": false,
  "s3_files_uploaded": false,
  "specialists_complete": false,
  "gateway_complete": false,
  "runtime_image_pushed": false,
  "runtime_deployed": false,
  "ui_image_pushed": false,
  "ecs_deployed": false,
  "gateway_url": ""
}
EOF
  fi
}

get_state() {
  local f
  f="$(state_file)"
  [ -f "${f}" ] || { echo ""; return; }
  python3 -c "import json,sys
try:
    d=json.load(open('${f}'))
except Exception:
    print(''); sys.exit(0)
print(d.get('$1',''))"
}

set_state() {
  local f
  f="$(state_file)"
  python3 -c "
import json
with open('${f}', 'r+') as fh:
    data = json.load(fh)
    data['$1'] = $2
    fh.seek(0)
    json.dump(data, fh, indent=2)
    fh.truncate()
"
}

mark_complete() {
  set_state "$1" "True"
  set_state "${1}_at" "\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\""
  log_success "Step complete: $1"
}

check_completed() {
  [ "$(get_state "$1")" = "True" ]
}

# Clears a completion flag and its timestamp so resume re-runs that step. Used when a
# later step's failure invalidates an earlier step's output — the caller logs why, so
# this stays quiet.
invalidate_state() {
  set_state "$1" "False"
  set_state "${1}_at" "\"\""
}

# ── ECS Express Gateway Service ────────────────────────────────────────────
# Express Gateway Services are a distinct resource management type: DeleteService
# rejects them with "has ResourceManagementType=ECS use DeleteExpressGatewayService".
# Echoes the service status, or GONE once it no longer exists — describe starts failing
# at that point, which is the signal deletion finished. The status key is located by
# walking the response rather than assuming a wrapper name.
express_service_status() {
  local arn="$1" json
  json="$(aws ecs describe-express-gateway-service --service-arn "${arn}" \
    --region "${AWS_REGION}" --output json 2>/dev/null)" || { echo "GONE"; return 0; }
  [ -z "${json}" ] && { echo "GONE"; return 0; }
  printf '%s' "${json}" | python3 -c "
import json, sys


def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'status' and isinstance(v, str):
                return v
            r = find(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r:
                return r
    return ''


try:
    print(find(json.load(sys.stdin)) or 'GONE')
except Exception:
    print('GONE')
" 2>/dev/null || echo "GONE"
}

# ── Deployment discovery ───────────────────────────────────────────────────
# Every state key a completed deployment sets, in execution order. Kept here so the
# state layer can report progress without each caller reimplementing the JSON reads.
STEP_KEYS=(
  layers_built
  infra_complete
  s3_files_uploaded
  specialists_complete
  gateway_complete
  runtime_image_pushed
  runtime_deployed
  ui_image_pushed
  ecs_deployed
)

# Echoes "done total last_activity" for one state file. last_activity is the newest
# *_at stamp, falling back to started_at — derived rather than tracked in a separate
# "last deployment" pointer, which would be one more thing to drift out of sync.
deployment_progress() {
  local f="$1"
  [ -f "${f}" ] || { echo "0 ${#STEP_KEYS[@]} "; return; }
  STEP_KEYS_CSV="$(IFS=,; echo "${STEP_KEYS[*]}")" python3 -c "
import json, os, sys
keys = os.environ['STEP_KEYS_CSV'].split(',')
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print('0 ' + str(len(keys)) + ' ')
    sys.exit(0)
done = sum(1 for k in keys if d.get(k) is True)
stamps = sorted(v for k, v in d.items()
                if k.endswith('_at') and isinstance(v, str) and v)
last = stamps[-1] if stamps else (d.get('started_at') or '')
print(str(done) + ' ' + str(len(keys)) + ' ' + last)
" "${f}"
}

# Every known deployment as "id done total last_activity", most recent activity first.
list_deployments() {
  local dir="${REPO_ROOT}/.deploy-state" f id
  [ -d "${dir}" ] || return 0
  for f in "${dir}"/*.json; do
    [ -f "${f}" ] || continue
    id="$(basename "${f}" .json)"
    echo "${id} $(deployment_progress "${f}")"
  done | sort -k4,4 -r
}

# DEPLOYMENT_ID becomes part of resource names ({DEPLOYMENT_ID}-{suffix}) and of the
# ECR repository name, so it has to be lowercase, start with a letter, and stay short.
_valid_deployment_id() {
  [[ "$1" =~ ^[a-z][a-z0-9-]{0,15}$ ]]
}

# Interactive selection: offer any unfinished deployments to resume, otherwise (or on
# request) prompt for a new ID. Sets and exports DEPLOYMENT_ID.
choose_deployment() {
  local -a ids=() dones=() totals=() lasts=()
  local id done total last
  # Completed deployments are listed too. Filtering them out left no way to re-run a
  # single step against a finished deployment — and "start a new one" refuses an ID that
  # already has state, so a 9/9 deployment became unreachable through this menu.
  while read -r id done total last; do
    [ -n "${id}" ] || continue
    ids+=("${id}"); dones+=("${done}"); totals+=("${total}"); lasts+=("${last}")
  done < <(list_deployments)

  if [ "${#ids[@]}" -gt 0 ]; then
    echo ""
    log_warn "Deployments found — most recent activity first:"
    echo ""
    local i mark
    for i in "${!ids[@]}"; do
      if [ "${dones[i]}" = "${totals[i]}" ]; then mark="complete"; else mark="in progress"; fi
      printf "   %2d) %-18s %s/%s steps  %-12s last activity %s\n" \
        "$((i + 1))" "${ids[i]}" "${dones[i]}" "${totals[i]}" "${mark}" "${lasts[i]:-unknown}"
    done
    echo ""
    echo "    n) Start a new deployment instead"
    echo ""
    local pick
    read -rp "Select which? (1-${#ids[@]}, or n) [1]: " pick
    pick="${pick:-1}"
    case "${pick}" in
      n|N) ;;
      ''|*[!0-9]*)
        log_error "Invalid selection: ${pick}"
        return 1 ;;
      *)
        if [ "${pick}" -lt 1 ] || [ "${pick}" -gt "${#ids[@]}" ]; then
          log_error "Selection out of range: ${pick}"
          return 1
        fi
        DEPLOYMENT_ID="${ids[$((pick - 1))]}"
        export DEPLOYMENT_ID
        log_success "Selected deployment: ${DEPLOYMENT_ID}"
        return 0 ;;
    esac
  else
    echo ""
    log_info "No existing deployments found."
  fi

  local new
  while true; do
    read -rp "New deployment ID (e.g. dev, demo, b2): " new
    if ! _valid_deployment_id "${new}"; then
      log_error "Invalid ID — must match ^[a-z][a-z0-9-]{0,15}\$ (lowercase, starts with a letter)."
      continue
    fi
    if [ -f "${REPO_ROOT}/.deploy-state/${new}.json" ]; then
      log_error "${new} already has deployment state. Re-run and pick it from the list to resume it."
      continue
    fi
    break
  done
  DEPLOYMENT_ID="${new}"
  export DEPLOYMENT_ID
  log_success "Starting new deployment: ${DEPLOYMENT_ID}"
}

# ── Deployed-stack discovery (teardown) ────────────────────────────────────
# Teardown discovers what exists in CloudFormation rather than trusting local state: a
# state file can be deleted while the stacks are still live, and stack names carry only
# the suffix, so a wrong DEPLOYMENT_ID still resolves real stacks.
_DEPLOYED_STACK_STATES="CREATE_COMPLETE UPDATE_COMPLETE UPDATE_ROLLBACK_COMPLETE ROLLBACK_COMPLETE CREATE_FAILED DELETE_FAILED UPDATE_FAILED UPDATE_ROLLBACK_FAILED"

# Echoes "id suffix stack_count newest_creation" per deployed deployment, newest first.
# Identity now comes straight off the stack names, so this needs one list-stacks call
# and no per-deployment describe-stacks to recover the id.
list_deployed_deployments() {
  # shellcheck disable=SC2086
  aws cloudformation list-stacks --region "${AWS_REGION}" \
    --stack-status-filter ${_DEPLOYED_STACK_STATES} \
    --query "StackSummaries[?starts_with(StackName, '${STACK_PREFIX}-')].[StackName,CreationTime]" \
    --output text 2>/dev/null \
  | _group_stacks_by_deployment
}

# Split out so the grouping can be tested without calling AWS.
_group_stacks_by_deployment() {
  STACK_PREFIX="${STACK_PREFIX}" STACK_BASE_NAMES="${STACK_BASE_NAMES}" python3 -c "
import os, re, sys
prefix = os.environ['STACK_PREFIX']
bases = os.environ['STACK_BASE_NAMES'].split()
alts = '|'.join(sorted((re.escape(b) for b in bases), key=len, reverse=True))
pat = re.compile('^' + re.escape(prefix) + '-(?:' + alts + ')-(.+)-([A-Za-z0-9]{1,8})\$')
groups = {}
for line in sys.stdin:
    parts = line.split()
    if len(parts) < 2:
        continue
    name, created = parts[0], parts[1]
    m = pat.match(name)
    if not m:
        continue
    key = (m.group(1), m.group(2))
    g = groups.setdefault(key, {'n': 0, 'newest': ''})
    g['n'] += 1
    if created > g['newest']:
        g['newest'] = created
for (dep_id, suffix), g in sorted(
    groups.items(), key=lambda kv: kv[1]['newest'], reverse=True
):
    print(dep_id + ' ' + suffix + ' ' + str(g['n']) + ' ' + g['newest'])
"
}

# Interactive teardown target selection. Sets and exports DEPLOYMENT_ID and
# STACK_SUFFIX from what is actually deployed.
choose_deployment_to_destroy() {
  local -a ids=() sufs=() counts=() createds=()
  local id suffix count created
  echo ""
  log_info "Scanning CloudFormation for ${STACK_PREFIX}-*-{id}-{suffix} stacks..."
  while read -r id suffix count created; do
    [ -n "${id}" ] || continue
    ids+=("${id}"); sufs+=("${suffix}"); counts+=("${count}"); createds+=("${created}")
  done < <(list_deployed_deployments)

  if [ "${#ids[@]}" -eq 0 ]; then
    log_error "No ${STACK_PREFIX}-*-{id}-{suffix} stacks found in ${AWS_REGION}."
    log_error "Nothing to destroy. Stacks named under an older convention are not matched"
    log_error "here and need deleting directly."
    return 1
  fi

  echo ""
  log_warn "Deployments found in ${AWS_REGION} — newest first:"
  echo ""
  local i
  for i in "${!ids[@]}"; do
    printf "   %2d) %-20s suffix %-6s  %s stack(s)   created %s\n" \
      "$((i + 1))" "${ids[i]}" "${sufs[i]}" "${counts[i]}" "${createds[i]}"
  done
  echo ""
  local pick
  read -rp "Destroy which? (1-${#ids[@]}, or q to abort): " pick
  case "${pick}" in
    q|Q|"") log_info "Aborted."; return 1 ;;
    *[!0-9]*) log_error "Invalid selection: ${pick}"; return 1 ;;
  esac
  if [ "${pick}" -lt 1 ] || [ "${pick}" -gt "${#ids[@]}" ]; then
    log_error "Selection out of range: ${pick}"
    return 1
  fi

  i=$((pick - 1))
  DEPLOYMENT_ID="${ids[i]}"
  STACK_SUFFIX="${sufs[i]}"
  export DEPLOYMENT_ID STACK_SUFFIX
  log_success "Selected: ${DEPLOYMENT_ID} (suffix ${STACK_SUFFIX}, ${counts[i]} stacks)"
}

# ── Stack suffix ───────────────────────────────────────────────────────────
# Generated once per DEPLOYMENT_ID and persisted, so re-runs target the same stacks.
_generate_suffix() {
  python3 -c "import secrets; print(secrets.token_hex(2)[:3])"
}

ensure_suffix() {
  require_deployment_id
  init_state
  local existing
  existing="$(get_state "stack_suffix")"
  if [ -n "${existing}" ]; then
    STACK_SUFFIX="${existing}"
  else
    STACK_SUFFIX="${STACK_SUFFIX:-$(_generate_suffix)}"
    set_state "stack_suffix" "\"${STACK_SUFFIX}\""
  fi
  export STACK_SUFFIX
}

# Read-only variant: fail rather than invent a suffix. Used by destroy and by the
# helper scripts, where generating a new suffix would silently target nothing.
load_suffix() {
  require_deployment_id
  if [ -z "${STACK_SUFFIX:-}" ]; then
    STACK_SUFFIX="$(get_state "stack_suffix")"
  fi
  if [ -z "${STACK_SUFFIX:-}" ]; then
    log_error "STACK_SUFFIX not found in $(state_file) and not set as an env var."
    log_error "Set it explicitly: STACK_SUFFIX=a1b $0"
    exit 1
  fi
  export STACK_SUFFIX
}

# ── Names ──────────────────────────────────────────────────────────────────
# Stack name: BADGERS-<Name>-<suffix>
# Stack names are BADGERS-{Name}-{DEPLOYMENT_ID}-{suffix} — see the rationale in
# deployment/app.py. Must stay in step with _sn() there.
_sn() { echo "${STACK_PREFIX}-$1-${DEPLOYMENT_ID}-${STACK_SUFFIX}"; }

# Every stack base name this project deploys, used to parse identity back out of a
# stack name. Kept beside _sn so the two cannot drift.
STACK_BASE_NAMES="CustomSpecialists ECS RuntimeWebSocket Gateway Lambda Memory XRay ECR InferenceProfiles IAM DynamoDB S3 Cognito Vpc"

# Splits BADGERS-{Name}-{id}-{suffix} into "id suffix". The base name is matched from
# the known set, so an id containing hyphens (my-app) parses unambiguously. Echoes
# nothing when the name is not one of ours.
parse_stack_name() {
  STACK_PREFIX="${STACK_PREFIX}" STACK_BASE_NAMES="${STACK_BASE_NAMES}" \
    python3 -c "
import os, re, sys
prefix = os.environ['STACK_PREFIX']
bases = os.environ['STACK_BASE_NAMES'].split()
alts = '|'.join(sorted((re.escape(b) for b in bases), key=len, reverse=True))
pat = re.compile('^' + re.escape(prefix) + '-(?:' + alts + ')-(.+)-([A-Za-z0-9]{1,8})\$')
m = pat.match(sys.argv[1])
print(m.group(1) + ' ' + m.group(2) if m else '')
" "$1"
}

# Resource-name id: <DEPLOYMENT_ID>-<suffix>, matching app.py's composite.
resource_id() { echo "${DEPLOYMENT_ID}-${STACK_SUFFIX}"; }

# ── AWS helpers ────────────────────────────────────────────────────────────
ensure_account() {
  if [ -z "${ACCOUNT_ID:-}" ]; then
    ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
    export ACCOUNT_ID
  fi
}

# stack_output <StackName> <OutputKey> -> value, or empty string
stack_output() {
  aws cloudformation describe-stacks \
    --stack-name "$1" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" \
    --output text 2>/dev/null || echo ""
}

stack_status() {
  aws cloudformation describe-stacks \
    --stack-name "$1" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].StackStatus' \
    --output text 2>/dev/null || echo "DELETED"
}

# ── CDK ────────────────────────────────────────────────────────────────────
ensure_cdk_deps() {
  if ! command -v uv &> /dev/null; then
    log_error "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  log_info "Syncing Python dependencies..."
  unset VIRTUAL_ENV
  (cd "${REPO_ROOT}" && uv sync --quiet)
  export VIRTUAL_ENV="${REPO_ROOT}/.venv"
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
}

export_cdk_env() {
  ensure_account
  export DEPLOYMENT_ID
  export STACK_SUFFIX
  export CDK_DEFAULT_ACCOUNT="${ACCOUNT_ID}"
  export CDK_DEFAULT_REGION="${AWS_REGION}"
  export TYPEGUARD_DISABLE=1
  export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"
}

cdk_deploy() {
  (unset VIRTUAL_ENV; cd "${DEPLOYMENT_DIR}" \
    && uv run cdk deploy --app "python app.py" --require-approval never "$@")
}

cdk_destroy() {
  (unset VIRTUAL_ENV; cd "${DEPLOYMENT_DIR}" \
    && uv run cdk destroy --app "python app.py" --force "$@")
}

# ── X-Ray Transaction Search preflight ─────────────────────────────────────
# Transaction Search is an account-and-region singleton, and enabling it needs a
# CloudWatch Logs resource policy. That policy competes for a hard, NON-ADJUSTABLE
# quota of 10 per region (Service Quotas code L-89892494), shared with every other
# project in the account. Left unchecked, CloudFormation fails mid-deploy with a bare
# "Resource limit exceeded", after other stacks have already been created.
LOGS_RESOURCE_POLICY_QUOTA_CODE="L-89892494"
LOGS_RESOURCE_POLICY_QUOTA_FALLBACK=10

# There are only two things this repository can legitimately know about the policies in
# somebody else's account:
#
#   1. Which ones AWS creates itself, identifiable by AWS's own naming convention.
#   2. Which one this project creates.
#
# Everything else belongs to whoever deployed it. The account this runs in is unknown at
# authoring time, so the script reports what it finds and leaves the judgement about
# other people's policies to the operator, who can go and identify the owners.
LOGS_RESOURCE_POLICY_AWS_MANAGED="XRayToLogsIngestion_DO-NOT-EDIT_*"

# The single policy this project creates — see stacks/xray_transaction_search_stack.py.
LOGS_RESOURCE_POLICY_OWN="TransactionSearchAccess"

# Echoes the count, or nothing at all when the call fails. An empty result means
# "could not determine", which callers must not conflate with zero — treating a
# credentials or permissions failure as "0 policies, plenty of room" is how a
# preflight ends up green immediately before CloudFormation fails on the quota.
logs_resource_policy_count() {
  aws logs describe-resource-policies --region "${AWS_REGION}" \
    --query 'length(resourcePolicies)' --output text 2>/dev/null || echo ""
}

logs_resource_policy_quota() {
  local v
  v="$(aws service-quotas get-service-quota \
    --service-code logs --quota-code "${LOGS_RESOURCE_POLICY_QUOTA_CODE}" \
    --region "${AWS_REGION}" --query 'Quota.Value' --output text 2>/dev/null || echo "")"
  # Falls back when the caller lacks servicequotas:GetServiceQuota.
  if [ -z "${v}" ] || [ "${v}" = "None" ]; then
    echo "${LOGS_RESOURCE_POLICY_QUOTA_FALLBACK}"
  else
    printf '%.0f\n' "${v}"
  fi
}

# Prints the existing policies oldest-first with their age, plus a hint about what each
# one appears to be for. Shown whenever the quota is full or about to be, because the
# useful question is not "how many" but "which of these can go".
# Summarises each policy by what it actually grants — principal and target log groups —
# rather than judging whose it is. Which of these can go is the operator's call, and the
# facts needed to make it are in the policy documents, not in this repository.
#
# A trailing (*) marks the policy this project creates, purely so it can be told apart
# from the rest; it carries no implication about which is safe to remove.
logs_resource_policy_report() {
  aws logs describe-resource-policies --region "${AWS_REGION}" --output json 2>/dev/null \
    | BADGERS_OWN="${LOGS_RESOURCE_POLICY_OWN}" \
      python3 -c '
import json, os, sys
from datetime import datetime, timezone

try:
    pols = json.load(sys.stdin).get("resourcePolicies", [])
except Exception:
    print("      (could not read the policy list)")
    raise SystemExit

OWN = os.environ.get("BADGERS_OWN", "")


def summarise(doc):
    """Pull the principals and target log groups out of a policy document."""
    try:
        d = json.loads(doc)
    except Exception:
        return "<unparseable document>", "<unparseable>"

    principals, groups = set(), set()
    for st in d.get("Statement", []):
        pr = st.get("Principal", {})
        if isinstance(pr, dict):
            for v in pr.values():
                principals.update([v] if isinstance(v, str) else v)
        elif isinstance(pr, str):
            principals.add(pr)

        res = st.get("Resource", [])
        for r in [res] if isinstance(res, str) else res:
            groups.add(r.split(":log-group:")[-1] if ":log-group:" in r else r)

    return (
        ",".join(sorted(principals)) or "-",
        ",".join(sorted(groups)) or "-",
    )


now = datetime.now(timezone.utc)
rows = sorted(pols, key=lambda p: p.get("lastUpdatedTime", 0))
if not rows:
    print("      (no resource policies found)")
    raise SystemExit

for p in rows:
    name = p["policyName"]
    dt = datetime.fromtimestamp(p.get("lastUpdatedTime", 0) / 1000, tz=timezone.utc)
    who, where = summarise(p.get("policyDocument", ""))
    mark = "  (*) created by this deployment" if name == OWN else ""
    print(f"      {name}{mark}")
    print(f"          updated {dt:%Y-%m-%d} ({(now - dt).days}d ago)")
    print(f"          grants  {who}")
    print(f"          on      {where}")
'
}

# Echoes ACTIVE when Transaction Search already sends segments to CloudWatch Logs,
# INACTIVE when it demonstrably does not, or UNAVAILABLE when the state could not be
# read at all. One call, so a transient failure cannot produce a half-read answer.
xray_transaction_search_status() {
  local json dest status
  json="$(aws xray get-trace-segment-destination --region "${AWS_REGION}" \
    --output json 2>/dev/null)" || { echo "UNAVAILABLE"; return 0; }
  [ -z "${json}" ] && { echo "UNAVAILABLE"; return 0; }

  dest="$(printf '%s' "${json}" | python3 -c \
    'import json,sys; print(json.load(sys.stdin).get("Destination",""))' 2>/dev/null)"
  status="$(printf '%s' "${json}" | python3 -c \
    'import json,sys; print(json.load(sys.stdin).get("Status",""))' 2>/dev/null)"

  if [ "${dest}" = "CloudWatchLogs" ] && [ "${status}" = "ACTIVE" ]; then
    echo "ACTIVE"
  elif [ -n "${status}" ]; then
    echo "INACTIVE (destination=${dest:-none}, status=${status})"
  else
    echo "UNAVAILABLE"
  fi
}

# Decides whether the X-Ray stack should be deployed at all, and refuses to start a
# deploy that is going to fail on the resource-policy quota.
#
# Exports BADGERS_SKIP_XRAY=1 when Transaction Search is already enabled; app.py reads
# it and leaves the stack out of the app entirely, which also drops the Runtime stack's
# dependency on it (otherwise cdk would pull the stack in regardless).
#
# Returns non-zero when the deploy should not proceed.
# Every step that runs cdk needs the X-Ray decision made, not just the one that owns the
# stack: RuntimeWebSocket depends on the XRay stack, and `cdk deploy` includes a stack's
# dependencies, so deploying ECS or Runtime with BADGERS_SKIP_XRAY unset pulls the XRay
# stack in and attempts a CloudWatch Logs resource policy against a full quota.
#
# Idempotent — an already-set BADGERS_SKIP_XRAY (by preflight or by hand) is preserved,
# so this can be called from every cdk step without re-running the checks or clobbering
# an explicit choice.
ensure_xray_decision() {
  if [ -n "${BADGERS_SKIP_XRAY:-}" ]; then
    return 0
  fi
  preflight_xray
}

preflight_xray() {
  local status count quota

  # Preserve an explicit skip. Falling through would reset it to 0 below and then run
  # quota checks the caller has already opted out of.
  if [ "${BADGERS_SKIP_XRAY:-}" = "1" ]; then
    log_warn "BADGERS_SKIP_XRAY=1 — skipping $(_sn XRay) (tracing will not be enabled)"
    return 0
  fi

  status="$(xray_transaction_search_status)"

  if [ "${status}" = "ACTIVE" ]; then
    log_success "X-Ray Transaction Search is already ACTIVE in ${AWS_REGION} — skipping $(_sn XRay)"
    log_info "  Nothing to enable, and skipping it avoids consuming a CloudWatch Logs"
    log_info "  resource policy slot for a setting that is already on."
    export BADGERS_SKIP_XRAY=1
    return 0
  fi

  export BADGERS_SKIP_XRAY=0
  count="$(logs_resource_policy_count)"
  quota="$(logs_resource_policy_quota)"

  # Could not read the account state — almost always expired credentials or a missing
  # logs:DescribeResourcePolicies / xray:GetTraceSegmentDestination permission. Say so
  # rather than assuming there is room, which would just move the failure into
  # CloudFormation after other stacks are already created.
  if [ "${status}" = "UNAVAILABLE" ] || [ -z "${count}" ]; then
    echo ""
    log_error "Preflight could not determine the X-Ray / CloudWatch Logs state in ${AWS_REGION}."
    log_error "  Transaction Search status : ${status}"
    log_error "  Resource policy count     : ${count:-<unreadable>}"
    echo ""
    log_error "Usually expired credentials, or the caller lacks:"
    log_error "      logs:DescribeResourcePolicies"
    log_error "      xray:GetTraceSegmentDestination"
    echo ""
    log_error "Verify with:  aws sts get-caller-identity"
    log_error "Then re-run. To deploy without tracing and skip this check entirely:"
    log_error "      BADGERS_SKIP_XRAY=1 ./deploy.sh 2"
    echo ""
    return 1
  fi

  # BADGERS' X-Ray stack creates exactly one resource policy.
  local projected=$((count + 1))
  log_info "Transaction Search status: ${status}"
  log_info "CloudWatch Logs resource policies: ${count}/${quota} (BADGERS would make it ${projected})"

  if [ "${count}" -ge "${quota}" ]; then
    echo ""
    log_error "Cannot enable X-Ray Transaction Search: the CloudWatch Logs resource"
    log_error "policy quota for ${AWS_REGION} is already full (${count}/${quota})."
    echo ""
    log_error "This quota is NOT adjustable — a support request will not raise it."
    echo ""
    logs_resource_policy_report
    echo ""
    log_error "These are account-and-region scoped, attach to nothing, and do not appear"
    log_error "in the console. 'aws logs describe-resource-policies' is the only way to"
    log_error "review them."
    echo ""
    log_error "Review the above, free a slot in this account, then re-run:"
    log_error "      ./deploy.sh 2"
    echo ""
    log_error "Or deploy without tracing, which needs no policy at all:"
    log_error "      BADGERS_SKIP_XRAY=1 ./deploy.sh 2"
    echo ""
    return 1

  elif [ "${projected}" -ge "${quota}" ]; then
    # Not blocking, but this deploy consumes the last slot, and the next project to
    # need one will be the one that fails. Worth knowing before it happens.
    echo ""
    log_warn "BADGERS is about to take the LAST CloudWatch Logs resource policy slot"
    log_warn "in ${AWS_REGION} (${count}/${quota} used, ${projected}/${quota} after this deploy)."
    echo ""
    logs_resource_policy_report
    echo ""
    log_warn "The quota is not adjustable. After this, the next thing in this account and"
    log_warn "region that needs a resource policy will fail — possibly something you did"
    log_warn "not deploy."
    echo ""
    log_warn "To leave the slot free, deploy without tracing instead:"
    log_warn "      BADGERS_SKIP_XRAY=1 ./deploy.sh 2"
    echo ""
    _confirm "Continue and use the last slot? (y/n): " || {
      log_info "Aborted at your request. Nothing was deployed."
      return 1
    }
  fi

  if [ "${projected}" -ge "${quota}" ]; then
    log_warn "Proceeding — $(_sn XRay) will take the last slot (${projected}/${quota})."
  else
    log_info "Headroom available (${projected}/${quota} after deploy) — $(_sn XRay) will be deployed."
  fi
  return 0
}

ecr_login() {
  ensure_account
  log_info "Logging into ECR..."
  aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" 2>/dev/null
}
