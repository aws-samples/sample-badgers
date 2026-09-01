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

# ── Docker cross-platform build check ─────────────────────────────────────────
# Ensures QEMU binfmt is registered when building for a different architecture
# than the host. On WSL/Linux, also ensures qemu-user-static is installed
# (provides the actual interpreter binaries on the host filesystem).
# Accepts a target platform argument:
#   preflight_docker_cross_platform arm64   (for runtime image)
#   preflight_docker_cross_platform amd64   (for UI image)
# If no argument, defaults to arm64 (runtime).
preflight_docker_cross_platform() {
  local target_arch="${1:-arm64}"
  local host_arch
  host_arch="$(uname -m)"

  # Normalize host arch names
  local host_normalized
  case "${host_arch}" in
    x86_64|amd64)   host_normalized="amd64" ;;
    aarch64|arm64)   host_normalized="arm64" ;;
    *)               host_normalized="${host_arch}" ;;
  esac

  # If host matches target, no emulation needed
  if [[ "${host_normalized}" == "${target_arch}" ]]; then
    log_info "Docker platform check: native ${target_arch} host — no emulation needed"
    return 0
  fi

  log_info "Docker platform check: host is ${host_normalized} (${host_arch}), target is linux/${target_arch}"

  # Check if Docker is available at all
  if ! command -v docker &>/dev/null; then
    log_error "Docker not found in PATH. Install Docker Desktop or Docker Engine first."
    return 1
  fi

  # On WSL/Linux: ensure qemu-user-static is installed (provides the interpreter
  # binaries that binfmt_misc points to). Without this, binfmt registers but
  # execution fails with "No such file or directory" for /bin/sh.
  if [[ "$(uname -s)" == "Linux" ]] && ! command -v qemu-aarch64-static &>/dev/null && [[ "${target_arch}" == "arm64" ]]; then
    log_warn "qemu-user-static not found — installing (provides /usr/bin/qemu-aarch64-static)..."
    sudo apt-get update -qq && sudo apt-get install -y -qq qemu-user-static \
      || { log_error "Failed to install qemu-user-static."; \
           log_error "Run manually: sudo apt-get install qemu-user-static"; return 1; }
  elif [[ "$(uname -s)" == "Linux" ]] && ! command -v qemu-x86_64-static &>/dev/null && [[ "${target_arch}" == "amd64" ]]; then
    log_warn "qemu-user-static not found — installing (provides /usr/bin/qemu-x86_64-static)..."
    sudo apt-get update -qq && sudo apt-get install -y -qq qemu-user-static \
      || { log_error "Failed to install qemu-user-static."; \
           log_error "Run manually: sudo apt-get install qemu-user-static"; return 1; }
  fi

  # Test if target platform emulation works
  if docker run --rm --platform "linux/${target_arch}" public.ecr.aws/docker/library/alpine:3 true 2>/dev/null; then
    log_success "${target_arch} emulation: OK (QEMU already registered)"
    return 0
  fi

  # Not registered — attempt auto-install
  log_warn "${target_arch} emulation not available — registering QEMU binfmt handlers..."
  if docker run --privileged --rm tonistiigi/binfmt --install "${target_arch}"; then
    log_success "QEMU ${target_arch} emulation registered successfully."
    return 0
  else
    log_error "Failed to register QEMU binfmt for ${target_arch}."
    log_error "Try manually: docker run --privileged --rm tonistiigi/binfmt --install ${target_arch}"
    log_error "On Docker Desktop: ensure 'Use Rosetta / QEMU' is enabled in Settings → General."
    return 1
  fi
}

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

# ── Windows VDI: strip carriage returns from tool output ─────────────────────
# These scripts may be run from a Windows VDI under Git Bash. The native Windows
# builds of python and the aws CLI open stdout in text mode, so every "\n" they
# print goes out as "\r\n". Bash command substitution "$(…)" strips a trailing
# "\n" but NOT the "\r", so captured values silently gain a trailing carriage
# return. That stray "\r" breaks string comparisons ("True\r" != "True"),
# JMESPath equality filters (modelId=='...\r' matches nothing), JSON, and
# resource names built from the value.
#
# Route both tools through a wrapper that removes CRs so every caller gets clean
# output, instead of sprinkling `tr -d '\r'` on each capture site (and missing new
# ones). `command` avoids recursing back into these wrappers. All scripts here run
# with `set -o pipefail`, so the pipeline still reports the tool's own exit status
# and `|| echo default` style fallbacks keep working.
#
# Safe here because no `aws`/`python3` call streams binary to stdout; the only
# raw-token case (`aws ecr get-login-password`) benefits from CR removal too.
aws()     { command aws     "$@" | tr -d '\r'; }

# python3 wrapper: use python3 if available, else python (Windows Git Bash).
# Also strips \r from output (Git Bash / Windows line endings).
if command -v python3 &>/dev/null; then
  _PYTHON_CMD=python3
elif command -v python &>/dev/null; then
  _PYTHON_CMD=python
else
  echo "ERROR: Neither python3 nor python found in PATH." >&2
  exit 1
fi
python3() { command ${_PYTHON_CMD} "$@" | tr -d '\r'; }

# ── Paths ──────────────────────────────────────────────────────────────────
# This file lives at <repo>/deployment/scripts/common.sh
BADGERS_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(cd "${BADGERS_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOYMENT_DIR}/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────
# Capture a caller-pinned region before applying any fallback, so ensure_region()
# can tell an explicit choice from an unset variable. AWS_REGION stays populated
# either way: scripts that source this file without calling ensure_region() still
# need a usable value.
_AWS_REGION_FROM_ENV="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
BADGERS_DEFAULT_REGION="us-west-2"
AWS_REGION="${_AWS_REGION_FROM_ENV:-${BADGERS_DEFAULT_REGION}}"
STACK_PREFIX="${STACK_PREFIX:-BADGERS}"

# ── Deployment identity ────────────────────────────────────────────────────
require_deployment_id() {
  if [ -z "${DEPLOYMENT_ID:-}" ]; then
    log_error "DEPLOYMENT_ID is not set. ./deploy.sh prompts for it; other scripts take it"
    log_error "explicitly, e.g. DEPLOYMENT_ID=dev ./destroy.sh"
    exit 1
  fi
}

# Convert MSYS/Git Bash paths (/c/...) to Windows-compatible paths (C:/...) for
# Python subprocesses. On Linux/macOS this is a no-op.
_py_path() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) cygpath -m "$1" ;;
    *) echo "$1" ;;
  esac
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
with open('$(_py_path "${f}")') as fh:
    d=json.load(fh)
print(d.get(sys.argv[1],''))" "$1"
}

set_state() {
  local f key val
  f="$(state_file)"
  key="$1"; val="$2"
  python3 -c "import json
f='$(_py_path "${f}")'
with open(f, 'r+') as fh:
    data = json.load(fh)
    data['${key}'] = ${val}
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
    read -rp "Select which? (1-${#ids[@]}, n, or q to quit) [1]: " pick
    pick="${pick:-1}"
    case "${pick}" in
      q|Q|0) log_info "Aborted."; return 1 ;;
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
# Region resolution ladder:
#   1. AWS_REGION / AWS_DEFAULT_REGION from the environment — no prompt
#   2. the active credential profile's configured region
#   3. ask
#   4. warn that us-west-2 is the fallback, then confirm
# Must run before ensure_account, which calls STS with --region "${AWS_REGION}".

# Validate an AWS region name (e.g. us-west-2).
_valid_region() {
  [[ "$1" =~ ^[a-z]{2}(-[a-z]+)+-[0-9]$ ]]
}

ensure_region() {
  local source_desc=""

  if [ -n "${_AWS_REGION_FROM_ENV}" ]; then
    AWS_REGION="${_AWS_REGION_FROM_ENV}"
    source_desc="environment"
  else
    # Bare `aws configure get region` already honours an exported AWS_PROFILE.
    local cred_region
    cred_region="$(aws configure get region 2>/dev/null || true)"
    if [ -n "${cred_region}" ]; then
      AWS_REGION="${cred_region}"
      source_desc="AWS profile ${AWS_PROFILE:-default}"
    fi
  fi

  if [ -z "${source_desc}" ]; then
    local can_prompt=true
    [ -t 0 ] || can_prompt=false

    local reply=""
    if [ "${can_prompt}" = true ]; then
      read -rp "  AWS region to deploy into (Enter if you are not sure): " reply || reply=""
      reply="${reply//$'\r'/}"
    else
      log_warn "No region in the environment or AWS profile."
    fi

    if [ -n "${reply}" ]; then
      AWS_REGION="${reply}"
      source_desc="entered"
    else
      AWS_REGION="${BADGERS_DEFAULT_REGION}"
      log_warn "Defaulting to ${AWS_REGION}."
      log_warn "Model availability varies by region; ${AWS_REGION} is where this stack is tested."
      if [ "${can_prompt}" = true ]; then
        _confirm "  Deploy into ${AWS_REGION}? (y/n): " || {
          log_error "Cancelled. Pin a region explicitly: AWS_REGION=<region> $0"
          exit 1
        }
      fi
      source_desc="default"
    fi
  fi

  if ! _valid_region "${AWS_REGION}"; then
    log_error "'${AWS_REGION}' is not a valid AWS region name (expected e.g. us-west-2)."
    exit 1
  fi

  export AWS_REGION
  log_info "Region ${AWS_REGION} (${source_desc})."
}

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

# ── CDK Bootstrap Preflight ──────────────────────────────────────────────────
# CDK requires a bootstrap stack (CDKToolkit) in the target account/region before
# any deployment. This check runs before the first cdk deploy and gives clear
# remediation steps if the toolkit stack is missing or in a bad state.
preflight_bootstrap() {
  log_info "Checking CDK bootstrap status for ${ACCOUNT_ID}/${AWS_REGION}..."

  local status
  status="$(aws cloudformation describe-stacks --stack-name CDKToolkit \
    --region "${AWS_REGION}" --query "Stacks[0].StackStatus" \
    --output text 2>/dev/null || echo "NOT_FOUND")"

  case "${status}" in
    CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE)
      # Check bootstrap version — modern CDK needs version >= 6 (ideally latest)
      local version
      version="$(aws ssm get-parameter \
        --name "/cdk-bootstrap/${CDK_QUALIFIER:-hnb659fds}/${ACCOUNT_ID}/${AWS_REGION}" \
        --region "${AWS_REGION}" --query "Parameter.Value" \
        --output text 2>/dev/null || echo "")"
      if [ -n "${version}" ] && [ "${version}" != "None" ]; then
        log_success "CDK bootstrap v${version} found in ${AWS_REGION} — ready to deploy."
      else
        log_success "CDK bootstrap stack found (${status}) — ready to deploy."
      fi
      return 0
      ;;
    NOT_FOUND|"")
      echo ""
      log_error "CDK has not been bootstrapped in this account/region."
      echo ""
      log_error "  Account: ${ACCOUNT_ID}"
      log_error "  Region:  ${AWS_REGION}"
      echo ""
      log_error "Run the following command first (requires AdministratorAccess or equivalent):"
      echo ""
      log_error "      cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}"
      echo ""
      log_error "Or with an explicit profile:"
      log_error "      cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION} --profile <your-profile>"
      echo ""
      log_error "Then re-run ./deploy.sh"
      echo ""
      return 1
      ;;
    *)
      log_warn "CDK bootstrap stack exists but is in state: ${status}"
      log_warn "This may indicate a previous bootstrap failed. Consider re-running:"
      log_warn "      cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}"
      _confirm "Attempt deploy anyway? (y/n): " || return 1
      ;;
  esac
}

export_cdk_env() {
  ensure_account
  export DEPLOYMENT_ID
  export STACK_SUFFIX
  export STACK_PREFIX
  export CDK_DEFAULT_ACCOUNT="${ACCOUNT_ID}"
  export CDK_DEFAULT_REGION="${AWS_REGION}"
  export TYPEGUARD_DISABLE=1
  export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"
  # app.py reads RUNTIME_IMAGE_TAG to point the runtime at the image the build
  # step pushed. Default it here so CDK-only entry points stay in sync with the
  # build without every caller having to set it.
  export RUNTIME_IMAGE_TAG="${RUNTIME_IMAGE_TAG:-websocket}"
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

# ── Service Quota Checks ─────────────────────────────────────────────────────
# Verifies headroom exists for all resources this deployment creates.
# Each check: current count vs quota, fail if count+needed >= quota.

_check_quota() {
  # Usage: _check_quota <service_code> <quota_code> <count> <needed> <label>
  local service="$1" quota_code="$2" count="$3" needed="$4" label="$5"
  local quota

  quota="$(aws service-quotas get-service-quota \
    --service-code "${service}" --quota-code "${quota_code}" \
    --region "${AWS_REGION}" --query 'Quota.Value' --output text 2>/dev/null || echo "")"

  # If Service Quotas unavailable, try default
  if [ -z "${quota}" ] || [ "${quota}" = "None" ]; then
    quota="$(aws service-quotas get-aws-default-service-quota \
      --service-code "${service}" --quota-code "${quota_code}" \
      --region "${AWS_REGION}" --query 'Quota.Value' --output text 2>/dev/null || echo "")"
  fi

  if [ -z "${quota}" ] || [ "${quota}" = "None" ]; then
    echo "    ? ${label}: could not determine quota (skipping)"
    return 0
  fi

  # Convert quota to int (it may be a float like 100.0)
  quota="$(printf '%.0f' "${quota}")"

  local remaining=$(( quota - count ))
  if [ "${remaining}" -lt "${needed}" ]; then
    log_error "${label}: ${count}/${quota} used, need ${needed} more — NOT ENOUGH HEADROOM"
    log_error "      Request a quota increase: https://console.aws.amazon.com/servicequotas/"
    return 1
  fi

  echo "    ✓ ${label}: ${count}/${quota} (need ${needed}, ${remaining} available)"
  return 0
}

preflight_service_quotas() {
  log_info "Checking service quotas for deployment headroom..."
  local failed=0

  # VPCs (we create 1)
  local vpc_count
  vpc_count="$(aws ec2 describe-vpcs --region "${AWS_REGION}" \
    --query 'length(Vpcs)' --output text 2>/dev/null || echo "0")"
  _check_quota "vpc" "L-F678F1CE" "${vpc_count}" 1 "VPCs per Region" || ((failed++))

  # Internet Gateways (we create 1)
  local igw_count
  igw_count="$(aws ec2 describe-internet-gateways --region "${AWS_REGION}" \
    --query 'length(InternetGateways)' --output text 2>/dev/null || echo "0")"
  _check_quota "vpc" "L-A4707A72" "${igw_count}" 1 "Internet gateways per Region" || ((failed++))

  # S3 Buckets (we create 3: config, source, output)
  local bucket_count
  bucket_count="$(aws s3api list-buckets --query 'length(Buckets)' --output text 2>/dev/null || echo "0")"
  _check_quota "s3" "L-DC2B2D3D" "${bucket_count}" 3 "S3 buckets" || ((failed++))

  # Cognito User Pools (we create 1)
  local pool_count
  pool_count="$(aws cognito-idp list-user-pools --max-results 60 --region "${AWS_REGION}" \
    --query 'length(UserPools)' --output text 2>/dev/null || echo "0")"
  _check_quota "cognito-idp" "L-F9A5A2F3" "${pool_count}" 1 "Cognito user pools" || ((failed++))

  # DynamoDB Tables (we create 1: jobs)
  local table_count
  table_count="$(aws dynamodb list-tables --region "${AWS_REGION}" \
    --query 'length(TableNames)' --output text 2>/dev/null || echo "0")"
  _check_quota "dynamodb" "L-F98FE922" "${table_count}" 1 "DynamoDB tables" || ((failed++))

  # ECR Repositories (we create 1)
  local ecr_count
  ecr_count="$(aws ecr describe-repositories --region "${AWS_REGION}" \
    --query 'length(repositories)' --output text 2>/dev/null || echo "0")"
  _check_quota "ecr" "L-CFEB8E8D" "${ecr_count}" 1 "ECR repositories" || ((failed++))

  # Lambda Functions (we create ~26 specialist functions)
  local lambda_count
  lambda_count="$(aws lambda list-functions --region "${AWS_REGION}" \
    --query 'length(Functions)' --output text 2>/dev/null || echo "0")"
  _check_quota "lambda" "L-B99A9384" "${lambda_count}" 26 "Lambda functions" || ((failed++))

  # IAM Roles (we create ~5: lambda execution, ecs task, CDK roles)
  local role_count
  role_count="$(aws iam list-roles --query 'length(Roles)' --output text 2>/dev/null || echo "0")"
  _check_quota "iam" "L-FE177D64" "${role_count}" 5 "IAM roles" || ((failed++))

  # KMS Keys (we create 1)
  local kms_count
  kms_count="$(aws kms list-keys --region "${AWS_REGION}" \
    --query 'length(Keys)' --output text 2>/dev/null || echo "0")"
  _check_quota "kms" "L-C2F1777E" "${kms_count}" 1 "KMS keys" || ((failed++))

  if [ "${failed}" -gt 0 ]; then
    log_error "${failed} quota check(s) failed — cannot proceed."
    return 1
  fi

  log_success "Service quotas OK — headroom confirmed for all resources"
  return 0
}

# ── Preflight: Bedrock Model Access ──────────────────────────────────────────
# Verifies that the default models used by BADGERS are available in the target
# region. Models with a "us." prefix are cross-region inference profiles and
# always pass.
#
# BADGERS uses these models by default (configurable via inference profiles):
BADGERS_DEFAULT_MODELS=(
  "us.anthropic.claude-sonnet-4-5-20250514-v1:0"
)

preflight_model_access() {
  log_info "Checking Bedrock model access in ${AWS_REGION}..."

  local missing=() model_id status

  for model_id in "${BADGERS_DEFAULT_MODELS[@]}"; do
    # Cross-region inference profiles (us.*, eu.*) are always available.
    if [[ "${model_id}" == us.* ]] || [[ "${model_id}" == eu.* ]]; then
      echo "    ✓ ${model_id} (cross-region)"
      continue
    fi

    # In-region model — verify it exists in the target region.
    status="$(aws bedrock list-foundation-models --region "${AWS_REGION}" \
      --query "modelSummaries[?modelId=='${model_id}'].modelLifecycle.status | [0]" \
      --output text 2>/dev/null | tr -d '\r' || echo "NOT_FOUND")"
    status="${status%%[[:space:]]}"  # strip trailing whitespace/CR

    if [ "${status}" = "ACTIVE" ] || [ "${status}" = "LEGACY" ]; then
      echo "    ✓ ${model_id}"
    else
      echo "    ✗ ${model_id} (not found in ${AWS_REGION})"
      missing+=("${model_id}")
    fi
  done

  if [ ${#missing[@]} -gt 0 ]; then
    echo ""
    log_error "The following models are NOT available in ${AWS_REGION}:"
    for m in "${missing[@]}"; do
      log_error "    ${m}"
    done
    echo ""
    log_error "Options:"
    log_error "  1. Deploy to a region where these models are available"
    log_error "  2. Use cross-region inference profiles (us.anthropic.* prefix)"
    log_error ""
    log_error "Model catalog: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html"
    return 1
  fi

  log_success "All configured models are accessible in ${AWS_REGION}"
  return 0
}

# ── Model Invocation Test ────────────────────────────────────────────────────
# Actually invokes each model with a minimal payload to confirm marketplace
# subscriptions, inference profiles, and IAM permissions are all in place.
preflight_model_invocation() {
  log_info "Testing model invocations (this may take 10-20 seconds)..."
  local failed=0

  local claude_body='{"anthropic_version":"bedrock-2023-05-31","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'

  for model_id in "${BADGERS_DEFAULT_MODELS[@]}"; do
    local response
    response="$(aws bedrock-runtime invoke-model \
      --model-id "${model_id}" \
      --region "${AWS_REGION}" \
      --content-type "application/json" \
      --body "${claude_body}" \
      --output json \
      /dev/null 2>&1)" || true

    if echo "${response}" | grep -qi "AccessDeniedException\|ValidationException\|ResourceNotFoundException\|ServiceUnavailableException"; then
      local err_msg
      err_msg="$(echo "${response}" | grep -o '"[Mm]essage":"[^"]*"' | head -1 || echo "${response}")"
      echo "    ✗ ${model_id}"
      echo "      Error: ${err_msg}"
      ((failed++))
    else
      echo "    ✓ ${model_id}"
    fi
  done

  if [ "${failed}" -gt 0 ]; then
    log_error "${failed} model(s) failed invocation test."
    log_error "Common fixes:"
    log_error "  - Marketplace models (Anthropic): Subscribe at https://console.aws.amazon.com/bedrock/home#/modelaccess"
    log_error "  - IAM: Ensure your deploy role has bedrock:InvokeModel on the model ARN"
    log_error "  - Region: Model may not be available in ${AWS_REGION}"
    return 1
  fi

  log_success "All models respond to invocation"
  return 0
}

# ── ECS Service-Linked Role ──────────────────────────────────────────────────
# Ensure the ECS service-linked role exists, creating it only if genuinely absent.
preflight_ecs_slr() {
  log_info "Checking ECS service-linked role..."

  local slr_arn get_err get_rc
  get_err="$(aws iam get-role \
    --role-name AWSServiceRoleForECS \
    --query 'Role.Arn' --output text 2>&1 >/dev/null)"
  get_rc=$?
  slr_arn="$(aws iam get-role \
    --role-name AWSServiceRoleForECS \
    --query 'Role.Arn' --output text 2>/dev/null)"

  if [ "${get_rc}" -eq 0 ] && [ -n "${slr_arn}" ] && [ "${slr_arn}" != "None" ]; then
    log_info "  ✓ ECS service-linked role exists: ${slr_arn}"
    return 0
  fi

  # get-role failed. Only "NoSuchEntity" means it truly isn't there — anything
  # else (AccessDenied, wrong account/profile, expired creds) means we can't
  # see it, and creating would be wrong.
  if ! printf '%s' "${get_err}" | grep -qi 'NoSuchEntity'; then
    log_error "  Cannot verify ECS service-linked role — get-role did not return NoSuchEntity."
    log_error "  AWS error: ${get_err:-<none>}"
    log_error "  Confirm the CLI is using the same account/region as the console:"
    aws sts get-caller-identity \
      --query '{Account:Account,Arn:Arn}' --output text 2>&1 | sed 's/^/    /'
    log_error "  Region: ${AWS_REGION:-$(aws configure get region 2>/dev/null)}"
    return 1
  fi

  log_warn "  ECS service-linked role not found — creating..."
  local create_err
  create_err="$(aws iam create-service-linked-role \
    --aws-service-name ecs.amazonaws.com 2>&1 >/dev/null)"
  if [ $? -eq 0 ]; then
    log_success "  Created ECS service-linked role."
    return 0
  fi

  # Create failed. "already exists" is benign (race, or it was there all along).
  if printf '%s' "${create_err}" | grep -qiE 'has been taken|InvalidInput|EntityAlreadyExists'; then
    log_info "  ✓ ECS service-linked role already exists (confirmed)."
    return 0
  fi

  log_error "  Failed to create ECS service-linked role."
  log_error "  AWS error: ${create_err:-<none>}"
  log_error "  Retry manually: aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com"
  return 1
}

ecr_login() {
  ensure_account
  log_info "Logging into ECR..."
  aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" 2>/dev/null
}
