#!/bin/bash
#
# Deploy a single analyzer without redeploying the entire stack.
#
# Scans lambdas/code/ for analyzer directories, presents an interactive
# picker, then runs: S3 sync → Lambda stack → Gateway stack.
#
# Usage: ./deploy_analyzer.sh [--all]
#   --all   Skip picker, deploy all stacks (S3 + Lambda + Gateway)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STACK_PREFIX="badgers"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; }

handle_error() { log_error "$1"; exit 1; }

# ── Prerequisites ──────────────────────────────────────────────────

export TYPEGUARD_DISABLE=1
export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"

if command -v uv &> /dev/null; then
    _CDK_CMD="uv run cdk"
else
    _CDK_CMD="cdk"
fi

# ── Collect analyzers ──────────────────────────────────────────────

LAMBDAS_DIR="$SCRIPT_DIR/lambdas/code"
if [ ! -d "$LAMBDAS_DIR" ]; then
    handle_error "lambdas/code directory not found"
fi

ANALYZERS=()
while IFS= read -r dir; do
    ANALYZERS+=("$(basename "$dir")")
done < <(find "$LAMBDAS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [ ${#ANALYZERS[@]} -eq 0 ]; then
    handle_error "No analyzers found in lambdas/code/"
fi

# ── Interactive picker ─────────────────────────────────────────────

pick_analyzer() {
    local selected=0
    local count=${#ANALYZERS[@]}

    # Hide cursor
    tput civis 2>/dev/null || true
    trap 'tput cnorm 2>/dev/null; exit' EXIT INT TERM

    while true; do
        # Move cursor to top of list (clear previous render)
        if [ "$first_render" != "true" ]; then
            tput cuu "$count" 2>/dev/null || true
        fi
        first_render=false

        # Render list
        for i in "${!ANALYZERS[@]}"; do
            local name="${ANALYZERS[$i]}"
            local has_manifest=" "
            local has_schema=" "
            local has_prompts=" "

            [ -f "$SCRIPT_DIR/s3_files/manifests/${name}.json" ] && has_manifest="${GREEN}✓${NC}"
            [ -f "$SCRIPT_DIR/s3_files/schemas/${name}.json" ] && has_schema="${GREEN}✓${NC}"
            [ -d "$SCRIPT_DIR/s3_files/prompts/${name}" ] && has_prompts="${GREEN}✓${NC}"

            if [ "$i" -eq "$selected" ]; then
                echo -e "  ${CYAN}▸${NC} ${BOLD}${name}${NC}  ${DIM}[manifest:${NC}${has_manifest}${DIM} schema:${NC}${has_schema}${DIM} prompts:${NC}${has_prompts}${DIM}]${NC}  "
            else
                echo -e "    ${DIM}${name}${NC}  ${DIM}[manifest:${NC}${has_manifest}${DIM} schema:${NC}${has_schema}${DIM} prompts:${NC}${has_prompts}${DIM}]${NC}  "
            fi
        done

        # Read keypress
        IFS= read -rsn1 key
        case "$key" in
            $'\x1b')
                read -rsn2 rest
                case "$rest" in
                    '[A') # Up arrow
                        ((selected > 0)) && ((selected--))
                        ;;
                    '[B') # Down arrow
                        ((selected < count - 1)) && ((selected++))
                        ;;
                esac
                ;;
            '') # Enter
                tput cnorm 2>/dev/null || true
                SELECTED_ANALYZER="${ANALYZERS[$selected]}"
                return
                ;;
            'q'|'Q')
                tput cnorm 2>/dev/null || true
                echo ""
                log_info "Cancelled."
                exit 0
                ;;
        esac
    done
}

# ── Validation ─────────────────────────────────────────────────────

validate_analyzer() {
    local name="$1"
    local missing=()

    [ ! -f "$SCRIPT_DIR/s3_files/manifests/${name}.json" ] && missing+=("manifest")
    [ ! -f "$SCRIPT_DIR/s3_files/schemas/${name}.json" ] && missing+=("schema")
    [ ! -d "$SCRIPT_DIR/s3_files/prompts/${name}" ] && missing+=("prompts")
    [ ! -f "$SCRIPT_DIR/lambdas/code/${name}/lambda_handler.py" ] && missing+=("lambda_handler.py")

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Analyzer '${name}' is missing: ${missing[*]}"
        echo ""
        echo "  Required files:"
        echo "    s3_files/manifests/${name}.json"
        echo "    s3_files/schemas/${name}.json"
        echo "    s3_files/prompts/${name}/  (directory with prompt XMLs)"
        echo "    lambdas/code/${name}/lambda_handler.py"
        echo ""
        return 1
    fi
    return 0
}

# ── Deploy ─────────────────────────────────────────────────────────

deploy_analyzer() {
    local name="$1"

    # Get deployment ID
    log_info "Fetching deployment ID..."
    DEPLOYMENT_ID=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-s3 \
        --query "Stacks[0].Tags[?Key=='deployment_id'].Value" \
        --output text 2>/dev/null || echo "")

    if [ -z "$DEPLOYMENT_ID" ] || [ "$DEPLOYMENT_ID" == "None" ]; then
        # Fallback: extract from bucket name
        CONFIG_BUCKET=$(aws cloudformation describe-stacks \
            --stack-name ${STACK_PREFIX}-s3 \
            --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
            --output text 2>/dev/null || echo "")
        if [ -n "$CONFIG_BUCKET" ]; then
            DEPLOYMENT_ID=$(echo "$CONFIG_BUCKET" | sed 's/badgers-config-//')
        fi
    fi

    if [ -z "$DEPLOYMENT_ID" ]; then
        handle_error "Could not determine deployment ID. Is the base stack deployed?"
    fi

    _CDK_CONTEXT="-c deployment_id=$DEPLOYMENT_ID"
    log_info "Deployment ID: ${BOLD}$DEPLOYMENT_ID${NC}"
    echo ""

    # Step 1: Sync S3 files
    log_info "Step 1/3: Syncing S3 files (prompts, manifest, schema)..."
    CONFIG_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-s3 \
        --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
        --output text)

    if [ -z "$CONFIG_BUCKET" ] || [ "$CONFIG_BUCKET" == "None" ]; then
        handle_error "Could not get config bucket name"
    fi

    # Upload only the files for this analyzer + shared prompts
    aws s3 sync "$SCRIPT_DIR/s3_files/manifests/" "s3://$CONFIG_BUCKET/manifests/" \
        --exclude "*" --include "${name}.json" --quiet
    aws s3 sync "$SCRIPT_DIR/s3_files/schemas/" "s3://$CONFIG_BUCKET/schemas/" \
        --exclude "*" --include "${name}.json" --quiet
    aws s3 sync "$SCRIPT_DIR/s3_files/prompts/${name}/" "s3://$CONFIG_BUCKET/prompts/${name}/" --quiet
    aws s3 sync "$SCRIPT_DIR/s3_files/prompts/shared/" "s3://$CONFIG_BUCKET/prompts/shared/" --quiet
    log_success "S3 files uploaded to s3://$CONFIG_BUCKET"

    # Step 2: Deploy Lambda stack
    log_info "Step 2/3: Deploying Lambda stack (creates function if new)..."
    if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    $_CDK_CMD deploy ${STACK_PREFIX}-lambda $_CDK_CONTEXT --require-approval never --exclusively \
        || handle_error "Deploy Lambda stack"
    log_success "Lambda stack deployed"

    # Step 3: Deploy Gateway stack
    log_info "Step 3/3: Deploying Gateway stack (wires target)..."
    $_CDK_CMD deploy ${STACK_PREFIX}-gateway $_CDK_CONTEXT --require-approval never --exclusively \
        || handle_error "Deploy Gateway stack"
    log_success "Gateway stack deployed"

    echo ""
    echo -e "${GREEN}=========================================="
    echo -e "  ${name} deployed!"
    echo -e "==========================================${NC}"
    echo ""
}

# ── Main ───────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}  Deploy Single Analyzer${NC}"
echo -e "  ${DIM}Select an analyzer to deploy (↑/↓ to move, Enter to select, q to quit)${NC}"
echo ""

first_render=true
pick_analyzer

echo ""
log_info "Selected: ${BOLD}${SELECTED_ANALYZER}${NC}"
echo ""

# Validate
if ! validate_analyzer "$SELECTED_ANALYZER"; then
    exit 1
fi

# Confirm
echo -e "  This will:"
echo -e "    1. Upload S3 files (prompts, manifest, schema)"
echo -e "    2. Deploy ${BOLD}badgers-lambda${NC} stack"
echo -e "    3. Deploy ${BOLD}badgers-gateway${NC} stack"
echo ""
read -p "  Proceed? (Y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Nn]$ ]]; then
    log_info "Cancelled."
    exit 0
fi

echo ""
deploy_analyzer "$SELECTED_ANALYZER"
