#!/bin/bash
#
# Destroy BADGERS frontend infrastructure (frontend + VPC stacks)
# Destroys in reverse dependency order: frontend first, then VPC
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STACK_PREFIX="badgers"

echo ""
echo "=========================================="
echo "  BADGERS — DESTROY FRONTEND"
echo "=========================================="
echo ""
log_warn "This will DELETE the frontend and VPC stacks!"
echo ""

read -rp "Are you sure you want to destroy frontend stacks? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Check for uv
if command -v uv &> /dev/null; then
    _CDK_CMD="uv run cdk"
else
    _CDK_CMD="cdk"
fi

# Activate venv if present
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

export TYPEGUARD_DISABLE=1
export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"

# Destroy in reverse dependency order:
#   frontend -> vpc, cognito, ecr
#   vpc -> (none)
STACKS=(
    "${STACK_PREFIX}-frontend"
    "${STACK_PREFIX}-vpc"
)

for STACK in "${STACKS[@]}"; do
    log_info "Destroying $STACK..."
    if aws cloudformation describe-stacks --stack-name "$STACK" &>/dev/null; then
        $_CDK_CMD destroy "$STACK" --force --exclusively || log_warn "Failed to destroy $STACK, continuing..."
        log_success "$STACK destroyed"
    else
        log_warn "$STACK not found, skipping"
    fi
done

echo ""
echo "=========================================="
log_success "Frontend destroy complete!"
echo "=========================================="
echo ""
