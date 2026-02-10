#!/bin/bash
#
# Destroy all deployed resources for PDF Extractor POC
# Destroys stacks in reverse dependency order
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
echo "  PDF Extractor POC - DESTROY"
echo "=========================================="
echo ""
log_warn "This will DELETE all deployed resources!"
echo ""

read -rp "Are you sure you want to destroy all stacks? (yes/no): " CONFIRM
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

# Destroy in reverse dependency order
# Dependency graph:
#   custom-analyzers -> gateway, lambda, iam, s3 (via Fn.import_value)
#   runtime-websocket -> ecr, gateway, cognito, memory, inference-profiles
#   gateway -> lambda, cognito
#   lambda -> ecr, inference-profiles, iam, s3
#   iam -> s3
STACKS=(
    "${STACK_PREFIX}-custom-analyzers"
    "${STACK_PREFIX}-runtime-websocket"
    "${STACK_PREFIX}-gateway"
    "${STACK_PREFIX}-lambda"
    "${STACK_PREFIX}-memory"
    "${STACK_PREFIX}-ecr"
    "${STACK_PREFIX}-inference-profiles"
    "${STACK_PREFIX}-iam"
    "${STACK_PREFIX}-cognito"
    "${STACK_PREFIX}-s3"
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
log_success "Destroy complete!"
echo "=========================================="
echo ""
