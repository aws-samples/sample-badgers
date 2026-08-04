#!/bin/bash
#
# Deploy custom specialists stack
# Syncs from S3 and deploys via CDK
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }

handle_error() {
    log_error "Deployment failed: $1"
    exit 1
}

# Turn off TypeGuard Checks
export TYPEGUARD_DISABLE=1
export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/common.sh"
load_suffix
cd "$SCRIPT_DIR"



echo ""
echo "=========================================="
echo "  Custom Specialists Deployment"
echo "=========================================="
echo ""

# Check for uv
if command -v uv &> /dev/null; then
    _CDK_CMD="uv run cdk"
else
    _CDK_CMD="cdk"
fi

# Get deployment ID from existing stack
log_info "Getting deployment ID from existing stacks..."
DEPLOYMENT_ID=$(aws cloudformation describe-stacks \
    --stack-name "$(_sn S3)" \
    --query "Stacks[0].Tags[?Key=='deployment_id'].Value" \
    --output text 2>/dev/null || echo "")

if [ -z "$DEPLOYMENT_ID" ] || [ "$DEPLOYMENT_ID" == "None" ]; then
    # Try to extract from bucket name
    CONFIG_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$(_sn S3)" \
        --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
        --output text 2>/dev/null || echo "")
    if [ -n "$CONFIG_BUCKET" ]; then
        # Extract ID from bucket name (badgers-config-XXXXXXXX)
        DEPLOYMENT_ID=$(echo "$CONFIG_BUCKET" | sed 's/badgers-config-//')
    fi
fi

if [ -z "$DEPLOYMENT_ID" ]; then
    handle_error "Could not determine deployment ID. Is the base stack deployed?"
fi

log_info "Using deployment ID: $DEPLOYMENT_ID"
_CDK_CONTEXT="-c deployment_id=$DEPLOYMENT_ID"

# Check if there are specialists to deploy
if [ -f "custom_specialists/specialist_registry.json" ]; then
    SPECIALIST_COUNT=$(jq '.specialists | length' custom_specialists/specialist_registry.json 2>/dev/null || echo "0")
    if [ "$SPECIALIST_COUNT" -eq 0 ]; then
        log_warn "No custom specialists found in registry. Nothing to deploy."
        exit 0
    fi
    log_info "Found $SPECIALIST_COUNT custom specialist(s) to deploy"
else
    log_warn "No specialist registry found. Create an specialist via the wizard first."
    exit 0
fi

# Deploy custom specialists stack (exclusively - don't update dependencies)
log_info "Deploying custom specialists stack..."
$_CDK_CMD deploy "$(_sn CustomSpecialists)" $_CDK_CONTEXT --require-approval never --exclusively || handle_error "Deploy custom specialists stack"

echo ""
echo "=========================================="
echo "  Custom Specialists Deployed!"
echo "=========================================="
echo ""
log_success "Custom specialists stack deployed successfully"

# List deployed specialists
log_info "Deployed specialists:"
jq -r '.specialists[].name' custom_specialists/specialist_registry.json 2>/dev/null | while read name; do
    echo "  - $name"
done
echo ""
