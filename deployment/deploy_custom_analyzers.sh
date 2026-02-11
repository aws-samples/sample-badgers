#!/bin/bash
#
# Deploy custom analyzers stack
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
cd "$SCRIPT_DIR"

STACK_PREFIX="badgers"

echo ""
echo "=========================================="
echo "  Custom Analyzers Deployment"
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
    --stack-name ${STACK_PREFIX}-s3 \
    --query "Stacks[0].Tags[?Key=='deployment_id'].Value" \
    --output text 2>/dev/null || echo "")

if [ -z "$DEPLOYMENT_ID" ] || [ "$DEPLOYMENT_ID" == "None" ]; then
    # Try to extract from bucket name
    CONFIG_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-s3 \
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

# Check if there are analyzers to deploy
if [ -f "custom_analyzers/analyzer_registry.json" ]; then
    ANALYZER_COUNT=$(jq '.analyzers | length' custom_analyzers/analyzer_registry.json 2>/dev/null || echo "0")
    if [ "$ANALYZER_COUNT" -eq 0 ]; then
        log_warn "No custom analyzers found in registry. Nothing to deploy."
        exit 0
    fi
    log_info "Found $ANALYZER_COUNT custom analyzer(s) to deploy"
else
    log_warn "No analyzer registry found. Create an analyzer via the wizard first."
    exit 0
fi

# Deploy custom analyzers stack (exclusively - don't update dependencies)
log_info "Deploying custom analyzers stack..."
$_CDK_CMD deploy ${STACK_PREFIX}-custom-analyzers $_CDK_CONTEXT --require-approval never --exclusively || handle_error "Deploy custom analyzers stack"

echo ""
echo "=========================================="
echo "  Custom Analyzers Deployed!"
echo "=========================================="
echo ""
log_success "Custom analyzers stack deployed successfully"

# List deployed analyzers
log_info "Deployed analyzers:"
jq -r '.analyzers[].name' custom_analyzers/analyzer_registry.json 2>/dev/null | while read name; do
    echo "  - $name"
done
echo ""
