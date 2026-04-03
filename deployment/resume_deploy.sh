#!/bin/bash
#
# Resume BADGERS deployment from any step
#
# Usage: ./resume_deploy.sh <deployment_id> [step_number]
#   If step_number is omitted, you'll be prompted to choose.
#
# Example: ./resume_deploy.sh 5bdb4f33
#          ./resume_deploy.sh 5bdb4f33 7
#

set -e

DEPLOYMENT_ID="${1:-}"
START_STEP="${2:-}"

if [ -z "$DEPLOYMENT_ID" ]; then
    echo "Usage: $0 <deployment_id> [step_number]"
    exit 1
fi

STACK_PREFIX="badgers"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDK_DIR="$SCRIPT_DIR"

export TYPEGUARD_DISABLE=1
export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"

if command -v uv &> /dev/null; then
    _CDK_CMD="uv run cdk"
else
    _CDK_CMD="cdk"
fi

_CDK_CONTEXT="-c deployment_id=$DEPLOYMENT_ID"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') - $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}   $(date '+%H:%M:%S') - $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') - $1"; }
handle_error() { echo -e "${RED}[FAIL]${NC} $(date '+%H:%M:%S') - $1"; exit 1; }

# Show menu if no step provided
if [ -z "$START_STEP" ]; then
    echo ""
    echo "=========================================="
    echo "  BADGERS - Resume Deploy"
    echo "  Deployment ID: $DEPLOYMENT_ID"
    echo "=========================================="
    echo ""
    echo "  1)  Install Python dependencies"
    echo "  2)  Build lambda layers"
    echo "  3)  CDK bootstrap"
    echo "  4)  Deploy S3 stack"
    echo "  5)  Upload schemas to S3"
    echo "  6)  Deploy Cognito, IAM, ECR, Memory, Inference-Profiles"
    echo "  65) Build & push container Lambda images"
    echo "  7)  Deploy Lambda stack"
    echo "  8)  Deploy Gateway stack"
    echo "  82) Deploy X-Ray Transaction Search"
    echo "  85) Configure Gateway observability"
    echo "  9)  Build & deploy WebSocket Runtime"
    echo "  10) Update frontend .env"
    echo ""
    read -p "Resume from step: " START_STEP
    echo ""
fi

if [ -z "$START_STEP" ]; then
    echo "No step selected. Exiting."
    exit 1
fi

cd "$CDK_DIR"

if [ -f "$CDK_DIR/.venv/bin/activate" ]; then
    source "$CDK_DIR/.venv/bin/activate"
fi

echo ""
log_info "Resuming from step $START_STEP with deployment ID: $DEPLOYMENT_ID"
echo ""

# --- Step functions ---

step_1() {
    log_info "Step 1: Installing Python dependencies..."
    if command -v uv &> /dev/null; then
        uv pip install -r requirements.txt || handle_error "Install dependencies"
    else
        pip install -r requirements.txt || handle_error "Install dependencies"
    fi
    log_success "Dependencies installed"
}

step_2() {
    log_info "Step 2: Building lambda layers..."
    cd "$CDK_DIR/lambdas"
    if [ -f "build_foundation_layer.sh" ]; then
        chmod +x build_foundation_layer.sh
        ./build_foundation_layer.sh || handle_error "Build foundation layer"
        log_success "Foundation layer built"
    fi
    if [ -f "build_poppler_qdf_layer.sh" ]; then
        chmod +x build_poppler_qdf_layer.sh
        ./build_poppler_qdf_layer.sh || handle_error "Build poppler layer"
        log_success "Poppler layer built"
    fi
    if [ -f "build_pdf_processing_layer.sh" ]; then
        chmod +x build_pdf_processing_layer.sh
        ./build_pdf_processing_layer.sh || handle_error "Build pdf-processing layer"
        log_success "PDF processing layer built"
    fi
    cd "$CDK_DIR"
}

step_3() {
    log_info "Step 3: CDK bootstrap..."
    $_CDK_CMD bootstrap || log_warn "Bootstrap may already exist, continuing..."
    log_success "CDK bootstrap complete"
}

step_4() {
    log_info "Step 4: Deploying S3 stack..."
    $_CDK_CMD deploy ${STACK_PREFIX}-s3 $_CDK_CONTEXT --require-approval never || handle_error "Deploy S3 stack"
    log_success "S3 stack deployed"
}

step_5() {
    log_info "Step 5: Uploading schemas to S3..."
    CONFIG_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-s3 \
        --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
        --output text)
    if [ -z "$CONFIG_BUCKET" ] || [ "$CONFIG_BUCKET" == "None" ]; then
        handle_error "Could not get config bucket name"
    fi
    aws s3 sync "$CDK_DIR/s3_files/" "s3://$CONFIG_BUCKET/" || handle_error "Upload schemas"
    log_success "Schemas uploaded"
}

step_6() {
    log_info "Step 6: Deploying Cognito, IAM, ECR, Memory, Inference-Profiles..."
    $_CDK_CMD deploy ${STACK_PREFIX}-cognito $_CDK_CONTEXT --require-approval never || handle_error "Deploy Cognito"
    $_CDK_CMD deploy ${STACK_PREFIX}-iam $_CDK_CONTEXT --require-approval never || handle_error "Deploy IAM"
    $_CDK_CMD deploy ${STACK_PREFIX}-ecr $_CDK_CONTEXT --require-approval never || handle_error "Deploy ECR"
    $_CDK_CMD deploy ${STACK_PREFIX}-memory $_CDK_CONTEXT --require-approval never || handle_error "Deploy Memory"
    $_CDK_CMD deploy ${STACK_PREFIX}-inference-profiles $_CDK_CONTEXT --require-approval never || handle_error "Deploy Inference-Profiles"
    log_success "Cognito, IAM, ECR, Memory, Inference-Profiles deployed"
}

step_65() {
    log_info "Step 6.5: Building container Lambda images..."
    cd "$CDK_DIR/lambdas"
    if [ -f "build_container_lambdas.sh" ]; then
        chmod +x build_container_lambdas.sh
        ./build_container_lambdas.sh "$DEPLOYMENT_ID" || handle_error "Build container Lambda images"
        log_success "Container Lambda images built and pushed"
    else
        log_warn "build_container_lambdas.sh not found"
    fi
    cd "$CDK_DIR"
}

step_7() {
    log_info "Step 7: Deploying Lambda stack..."
    $_CDK_CMD deploy ${STACK_PREFIX}-lambda $_CDK_CONTEXT --require-approval never || handle_error "Deploy Lambda stack"
    log_success "Lambda stack deployed"
}

step_8() {
    log_info "Step 8: Deploying Gateway stack..."
    $_CDK_CMD deploy ${STACK_PREFIX}-gateway $_CDK_CONTEXT --require-approval never || handle_error "Deploy Gateway stack"
    log_success "Gateway stack deployed"
}

step_82() {
    log_info "Step 8.25: Deploying X-Ray Transaction Search..."
    $_CDK_CMD deploy ${STACK_PREFIX}-xray $_CDK_CONTEXT --require-approval never || log_warn "X-Ray may already be enabled"
    log_success "X-Ray deployed"
}

step_85() {
    log_info "Step 8.5: Configuring Gateway observability..."
    GATEWAY_ID=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-gateway \
        --query "Stacks[0].Outputs[?OutputKey=='GatewayId'].OutputValue" \
        --output text 2>/dev/null || echo "")
    GATEWAY_ARN=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-gateway \
        --query "Stacks[0].Outputs[?OutputKey=='GatewayArn'].OutputValue" \
        --output text 2>/dev/null || echo "")
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=$(aws configure get region || echo "us-west-2")

    if [ -z "$GATEWAY_ID" ] || [ "$GATEWAY_ID" == "None" ]; then
        log_warn "Could not get Gateway ID, skipping observability"
        return
    fi

    LOG_GROUP_NAME="/aws/vendedlogs/bedrock-agentcore/gateway/APPLICATION_LOGS/${GATEWAY_ID}"
    aws logs create-log-group --log-group-name "$LOG_GROUP_NAME" 2>/dev/null || true
    sleep 2

    aws logs put-delivery-source --name "${GATEWAY_ID}-logs-source" --log-type "APPLICATION_LOGS" --resource-arn "$GATEWAY_ARN" 2>/dev/null || true
    aws logs put-delivery-source --name "${GATEWAY_ID}-traces-source" --log-type "TRACES" --resource-arn "$GATEWAY_ARN" 2>/dev/null || true

    LOG_GROUP_ARN="arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT_ID}:log-group:${LOG_GROUP_NAME}"
    aws logs put-delivery-destination --name "${GATEWAY_ID}-logs-destination" --delivery-destination-type "CWL" \
        --delivery-destination-configuration "{\"destinationResourceArn\":\"${LOG_GROUP_ARN}\"}" 2>/dev/null || true
    aws logs put-delivery-destination --name "${GATEWAY_ID}-traces-destination" --delivery-destination-type "XRAY" \
        --delivery-destination-configuration "{}" 2>/dev/null || true
    sleep 2

    LOGS_DEST_ARN=$(aws logs describe-delivery-destinations \
        --query "deliveryDestinations[?name=='${GATEWAY_ID}-logs-destination'].arn" --output text 2>/dev/null || echo "")
    if [ -n "$LOGS_DEST_ARN" ] && [ "$LOGS_DEST_ARN" != "None" ]; then
        aws logs create-delivery --delivery-source-name "${GATEWAY_ID}-logs-source" --delivery-destination-arn "$LOGS_DEST_ARN" >/dev/null 2>&1 || true
    fi

    TRACES_DEST_ARN=$(aws logs describe-delivery-destinations \
        --query "deliveryDestinations[?name=='${GATEWAY_ID}-traces-destination'].arn" --output text 2>/dev/null || echo "")
    if [ -n "$TRACES_DEST_ARN" ] && [ "$TRACES_DEST_ARN" != "None" ]; then
        aws logs create-delivery --delivery-source-name "${GATEWAY_ID}-traces-source" --delivery-destination-arn "$TRACES_DEST_ARN" >/dev/null 2>&1 || true
    fi

    log_success "Gateway observability configured"
}

step_9() {
    log_info "Step 9: Building and deploying WebSocket Runtime..."
    cd "$CDK_DIR/runtime"
    if [ -f "build_and_push_websocket.sh" ]; then
        chmod +x build_and_push_websocket.sh
        ./build_and_push_websocket.sh || handle_error "Build and push WebSocket runtime"
    else
        log_warn "build_and_push_websocket.sh not found"
    fi
    cd "$CDK_DIR"
    $_CDK_CMD deploy ${STACK_PREFIX}-runtime-websocket $_CDK_CONTEXT --require-approval never || handle_error "Deploy WebSocket Runtime"
    log_success "WebSocket Runtime deployed"
}

step_10() {
    log_info "Step 10: Updating frontend .env..."
    if [ -f "$SCRIPT_DIR/update_frontend_env.sh" ]; then
        chmod +x "$SCRIPT_DIR/update_frontend_env.sh"
        "$SCRIPT_DIR/update_frontend_env.sh" || log_warn "Failed to update frontend .env"
        log_success "Frontend .env updated"
    else
        log_warn "update_frontend_env.sh not found"
    fi
}

# --- Ordered step list ---
STEPS=(1 2 3 4 5 6 65 7 8 82 85 9 10)

# Find starting index
STARTED=false
for S in "${STEPS[@]}"; do
    if [ "$S" == "$START_STEP" ]; then
        STARTED=true
    fi
    if [ "$STARTED" = true ]; then
        step_$S
    fi
done

if [ "$STARTED" = false ]; then
    echo "Invalid step: $START_STEP"
    echo "Valid steps: ${STEPS[*]}"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Resume Deploy Complete!"
echo "=========================================="
echo ""
