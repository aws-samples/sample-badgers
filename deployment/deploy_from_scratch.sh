#!/bin/bash
#
# Full deployment script for PDF Extractor POC
# Deploys all CDK stacks from scratch with proper ordering
#
# Usage: ./deploy_from_scratch.sh [--websocket-only]
#   --websocket-only  Skip regular runtime, deploy only websocket runtime
#

set -e  # Exit on error

# Parse arguments
WEBSOCKET_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --websocket-only)
            WEBSOCKET_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./deploy_from_scratch.sh [--websocket-only]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Error handler
handle_error() {
    log_error "Deployment failed at step: $1"
    log_error "Check the logs above for details"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi

    # Check CDK
    if ! command -v cdk &> /dev/null; then
        log_error "AWS CDK not found. Install with: npm install -g aws-cdk"
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi

    # Check Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker."
        exit 1
    fi

    # Check Python/uv
    if command -v uv &> /dev/null; then
        _PIP_CMD="uv pip install"
        _CDK_CMD="uv run cdk"
    else
        _PIP_CMD="pip install"
        _CDK_CMD="cdk"
    fi

    log_success "All prerequisites met"
}

# Turn off TypeGuard Checks
export TYPEGUARD_DISABLE=1
export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDK_DIR="$SCRIPT_DIR"

# Stack name prefix
STACK_PREFIX="badgers"

# Main deployment
main() {
    echo ""
    echo "=========================================="
    echo "  BADGERS POC - Full Deployment"
    echo "=========================================="
    echo ""

    check_prerequisites

    cd "$CDK_DIR"

    # Check for existing stacks - this script is for fresh deployments only
    log_info "Checking for existing stacks..."
    EXISTING_STACKS=$(aws cloudformation list-stacks \
        --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE UPDATE_ROLLBACK_COMPLETE \
        --query "StackSummaries[?starts_with(StackName, '${STACK_PREFIX}-')].StackName" \
        --output text 2>/dev/null || echo "")

    if [ -n "$EXISTING_STACKS" ] && [ "$EXISTING_STACKS" != "None" ]; then
        log_warn "Found existing stacks: $EXISTING_STACKS"
        log_warn "This script is designed for fresh deployments."
        log_warn "For existing deployments, run './destroy.sh' first, or use 'cdk deploy --all' for updates."
        echo ""
        read -p "Do you want to run destroy.sh first? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Running destroy.sh..."
            if [ -f "$SCRIPT_DIR/destroy.sh" ]; then
                chmod +x "$SCRIPT_DIR/destroy.sh"
                "$SCRIPT_DIR/destroy.sh" || handle_error "Destroy existing stacks"
                log_success "Existing stacks destroyed"
            else
                handle_error "destroy.sh not found"
            fi
        else
            log_error "Aborting deployment. Please destroy existing stacks first or use 'cdk deploy --all'."
            exit 1
        fi
    fi

    # Generate deployment ID once for all stacks
    DEPLOYMENT_ID=$(uuidgen | cut -c1-8 | tr '[:upper:]' '[:lower:]')
    log_info "Using deployment ID: $DEPLOYMENT_ID"
    _CDK_CONTEXT="-c deployment_id=$DEPLOYMENT_ID"

    # Step 1: Install Python dependencies
    log_info "Step 1/11: Installing Python dependencies..."
    $_PIP_CMD -r requirements.txt || handle_error "Install dependencies"
    log_success "Dependencies installed"

    # Step 2: Build lambda layers
    log_info "Step 2/11: Building lambda layers..."
    cd "$CDK_DIR/lambdas"

    # Build foundation layer
    if [ -f "build_foundation_layer.sh" ]; then
        chmod +x build_foundation_layer.sh
        ./build_foundation_layer.sh || handle_error "Build foundation lambda layer"
        log_success "Foundation lambda layer built"
    else
        log_warn "build_foundation_layer.sh not found, checking for existing layer.zip"
        if [ ! -f "layer.zip" ]; then
            handle_error "Foundation lambda layer not found"
        fi
    fi

    # Build poppler layer
    if [ -f "build_poppler_layer.sh" ]; then
        chmod +x build_poppler_layer.sh
        ./build_poppler_layer.sh || handle_error "Build poppler lambda layer"
        log_success "Poppler lambda layer built"
    else
        log_warn "build_poppler_layer.sh not found, checking for existing poppler-layer.zip"
        if [ ! -f "poppler-layer.zip" ]; then
            handle_error "Poppler lambda layer not found"
        fi
    fi

    # Build enhancement layer
    if [ -f "build_enhancement_layer.sh" ]; then
        chmod +x build_enhancement_layer.sh
        ./build_enhancement_layer.sh || handle_error "Build enhancement lambda layer"
        log_success "Enhancement lambda layer built"
    else
        log_warn "build_enhancement_layer.sh not found, checking for existing enhancement-layer.zip"
        if [ ! -f "enhancement-layer.zip" ]; then
            handle_error "Enhancement lambda layer not found"
        fi
    fi

    # Build pdf-processing layer
    if [ -f "build_pdf_processing_layer.sh" ]; then
        chmod +x build_pdf_processing_layer.sh
        ./build_pdf_processing_layer.sh || handle_error "Build pdf-processing lambda layer"
        log_success "PDF processing lambda layer built"
    else
        log_warn "build_pdf_processing_layer.sh not found, checking for existing pdf-processing-layer.zip"
        if [ ! -f "pdf-processing-layer.zip" ]; then
            log_warn "PDF processing lambda layer not found - remediation_analyzer may fail"
        fi
    fi
    cd "$CDK_DIR"

    # Step 3: Bootstrap CDK (if needed)
    log_info "Step 3/11: Checking CDK bootstrap..."
    $_CDK_CMD bootstrap || log_warn "Bootstrap may already exist, continuing..."
    log_success "CDK bootstrap complete"

    # Step 4: Deploy S3 Stack
    log_info "Step 4/11: Deploying S3 Stack..."
    $_CDK_CMD deploy ${STACK_PREFIX}-s3 $_CDK_CONTEXT --require-approval never || handle_error "Deploy S3 Stack"
    log_success "S3 Stack deployed"

    # Step 5: Upload schemas to S3
    log_info "Step 5/11: Uploading schemas to S3..."
    CONFIG_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-s3 \
        --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
        --output text)

    if [ -z "$CONFIG_BUCKET" ] || [ "$CONFIG_BUCKET" == "None" ]; then
        handle_error "Could not get config bucket name"
    fi

    log_info "Uploading to bucket: $CONFIG_BUCKET"
    aws s3 sync "$CDK_DIR/s3_files/" "s3://$CONFIG_BUCKET/" || handle_error "Upload schemas"
    log_success "Schemas uploaded to S3"

    # Step 6: Deploy cognito, iam, ecr, memory, and inference-profiles Stacks
    log_info "Step 6/11: Deploying cognito, iam, ecr, memory, and inference-profiles Stacks..."
    $_CDK_CMD deploy ${STACK_PREFIX}-cognito $_CDK_CONTEXT --require-approval never || handle_error "Deploy cognito Stack"
    $_CDK_CMD deploy ${STACK_PREFIX}-iam $_CDK_CONTEXT --require-approval never || handle_error "Deploy iam Stack"
    $_CDK_CMD deploy ${STACK_PREFIX}-ecr $_CDK_CONTEXT --require-approval never || handle_error "Deploy ecr Stack"
    $_CDK_CMD deploy ${STACK_PREFIX}-memory $_CDK_CONTEXT --require-approval never || handle_error "Deploy memory Stack"
    $_CDK_CMD deploy ${STACK_PREFIX}-inference-profiles $_CDK_CONTEXT --require-approval never || handle_error "Deploy inference-profiles Stack"
    log_success "cognito, iam, ecr, memory, and inference-profiles Stacks deployed"

    # Step 6.5: Build and push container Lambda images
    log_info "Step 6.5/11: Building container Lambda images..."
    cd "$CDK_DIR/lambdas"
    if [ -f "build_container_lambdas.sh" ]; then
        chmod +x build_container_lambdas.sh
        ./build_container_lambdas.sh "$DEPLOYMENT_ID" || handle_error "Build container Lambda images"
        log_success "Container Lambda images built and pushed"
    else
        log_warn "build_container_lambdas.sh not found, skipping container Lambda build"
    fi
    cd "$CDK_DIR"

    # Step 7: Deploy lambda Stack
    log_info "Step 7/11: Deploying lambda Stack..."
    $_CDK_CMD deploy ${STACK_PREFIX}-lambda $_CDK_CONTEXT --require-approval never || handle_error "Deploy lambda Stack"
    log_success "lambda Stack deployed"

    # Step 8: Deploy gateway Stack
    log_info "Step 8/11: Deploying gateway Stack..."
    $_CDK_CMD deploy ${STACK_PREFIX}-gateway $_CDK_CONTEXT --require-approval never || handle_error "Deploy gateway Stack"
    log_success "gateway Stack deployed"

    # Step 8.5: Configure Gateway Observability (logging and tracing)
    log_info "Step 8.5/11: Configuring Gateway observability..."
    GATEWAY_ID=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-gateway \
        --query "Stacks[0].Outputs[?OutputKey=='GatewayId'].OutputValue" \
        --output text)
    GATEWAY_ARN=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-gateway \
        --query "Stacks[0].Outputs[?OutputKey=='GatewayArn'].OutputValue" \
        --output text)
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=$(aws configure get region || echo "us-west-2")

    if [ -n "$GATEWAY_ID" ] && [ "$GATEWAY_ID" != "None" ]; then
        log_info "Configuring observability for gateway: $GATEWAY_ID"

        # Create log group for gateway logs
        LOG_GROUP_NAME="/aws/vendedlogs/bedrock-agentcore/gateway/APPLICATION_LOGS/${GATEWAY_ID}"
        aws logs create-log-group --log-group-name "$LOG_GROUP_NAME" 2>/dev/null || log_warn "Log group may already exist"

        # Small delay to ensure log group is available
        sleep 2

        # Configure delivery source for logs
        aws logs put-delivery-source \
            --name "${GATEWAY_ID}-logs-source" \
            --log-type "APPLICATION_LOGS" \
            --resource-arn "$GATEWAY_ARN" 2>/dev/null || log_warn "Logs delivery source may already exist"

        # Configure delivery source for traces
        aws logs put-delivery-source \
            --name "${GATEWAY_ID}-traces-source" \
            --log-type "TRACES" \
            --resource-arn "$GATEWAY_ARN" 2>/dev/null || log_warn "Traces delivery source may already exist"

        # Configure delivery destination for logs (CloudWatch Logs)
        LOG_GROUP_ARN="arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT_ID}:log-group:${LOG_GROUP_NAME}"
        aws logs put-delivery-destination \
            --name "${GATEWAY_ID}-logs-destination" \
            --delivery-destination-type "CWL" \
            --delivery-destination-configuration "{\"destinationResourceArn\":\"${LOG_GROUP_ARN}\"}" 2>/dev/null || log_warn "Logs delivery destination may already exist"

        # Configure delivery destination for traces (X-Ray)
        aws logs put-delivery-destination \
            --name "${GATEWAY_ID}-traces-destination" \
            --delivery-destination-type "XRAY" \
            --delivery-destination-configuration "{}" 2>/dev/null || log_warn "Traces delivery destination may already exist"

        # Small delay to ensure destinations are registered
        sleep 2

        # Create delivery for logs
        LOGS_DEST_ARN=$(aws logs describe-delivery-destinations \
            --query "deliveryDestinations[?name=='${GATEWAY_ID}-logs-destination'].arn" \
            --output text 2>/dev/null)
        if [ -n "$LOGS_DEST_ARN" ] && [ "$LOGS_DEST_ARN" != "None" ]; then
            aws logs create-delivery \
                --delivery-source-name "${GATEWAY_ID}-logs-source" \
                --delivery-destination-arn "$LOGS_DEST_ARN" >/dev/null 2>&1 || log_warn "Logs delivery may already exist"
        else
            log_warn "Could not get logs destination ARN"
        fi

        # Create delivery for traces
        TRACES_DEST_ARN=$(aws logs describe-delivery-destinations \
            --query "deliveryDestinations[?name=='${GATEWAY_ID}-traces-destination'].arn" \
            --output text 2>/dev/null)
        if [ -n "$TRACES_DEST_ARN" ] && [ "$TRACES_DEST_ARN" != "None" ]; then
            aws logs create-delivery \
                --delivery-source-name "${GATEWAY_ID}-traces-source" \
                --delivery-destination-arn "$TRACES_DEST_ARN" >/dev/null 2>&1 || log_warn "Traces delivery may already exist"
        else
            log_warn "Could not get traces destination ARN"
        fi

        log_success "Gateway observability configured"
    else
        log_warn "Could not get gateway ID, skipping observability configuration"
    fi

    # Step 9: Build Runtime containers and deploy Runtime
    log_info "Step 9/11: Building Runtime containers and deploying Runtime..."

    # Build and push container
    log_info "Building and pushing Runtime container..."
    cd "$CDK_DIR/runtime"
    if [ -f "build_and_push_websocket.sh" ]; then
        chmod +x build_and_push_websocket.sh
        ./build_and_push_websocket.sh || handle_error "Build and push container"
    else
        log_warn "build_and_push_websocket.sh not found, skipping container build"
    fi
    cd "$CDK_DIR"

    # Deploy WebSocket Runtime
    log_info "Deploying WebSocket Runtime Stack..."
    $_CDK_CMD deploy ${STACK_PREFIX}-runtime-websocket $_CDK_CONTEXT --require-approval never || handle_error "Deploy WebSocket Runtime Stack"
    log_success "WebSocket Runtime Stack deployed"

    # Final summary
    echo ""
    echo "=========================================="
    echo "  Deployment Complete!"
    echo "=========================================="
    echo ""
    log_success "All stacks deployed successfully"
    echo ""

    # Print outputs
    log_info "Key outputs:"
    echo ""
    GATEWAY_URL=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-gateway \
        --query "Stacks[0].Outputs[?OutputKey=='GatewayUrl'].OutputValue" \
        --output text 2>/dev/null || echo "N/A")
    echo "Gateway URL: $GATEWAY_URL"

    echo "Runtime URL: WebSocket only (no sync runtime)"

    USER_POOL_ID=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-cognito \
        --query "Stacks[0].Outputs[?OutputKey=='UserPoolId'].OutputValue" \
        --output text)
    echo "cognito User Pool ID: $USER_POOL_ID"

    CLIENT_ID=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-cognito \
        --query "Stacks[0].Outputs[?OutputKey=='UserPoolClientId'].OutputValue" \
        --output text)
    echo "cognito Client ID: $CLIENT_ID"

    echo ""
    log_info "Log locations:"
    echo "  gateway: /aws/vendedlogs/bedrock-agentcore/gateway/"
    echo "  Runtime: /aws/bedrock-agentcore/runtimes/"
    echo ""

    # Step 10: Update frontend .env with deployment outputs
    log_info "Step 10/11: Updating frontend .env..."
    if [ -f "$SCRIPT_DIR/update_frontend_env.sh" ]; then
        chmod +x "$SCRIPT_DIR/update_frontend_env.sh"
        "$SCRIPT_DIR/update_frontend_env.sh" || log_warn "Failed to update frontend .env"
        log_success "Frontend .env updated"
    else
        log_warn "update_frontend_env.sh not found, skipping frontend .env update"
    fi
    echo ""
}

# Run main
main "$@"
