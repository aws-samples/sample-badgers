#!/bin/bash
#
# Deploy BADGERS frontend infrastructure (VPC + ALB + Fargate + Cognito auth)
# Requires the core stacks (s3, cognito, ecr) to already be deployed.
#
# Usage: ./deploy_frontend.sh [--skip-build] [--force]
#

set -e

# Parse arguments
SKIP_BUILD=false
FORCE_UPDATE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./deploy_frontend.sh [--skip-build] [--force]"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }

handle_error() {
    log_error "Failed at: $1"
    exit 1
}

export TYPEGUARD_DISABLE=1
export PYTHONWARNINGS="ignore::UserWarning:aws_cdk"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDK_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/unified-ui"
STACK_PREFIX="badgers"

# ── Resolve deployment ID from existing core stacks ──────────────────────────

resolve_deployment_id() {
    log_info "Resolving deployment ID from existing stacks..."

    # Try tag first
    DEPLOYMENT_ID=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-s3 \
        --query "Stacks[0].Tags[?Key=='deployment_id'].Value" \
        --output text 2>/dev/null || echo "")

    if [ -z "$DEPLOYMENT_ID" ] || [ "$DEPLOYMENT_ID" == "None" ]; then
        # Fallback: extract from config bucket name (badgers-config-XXXXXXXX)
        CONFIG_BUCKET=$(aws cloudformation describe-stacks \
            --stack-name ${STACK_PREFIX}-s3 \
            --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
            --output text 2>/dev/null || echo "")
        if [ -n "$CONFIG_BUCKET" ]; then
            DEPLOYMENT_ID=$(echo "$CONFIG_BUCKET" | sed 's/badgers-config-//')
        fi
    fi

    if [ -z "$DEPLOYMENT_ID" ]; then
        handle_error "Could not determine deployment ID. Are the core stacks deployed?"
    fi

    log_info "Deployment ID: $DEPLOYMENT_ID"
    _CDK_CONTEXT="-c deployment_id=$DEPLOYMENT_ID"
}

# ── Verify prerequisite stacks exist ─────────────────────────────────────────

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Docker
    if ! command -v docker &> /dev/null; then
        handle_error "Docker not found"
    fi
    if ! docker info &> /dev/null; then
        handle_error "Docker is not running"
    fi

    # CDK
    if command -v uv &> /dev/null && uv run cdk --version &> /dev/null; then
        _CDK_CMD="uv run cdk"
    elif command -v cdk &> /dev/null; then
        _CDK_CMD="cdk"
    else
        handle_error "AWS CDK not found"
    fi

    # Node/npm for frontend build
    if [ "$SKIP_BUILD" != true ]; then
        if ! command -v node &> /dev/null; then
            handle_error "Node.js not found (needed for frontend build)"
        fi
    fi

    # Required stacks
    for stack in s3 cognito ecr; do
        STATUS=$(aws cloudformation describe-stacks \
            --stack-name ${STACK_PREFIX}-${stack} \
            --query "Stacks[0].StackStatus" \
            --output text 2>/dev/null || echo "MISSING")
        if [[ "$STATUS" == "MISSING" || "$STATUS" == *"DELETE"* ]]; then
            handle_error "Required stack ${STACK_PREFIX}-${stack} not found (status: $STATUS). Deploy core stacks first."
        fi
    done

    log_success "Prerequisites met"
}

# ── Build frontend assets and Docker image ───────────────────────────────────

build_and_push_frontend() {
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region || echo "us-west-2")

    REPOSITORY_NAME=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-ecr \
        --query "Stacks[0].Outputs[?OutputKey=='RepositoryName'].OutputValue" \
        --output text 2>/dev/null)

    if [ -z "$REPOSITORY_NAME" ] || [ "$REPOSITORY_NAME" == "None" ]; then
        handle_error "Could not get ECR repository name from ${STACK_PREFIX}-ecr stack"
    fi

    ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}"
    IMAGE_TAG="frontend"

    log_info "ECR repository: $REPOSITORY_NAME"
    log_info "Image tag: $IMAGE_TAG"

    # Step 1: Build Vite production bundle
    if [ "$SKIP_BUILD" != true ]; then
        log_info "Building frontend assets (npm run build)..."
        cd "$FRONTEND_DIR"
        npm ci || handle_error "npm ci"
        npm run build || handle_error "npm run build"
        cd "$CDK_DIR"
        log_success "Frontend assets built"
    else
        log_warn "Skipping frontend build (--skip-build)"
        if [ ! -d "$FRONTEND_DIR/dist" ]; then
            handle_error "No dist/ directory found. Run without --skip-build first."
        fi
    fi

    # Step 2: Docker build & push
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region "$REGION" | \
        docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

    log_info "Building Docker image..."
    cd "$FRONTEND_DIR"
    docker build --platform linux/amd64 -t "${REPOSITORY_NAME}:${IMAGE_TAG}" . || handle_error "Docker build"

    docker tag "${REPOSITORY_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

    log_info "Pushing to ECR..."
    docker push "${ECR_URI}:${IMAGE_TAG}" || handle_error "Docker push"
    cd "$CDK_DIR"

    log_success "Frontend image pushed: ${ECR_URI}:${IMAGE_TAG}"
}

# ── Deploy CDK stacks ────────────────────────────────────────────────────────

deploy_stacks() {
    cd "$CDK_DIR"

    # Activate venv if present
    if [ -f "$CDK_DIR/.venv/bin/activate" ]; then
        source "$CDK_DIR/.venv/bin/activate"
    fi

    # VPC
    log_info "Deploying ${STACK_PREFIX}-vpc..."
    $_CDK_CMD deploy ${STACK_PREFIX}-vpc $_CDK_CONTEXT --require-approval never || handle_error "Deploy VPC stack"
    log_success "VPC stack deployed"

    # Frontend (ALB + Fargate + ACM + Route53)
    log_info "Deploying ${STACK_PREFIX}-frontend..."
    $_CDK_CMD deploy ${STACK_PREFIX}-frontend $_CDK_CONTEXT --require-approval never || handle_error "Deploy frontend stack"
    log_success "Frontend stack deployed"
}

# ── Print outputs ────────────────────────────────────────────────────────────

print_outputs() {
    echo ""
    echo "=========================================="
    echo "  Frontend Deployment Complete"
    echo "=========================================="
    echo ""

    FRONTEND_URL=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-frontend \
        --query "Stacks[0].Outputs[?OutputKey=='FrontendUrl'].OutputValue" \
        --output text 2>/dev/null || echo "N/A")

    ALB_DNS=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-frontend \
        --query "Stacks[0].Outputs[?OutputKey=='AlbDnsName'].OutputValue" \
        --output text 2>/dev/null || echo "N/A")

    ALB_CLIENT_ID=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_PREFIX}-frontend \
        --query "Stacks[0].Outputs[?OutputKey=='AlbAppClientId'].OutputValue" \
        --output text 2>/dev/null || echo "N/A")

    echo "  Frontend URL:     $FRONTEND_URL"
    echo "  ALB DNS:          $ALB_DNS"
    echo "  ALB Client ID:    $ALB_CLIENT_ID"
    echo "  Deployment ID:    $DEPLOYMENT_ID"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "=========================================="
    echo "  BADGERS — Frontend Deployment"
    echo "=========================================="
    echo ""

    check_prerequisites
    resolve_deployment_id
    build_and_push_frontend
    deploy_stacks
    print_outputs
}

main "$@"
