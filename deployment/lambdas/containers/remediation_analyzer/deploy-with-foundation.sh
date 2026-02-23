#!/usr/bin/env bash
#
# Quick deployment script for remediation_analyzer with foundation library
#
# This script handles the foundation library setup and builds the container.
# It provides options to bundle foundation in the container or use a Lambda layer.
#
# Usage:
#   ./deploy-with-foundation.sh [OPTIONS]
#
# Options:
#   --bundle-foundation    Bundle foundation library in the container image
#   --use-layer            Use Lambda layer for foundation (layer must exist)
#   --build-only           Build Docker image only, don't push
#   --push                 Build and push to ECR
#   --update-lambda        Build, push, and update Lambda function
#   --region REGION        AWS region (default: us-west-2)
#   --profile PROFILE      AWS profile to use (default: none)
#   --help                 Show this help message
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_PROFILE="${AWS_PROFILE:-}"
ACTION="build-only"
FOUNDATION_MODE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bundle-foundation)
            FOUNDATION_MODE="bundle"
            shift
            ;;
        --use-layer)
            FOUNDATION_MODE="layer"
            shift
            ;;
        --build-only)
            ACTION="build-only"
            shift
            ;;
        --push)
            ACTION="push"
            shift
            ;;
        --update-lambda)
            ACTION="update-lambda"
            shift
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        --help)
            head -n 20 "$0" | tail -n +3
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Remediation Analyzer - Container Lambda Deployment"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if foundation mode is specified
if [ -z "$FOUNDATION_MODE" ]; then
    print_warning "Foundation library deployment mode not specified"
    echo ""
    echo "You must choose how to deploy the foundation library:"
    echo ""
    echo "  1. --bundle-foundation  : Bundle foundation in the container"
    echo "                            (Simpler, but larger image)"
    echo ""
    echo "  2. --use-layer          : Use Lambda layer for foundation"
    echo "                            (Smaller image, shared across analyzers)"
    echo ""
    read -p "Choose option (1 or 2): " choice
    case $choice in
        1)
            FOUNDATION_MODE="bundle"
            ;;
        2)
            FOUNDATION_MODE="layer"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
fi

# Navigate to the script directory
cd "$SCRIPT_DIR"

# Handle foundation library setup
if [ "$FOUNDATION_MODE" = "bundle" ]; then
    print_info "Bundling foundation library in container..."

    # Find foundation library location
    FOUNDATION_SOURCE="$PROJECT_ROOT/deployment/badgers-foundation/foundation"

    if [ ! -d "$FOUNDATION_SOURCE" ]; then
        print_error "Foundation library not found at: $FOUNDATION_SOURCE"
        echo ""
        echo "Expected foundation library at:"
        echo "  $FOUNDATION_SOURCE"
        echo ""
        echo "Please ensure the foundation library is available."
        exit 1
    fi

    # Copy foundation library
    print_info "Copying foundation from: $FOUNDATION_SOURCE"
    rm -rf foundation/
    cp -r "$FOUNDATION_SOURCE" ./foundation/
    print_success "Foundation library copied"

    # Update Dockerfile to include foundation
    if ! grep -q "^COPY foundation/ \${LAMBDA_TASK_ROOT}/foundation/" Dockerfile; then
        print_info "Updating Dockerfile to include foundation..."

        # Create a backup
        cp Dockerfile Dockerfile.backup

        # Uncomment the COPY foundation line
        sed -i.tmp 's/^# COPY foundation\/ ${LAMBDA_TASK_ROOT}\/foundation\/$/COPY foundation\/ ${LAMBDA_TASK_ROOT}\/foundation\//' Dockerfile
        rm -f Dockerfile.tmp

        print_success "Dockerfile updated"
    else
        print_success "Dockerfile already configured for bundled foundation"
    fi

elif [ "$FOUNDATION_MODE" = "layer" ]; then
    print_info "Using Lambda layer for foundation library"

    # Check if foundation layer exists
    LAYER_DIR="$PROJECT_ROOT/deployment/lambdas/layer/python/foundation"

    if [ ! -d "$LAYER_DIR" ]; then
        print_warning "Foundation layer not found at: $LAYER_DIR"
        print_info "You'll need to create and attach the layer manually after deployment"
    else
        print_success "Foundation layer found at: $LAYER_DIR"
    fi

    # Ensure Dockerfile does NOT include foundation
    if grep -q "^COPY foundation/ \${LAMBDA_TASK_ROOT}/foundation/" Dockerfile; then
        print_info "Updating Dockerfile to exclude foundation (using layer)..."

        # Create a backup
        cp Dockerfile Dockerfile.backup

        # Comment out the COPY foundation line
        sed -i.tmp 's/^COPY foundation\/ ${LAMBDA_TASK_ROOT}\/foundation\/$/# COPY foundation\/ ${LAMBDA_TASK_ROOT}\/foundation\//' Dockerfile
        rm -f Dockerfile.tmp

        print_success "Dockerfile updated"
    else
        print_success "Dockerfile already configured for layer-based foundation"
    fi

    # Remove any bundled foundation if it exists
    if [ -d "foundation" ]; then
        print_info "Removing bundled foundation directory..."
        rm -rf foundation/
        print_success "Bundled foundation removed"
    fi
fi

echo ""
print_info "Foundation mode: $FOUNDATION_MODE"
print_info "Action: $ACTION"
print_info "Region: $AWS_REGION"
if [ -n "$AWS_PROFILE" ]; then
    print_info "Profile: $AWS_PROFILE"
fi
echo ""

# Build the container using the existing build.sh script
EXTRA_ARGS=""
if [ -n "$AWS_PROFILE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --profile $AWS_PROFILE"
fi
if [ -n "$AWS_REGION" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --region $AWS_REGION"
fi

case $ACTION in
    "build-only")
        ./build.sh --build-only $EXTRA_ARGS
        ;;
    "push")
        ./build.sh --push $EXTRA_ARGS
        ;;
    "update-lambda")
        ./build.sh --update-lambda $EXTRA_ARGS
        ;;
esac

echo ""
print_success "Deployment script completed"

# Show next steps
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Next Steps"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ "$ACTION" = "build-only" ]; then
    print_info "To test locally:"
    echo "  ./test-local.sh"
    echo ""
    print_info "To push to ECR:"
    echo "  ./deploy-with-foundation.sh --${FOUNDATION_MODE/bundle/bundle-foundation}${FOUNDATION_MODE/layer/use-layer} --push"
fi

if [ "$ACTION" = "push" ] || [ "$ACTION" = "update-lambda" ]; then
    if [ "$FOUNDATION_MODE" = "layer" ]; then
        echo ""
        print_warning "Remember to attach the foundation Lambda layer:"
        echo ""
        echo "  aws lambda update-function-configuration \\"
        echo "    --function-name remediation-analyzer \\"
        echo "    --layers arn:aws:lambda:\${AWS_REGION}:\${AWS_ACCOUNT_ID}:layer:badgers-foundation:VERSION"
        echo ""
    fi

    print_info "To test the deployed function:"
    echo "  aws lambda invoke \\"
    echo "    --function-name remediation-analyzer \\"
    echo "    --payload file://test-event.json \\"
    echo "    response.json"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
