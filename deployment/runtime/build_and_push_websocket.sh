#!/bin/bash
# Build and push WebSocket AgentCore Runtime container to ECR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/common.sh"
load_suffix

echo "🐳 Building and Pushing WebSocket Runtime Container"
echo "===================================================="
echo ""

# Get AWS account and region
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo "us-west-2")

# Get repository name from CloudFormation stack output
REPOSITORY_NAME=$(aws cloudformation describe-stacks \
    --stack-name "$(_sn ECR)" \
    --query "Stacks[0].Outputs[?OutputKey=='RepositoryName'].OutputValue" \
    --output text 2>/dev/null)

if [ -z "$REPOSITORY_NAME" ] || [ "$REPOSITORY_NAME" == "None" ]; then
    echo "❌ Error: Could not get repository name from ECR stack"
    echo "   Deploy the ECR stack first"
    exit 1
fi

echo "📋 Configuration:"
echo "   Account: $ACCOUNT"
echo "   Region: $REGION"
echo "   Repository: $REPOSITORY_NAME"
echo "   Tag: websocket"
echo ""

# Login to ECR
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "✅ Logged in"
echo ""

# Copy the foundation module into the build context. The agent imports
# foundation.job_state to create the job-level record when it mints a job_id, and
# Dockerfile.websocket only copies this directory. Sourced from
# ../badgers-foundation/foundation (the tracked source of truth) rather than from
# ../lambdas/layer/python/foundation the way build_container_lambdas.sh does, so
# this build does not depend on build_foundation_layer.sh having run first.
FOUNDATION_SRC="../badgers-foundation/foundation"
FOUNDATION_DEST="./agent/foundation"
echo "📋 Copying foundation module to build context..."
if [ ! -d "$FOUNDATION_SRC" ]; then
    echo "❌ Error: foundation module not found at $FOUNDATION_SRC"
    exit 1
fi
# Destination is agent/ rather than the build-context root on purpose: the image
# runs "python agent/main-websocket.py", so sys.path[0] is /app/agent. This puts
# foundation beside the handler, the same layout the container Lambdas use.
rm -rf "$FOUNDATION_DEST"
cp -r "$FOUNDATION_SRC" "$FOUNDATION_DEST"
echo "✅ foundation copied to $FOUNDATION_DEST"
echo ""

# Remove the copy on any exit path so it never lingers in the working tree.
cleanup_foundation() {
    rm -rf "$FOUNDATION_DEST"
}
trap cleanup_foundation EXIT

# Build WebSocket image
echo "🏗️  Building Docker image (WebSocket)..."
docker build --platform linux/arm64 -t "$REPOSITORY_NAME:websocket" -f Dockerfile.websocket .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Image built"
echo ""

# Tag and push
echo "🏷️  Tagging image..."
docker tag "$REPOSITORY_NAME:websocket" \
    "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:websocket"

echo "📤 Pushing to ECR..."
docker push "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:websocket"

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi

echo ""
echo "===================================================="
echo "✅ WebSocket image pushed!"
echo ""
echo "📝 Image URI:"
echo "   $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:websocket"
echo ""
echo "📝 Next step:"
echo "   cd .. && uv run cdk deploy $(_sn RuntimeWebSocket) --require-approval never"
echo ""
