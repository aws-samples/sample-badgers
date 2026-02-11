#!/bin/bash
# Build and push WebSocket AgentCore Runtime container to ECR

set -e

echo "🐳 Building and Pushing WebSocket Runtime Container"
echo "===================================================="
echo ""

# Get AWS account and region
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo "us-west-2")

# Get repository name from CloudFormation stack output
REPOSITORY_NAME=$(aws cloudformation describe-stacks \
    --stack-name badgers-ecr \
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
echo "   cd .. && uv run cdk deploy badgers-runtime-websocket --require-approval never"
echo ""
