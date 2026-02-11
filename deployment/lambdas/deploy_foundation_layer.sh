#!/bin/bash
# Deploy the analyzer foundation Lambda layer to AWS

set -e

LAYER_NAME="analyzer-foundation"
DESCRIPTION="Reusable foundation framework for analyzer Lambda functions"
COMPATIBLE_RUNTIMES="python3.12"
REGION="${AWS_REGION:-us-west-2}"

echo "☁️  Deploying Analyzer Foundation Lambda Layer..."

# Check if layer.zip exists
if [ ! -f "layer.zip" ]; then
    echo "❌ layer.zip not found. Run ./build_foundation_layer.sh first."
    exit 1
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first."
    exit 1
fi

# Get layer size
LAYER_SIZE=$(du -h layer.zip | cut -f1)
echo "📦 Layer size: $LAYER_SIZE"

# Publish layer
echo "🚀 Publishing layer to AWS..."
LAYER_VERSION=$(aws lambda publish-layer-version \
    --layer-name "$LAYER_NAME" \
    --description "$DESCRIPTION" \
    --zip-file fileb://layer.zip \
    --compatible-runtimes "$COMPATIBLE_RUNTIMES" \
    --compatible-architectures x86_64 arm64 \
    --region "$REGION" \
    --query 'Version' \
    --output text)

if [ $? -eq 0 ]; then
    echo "✅ Layer published successfully!"
    echo ""

    # Get account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    LAYER_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:layer:$LAYER_NAME:$LAYER_VERSION"

    echo "📋 Layer Details:"
    echo "   Name: $LAYER_NAME"
    echo "   Version: $LAYER_VERSION"
    echo "   Region: $REGION"
    echo "   ARN: $LAYER_ARN"
    echo ""

    # Save ARN to file for easy reference
    echo "$LAYER_ARN" > layer_arn.txt
    echo "💾 Layer ARN saved to: layer_arn.txt"
    echo ""

    echo "🎯 Use this layer in your functions:"
    echo ""
    echo "aws lambda update-function-configuration \\"
    echo "    --function-name YOUR_FUNCTION_NAME \\"
    echo "    --layers $LAYER_ARN \\"
    echo "    --region $REGION"
    echo ""

    echo "Or in your function code:"
    echo "from foundation.analyzer_foundation import AnalyzerFoundation"
    echo ""

    # List all versions
    echo "📚 All versions of this layer:"
    aws lambda list-layer-versions \
        --layer-name "$LAYER_NAME" \
        --region "$REGION" \
        --query 'LayerVersions[*].[Version,CreatedDate]' \
        --output table
else
    echo "❌ Failed to publish layer"
    exit 1
fi
