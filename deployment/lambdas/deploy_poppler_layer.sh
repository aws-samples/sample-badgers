#!/bin/bash
# Deploy the poppler Lambda layer to AWS

set -e

LAYER_NAME="poppler-utils"
DESCRIPTION="Poppler utilities for PDF processing (pdftoppm, pdfinfo, pdftotext, pdfimages)"
COMPATIBLE_RUNTIMES="python3.12"
REGION="${AWS_REGION:-us-west-2}"
LAYER_FILE="poppler-layer.zip"

echo "☁️  Deploying Poppler Lambda Layer..."

# Check if layer zip exists
if [ ! -f "$LAYER_FILE" ]; then
    echo "❌ $LAYER_FILE not found. Run ./build_poppler_layer.sh first."
    exit 1
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first."
    exit 1
fi

# Get layer size
LAYER_SIZE=$(du -h "$LAYER_FILE" | cut -f1)
echo "📦 Layer size: $LAYER_SIZE"

# Publish layer
echo "🚀 Publishing layer to AWS..."
LAYER_VERSION=$(aws lambda publish-layer-version \
    --layer-name "$LAYER_NAME" \
    --description "$DESCRIPTION" \
    --zip-file "fileb://$LAYER_FILE" \
    --compatible-runtimes "$COMPATIBLE_RUNTIMES" \
    --compatible-architectures x86_64 \
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
