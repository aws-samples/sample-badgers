#!/bin/bash
# Sync custom specialists from S3 to local for CDK deployment

set -e

# Try to get config bucket from frontend .env first (faster)
if [ -f "../frontend/.env" ]; then
    CONFIG_BUCKET=$(grep "^S3_UPLOAD_BUCKET=" ../frontend/.env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
fi

# Fall back to CloudFormation if not found
if [ -z "$CONFIG_BUCKET" ]; then
    CONFIG_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name badgers-s3 \
        --query 'Stacks[0].Outputs[?OutputKey==`ConfigBucketName`].OutputValue' \
        --output text 2>/dev/null)
fi

if [ -z "$CONFIG_BUCKET" ] || [ "$CONFIG_BUCKET" == "None" ]; then
    echo "❌ Could not find config bucket. Is badgers-s3 stack deployed?"
    exit 1
fi

echo "📦 Config bucket: $CONFIG_BUCKET"

# Create local directory
mkdir -p custom_specialists

# Download registry
echo "📥 Downloading specialist registry..."
aws s3 cp "s3://${CONFIG_BUCKET}/custom-specialists/specialist_registry.json" \
    custom_specialists/specialist_registry.json 2>/dev/null || {
    echo "⚠️  No custom specialists found in S3"
    echo '{"specialists": []}' > custom_specialists/specialist_registry.json
    exit 0
}

# Check if registry has specialists
SPECIALIST_COUNT=$(jq '.specialists | length' custom_specialists/specialist_registry.json)

if [ "$SPECIALIST_COUNT" -eq 0 ]; then
    echo "⚠️  No custom specialists in registry"
    exit 0
fi

echo "📋 Found $SPECIALIST_COUNT custom specialist(s)"

# Download manifests
echo "📥 Downloading manifests..."
aws s3 sync "s3://${CONFIG_BUCKET}/custom-specialists/manifests/" \
    custom_specialists/manifests/ --quiet

# Download schemas
echo "📥 Downloading schemas..."
aws s3 sync "s3://${CONFIG_BUCKET}/custom-specialists/schemas/" \
    custom_specialists/schemas/ --quiet

echo "✅ Custom specialists synced successfully"
echo ""
echo "Next steps:"
echo "  cdk deploy badgers-custom-specialists"
