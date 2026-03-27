#!/bin/bash
# Build and push the image_enhancer container to ECR
# Usage: ./deploy_analyzer_container.sh [--profile PROFILE]

set -e

STACK_PREFIX="badgers"
REGION="${AWS_REGION:-us-west-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
_AWS_PROFILE_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            _AWS_PROFILE_ARG="--profile $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Get deployment ID from stack tags (same approach as sync_s3_files.sh)
echo "Fetching deployment ID from CDK outputs..."
CONFIG_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_PREFIX}-s3" \
    --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
    --output text $_AWS_PROFILE_ARG 2>/dev/null || echo "")

if [[ -z "$CONFIG_BUCKET" ]]; then
    echo "Error: Could not fetch config bucket from stack ${STACK_PREFIX}-s3"
    exit 1
fi

# Extract deployment ID from bucket name (badgers-config-XXXXXXXX)
DEPLOYMENT_ID=$(echo "$CONFIG_BUCKET" | sed 's/badgers-config-//')

if [[ -z "$DEPLOYMENT_ID" ]]; then
    echo "Error: Could not extract deployment ID from bucket name: $CONFIG_BUCKET"
    exit 1
fi

echo "Using deployment ID: $DEPLOYMENT_ID"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text $_AWS_PROFILE_ARG)
ECR_REPO="badgers-${DEPLOYMENT_ID}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

# ECR login
aws ecr get-login-password --region ${REGION} $_AWS_PROFILE_ARG | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build and push image_enhancer
cd "$SCRIPT_DIR/lambdas"
docker build --platform linux/amd64 --provenance=false -t "${ECR_URI}:image_enhancer" ./containers/image_enhancer
docker push "${ECR_URI}:image_enhancer"

echo "Done! Pushed ${ECR_URI}:image_enhancer"
