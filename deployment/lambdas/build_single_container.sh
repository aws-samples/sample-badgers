#!/bin/bash
# Build and push a single container Lambda image
# Usage: ./build_single_container.sh <container_name> <deployment_id>

set -e

CONTAINER_NAME="${1:-}"
DEPLOYMENT_ID="${2:-}"
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$CONTAINER_NAME" ] || [ -z "$DEPLOYMENT_ID" ]; then
    echo "Usage: $0 <container_name> <deployment_id>"
    echo "Example: $0 image_enhancer 1dd02ffa"
    exit 1
fi

ECR_REPO="badgers-${DEPLOYMENT_ID}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
CONTAINER_DIR="./containers/${CONTAINER_NAME}"

if [ ! -d "$CONTAINER_DIR" ]; then
    echo "Error: Container directory not found: ${CONTAINER_DIR}"
    exit 1
fi

IMAGE_TAG="${CONTAINER_NAME}"
FULL_URI="${ECR_URI}:${IMAGE_TAG}"

echo "=========================================="
echo "Building: ${CONTAINER_NAME}"
echo "Tag: ${IMAGE_TAG}"
echo "URI: ${FULL_URI}"
echo "=========================================="

# ECR login
echo "Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Copy foundation and config modules to container build context
echo "Copying foundation and config modules to build context..."
if [ -d "./layer/python/foundation" ]; then
    rm -rf "${CONTAINER_DIR}/foundation"
    cp -r "./layer/python/foundation" "${CONTAINER_DIR}/foundation"
else
    echo "Error: foundation module not found at ./layer/python/foundation"
    exit 1
fi

if [ -d "./layer/python/config" ]; then
    rm -rf "${CONTAINER_DIR}/config"
    cp -r "./layer/python/config" "${CONTAINER_DIR}/config"
else
    echo "Error: config module not found at ./layer/python/config"
    exit 1
fi

# Build for x86_64 (Lambda runtime)
# --provenance=false prevents OCI attestation manifests that Lambda doesn't support
docker build \
    --platform linux/amd64 \
    --provenance=false \
    -t "${FULL_URI}" \
    "${CONTAINER_DIR}"

# Clean up foundation and config copies
rm -rf "${CONTAINER_DIR}/foundation"
rm -rf "${CONTAINER_DIR}/config"

# Push to ECR
echo "Pushing ${FULL_URI}..."
docker push "${FULL_URI}"

echo ""
echo "=========================================="
echo "✓ ${CONTAINER_NAME} pushed successfully!"
echo "URI: ${FULL_URI}"
echo "=========================================="
