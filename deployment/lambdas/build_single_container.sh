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
LAYER_SOURCE="./layer/python"
LAYER_STAGE_DIR=""
if [ ! -d "${LAYER_SOURCE}/foundation" ] || [ ! -d "${LAYER_SOURCE}/config" ]; then
    if [ ! -f "./layer.zip" ]; then
        echo "Error: foundation layer not found. Run ./build_foundation_layer.sh first."
        exit 1
    fi
    LAYER_STAGE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/badgers-container-layer.XXXXXX")
    unzip -q "./layer.zip" "python/foundation/*" "python/config/*" -d "${LAYER_STAGE_DIR}"
    LAYER_SOURCE="${LAYER_STAGE_DIR}/python"
fi

if [ -d "${LAYER_SOURCE}/foundation" ]; then
    rm -rf "${CONTAINER_DIR}/foundation"
    cp -r "${LAYER_SOURCE}/foundation" "${CONTAINER_DIR}/foundation"
else
    echo "Error: foundation module not found at ./layer/python/foundation"
    exit 1
fi

if [ -d "${LAYER_SOURCE}/config" ]; then
    rm -rf "${CONTAINER_DIR}/config"
    cp -r "${LAYER_SOURCE}/config" "${CONTAINER_DIR}/config"
else
    echo "Error: config module not found at ./layer/python/config"
    exit 1
fi
if [ -n "${LAYER_STAGE_DIR}" ]; then
    rm -rf "${LAYER_STAGE_DIR}"
fi

if [ "$CONTAINER_NAME" = "docx_processor" ]; then
    echo "Copying BADGERS schemas to DOCX processor build context..."
    if [ -d "../../schemas" ]; then
        rm -rf "${CONTAINER_DIR}/schemas"
        cp -r "../../schemas" "${CONTAINER_DIR}/schemas"
    else
        echo "Error: BADGERS schemas not found at ../../schemas"
        exit 1
    fi
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
if [ "$CONTAINER_NAME" = "docx_processor" ]; then
    rm -rf "${CONTAINER_DIR}/schemas"
fi

# Push to ECR
echo "Pushing ${FULL_URI}..."
docker push "${FULL_URI}"

echo ""
echo "=========================================="
echo "✓ ${CONTAINER_NAME} pushed successfully!"
echo "URI: ${FULL_URI}"
echo "=========================================="
