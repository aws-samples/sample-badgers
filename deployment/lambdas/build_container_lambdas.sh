#!/bin/bash
# Build and push container Lambda images to the badgers ECR repository
# Usage: ./build_container_lambdas.sh <deployment_id>
#
# This script builds the image_enhancer, remediation_specialist, and docx_processor containers
# and pushes them to the shared badgers-<deployment_id> ECR repository.

set -e

DEPLOYMENT_ID="${1:-}"
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$DEPLOYMENT_ID" ]; then
    echo "Usage: $0 <deployment_id>"
    echo "Example: $0 07a6cffb"
    exit 1
fi

ECR_REPO="badgers-${DEPLOYMENT_ID}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

echo "Building container Lambdas for ECR: ${ECR_URI}"

# ECR login
echo "Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Container functions to build
CONTAINERS=("image_enhancer" "remediation_specialist" "docx_processor")

for CONTAINER in "${CONTAINERS[@]}"; do
    CONTAINER_DIR="./containers/${CONTAINER}"

    if [ ! -d "$CONTAINER_DIR" ]; then
        echo "Warning: Container directory not found: ${CONTAINER_DIR}"
        continue
    fi

    IMAGE_TAG="${CONTAINER}"
    FULL_URI="${ECR_URI}:${IMAGE_TAG}"

    echo ""
    echo "=========================================="
    echo "Building: ${CONTAINER}"
    echo "Tag: ${IMAGE_TAG}"
    echo "URI: ${FULL_URI}"
    echo "=========================================="

    # Copy the foundation module into every container build context. Both
    # containers now import foundation.job_state to record their subtask state in
    # DynamoDB, so remediation_specialist is no longer self-contained. The
    # layer builder emits layer.zip directly to avoid macOS Finder recreating
    # .DS_Store during its cleanup, so read its contents from the archive when
    # an expanded legacy layer/ directory is not present.
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

    echo "Copying foundation module to build context..."
    if [ -d "${LAYER_SOURCE}/foundation" ]; then
        rm -rf "${CONTAINER_DIR}/foundation"
        cp -r "${LAYER_SOURCE}/foundation" "${CONTAINER_DIR}/foundation"
    else
        echo "Error: foundation module not found at ./layer/python/foundation"
        exit 1
    fi

    # config is required by every container that imports foundation, not just the
    # ones that read specialist manifests: foundation/__init__.py eagerly imports
    # specialist_foundation, which resolves SPECIALIST_CONFIG_PATH from
    # config.config at module import time. Skipping it produces
    # "Runtime.ImportModuleError: No module named 'config'" at cold start.
    echo "Copying config module to build context..."
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

    # docx_processor validates native output against the canonical BADGERS
    # content-tree XSD inside the Lambda image. Stage the schemas only for that
    # container so other image contexts remain unchanged.
    if [ "${CONTAINER}" = "docx_processor" ]; then
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
    if [ "${CONTAINER}" = "docx_processor" ]; then
        rm -rf "${CONTAINER_DIR}/schemas"
    fi

    # Push to ECR
    echo "Pushing ${FULL_URI}..."
    docker push "${FULL_URI}"

    echo "✓ ${CONTAINER} pushed successfully"
done

echo ""
echo "=========================================="
echo "All container Lambdas built and pushed!"
echo "ECR Repository: ${ECR_REPO}"
echo "Images:"
for CONTAINER in "${CONTAINERS[@]}"; do
    echo "  - ${ECR_URI}:${CONTAINER}"
done
echo "=========================================="
