#!/bin/bash
# Build and push container Lambda images to the badgers ECR repository
# Usage: ./build_container_lambdas.sh <deployment_id>
#
# This script builds the image_enhancer and remediation_specialist containers
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
CONTAINERS=("image_enhancer" "remediation_specialist")

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
    # DynamoDB, so remediation_specialist is no longer self-contained.
    echo "Copying foundation module to build context..."
    if [ -d "./layer/python/foundation" ]; then
        rm -rf "${CONTAINER_DIR}/foundation"
        cp -r "./layer/python/foundation" "${CONTAINER_DIR}/foundation"
    else
        echo "Error: foundation module not found at ./layer/python/foundation"
        exit 1
    fi

    # config is only needed by the containers that load specialist manifests;
    # remediation_specialist reads none.
    if [ "$CONTAINER" != "remediation_specialist" ]; then
        echo "Copying config module to build context..."
        if [ -d "./layer/python/config" ]; then
            rm -rf "${CONTAINER_DIR}/config"
            cp -r "./layer/python/config" "${CONTAINER_DIR}/config"
        else
            echo "Error: config module not found at ./layer/python/config"
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

    # Clean up foundation and config copies (if they were added)
    if [ "$CONTAINER" != "remediation_specialist" ]; then
        rm -rf "${CONTAINER_DIR}/foundation"
        rm -rf "${CONTAINER_DIR}/config"
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
