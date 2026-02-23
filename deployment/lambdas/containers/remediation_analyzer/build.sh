#!/usr/bin/env bash
#
# Build and deploy script for remediation_analyzer container Lambda
#
# Usage:
#   ./build.sh [OPTIONS]
#
# Options:
#   --build-only       Build Docker image only, don't push
#   --push             Build and push to ECR
#   --update-lambda    Build, push, and update Lambda function
#   --region REGION    AWS region (default: us-west-2)
#   --profile PROFILE  AWS profile to use (default: none)
#
# Examples:
#   ./build.sh --build-only
#   ./build.sh --push --region us-west-2
#   ./build.sh --update-lambda --region us-west-2 --profile my-profile
#

set -e

# Default values
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_PROFILE="${AWS_PROFILE:-}"
ACTION="build-only"
IMAGE_NAME="remediation-analyzer"
IMAGE_TAG="latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            ACTION="build-only"
            shift
            ;;
        --push)
            ACTION="push"
            shift
            ;;
        --update-lambda)
            ACTION="update-lambda"
            shift
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        --help)
            head -n 20 "$0" | tail -n +3
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build AWS CLI command prefix
AWS_CMD="aws"
if [ -n "$AWS_PROFILE" ]; then
    AWS_CMD="aws --profile $AWS_PROFILE"
fi
if [ -n "$AWS_REGION" ]; then
    AWS_CMD="$AWS_CMD --region $AWS_REGION"
fi

echo "==> Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${IMAGE_TAG}" .

if [ "$ACTION" = "build-only" ]; then
    echo "==> Build complete. Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "    To test locally, run:"
    echo "    docker run -p 9000:8080 ${IMAGE_NAME}:${IMAGE_TAG}"
    exit 0
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$($AWS_CMD sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"

echo "==> ECR repository: ${ECR_REPO}"

# Create ECR repository if it doesn't exist
if ! $AWS_CMD ecr describe-repositories --repository-names "${IMAGE_NAME}" &>/dev/null; then
    echo "==> Creating ECR repository: ${IMAGE_NAME}"
    $AWS_CMD ecr create-repository --repository-name "${IMAGE_NAME}"
fi

# Authenticate Docker to ECR
echo "==> Authenticating Docker to ECR"
$AWS_CMD ecr get-login-password | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Tag and push image
echo "==> Tagging image for ECR"
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ECR_REPO}:${IMAGE_TAG}"

echo "==> Pushing image to ECR"
docker push "${ECR_REPO}:${IMAGE_TAG}"

IMAGE_URI="${ECR_REPO}:${IMAGE_TAG}"
echo "==> Image pushed successfully: ${IMAGE_URI}"

if [ "$ACTION" = "update-lambda" ]; then
    # Prompt for Lambda function name
    read -p "Enter Lambda function name (default: remediation-analyzer): " LAMBDA_FUNCTION_NAME
    LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-remediation-analyzer}"

    echo "==> Updating Lambda function: ${LAMBDA_FUNCTION_NAME}"
    $AWS_CMD lambda update-function-code \
        --function-name "${LAMBDA_FUNCTION_NAME}" \
        --image-uri "${IMAGE_URI}"

    echo "==> Waiting for Lambda function to update..."
    $AWS_CMD lambda wait function-updated --function-name "${LAMBDA_FUNCTION_NAME}"

    echo "==> Lambda function updated successfully"
fi

echo ""
echo "==> Done!"
echo "    Image URI: ${IMAGE_URI}"
if [ "$ACTION" = "update-lambda" ]; then
    echo "    Lambda function: ${LAMBDA_FUNCTION_NAME}"
fi
