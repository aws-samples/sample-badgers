#!/usr/bin/env bash
#
# Local testing script for remediation analyzer container
#
# This script runs the container locally and allows you to test
# Lambda invocations without deploying to AWS.
#
# Prerequisites:
#   - Docker installed and running
#   - Container image built (run: ./build.sh --build-only)
#   - AWS credentials available (for Bedrock and S3 access)
#
# Usage:
#   ./test-local.sh
#
# Then in another terminal:
#   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#     -d @test-event.json
#

set -e

IMAGE_NAME="remediation-analyzer"
IMAGE_TAG="latest"
PORT="${PORT:-9000}"

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Error: Docker image not found: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Build it first with: ./build.sh --build-only"
    exit 1
fi

# Load AWS credentials from environment or AWS CLI
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "==> Loading AWS credentials from default profile..."
    AWS_PROFILE="${AWS_PROFILE:-default}"
    eval $(aws configure export-credentials --profile $AWS_PROFILE --format env | sed 's/^/export /')
fi

# Environment variables for the container
ENV_VARS=(
    "-e" "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
    "-e" "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
    "-e" "AWS_REGION=${AWS_REGION:-us-west-2}"
    "-e" "CONFIG_BUCKET=${CONFIG_BUCKET:-}"
    "-e" "OUTPUT_BUCKET=${OUTPUT_BUCKET:-}"
    "-e" "ANALYZER_NAME=${ANALYZER_NAME:-remediation_analyzer}"
    "-e" "LOGGING_LEVEL=${LOGGING_LEVEL:-INFO}"
)

# Add session token if available (for temporary credentials)
if [ -n "$AWS_SESSION_TOKEN" ]; then
    ENV_VARS+=("-e" "AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN")
fi

echo "==> Starting container on port ${PORT}..."
echo "    Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "    AWS Region: ${AWS_REGION:-us-west-2}"
echo "    Config Bucket: ${CONFIG_BUCKET:-(not set)}"
echo "    Output Bucket: ${OUTPUT_BUCKET:-(not set)}"
echo ""
echo "==> Test the Lambda function with:"
echo "    curl -XPOST \"http://localhost:${PORT}/2015-03-31/functions/function/invocations\" \\"
echo "      -d '{\"pdf_path\": \"s3://bucket/test.pdf\", \"session_id\": \"test-001\"}'"
echo ""
echo "==> Or use the test event file:"
echo "    curl -XPOST \"http://localhost:${PORT}/2015-03-31/functions/function/invocations\" \\"
echo "      -d @test-event.json"
echo ""
echo "Press Ctrl+C to stop the container"
echo ""

# Run container
docker run --rm -it \
    -p "${PORT}:8080" \
    "${ENV_VARS[@]}" \
    "${IMAGE_NAME}:${IMAGE_TAG}"
