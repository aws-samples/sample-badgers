#!/bin/bash
# Syncs s3_files directory to the deployed S3 bucket
# Usage: ./sync_s3_files.sh [--profile PROFILE]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S3_FILES_DIR="$SCRIPT_DIR/s3_files"
STACK_PREFIX="badgers"

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

echo "Fetching S3 bucket name from CDK outputs..."

# Get bucket name from stack output
S3_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_PREFIX}-s3" \
    --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
    --output text $_AWS_PROFILE_ARG 2>/dev/null || echo "")

if [[ -z "$S3_BUCKET" ]]; then
    echo "Error: Could not fetch S3 bucket name from stack ${STACK_PREFIX}-s3"
    exit 1
fi

echo "Syncing $S3_FILES_DIR to s3://$S3_BUCKET..."
aws s3 sync "$S3_FILES_DIR" "s3://$S3_BUCKET" $_AWS_PROFILE_ARG

echo "Done! Files synced to s3://$S3_BUCKET"
