#!/bin/bash
# Check status of the remediation_analyzer Lambda
# Usage: ./check_remediation_status.sh [deployment_id]
set -e
export AWS_PAGER=""

DEPLOYMENT_ID="${1:-89a27522}"
REGION="${AWS_REGION:-us-west-2}"
FUNCTION_NAME="badgers_remediation_analyzer"

echo "=== Lambda Function Status ==="
aws lambda get-function-configuration \
  --function-name "${FUNCTION_NAME}" \
  --region ${REGION} \
  --query '{State: State, LastUpdateStatus: LastUpdateStatus, Runtime: PackageType, MemorySize: MemorySize, Timeout: Timeout, ImageUri: Code.ImageUri, LastModified: LastModified}' \
  --output table

echo ""
echo "=== IAM Role Inline Policies ==="
ROLE_NAME="lambda-analyzer-role-${DEPLOYMENT_ID}"
aws iam list-role-policies --role-name "${ROLE_NAME}" --output table

echo ""
echo "=== Bedrock Invoke Policy ==="
aws iam get-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name bedrock-invoke \
  --query 'PolicyDocument' \
  --output json 2>/dev/null || echo "(no bedrock-invoke policy found)"
