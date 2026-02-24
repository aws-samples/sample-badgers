#!/bin/bash
# Deploy remediation_analyzer container and update Lambda + IAM
# Usage: ./deploy_remediation_analyzer.sh <deployment_id>
set -e

# Disable AWS CLI pager (avoids getting stuck in less/more)
export AWS_PAGER=""

DEPLOYMENT_ID="${1:?Usage: $0 <deployment_id>}"
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_NAME="lambda-analyzer-role-${DEPLOYMENT_ID}"
ECR_REPO="badgers-${DEPLOYMENT_ID}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
FUNCTION_NAME="badgers_remediation_analyzer"

echo "Deployment ID: ${DEPLOYMENT_ID}"
echo "Region: ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "ECR: ${ECR_URI}"

# 1. Add bedrock:InvokeModel permission to the Lambda role
echo ""
echo "=== Updating IAM policy ==="
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name bedrock-invoke \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:'"${ACCOUNT_ID}"':inference-profile/*"
      ]
    }]
  }'
echo "✓ IAM policy updated"

# 2. ECR login
echo ""
echo "=== ECR login ==="
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# 3. Copy foundation + config into build context
echo ""
echo "=== Preparing build context ==="
cp -r ./layer/python/foundation ./containers/remediation_analyzer/foundation
cp -r ./layer/python/config ./containers/remediation_analyzer/config

# 4. Build
echo ""
echo "=== Building container ==="
docker build \
  --platform linux/amd64 \
  --provenance=false \
  -t "${ECR_URI}:remediation_analyzer" \
  ./containers/remediation_analyzer

# 5. Clean up build context copies
rm -rf ./containers/remediation_analyzer/foundation ./containers/remediation_analyzer/config

# 6. Push
echo ""
echo "=== Pushing to ECR ==="
docker push "${ECR_URI}:remediation_analyzer"
echo "✓ Image pushed"

# 7. Update Lambda
echo ""
echo "=== Updating Lambda function ==="
aws lambda update-function-code \
  --function-name "${FUNCTION_NAME}" \
  --image-uri "${ECR_URI}:remediation_analyzer" \
  --region ${REGION}
echo "✓ Lambda updated"

echo ""
echo "=== Done ==="

#!/bin/bash
# Check status of the remediation_analyzer Lambda
# Usage: ./check_remediation_status.sh [deployment_id]
set -e
export AWS_PAGER=""

DEPLOYMENT_ID="${1:?Usage: $0 <deployment_id>}"
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
