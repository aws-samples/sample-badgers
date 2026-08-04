#!/bin/bash
# Targeted cleanup for a stuck AgentCore Runtime stack: deletes the runtime, then
# the stack. ./destroy.sh does this as part of a full teardown; this is for when
# only the runtime stack is wedged.
#
# Usage: DEPLOYMENT_ID=dev ./cleanup-stack.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/common.sh"
load_suffix

REGION="${AWS_REGION}"
STACK_NAME="$(_sn RuntimeWebSocket)"

echo "=== Cleaning up ${STACK_NAME} ==="

# Step 1: Get the Runtime ID from the stack
echo "Step 1: Getting Runtime ID from CloudFormation stack..."
RUNTIME_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --region $REGION \
  --query 'Stacks[0].Outputs[?OutputKey==`RuntimeId`].OutputValue' \
  --output text 2>/dev/null || echo "")

if [ -z "$RUNTIME_ID" ]; then
  echo "  Could not find Runtime ID in stack outputs. Trying to find it directly..."
  RUNTIME_ID=$(aws cloudformation describe-stack-resources \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'StackResources[?ResourceType==`AWS::BedrockAgentCore::Runtime`].PhysicalResourceId' \
    --output text 2>/dev/null || echo "")
fi

if [ -n "$RUNTIME_ID" ]; then
  echo "  Found Runtime ID: $RUNTIME_ID"

  # Step 2: Delete the BedrockAgentCore Runtime manually
  echo "Step 2: Deleting BedrockAgentCore Runtime..."
  aws bedrock-agentcore-control delete-runtime \
    --runtime-id "$RUNTIME_ID" \
    --region $REGION 2>/dev/null || echo "  Runtime already deleted or doesn't exist"

  echo "  Waiting for runtime deletion to complete..."
  sleep 5
else
  echo "  No Runtime ID found - it may already be deleted"
fi

# Step 3: Delete the CloudFormation stack
echo "Step 3: Deleting CloudFormation stack..."
aws cloudformation delete-stack \
  --stack-name $STACK_NAME \
  --region $REGION

echo "  Waiting for stack deletion..."
aws cloudformation wait stack-delete-complete \
  --stack-name $STACK_NAME \
  --region $REGION

echo "=== Stack deletion complete! ==="
