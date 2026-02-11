 n#!/bin/bash
# Update foundation layer version for all analyzer Lambdas
# Usage: ./update_layer_version.sh <layer_arn>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <layer_arn>"
    echo "Example: $0 'arn:aws:lambda:us-west-2:123456789012:layer:analyzer-foundation:88'"
    exit 1
fi

LAYER_ARN=$1
REGION=${AWS_REGION:-us-west-2}

echo "Updating all analyzer Lambdas to layer:"
echo "${LAYER_ARN}"
echo "Region: ${REGION}"
echo ""

# Get all Lambda functions with 'analyzer' in the name
FUNCTIONS=$(aws lambda list-functions \
    --region ${REGION} \
    --query "Functions[?contains(FunctionName, 'analyzer')].FunctionName" \
    --output text)

if [ -z "$FUNCTIONS" ]; then
    echo "No analyzer Lambda functions found"
    exit 1
fi

echo "Found analyzer functions:"
echo "$FUNCTIONS" | tr '\t' '\n'
echo ""

# Update each function
for FUNCTION in $FUNCTIONS; do
    # Skip container-based functions
    if [[ $FUNCTION == "remediation_analyzer" ]]; then
        echo "Skipping ${FUNCTION} (container-based)..."
        continue
    fi

    echo "Updating ${FUNCTION}..."

    # Get current layer configuration
    CURRENT_LAYERS=$(aws lambda get-function-configuration \
        --function-name ${FUNCTION} \
        --region ${REGION} \
        --query 'Layers[].LayerArn' \
        --output text)

    # Build new layer list (replace foundation layer, keep others)
    NEW_LAYERS=""
    FOUND_FOUNDATION=false

    if [ -n "$CURRENT_LAYERS" ]; then
        for LAYER in $CURRENT_LAYERS; do
            if [[ $LAYER == *"foundation"* ]] || [[ $LAYER == *"analyzer-foundation"* ]] || [[ $LAYER == *"badgers-foundation"* ]]; then
                NEW_LAYERS="${NEW_LAYERS} ${LAYER_ARN}"
                FOUND_FOUNDATION=true
            else
                NEW_LAYERS="${NEW_LAYERS} ${LAYER}"
            fi
        done
    fi

    # If no foundation layer was found, add it
    if [ "$FOUND_FOUNDATION" = false ]; then
        NEW_LAYERS="${LAYER_ARN}"
    fi

    # Trim leading/trailing spaces
    NEW_LAYERS=$(echo ${NEW_LAYERS} | xargs)

    # Update the function
    if [ -n "$NEW_LAYERS" ]; then
        aws lambda update-function-configuration \
            --function-name ${FUNCTION} \
            --region ${REGION} \
            --layers ${NEW_LAYERS} \
            --output text > /dev/null
        echo "  ✓ Updated ${FUNCTION}"
    else
        echo "  ⚠ No layers to update for ${FUNCTION}"
    fi
done

echo ""
echo "All analyzer Lambdas updated successfully"
