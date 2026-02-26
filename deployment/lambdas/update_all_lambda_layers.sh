#!/bin/bash
# Update the foundation layer to the latest version on all non-container badger lambdas.
# Usage: ./update_all_lambda_layers.sh [layer_version]
#   If layer_version is omitted, the latest published version is used.

set -e

LAYER_NAME="analyzer-foundation"
REGION="${AWS_REGION:-us-west-2}"
FUNCTION_PREFIX="badgers_"

# Container-based lambdas — these don't use layers
SKIP_FUNCTIONS=("badgers_image_enhancer" "badgers_remediation_analyzer")

# ---------------------------------------------------------------------------
# Resolve layer ARN
# ---------------------------------------------------------------------------
if [ -n "$1" ]; then
    LAYER_VERSION="$1"
else
    echo "🔍 Finding latest version of layer '${LAYER_NAME}'..."
    LAYER_VERSION=$(aws lambda list-layer-versions \
        --layer-name "$LAYER_NAME" \
        --region "$REGION" \
        --query 'LayerVersions[0].Version' \
        --output text)

    if [ -z "$LAYER_VERSION" ] || [ "$LAYER_VERSION" = "None" ]; then
        echo "❌ No versions found for layer '${LAYER_NAME}'"
        exit 1
    fi
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAYER_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:layer:${LAYER_NAME}:${LAYER_VERSION}"

echo "📦 Layer ARN: ${LAYER_ARN}"
echo ""

# ---------------------------------------------------------------------------
# List badger lambdas
# ---------------------------------------------------------------------------
FUNCTIONS=$(aws lambda list-functions \
    --region "$REGION" \
    --query "Functions[?starts_with(FunctionName, '${FUNCTION_PREFIX}')].FunctionName" \
    --output text)

if [ -z "$FUNCTIONS" ]; then
    echo "❌ No functions found with prefix '${FUNCTION_PREFIX}'"
    exit 1
fi

UPDATED=0
SKIPPED=0

for FUNC in $FUNCTIONS; do
    # Skip container-based functions
    SKIP=false
    for S in "${SKIP_FUNCTIONS[@]}"; do
        if [ "$FUNC" = "$S" ]; then
            SKIP=true
            break
        fi
    done
    if [ "$SKIP" = true ]; then
        echo "⏭  Skipping ${FUNC} (container)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Get current layers
    CURRENT_LAYERS=$(aws lambda get-function-configuration \
        --function-name "$FUNC" \
        --region "$REGION" \
        --query 'Layers[].Arn' \
        --output text)

    # Build new layer list — swap foundation, keep everything else
    NEW_LAYERS=""
    FOUND=false
    if [ -n "$CURRENT_LAYERS" ]; then
        for L in $CURRENT_LAYERS; do
            if [[ $L == *"${LAYER_NAME}"* ]]; then
                NEW_LAYERS="${NEW_LAYERS} ${LAYER_ARN}"
                FOUND=true
            else
                NEW_LAYERS="${NEW_LAYERS} ${L}"
            fi
        done
    fi

    if [ "$FOUND" = false ]; then
        NEW_LAYERS="${LAYER_ARN} ${NEW_LAYERS}"
    fi

    NEW_LAYERS=$(echo "$NEW_LAYERS" | xargs)

    aws lambda update-function-configuration \
        --function-name "$FUNC" \
        --region "$REGION" \
        --layers $NEW_LAYERS \
        --output text > /dev/null

    echo "✅ ${FUNC}"
    UPDATED=$((UPDATED + 1))
done

echo ""
echo "Done — updated ${UPDATED}, skipped ${SKIPPED} container lambdas."
