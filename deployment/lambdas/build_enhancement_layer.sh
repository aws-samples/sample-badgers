#!/bin/bash
# Build the image enhancement Lambda layer using Docker
# NOTE: This layer is currently UNUSED. The active enhancement path is the
# container-based agentic enhancer at lambdas/containers/image_enhancer/.
# The enhancement_eligible_functions list in lambda_stack.py is empty.
# This script is retained for potential future use with non-container Lambdas.

set -e

LAYER_NAME="image-enhancement"
OUTPUT_FILE="enhancement-layer.zip"

echo "🏗️  Building Enhancement Lambda Layer..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf enhancement_build/ $OUTPUT_FILE

# Create build directory
mkdir -p enhancement_build

# Create Dockerfile for building Python dependencies
cat > enhancement_build/Dockerfile << 'EOF'
FROM public.ecr.aws/lambda/python:3.12

# Install build dependencies
RUN dnf install -y gcc gcc-c++ python3-devel && dnf clean all

# Install OpenCV (headless) and NumPy
RUN pip install opencv-python-headless numpy pillow -t /opt/python/ --quiet

CMD ["echo", "Build complete"]
EOF

# Build the Docker image for x86_64 (Lambda default architecture)
echo "🐳 Building Docker image for x86_64..."
docker build --platform linux/amd64 -t enhancement-layer-builder enhancement_build/

# Extract the layer files
echo "📦 Extracting Python packages..."
mkdir -p enhancement_build/layer/python
CONTAINER_ID=$(docker create enhancement-layer-builder)
docker cp "$CONTAINER_ID:/opt/python/." enhancement_build/layer/python/
docker rm "$CONTAINER_ID"

# Copy enhancement modules
# NOTE: The historical_document_enhancer module is no longer bundled in this layer.
# Enhancement logic now runs in the container-based agentic enhancer
# (lambdas/containers/image_enhancer/) which has its own dependencies.
# This layer only provides OpenCV/numpy/pillow for any Lambda that needs them.

# Create layer info file
echo "📝 Creating layer info..."
cat > enhancement_build/layer/python/ENHANCEMENT_LAYER_INFO.txt << EOF
Image Enhancement Lambda Layer
Built: $(date)
Python: 3.12
Architecture: x86_64

Contents:
- cv2/ (OpenCV headless ~45 MB)
- numpy/ (~30 MB)
- PIL/ (Pillow)

Note: Enhancement logic runs in the container-based agentic enhancer
(lambdas/containers/image_enhancer/). This layer only provides shared
image processing dependencies for any Lambda that needs them.

Version: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
EOF

# Create zip file
echo "🗜️  Creating layer package..."
cd enhancement_build/layer
zip -r "../../$OUTPUT_FILE" . -q
cd ../..

# Cleanup build directory
rm -rf enhancement_build/

# Get size info
LAYER_SIZE=$(du -h $OUTPUT_FILE | cut -f1)
echo ""
echo "✅ Layer built successfully!"
echo "   File: $OUTPUT_FILE"
echo "   Size: $LAYER_SIZE"
echo ""
echo "Next steps:"
echo "  1. Review layer contents: unzip -l $OUTPUT_FILE"
echo "  2. Deploy with CDK: cdk deploy"
