#!/bin/bash
# Build the image enhancement Lambda layer using Docker

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
echo "📋 Copying enhancement modules..."
mkdir -p enhancement_build/layer/python/enhancement
cp ../badgers-foundation/enhancement/historical_document_enhancer.py enhancement_build/layer/python/enhancement/

# Create __init__.py for the enhancement package
cat > enhancement_build/layer/python/enhancement/__init__.py << 'INITEOF'
"""Image enhancement module for historical/degraded documents."""

from .historical_document_enhancer import (
    HistoricalDocumentEnhancer,
    DocumentType,
    EnhancementLevel,
    EnhancementConfig,
    EnhancementResult,
    enhance_document,
    prepare_for_vision_llm,
)

__all__ = [
    "HistoricalDocumentEnhancer",
    "DocumentType",
    "EnhancementLevel",
    "EnhancementConfig",
    "EnhancementResult",
    "enhance_document",
    "prepare_for_vision_llm",
]
INITEOF

# Create layer info file
echo "📝 Creating layer info..."
cat > enhancement_build/layer/python/ENHANCEMENT_LAYER_INFO.txt << EOF
Image Enhancement Lambda Layer
Built: $(date)
Python: 3.12
Architecture: x86_64

Contents:
- enhancement/ (historical document enhancer)
- cv2/ (OpenCV headless ~45 MB)
- numpy/ (~30 MB)
- PIL/ (Pillow)

Usage:
  from enhancement import prepare_for_vision_llm, HistoricalDocumentEnhancer

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
