#!/bin/bash
# Build the specialist foundation Lambda layer using Docker

set -e

echo "🏗️  Building Specialist Foundation Lambda Layer..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf layer/ layer.zip foundation_build/

# Create build directory
mkdir -p foundation_build

# Copy requirements.txt to build context
cp requirements.txt foundation_build/

# Create Dockerfile for building Python dependencies
cat > foundation_build/Dockerfile << 'EOF'
FROM public.ecr.aws/lambda/python:3.12

# Install build dependencies for native packages (like pikepdf which needs qpdf)
RUN dnf install -y \
    gcc \
    gcc-c++ \
    python3-devel \
    qpdf-devel \
    qpdf-libs \
    && dnf clean all

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install -r /tmp/requirements.txt -t /opt/python/ --quiet

CMD ["echo", "Build complete"]
EOF

# Build the Docker image for x86_64 (Lambda default architecture)
echo "🐳 Building Docker image for x86_64..."
docker build --platform linux/amd64 -t foundation-layer-builder foundation_build/

# Extract the layer files
echo "📦 Extracting Python packages..."
mkdir -p layer/python
CONTAINER_ID=$(docker create foundation-layer-builder)
docker cp "$CONTAINER_ID:/opt/python/." layer/python/
docker rm "$CONTAINER_ID"

# Cleanup build directory
rm -rf foundation_build/

echo "✅ Packages installed"



# Copy foundation framework
echo "📋 Copying foundation framework..."
mkdir -p layer/python/foundation
cp -r ../badgers-foundation/foundation/*.py layer/python/foundation/
touch layer/python/foundation/__init__.py

# Copy config utilities
echo "📋 Copying config utilities..."
mkdir -p layer/python/config
cp ../badgers-foundation/config/config.py layer/python/config/
# Don't copy specialist_config.json - each function has its own manifest
touch layer/python/config/__init__.py

# Copy core system prompts (used by all specialists)
echo "📋 Copying core system prompts..."
mkdir -p layer/python/prompts/core_system_prompts
cp -r ../badgers-foundation/foundation/core_system_prompts/* layer/python/prompts/core_system_prompts/

# Create layer info file
echo "📝 Creating layer info..."
cat > layer/python/LAYER_INFO.txt << EOF
Specialist Foundation Lambda Layer
Built: $(date)
Python: 3.12
Architecture: x86_64, arm64

Contents:
- foundation/ (7 modules)
- config/ (1 module)
- prompts/core_system_prompts/ (system wrappers)
- boto3, pillow, botocore

Usage:
Import foundation in your specialist:
  from foundation.specialist_foundation import SpecialistFoundation

Version: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
EOF

# Create zip file
echo "🗜️  Creating layer package..."
cd layer
zip -r ../layer.zip . -q
cd ..

# Get size info
LAYER_SIZE=$(du -h layer.zip | cut -f1)
echo ""
echo "✅ Layer built successfully!"
echo "   File: layer.zip"
echo "   Size: $LAYER_SIZE"
echo ""
echo "Next steps:"
echo "  1. Review layer contents: unzip -l layer.zip"
echo "  2. Deploy layer: ./deploy_foundation_layer.sh"
echo "  3. Use in functions: see examples/"
