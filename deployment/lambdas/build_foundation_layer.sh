#!/bin/bash
# Build the specialist foundation Lambda layer using Docker

set -e

echo "🏗️  Building Specialist Foundation Lambda Layer..."

# Build in an isolated temporary directory.  On macOS, Finder/Spotlight can
# recreate layer/python/.DS_Store between `rm -rf layer/` and the next command,
# making the old in-place cleanup fail with "Directory not empty."  The layer
# zip is the only deployable output; staging it outside the watched workspace
# makes the build deterministic and leaves no mutable layer/ tree behind.
SCRIPT_DIR="$(pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/badgers-foundation-layer.XXXXXX")"
FOUNDATION_BUILD_DIR="${WORK_DIR}/foundation_build"
LAYER_DIR="${WORK_DIR}/layer"
cleanup() {
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

echo "🧹 Cleaning previous layer archive..."
rm -f layer.zip

# Create build directory
mkdir -p "${FOUNDATION_BUILD_DIR}"

# Copy requirements.txt to build context
cp requirements.txt "${FOUNDATION_BUILD_DIR}/"

# Create Dockerfile for building Python dependencies
cat > "${FOUNDATION_BUILD_DIR}/Dockerfile" << 'EOF'
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
docker build --platform linux/amd64 -t foundation-layer-builder "${FOUNDATION_BUILD_DIR}"

# Extract the layer files
echo "📦 Extracting Python packages..."
mkdir -p "${LAYER_DIR}/python"
CONTAINER_ID=$(docker create foundation-layer-builder)
docker cp "$CONTAINER_ID:/opt/python/." "${LAYER_DIR}/python/"
docker rm "$CONTAINER_ID"

echo "✅ Packages installed"



# Copy foundation framework
echo "📋 Copying foundation framework..."
mkdir -p "${LAYER_DIR}/python/foundation"
cp -r ../badgers-foundation/foundation/*.py "${LAYER_DIR}/python/foundation/"
touch "${LAYER_DIR}/python/foundation/__init__.py"

# Copy config utilities
echo "📋 Copying config utilities..."
mkdir -p "${LAYER_DIR}/python/config"
cp ../badgers-foundation/config/config.py "${LAYER_DIR}/python/config/"
# Don't copy specialist_config.json - each function has its own manifest
touch "${LAYER_DIR}/python/config/__init__.py"

# Copy core system prompts (used by all specialists)
echo "📋 Copying core system prompts..."
mkdir -p "${LAYER_DIR}/python/prompts/core_system_prompts"
cp -r ../badgers-foundation/foundation/core_system_prompts/* "${LAYER_DIR}/python/prompts/core_system_prompts/"

# Create layer info file
echo "📝 Creating layer info..."
cat > "${LAYER_DIR}/python/LAYER_INFO.txt" << EOF
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
(
    cd "${LAYER_DIR}"
    zip -r "${SCRIPT_DIR}/layer.zip" . -q
)

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
