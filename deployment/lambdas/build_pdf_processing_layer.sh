#!/bin/bash
# Build the PDF processing Lambda layer using Docker
# Contains: pymupdf, pikepdf, lxml (for PDF manipulation/accessibility tagging)

set -e

LAYER_NAME="pdf-processing"
OUTPUT_FILE="pdf-processing-layer.zip"

echo "🏗️  Building PDF Processing Lambda Layer..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pdf_processing_build/ $OUTPUT_FILE

# Create build directory
mkdir -p pdf_processing_build

# Create requirements file for PDF processing
cat > pdf_processing_build/requirements.txt << 'EOF'
pikepdf>=8.0.0
pymupdf>=1.23.0
EOF

# Create Dockerfile for building Python dependencies
cat > pdf_processing_build/Dockerfile << 'EOF'
FROM public.ecr.aws/lambda/python:3.12

# Install build dependencies for native packages (pikepdf needs qpdf)
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
docker build --platform linux/amd64 -t pdf-processing-layer-builder pdf_processing_build/

# Extract the layer files
echo "📦 Extracting Python packages..."
mkdir -p pdf_processing_build/layer/python
CONTAINER_ID=$(docker create pdf-processing-layer-builder)
docker cp "$CONTAINER_ID:/opt/python/." pdf_processing_build/layer/python/
docker rm "$CONTAINER_ID"

# Create layer info file
echo "📝 Creating layer info..."
cat > pdf_processing_build/layer/python/PDF_PROCESSING_LAYER_INFO.txt << EOF
PDF Processing Lambda Layer
Built: $(date)
Python: 3.12
Architecture: x86_64

Contents:
- pymupdf/fitz (PDF rendering and manipulation)
- pikepdf (PDF/A compliance and accessibility)
- lxml (XML processing for PDF metadata)

Usage:
  import fitz  # PyMuPDF
  import pikepdf

Attach to: remediation_analyzer (PDF accessibility tagging)

Version: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
EOF

# Create zip file
echo "🗜️  Creating layer package..."
cd pdf_processing_build/layer
zip -r "../../$OUTPUT_FILE" . -q
cd ../..

# Cleanup build directory
rm -rf pdf_processing_build/

# Get size info
LAYER_SIZE=$(du -h $OUTPUT_FILE | cut -f1)
echo ""
echo "✅ Layer built successfully!"
echo "   File: $OUTPUT_FILE"
echo "   Size: $LAYER_SIZE"
echo ""
echo "Attach to functions that need PDF manipulation (remediation_analyzer)"
