#!/bin/bash
# Build poppler Lambda layer using Docker with official Lambda base image

set -e

LAYER_NAME="poppler-utils"
OUTPUT_FILE="poppler-layer.zip"

echo "🏗️  Building Poppler Lambda Layer with Docker..."

# Clean previous builds
rm -rf poppler_build/ $OUTPUT_FILE

# Create build directory
mkdir -p poppler_build

# Create Dockerfile for building poppler
cat > poppler_build/Dockerfile << 'EOF'
FROM public.ecr.aws/lambda/python:3.12

# Install poppler and dependencies
RUN dnf install -y poppler-utils && \
    dnf clean all

# Create layer structure
RUN mkdir -p /opt/layer/bin /opt/layer/lib

# Copy poppler binaries
RUN cp /usr/bin/pdftoppm /opt/layer/bin/ && \
    cp /usr/bin/pdfinfo /opt/layer/bin/ && \
    cp /usr/bin/pdftotext /opt/layer/bin/ && \
    cp /usr/bin/pdfimages /opt/layer/bin/

# Copy required shared libraries
RUN for bin in /opt/layer/bin/*; do \
        ldd "$bin" 2>/dev/null | grep "=> /" | awk '{print $3}' | while read lib; do \
            cp -n "$lib" /opt/layer/lib/ 2>/dev/null || true; \
        done; \
    done

# Remove libs that are already in Lambda runtime
RUN cd /opt/layer/lib && rm -f libc.so* libm.so* libpthread.so* libdl.so* librt.so* 2>/dev/null || true

CMD ["echo", "Build complete"]
EOF

# Build the Docker image for x86_64 (Lambda default architecture)
echo "🐳 Building Docker image for x86_64..."
docker build --platform linux/amd64 -t poppler-layer-builder poppler_build/

# Extract the layer files
echo "📦 Extracting layer files..."
CONTAINER_ID=$(docker create poppler-layer-builder)
docker cp "$CONTAINER_ID:/opt/layer" poppler_build/layer
docker rm "$CONTAINER_ID"

# Create the layer zip
echo "🗜️  Creating layer zip..."
cd poppler_build/layer
zip -r "../../$OUTPUT_FILE" . -q
cd ../..

# Cleanup
rm -rf poppler_build/

# Get size info
LAYER_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "✅ Poppler layer built successfully!"
echo "   File: $OUTPUT_FILE"
echo "   Size: $LAYER_SIZE"
echo ""
echo "To deploy:"
echo "  aws lambda publish-layer-version \\"
echo "    --layer-name $LAYER_NAME \\"
echo "    --zip-file fileb://$OUTPUT_FILE \\"
echo "    --compatible-runtimes python3.12 \\"
echo "    --compatible-architectures x86_64"
