FROM debian:bookworm-slim

# Amazon Linux's Lambda repository does not publish LibreOffice. Use Debian's
# maintained headless Writer package and provide the Lambda Runtime Interface
# Client ourselves; Lambda supports custom base images when they implement RIC.
# This image exposes no listener beyond Lambda's invocation Runtime API.
#checkov:skip=CKV_DOCKER_2:Lambda invokes the Runtime API, not Docker HEALTHCHECK
#checkov:skip=CKV_DOCKER_3:Lambda provides the least-privileged execution user
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FUNCTION_DIR=/var/task \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3 \
        python3-venv \
        libreoffice-writer-nogui \
        fontconfig \
        fonts-dejavu-core \
    && python3 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${FUNCTION_DIR}

COPY requirements.txt .
RUN pip install --no-cache-dir \
        "awslambdaric>=3.1.1,<4" \
    && pip install --no-cache-dir -r requirements.txt

COPY docx_processor.py .
COPY lambda_handler.py .
COPY local_intake.py .
COPY schemas/ schemas/
COPY foundation/ foundation/
COPY config/ config/

RUN chmod 644 *.py

ENTRYPOINT ["/opt/venv/bin/python", "-m", "awslambdaric"]
CMD ["lambda_handler.lambda_handler"]
