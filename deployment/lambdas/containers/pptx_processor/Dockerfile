FROM python:3.12-slim

#checkov:skip=CKV_DOCKER_2:Lambda uses the Runtime API, not Docker HEALTHCHECK
#checkov:skip=CKV_DOCKER_3:Lambda provides a least-privileged default user

# LibreOffice is used only to render slides selected by the native page plan.
# The Amazon Linux Lambda image does not reliably publish LibreOffice RPMs, so
# use Debian's maintained packages with the Lambda Runtime Interface Client.
ENV LAMBDA_TASK_ROOT=/var/task
WORKDIR ${LAMBDA_TASK_ROOT}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-core \
        libreoffice-impress \
        fontconfig \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir awslambdaric -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["lambda_handler.lambda_handler"]
