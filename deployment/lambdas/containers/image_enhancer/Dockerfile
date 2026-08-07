FROM public.ecr.aws/lambda/python:3.12

# Lambda container image. Lambda manages the container lifecycle through the
# Runtime API rather than Docker healthchecks, and per the AWS Lambda docs
# (https://docs.aws.amazon.com/lambda/latest/dg/images-create.html) the service
# already runs function code as a least-privileged default Linux user, so a
# USER directive is neither required nor recommended here.
#checkov:skip=CKV_DOCKER_2:Lambda uses the Runtime API, not Docker HEALTHCHECK
#checkov:skip=CKV_DOCKER_3:Lambda provides a least-privileged default user

# Install system dependencies for OpenCV
RUN dnf install -y \
    mesa-libGL \
    && dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application code (new agentic modules)
COPY enhancement_tools.py ${LAMBDA_TASK_ROOT}/
COPY image_state.py ${LAMBDA_TASK_ROOT}/
COPY agentic_enhancer.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Copy the shared foundation package (job_state for DynamoDB job tracking).
# build_container_lambdas.sh / build_single_container.sh stage this directory
# into the build context from ./layer/python/foundation before docker build.
COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/

# Fix file permissions
RUN chmod 644 ${LAMBDA_TASK_ROOT}/*.py

CMD ["lambda_handler.lambda_handler"]
