FROM public.ecr.aws/lambda/python:3.12

# Lambda container image. Lambda manages the container lifecycle through the
# Runtime API rather than Docker healthchecks, and per the AWS Lambda docs
# (https://docs.aws.amazon.com/lambda/latest/dg/images-create.html) the service
# already runs function code as a least-privileged default Linux user, so a
# USER directive is neither required nor recommended here.
#checkov:skip=CKV_DOCKER_2:Lambda uses the Runtime API, not Docker HEALTHCHECK
#checkov:skip=CKV_DOCKER_3:Lambda provides a least-privileged default user

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy syntax repair (two-pass PDF fix before tagging)
COPY pdf_syntax_repair.py ${LAMBDA_TASK_ROOT}/

# Copy the utils package (all tagging modules)
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/

# Copy the shared foundation package (job_state for DynamoDB job tracking).
# The build scripts stage this directory into the build context from
# ./layer/python/foundation before docker build.
COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/

# Copy the Lambda handler
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_handler.lambda_handler"]
