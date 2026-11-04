FROM public.ecr.aws/lambda/python:3.12

# Native PDF inspection requires PyMuPDF, whose full supported dependency line
# can exceed the ZIP Lambda + layer package limit. This container is invoked
# only by Lambda's Runtime API; it does not expose a listener.
#checkov:skip=CKV_DOCKER_2:Lambda invokes the Runtime API, not Docker HEALTHCHECK
#checkov:skip=CKV_DOCKER_3:Lambda provides a least-privileged default user

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

RUN chmod 644 ${LAMBDA_TASK_ROOT}/lambda_handler.py

CMD ["lambda_handler.lambda_handler"]
