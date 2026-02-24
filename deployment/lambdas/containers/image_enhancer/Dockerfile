FROM public.ecr.aws/lambda/python:3.12

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

# Fix file permissions
RUN chmod 644 ${LAMBDA_TASK_ROOT}/*.py

CMD ["lambda_handler.lambda_handler"]
