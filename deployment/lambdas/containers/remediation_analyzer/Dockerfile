FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for PDF processing
# pymupdf and pikepdf may need additional libraries
RUN yum install -y \
    mupdf \
    mupdf-devel \
    && yum clean all \
    && rm -rf /var/cache/yum

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Foundation library provided via Lambda Layer at /opt/python/foundation/

# Copy application code (all Python modules)
COPY pdf_accessibility_models.py ${LAMBDA_TASK_ROOT}/
COPY pdf_accessibility_auditor.py ${LAMBDA_TASK_ROOT}/
COPY pdf_accessibility_tagger.py ${LAMBDA_TASK_ROOT}/
COPY cell_grid_resolver.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_handler.lambda_handler"]
