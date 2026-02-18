FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application code
COPY pdf_accessibility_models.py ${LAMBDA_TASK_ROOT}/
COPY pdf_accessibility_auditor.py ${LAMBDA_TASK_ROOT}/
COPY pdf_accessibility_tagger.py ${LAMBDA_TASK_ROOT}/
COPY cell_grid_resolver.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.lambda_handler"]