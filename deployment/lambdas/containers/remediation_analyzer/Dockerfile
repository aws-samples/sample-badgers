FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy foundation module (required for BedrockClient, AnalyzerFoundation, etc.)
COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/

# Copy config module (required by foundation)
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Copy application code
COPY pdf_accessibility_models.py ${LAMBDA_TASK_ROOT}/
COPY pdf_accessibility_auditor.py ${LAMBDA_TASK_ROOT}/
COPY pdf_accessibility_tagger.py ${LAMBDA_TASK_ROOT}/
COPY cell_grid_resolver.py ${LAMBDA_TASK_ROOT}/
COPY diagnostic_visualizer.py ${LAMBDA_TASK_ROOT}/
COPY pdf_syntax_repair.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_handler.lambda_handler"]
