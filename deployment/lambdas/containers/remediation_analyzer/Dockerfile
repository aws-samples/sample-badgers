FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy foundation code
COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/

# Copy the PDF accessibility tagger and handler
COPY pdf_accessibility_tagger.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.lambda_handler"]
