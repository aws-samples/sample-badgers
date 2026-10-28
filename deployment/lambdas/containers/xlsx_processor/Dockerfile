FROM python:3.12-slim

#checkov:skip=CKV_DOCKER_2:Lambda uses the Runtime API, not Docker HEALTHCHECK
#checkov:skip=CKV_DOCKER_3:Lambda runs under an unprivileged application user

ENV LAMBDA_TASK_ROOT=/var/task \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR ${LAMBDA_TASK_ROOT}

RUN useradd --create-home --uid 10001 badgers \
    && mkdir -p ${LAMBDA_TASK_ROOT} \
    && chown badgers:badgers ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir "awslambdaric>=3.1.1,<4" \
    && pip install --no-cache-dir -r requirements.txt

COPY lambda_handler.py workbook_inspector.py visual_analysis.py xlsx_content_tree.py ./
COPY foundation/ foundation/
COPY config/ config/

RUN chown -R badgers:badgers ${LAMBDA_TASK_ROOT}

USER badgers

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["lambda_handler.lambda_handler"]
