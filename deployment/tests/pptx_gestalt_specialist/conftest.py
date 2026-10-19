"""Load the PPTX Gestalt Lambda with only its pure geometry dependencies mocked."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def gestalt_module():
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *args, **kwargs: None
    sys.modules.setdefault("boto3", boto3)

    foundation = types.ModuleType("foundation")
    foundation.job_state = types.SimpleNamespace()
    sys.modules.setdefault("foundation", foundation)

    bedrock_client = types.ModuleType("foundation.bedrock_client")
    bedrock_client.BedrockClient = object
    sys.modules.setdefault("foundation.bedrock_client", bedrock_client)

    model_selection = types.ModuleType("foundation.model_selection")
    model_selection.max_tokens_from = lambda *args, **kwargs: 1
    model_selection.parse_model_selection = lambda *args, **kwargs: {}
    sys.modules.setdefault("foundation.model_selection", model_selection)

    config_loader = types.ModuleType("foundation.s3_config_loader")
    config_loader.load_manifest_from_s3 = lambda *args, **kwargs: {}
    sys.modules.setdefault("foundation.s3_config_loader", config_loader)

    path = (
        Path(__file__).resolve().parents[2]
        / "lambdas"
        / "code"
        / "pptx_gestalt_specialist"
        / "lambda_handler.py"
    )
    spec = importlib.util.spec_from_file_location("pptx_gestalt_handler", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
