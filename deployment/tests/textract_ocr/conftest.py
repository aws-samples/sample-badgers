"""Expose the Textract OCR Lambda and shared foundation package to pytest."""

import importlib.util
import sys
from pathlib import Path

import pytest

DEPLOYMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOYMENT / "badgers-foundation"))

_SPEC = importlib.util.spec_from_file_location(
    "textract_ocr_handler",
    DEPLOYMENT / "lambdas" / "code" / "textract_ocr" / "lambda_handler.py",
)
assert _SPEC and _SPEC.loader
textract_ocr_handler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = textract_ocr_handler
_SPEC.loader.exec_module(textract_ocr_handler)


@pytest.fixture(scope="session")
def textract_ocr():
    return textract_ocr_handler
