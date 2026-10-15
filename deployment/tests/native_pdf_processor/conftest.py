"""Expose the native PDF Lambda and the shared foundation package to pytest."""

import importlib.util
import sys
from pathlib import Path

import pytest

DEPLOYMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOYMENT / "badgers-foundation"))

_SPEC = importlib.util.spec_from_file_location(
    "native_pdf_processor_handler",
    DEPLOYMENT / "lambdas" / "code" / "native_pdf_processor" / "lambda_handler.py",
)
assert _SPEC and _SPEC.loader
native_pdf_processor_handler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = native_pdf_processor_handler
_SPEC.loader.exec_module(native_pdf_processor_handler)


@pytest.fixture(scope="session")
def native_pdf():
    return native_pdf_processor_handler
