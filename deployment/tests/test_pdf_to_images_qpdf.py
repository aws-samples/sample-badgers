"""qpdf exit-code handling for the Linux PDF converter Lambda."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT / "badgers-foundation"))
_SPEC = importlib.util.spec_from_file_location(
    "pdf_to_images_qpdf_handler",
    DEPLOYMENT / "lambdas" / "code" / "pdf_to_images_converter" / "lambda_handler.py",
)
if not (_SPEC and _SPEC.loader):
    raise AssertionError
handler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = handler
_SPEC.loader.exec_module(handler)


def _qpdf_result(returncode: int, output: bytes = b"repaired") -> object:
    def run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        Path(args[-1]).write_bytes(output)
        return SimpleNamespace(returncode=returncode, stdout="", stderr="warning")

    return run


@pytest.mark.parametrize("returncode", [0, 3])
def test_qpdf_accepts_success_and_success_with_warnings(
    monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    monkeypatch.setattr(handler.subprocess, "run", _qpdf_result(returncode))

    if not (handler._decrypt_pdf_if_needed(b"%PDF-test") == b"repaired"):
        raise AssertionError


def test_qpdf_rejects_a_real_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handler.subprocess, "run", _qpdf_result(2))

    with pytest.raises(RuntimeError, match="qpdf decryption failed"):
        handler._decrypt_pdf_if_needed(b"%PDF-test")


def test_qpdf_rejects_an_empty_warning_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handler.subprocess, "run", _qpdf_result(3, b""))

    with pytest.raises(RuntimeError, match="without producing"):
        handler._decrypt_pdf_if_needed(b"%PDF-test")
