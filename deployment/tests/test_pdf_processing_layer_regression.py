"""Regression coverage for the retired, unused v5 PDF-processing layer."""

from __future__ import annotations

from pathlib import Path


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOYMENT_DIR.parent


def test_unused_pdf_processing_layer_is_not_built_or_deployed() -> None:
    assert not (DEPLOYMENT_DIR / "lambdas" / "build_pdf_processing_layer.sh").exists()

    deploy_script = (REPO_ROOT / "deploy.sh").read_text(encoding="utf-8")
    lambda_stack = (DEPLOYMENT_DIR / "stacks" / "lambda_stack.py").read_text(
        encoding="utf-8"
    )

    assert "build_pdf_processing_layer.sh" not in deploy_script
    assert "PdfProcessingLayer" not in lambda_stack
