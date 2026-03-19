"""PDF syntax repair utilities.

Two-pass repair approach adapted from fix-corrupt-pdfs:
  Pass 1 (PyMuPDF): clean=True + garbage=4
  Pass 2 (pikepdf): linearize=True + object_stream_mode=generate

Both passes are fault-tolerant — if either fails, we fall back to the previous output.
"""

import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pikepdf

logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    """Result of a PDF repair operation."""

    input_path: str
    output_path: str
    original_size: int = 0
    repaired_size: int = 0
    pass1_ok: bool = False
    pass2_ok: bool = False
    logs: list[str] = field(default_factory=list)

    @property
    def size_delta(self) -> int:
        return self.repaired_size - self.original_size

    @property
    def any_repair_applied(self) -> bool:
        return self.pass1_ok or self.pass2_ok


def repair_pdf(pdf_path: str, work_dir: Path | None = None) -> RepairResult:
    """Two-pass PDF syntax repair. Returns a RepairResult with paths and diagnostics.

    Pass 1 (PyMuPDF): clean content streams + deduplicate objects.
    Pass 2 (pikepdf): rebuild xref table + normalize object streams.

    Each pass is fault-tolerant — failure falls back to previous output.
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="pdf_repair_"))

    current = pdf_path
    result = RepairResult(input_path=pdf_path, output_path=pdf_path)
    result.original_size = Path(pdf_path).stat().st_size

    # --- Pass 1: PyMuPDF ---
    pass1_out = str(work_dir / "pass1_pymupdf.pdf")
    try:
        doc = fitz.open(current)
        doc.save(pass1_out, clean=True, garbage=4)
        doc.close()
        current = pass1_out
        result.pass1_ok = True
        result.logs.append(f"Pass 1 (PyMuPDF): OK → {pass1_out}")
    except Exception as e:
        result.logs.append(f"Pass 1 (PyMuPDF): FAILED, skipping — {e}")

    # --- Pass 2: pikepdf ---
    pass2_out = str(work_dir / "pass2_pikepdf.pdf")
    try:
        pdf = pikepdf.open(current)
        pdf.save(
            pass2_out,
            linearize=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )
        pdf.close()
        current = pass2_out
        result.pass2_ok = True
        result.logs.append(f"Pass 2 (pikepdf): OK → {pass2_out}")
    except Exception as e:
        result.logs.append(f"Pass 2 (pikepdf): FAILED, skipping — {e}")

    result.output_path = current
    result.repaired_size = Path(current).stat().st_size

    for log_line in result.logs:
        logger.info(log_line)

    return result
