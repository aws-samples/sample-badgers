"""Run the DOCX native intake pipeline locally without AWS services.

Usage:
    python local_intake.py INPUT.docx OUTPUT_DIRECTORY

The command writes a page-plan JSON artifact, canonical native XML per page, the
rendered PDF used for pagination, and JPEGs only for pages routed to hybrid,
Vision, or Textract OCR
Vision AI. LibreOffice/soffice must be installed locally, or run this command
inside the docx_processor Lambda container.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from docx_processor import create_page_plan, render_docx_to_pdf, serializable_page_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local BADGERS DOCX page plan")
    parser.add_argument("document", type=Path, help="Input .docx file")
    parser.add_argument("output_directory", type=Path, help="Directory for local artifacts")
    parser.add_argument(
        "--rendered-pdf",
        type=Path,
        help="Use an existing rendered PDF instead of invoking LibreOffice",
    )
    parser.add_argument("--session-id", default="local-docx", help="Artifact session ID")
    parser.add_argument("--max-image-size-mb", type=float, default=4.0)
    arguments = parser.parse_args()

    payload = arguments.document.read_bytes()
    pdf_payload = (
        arguments.rendered_pdf.read_bytes()
        if arguments.rendered_pdf
        else render_docx_to_pdf(payload)
    )
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    schema_directory = Path(__file__).resolve().parents[4] / "schemas"
    page_plan, images = create_page_plan(
        payload,
        pdf_payload,
        arguments.session_id,
        schema_directory,
        max_image_size_mb=arguments.max_image_size_mb,
    )

    (output / "rendered.pdf").write_bytes(pdf_payload)
    native_directory = output / "native"
    image_directory = output / "images"
    native_directory.mkdir(exist_ok=True)
    image_directory.mkdir(exist_ok=True)
    for page in page_plan["pages"]:
        page_number = page["page"]
        native_xml = page.pop("native_output_xml")
        native_path = native_directory / f"page_{page_number:03d}.xml"
        native_path.write_text(native_xml, encoding="utf-8")
        page["native_output_uri"] = native_path.as_uri()
        if page_number in images:
            image_path = image_directory / f"page_{page_number:03d}.jpg"
            image_path.write_bytes(base64.b64decode(images[page_number]))
            page["image_uri"] = image_path.as_uri()
        else:
            page["image_uri"] = None
    (output / "page_plan.json").write_text(serializable_page_plan(page_plan), encoding="utf-8")

    native_count = sum(page["route"] == "native_docx" for page in page_plan["pages"])
    print(
        f"Wrote {page_plan['page_count']} pages to {output} "
        f"({native_count} native-only, {len(images)} visual/OCR candidate images)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
