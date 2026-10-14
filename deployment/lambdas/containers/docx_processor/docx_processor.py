"""Safe, page-aware DOCX inspection and BADGERS-native output generation.

This module deliberately keeps WordprocessingML extraction separate from page
rendering. OOXML provides semantic structure, while the rendered PDF provides
the only dependable page boundaries. The resulting page plan lets the
orchestrator skip Vision AI for simple pages without discarding native facts.
"""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import tempfile
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from lxml import etree

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
CONTENT_TREE_NS = "https://github.com/aws-samples/sample-badgers/schemas/content-tree"
NS = {
    "w": WORD_NS,
    "r": REL_NS,
    "wp": DRAWING_NS,
    "m": MATH_NS,
    "pr": PKG_REL_NS,
}

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2_000
MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100
MAX_RENDERED_PAGES = 250


class DocxProcessingError(ValueError):
    """Raised for an input that cannot safely produce a DOCX page plan."""


@dataclass
class NativeBlock:
    """One source-order Word block with normalized structural facts."""

    source_id: str
    kind: str
    text: str = ""
    style: str = ""
    level: int | None = None
    list_id: str | None = None
    table_rows: int = 0
    table_columns: int = 0
    has_merged_cells: bool = False
    table_cells: list[list[dict[str, Any]]] = field(default_factory=list)
    has_page_break: bool = False
    column_count: int = 1
    has_drawing: bool = False
    has_floating_drawing: bool = False
    has_equation: bool = False
    image_relationship_ids: list[str] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    alignment_confidence: float = 0.0


@dataclass
class PageLayout:
    """Rendered PDF facts needed to assign native blocks and calculate routing."""

    page: int
    text: str
    image_count: int
    drawing_count: int
    width: float
    height: float


def _xml_parser() -> etree.XMLParser:
    """Return a parser that refuses entities, network access, and huge trees."""
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
    )


def _parse_xml(payload: bytes, part_name: str) -> etree._Element:
    try:
        return etree.fromstring(payload, parser=_xml_parser())
    except etree.XMLSyntaxError as error:
        raise DocxProcessingError(f"Invalid XML in {part_name}: {error}") from error


def _safe_part_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and not name.startswith("/")


def validate_docx_package(payload: bytes) -> None:
    """Validate OOXML packaging before extracting or invoking LibreOffice."""
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise DocxProcessingError(
            f"DOCX exceeds the {MAX_ARCHIVE_BYTES // (1024 * 1024)} MiB input limit."
        )
    if payload.startswith(b"\xd0\xcf\x11\xe0"):
        raise DocxProcessingError(
            "Legacy .doc files are not supported. Save the document as .docx or PDF."
        )
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise DocxProcessingError("Input is not a valid OOXML .docx ZIP package.")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        names = {item.filename for item in infos}
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise DocxProcessingError(
                "Input is not a Word .docx package (required OOXML parts are missing)."
            )
        if "word/vbaProject.bin" in names:
            raise DocxProcessingError(
                "Macro-enabled Word documents are not supported. Remove macros and save as .docx."
            )
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise DocxProcessingError("DOCX contains too many package members.")

        uncompressed_size = 0
        for item in infos:
            if not _safe_part_name(item.filename):
                raise DocxProcessingError("DOCX contains an unsafe package path.")
            uncompressed_size += item.file_size
            if uncompressed_size > MAX_UNCOMPRESSED_BYTES:
                raise DocxProcessingError("DOCX expands beyond the allowed extraction size.")
            if item.compress_size and item.file_size / item.compress_size > MAX_COMPRESSION_RATIO:
                raise DocxProcessingError("DOCX contains a suspiciously compressed package member.")


def _normalized_text(value: str) -> str:
    """Normalize text for Word-to-rendered-PDF alignment.

    PDF text extraction regularly substitutes visually similar punctuation
    (for example, an em dash may become a middle dot). Treat punctuation as
    separators instead of literal evidence, while retaining letters and
    numbers that make a source block distinctive.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _paragraph_text(element: etree._Element) -> str:
    return "".join(element.xpath(".//w:t/text()", namespaces=NS)).strip()


def _paragraph_style(element: etree._Element) -> str:
    values = element.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return values[0] if values else ""


def _heading_level(style: str, element: etree._Element) -> int | None:
    outline = element.xpath("./w:pPr/w:outlineLvl/@w:val", namespaces=NS)
    if outline and outline[0].isdigit():
        return min(int(outline[0]) + 1, 6)
    match = re.search(r"(?:heading|title)\s*([1-6])?", style, flags=re.IGNORECASE)
    if match:
        return int(match.group(1) or "1")
    return None


def _relationship_targets(archive: zipfile.ZipFile) -> dict[str, str]:
    part_name = "word/_rels/document.xml.rels"
    if part_name not in archive.namelist():
        return {}
    root = _parse_xml(archive.read(part_name), part_name)
    targets: dict[str, str] = {}
    for relationship in root.xpath("./pr:Relationship", namespaces=NS):
        relation_id = relationship.get("Id")
        target = relationship.get("Target")
        if relation_id and target:
            targets[relation_id] = target
    return targets


def _section_columns(section_properties: etree._Element | None) -> int:
    """Return a section's declared column count, defaulting to one column."""
    if section_properties is None:
        return 1
    columns = section_properties.xpath("./w:cols/@w:num", namespaces=NS)
    return max((int(value) for value in columns if value.isdigit()), default=1)


def _document_page_dimensions(document: etree._Element) -> tuple[int, int]:
    """Return the first section's page size in EMUs for image-area scoring."""
    widths = document.xpath(".//w:sectPr/w:pgSz/@w:w", namespaces=NS)
    heights = document.xpath(".//w:sectPr/w:pgSz/@w:h", namespaces=NS)
    # OOXML page sizes are stored in twips. Letter is the Word default when
    # page settings are absent.
    width_twips = int(widths[0]) if widths and widths[0].isdigit() else 12_240
    height_twips = int(heights[0]) if heights and heights[0].isdigit() else 15_840
    return width_twips * 635, height_twips * 635


def _image_metadata(
    archive: zipfile.ZipFile, relationship_id: str, relationships: dict[str, str]
) -> dict[str, Any]:
    """Read image dimensions and identity without interpreting its visual content."""
    target = relationships.get(relationship_id, "")
    part_name = str(PurePosixPath("word") / target)
    metadata: dict[str, Any] = {
        "relationship_id": relationship_id,
        "target": target,
        "part_name": part_name,
        "pixel_width": 0,
        "pixel_height": 0,
        "byte_size": 0,
        "sha256": "",
    }
    if not target or not _safe_part_name(part_name) or part_name not in archive.namelist():
        metadata["warning"] = "missing_or_external_image_relationship"
        return metadata
    image_data = archive.read(part_name)
    metadata["byte_size"] = len(image_data)
    metadata["sha256"] = __import__("hashlib").sha256(image_data).hexdigest()
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_data)) as image:
            metadata["pixel_width"], metadata["pixel_height"] = image.size
            metadata["format"] = image.format or ""
    except Exception:
        # The relationship still matters to routing. Bad/corporate image formats
        # receive a warning rather than preventing all native text extraction.
        metadata["warning"] = "image_metadata_unavailable"
    return metadata


def _drawing_images(
    element: etree._Element,
    archive: zipfile.ZipFile,
    relationships: dict[str, str],
    page_area_emu: int,
) -> list[dict[str, Any]]:
    """Return OOXML drawing facts, display area, and image metadata."""
    drawing_ns = {**NS, "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    images: list[dict[str, Any]] = []
    for drawing in element.xpath(".//w:drawing", namespaces=NS):
        relationship_ids = drawing.xpath(".//a:blip/@r:embed", namespaces=drawing_ns)
        extents = drawing.xpath("./wp:inline/wp:extent | ./wp:anchor/wp:extent", namespaces=NS)
        width = int(extents[0].get("cx", "0")) if extents else 0
        height = int(extents[0].get("cy", "0")) if extents else 0
        graphic_types = drawing.xpath(".//a:graphicData/@uri", namespaces=drawing_ns)
        for relationship_id in relationship_ids:
            image = _image_metadata(archive, relationship_id, relationships)
            image.update(
                {
                    "display_width_emu": width,
                    "display_height_emu": height,
                    "display_area_ratio": round(
                        (width * height) / page_area_emu, 4
                    )
                    if page_area_emu and width and height
                    else 0.0,
                    "floating": bool(drawing.xpath("./wp:anchor", namespaces=NS)),
                    "graphic_type": graphic_types[0] if graphic_types else "picture",
                }
            )
            images.append(image)
    return images


def build_native_relationships(blocks: list[NativeBlock]) -> list[dict[str, Any]]:
    """Build the same explicit relationship graph used by native and visual pages."""
    relationships: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    visual_candidates: list[NativeBlock] = []
    table_candidates: list[NativeBlock] = []
    caption_pattern = re.compile(r"^(figure|fig\.?|table|chart|diagram)\s+([0-9]+)", re.I)
    reference_pattern = re.compile(r"\b(figure|fig\.?|table|chart|diagram)\s+([0-9]+)\b", re.I)

    def label_kind(value: str) -> str:
        normalized = value.lower().rstrip(".")
        return "figure" if normalized == "fig" else normalized

    for block in blocks:
        if block.has_drawing:
            visual_candidates.append(block)
            for image in block.images:
                relationships.append(
                    {
                        "type": "contains_figure",
                        "source_id": block.source_id,
                        "target_id": f"image:{image['relationship_id']}",
                        "evidence": image.get("target", ""),
                        "confidence": 1.0,
                        "provenance": "native_docx",
                    }
                )
        if block.kind == "table":
            table_candidates.append(block)
        caption_match = caption_pattern.match(block.text.strip())
        if caption_match:
            kind = label_kind(caption_match.group(1))
            label = f"{kind}-{caption_match.group(2)}"
            candidates = table_candidates if kind == "table" else visual_candidates
            if candidates:
                target = candidates[-1]
                labels[label] = target.source_id
                relationships.append(
                    {
                        "type": "captioned_by",
                        "source_id": target.source_id,
                        "target_id": block.source_id,
                        "evidence": block.text,
                        "confidence": 0.95,
                        "provenance": "native_docx",
                    }
                )

    for block in blocks:
        for kind, number in reference_pattern.findall(block.text):
            label = f"{label_kind(kind)}-{number}"
            target = labels.get(label)
            if target and target != block.source_id:
                relationships.append(
                    {
                        "type": "references",
                        "source_id": block.source_id,
                        "target_id": target,
                        "evidence": f"{kind} {number}",
                        "confidence": 0.98,
                        "provenance": "native_docx",
                    }
                )
    return relationships


def _python_docx_metadata(payload: bytes) -> dict[str, str]:
    """Use python-docx as a high-level parse cross-check when it is installed."""
    try:
        from docx import Document  # type: ignore[import-not-found]

        properties = Document(io.BytesIO(payload)).core_properties
        return {
            key: str(value)
            for key, value in {
                "title": properties.title,
                "subject": properties.subject,
                "author": properties.author,
                "language": properties.language,
            }.items()
            if value
        }
    except ImportError:
        # Unit tests and non-container local inspection can still exercise the
        # low-level parser. The deployed image declares python-docx explicitly.
        return {}
    except Exception as error:  # python-docx is a convenience layer, not authority
        return {"python_docx_warning": f"{type(error).__name__}: {error}"}


def inspect_docx(payload: bytes) -> tuple[list[NativeBlock], dict[str, Any]]:
    """Deeply inspect document.xml and relationships in source order."""
    validate_docx_package(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        document = _parse_xml(archive.read("word/document.xml"), "word/document.xml")
        relationships = _relationship_targets(archive)
        page_width_emu, page_height_emu = _document_page_dimensions(document)
        page_area_emu = page_width_emu * page_height_emu
        body = document.find(f"{{{WORD_NS}}}body")
        if body is None:
            raise DocxProcessingError("word/document.xml has no document body.")

        blocks: list[NativeBlock] = []
        body_blocks: list[tuple[int, NativeBlock]] = []
        for index, child in enumerate(body):
            local_name = etree.QName(child).localname
            source_id = f"word/document.xml:block-{index + 1}"
            if local_name == "p":
                style = _paragraph_style(child)
                list_ids = child.xpath("./w:pPr/w:numPr/w:numId/@w:val", namespaces=NS)
                block_images = _drawing_images(
                    child, archive, relationships, page_area_emu
                )
                descriptions = child.xpath(".//wp:docPr/@descr", namespaces=NS)
                has_inline = bool(child.xpath(".//wp:inline", namespaces=NS))
                has_anchor = bool(child.xpath(".//wp:anchor", namespaces=NS))
                has_equation = bool(child.xpath(".//m:oMath | .//m:oMathPara", namespaces=NS))
                has_page_break = bool(
                    child.xpath(
                        ".//w:br[@w:type='page'] | .//w:lastRenderedPageBreak",
                        namespaces=NS,
                    )
                )
                section_properties = child.xpath("./w:pPr/w:sectPr", namespaces=NS)
                section_types = child.xpath(
                    "./w:pPr/w:sectPr/w:type/@w:val", namespaces=NS
                )
                has_next_page_section_break = bool(section_properties) and (
                    not section_types or section_types[0] != "continuous"
                )
                has_layout_break = has_page_break or has_next_page_section_break
                level = _heading_level(style, child)
                text = _paragraph_text(child)
                kind = (
                    "page_break"
                    if has_layout_break and not text
                    else ("heading" if level else ("list" if list_ids else "paragraph"))
                )
                warnings: list[str] = []
                if child.xpath(".//w:object | .//w:pict | .//w:sdt", namespaces=NS):
                    warnings.append("unsupported_or_legacy_ooxml_object")
                block = NativeBlock(
                    source_id=source_id,
                    kind=kind,
                    text=text,
                    style=style,
                    level=level,
                    list_id=list_ids[0] if list_ids else None,
                    has_drawing=has_inline or has_anchor,
                    has_floating_drawing=has_anchor,
                    has_equation=has_equation,
                    has_page_break=has_layout_break,
                    image_relationship_ids=[
                        image["relationship_id"] for image in block_images
                    ],
                    image_descriptions=[item for item in descriptions if item],
                    images=block_images,
                    warnings=warnings,
                )
                blocks.append(block)
                body_blocks.append((index, block))
            elif local_name == "tbl":
                rows = child.xpath("./w:tr", namespaces=NS)
                column_counts = [
                    len(row.xpath("./w:tc", namespaces=NS))
                    for row in rows
                ]
                merged = bool(
                    child.xpath(".//w:gridSpan | .//w:vMerge", namespaces=NS)
                )
                cells = child.xpath(".//w:tc", namespaces=NS)
                table_cells = [
                    [
                        {
                            "text": _paragraph_text(cell),
                            "column_span": int(
                                (
                                    cell.xpath(
                                        "./w:tcPr/w:gridSpan/@w:val",
                                        namespaces=NS,
                                    )
                                    or ["1"]
                                )[0]
                            ),
                            "vertical_merge": bool(
                                cell.xpath("./w:tcPr/w:vMerge", namespaces=NS)
                            ),
                        }
                        for cell in row.xpath("./w:tc", namespaces=NS)
                    ]
                    for row in rows
                ]
                block = NativeBlock(
                    source_id=source_id,
                    kind="table",
                    text=" ".join(_paragraph_text(cell) for cell in cells).strip(),
                    table_rows=len(rows),
                    table_columns=max(column_counts, default=0),
                    has_merged_cells=merged,
                    table_cells=table_cells,
                )
                blocks.append(block)
                body_blocks.append((index, block))
            elif local_name != "sectPr":
                block = NativeBlock(
                    source_id=source_id,
                    kind="unsupported",
                    warnings=[f"unsupported_document_body_element:{local_name}"],
                )
                blocks.append(block)
                body_blocks.append((index, block))

        # In WordprocessingML, a paragraph's sectPr describes the section ending
        # at that paragraph, and body/sectPr describes the final section. Assign
        # each block its own section's columns so a two-column appendix does not
        # force unrelated prose pages through Vision AI.
        section_start = 0
        for index, child in enumerate(body):
            section_properties = child.xpath("./w:pPr/w:sectPr", namespaces=NS)
            if not section_properties:
                continue
            columns = _section_columns(section_properties[0])
            for block_index, block in body_blocks:
                if section_start <= block_index <= index:
                    block.column_count = columns
            section_start = index + 1
        final_section_properties = body.find(f"{{{WORD_NS}}}sectPr")
        final_columns = _section_columns(final_section_properties)
        for block_index, block in body_blocks:
            if block_index >= section_start:
                block.column_count = final_columns

        for block in blocks:
            for image in block.images:
                image["repeated_count"] = sum(
                    1
                    for candidate in blocks
                    for item in candidate.images
                    if item.get("sha256") and item.get("sha256") == image.get("sha256")
                )
        native_relationships = build_native_relationships(blocks)
        metadata = {
            "source_format": "docx",
            "columns": max((block.column_count for block in blocks), default=1),
            "page_width_emu": page_width_emu,
            "page_height_emu": page_height_emu,
            "relationships": relationships,
            "native_relationships": native_relationships,
            "python_docx": _python_docx_metadata(payload),
            "native_block_count": len(blocks),
        }
        return blocks, metadata


def render_docx_to_pdf(payload: bytes, timeout_seconds: int = 120) -> bytes:
    """Render DOCX through the pinned container LibreOffice installation."""
    with tempfile.TemporaryDirectory(prefix="badgers-docx-") as temporary_directory:
        temp_dir = Path(temporary_directory)
        input_path = temp_dir / "input.docx"
        output_dir = temp_dir / "out"
        output_dir.mkdir()
        input_path.write_bytes(payload)
        command = [
            "soffice",
            "--headless",
            "--nologo",
            "--nodefault",
            "--nofirststartwizard",
            "-env:UserInstallation=file:///tmp/badgers-libreoffice-profile",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(output_dir),
            str(input_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise DocxProcessingError("DOCX renderer is not installed in this runtime.") from error
        except subprocess.TimeoutExpired as error:
            raise DocxProcessingError("DOCX rendering exceeded the configured time limit.") from error
        pdf_path = output_dir / "input.pdf"
        if completed.returncode != 0 or not pdf_path.is_file():
            detail = (completed.stderr or completed.stdout).strip()
            raise DocxProcessingError(f"DOCX rendering failed: {detail[:500]}")
        return pdf_path.read_bytes()


def index_pdf_pages(pdf_payload: bytes) -> list[PageLayout]:
    """Return text and visual layout facts without rasterizing every page."""
    try:
        import fitz
    except ImportError as error:  # pragma: no cover - guarded by container requirements
        raise DocxProcessingError("PyMuPDF is required for DOCX page indexing.") from error

    document = fitz.open(stream=pdf_payload, filetype="pdf")
    try:
        if document.page_count > MAX_RENDERED_PAGES:
            raise DocxProcessingError(
                f"DOCX renders to more than the {MAX_RENDERED_PAGES}-page limit."
            )
        pages: list[PageLayout] = []
        for index, page in enumerate(document):
            rect = page.rect
            visible_image_count = 0
            for image in page.get_images(full=True):
                try:
                    visible_image_count += sum(
                        1
                        for image_rect in page.get_image_rects(image[0])
                        if image_rect.get_area() > 0
                    )
                except (RuntimeError, ValueError):
                    # A malformed or unused PDF image resource is not evidence
                    # that the visible page needs Vision AI.
                    continue
            pages.append(
                PageLayout(
                    page=index + 1,
                    text=page.get_text("text"),
                    image_count=visible_image_count,
                    drawing_count=len(page.get_drawings()),
                    width=rect.width,
                    height=rect.height,
                )
            )
        return pages
    finally:
        document.close()


def align_blocks_to_pages(blocks: list[NativeBlock], pages: list[PageLayout]) -> None:
    """Assign source-order blocks to rendered pages using normalized text matching."""
    page_text = [_normalized_text(page.text) for page in pages]
    cursor = 0
    for block in blocks:
        text = _normalized_text(block.text)
        if not pages:
            block.warnings.append("no_rendered_pages")
            continue
        if not text:
            # Word emits an empty paragraph for an explicit page break. It is a
            # deterministic pagination instruction, not unmapped content, and
            # must advance the source-order cursor without making either page
            # appear structurally uncertain.
            block.page_numbers = [min(cursor + 1, len(pages))]
            block.alignment_confidence = 1.0 if block.has_page_break else 0.4
            if block.has_page_break:
                cursor = min(cursor + 1, len(pages) - 1)
            continue

        candidates = [
            index for index in range(cursor, len(pages)) if text in page_text[index]
        ]
        if candidates:
            chosen = candidates[0]
            block.page_numbers = [chosen + 1]
            block.alignment_confidence = 0.98
            cursor = min(chosen + int(block.has_page_break), len(pages) - 1)
            continue

        # A long block can cross page boundaries. Match a distinctive opening token
        # sequence before assigning a low-confidence fallback.
        opening = text[: min(80, len(text))]
        candidates = [
            index for index in range(cursor, len(pages)) if opening and opening in page_text[index]
        ]
        if candidates:
            chosen = candidates[0]
            block.page_numbers = [chosen + 1]
            block.alignment_confidence = 0.7
            block.warnings.append("partial_text_alignment")
            cursor = min(chosen + int(block.has_page_break), len(pages) - 1)
        else:
            block.page_numbers = [min(cursor + 1, len(pages))]
            block.alignment_confidence = 0.35
            block.warnings.append("unmapped_native_block")


def decide_page_route(
    page: PageLayout,
    blocks: Iterable[NativeBlock],
    document_columns: int,
    relationships: Iterable[dict[str, Any]] = (),
) -> tuple[int, str, list[str]]:
    """Score one page deterministically and return route + explainable reasons."""
    score = 0
    reasons: list[str] = []
    page_blocks = list(blocks)
    page_block_ids = {block.source_id for block in page_blocks}
    page_relationships = [
        relation
        for relation in relationships
        if relation.get("source_id") in page_block_ids
        or relation.get("target_id") in page_block_ids
    ]

    # An image-only Word page is most often an inserted scan.  There is no
    # native text spine to preserve, but it need not consume a Vision model
    # immediately: Textract gets a bounded, low-cost OCR pass first.  Keep
    # known semantic visuals (charts, diagrams, captions, and floated layout)
    # on the visual path, because OCR cannot recover their relationships.
    native_text_length = len(
        " ".join(block.text.strip() for block in page_blocks if block.text.strip())
    )
    has_known_visual_semantics = any(
        block.has_equation
        or block.has_floating_drawing
        or any(image.get("graphic_type", "").endswith("/chart") for image in block.images)
        for block in page_blocks
    ) or any(
        relationship["type"] in {"references", "captioned_by"}
        for relationship in page_relationships
    )
    if (
        page.image_count > 0
        and native_text_length < 12
        and not has_known_visual_semantics
    ):
        return 15, "textract_ocr", [
            "image_only_page_without_native_text",
            "textract_ocr_first_pass",
        ]

    if document_columns > 1:
        score += 55
        reasons.append("multi_column_layout")
    if page.image_count:
        score += min(20 + page.image_count * 10, 45)
        reasons.append("rendered_page_images")
    # LibreOffice represents table rules as PDF vector drawings. They should be
    # judged by the table's native structure, not mistaken for a diagram.
    if page.drawing_count >= 8 and not any(
        block.kind == "table" for block in page_blocks
    ):
        score += 35
        reasons.append("dense_vector_drawings")

    for block in page_blocks:
        if block.kind == "table":
            if block.has_merged_cells or block.table_rows > 12 or block.table_columns > 6:
                score += 45
                reasons.append("complex_table")
            else:
                score += 10
                reasons.append("native_table")
        if block.has_equation:
            score += 50
            reasons.append("equation")
        if block.has_floating_drawing:
            score += 70
            reasons.append("floating_drawing")
        elif block.has_drawing:
            score += 40
            reasons.append("embedded_drawing")
        for image in block.images:
            if image.get("graphic_type", "").endswith("/chart"):
                score += 70
                reasons.append("native_chart")
            if image.get("display_area_ratio", 0) >= 0.5:
                score += 25
                reasons.append("page_dominant_image")
            elif image.get("display_area_ratio", 0) >= 0.25:
                score += 15
                reasons.append("large_embedded_image")
            if image.get("repeated_count", 0) > 1:
                score = max(0, score - 10)
                reasons.append("repeated_image_asset")
        if block.has_drawing and not block.image_descriptions:
            score += 20
            reasons.append("image_missing_description")
        if block.alignment_confidence < 0.6:
            score += 35
            reasons.append("low_alignment_confidence")
        if block.warnings:
            score += 40
            reasons.extend(block.warnings)
    if any(
        relationship["type"] in {"references", "captioned_by"}
        for relationship in page_relationships
    ):
        score += 15
        reasons.append("referenced_visual_or_table_content")

    score = min(score, 100)
    unique_reasons = list(dict.fromkeys(reasons)) or ["aligned_native_text"]
    if score >= 60:
        return score, "vision_ai", unique_reasons
    if score >= 30:
        return score, "hybrid", unique_reasons
    return score, "native_docx", unique_reasons


def _element(
    parent: etree._Element,
    source_id: str,
    tag: str,
    page: int,
    order: int,
    text: str = "",
) -> etree._Element:
    node = etree.SubElement(
        parent,
        f"{{{CONTENT_TREE_NS}}}element",
        id=source_id,
        tag=tag,
        page=str(page),
        order=str(order),
        placement="Block",
    )
    if text:
        etree.SubElement(node, f"{{{CONTENT_TREE_NS}}}text").text = text
    return node


def build_content_tree_xml(
    page_number: int,
    blocks: Iterable[NativeBlock],
    session_id: str,
    source_image: str = "native:docx",
    relationships: Iterable[dict[str, Any]] = (),
) -> bytes:
    """Build a canonical BADGERS content-tree XML page without losing provenance.

    The relationship graph remains in the JSON page plan for routing/audit use,
    and is duplicated here using the content-tree schema's ``enrichments`` so
    native-only pages retain the same semantic links as visual pages.
    """
    root = etree.Element(
        f"{{{CONTENT_TREE_NS}}}unified_document",
        nsmap={None: CONTENT_TREE_NS},
        session_id=session_id,
        page=str(page_number),
        schema_version="2.0",
    )
    metadata = etree.SubElement(root, f"{{{CONTENT_TREE_NS}}}metadata")
    etree.SubElement(metadata, f"{{{CONTENT_TREE_NS}}}source_image").text = source_image
    etree.SubElement(metadata, f"{{{CONTENT_TREE_NS}}}correlation_timestamp").text = (
        datetime.now(UTC).isoformat()
    )
    executed = etree.SubElement(metadata, f"{{{CONTENT_TREE_NS}}}specialists_executed")
    etree.SubElement(
        executed,
        f"{{{CONTENT_TREE_NS}}}specialist",
        name="docx_processor",
        s3_uri="native:docx_processor",
    )
    content_tree = etree.SubElement(root, f"{{{CONTENT_TREE_NS}}}content_tree")
    section = etree.SubElement(
        content_tree,
        f"{{{CONTENT_TREE_NS}}}sect",
        id=f"docx-page-{page_number}",
    )

    page_blocks = list(blocks)
    blocks_by_id = {block.source_id: block for block in page_blocks}
    relationships_by_source: dict[str, list[dict[str, Any]]] = {}
    for relationship in relationships:
        relationships_by_source.setdefault(
            str(relationship.get("source_id", "")), []
        ).append(relationship)

    for order, block in enumerate(page_blocks, start=1):
        if block.kind == "heading":
            tag = f"H{block.level or 1}"
        elif block.kind == "list":
            tag = "L"
        elif block.kind == "table":
            tag = "Table"
        elif block.has_equation:
            tag = "Formula"
        elif block.has_drawing:
            tag = "Figure"
        elif block.kind == "page_break":
            tag = "Artifact"
        elif block.kind == "unsupported":
            tag = "Artifact"
        else:
            tag = "P"
        node = _element(section, block.source_id, tag, page_number, order, block.text)
        if block.has_drawing and block.image_descriptions:
            etree.SubElement(node, f"{{{CONTENT_TREE_NS}}}alt_text").text = (
                block.image_descriptions[0]
            )
        caption_relationship = next(
            (
                relationship
                for relationship in relationships_by_source.get(block.source_id, [])
                if relationship.get("type") == "captioned_by"
            ),
            None,
        )
        if caption_relationship:
            caption_block = blocks_by_id.get(str(caption_relationship.get("target_id")))
            if caption_block and caption_block.text:
                etree.SubElement(node, f"{{{CONTENT_TREE_NS}}}caption").text = (
                    caption_block.text
                )
        if block.kind == "table" and block.table_cells:
            children = etree.SubElement(node, f"{{{CONTENT_TREE_NS}}}children")
            for row_index, row in enumerate(block.table_cells, start=1):
                row_id = f"{block.source_id}:row-{row_index}"
                row_node = _element(
                    children, row_id, "TR", page_number, row_index
                )
                row_children = etree.SubElement(
                    row_node, f"{{{CONTENT_TREE_NS}}}children"
                )
                for column_index, cell in enumerate(row, start=1):
                    cell_tag = "TH" if row_index == 1 else "TD"
                    cell_node = _element(
                        row_children,
                        f"{row_id}:cell-{column_index}",
                        cell_tag,
                        page_number,
                        column_index,
                        str(cell.get("text", "")),
                    )
                    if cell_tag == "TH":
                        cell_node.set("scope", "Column")
        source_relationships = relationships_by_source.get(block.source_id, [])
        if source_relationships:
            enrichments = etree.SubElement(node, f"{{{CONTENT_TREE_NS}}}enrichments")
            for relationship in source_relationships:
                cross_reference = etree.SubElement(
                    enrichments,
                    f"{{{CONTENT_TREE_NS}}}cross_reference",
                    target=str(relationship.get("target_id", "")),
                    type=str(relationship.get("type", "")),
                )
                etree.SubElement(
                    cross_reference, f"{{{CONTENT_TREE_NS}}}relationship"
                ).text = str(relationship.get("type", ""))
                evidence = relationship.get("evidence")
                if evidence:
                    etree.SubElement(
                        cross_reference, f"{{{CONTENT_TREE_NS}}}evidence"
                    ).text = str(evidence)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8")


def validate_content_tree(xml_payload: bytes, schema_directory: Path) -> tuple[bool, list[str]]:
    """Validate canonical native output against BADGERS' primary content-tree XSD."""
    schema_path = schema_directory / "badgers-content-tree.xsd"
    if not schema_path.is_file():
        return False, [f"Schema not found: {schema_path}"]
    try:
        schema = etree.XMLSchema(etree.parse(str(schema_path), parser=_xml_parser()))
        document = etree.fromstring(xml_payload, parser=_xml_parser())
        schema.assertValid(document)
        return True, []
    except (etree.XMLSchemaError, etree.DocumentInvalid, etree.XMLSyntaxError) as error:
        return False, [str(error)]


def rasterize_selected_pages(
    pdf_payload: bytes, page_numbers: set[int], max_image_size_mb: float
) -> dict[int, str]:
    """Rasterize only visual/hybrid pages to base64 JPEGs."""
    if not page_numbers:
        return {}
    try:
        import fitz
        from PIL import Image
    except ImportError as error:  # pragma: no cover - guarded by container requirements
        raise DocxProcessingError("PyMuPDF and Pillow are required for page rasterization.") from error

    images: dict[int, str] = {}
    max_size_bytes = int(max_image_size_mb * 1024 * 1024)
    document = fitz.open(stream=pdf_payload, filetype="pdf")
    try:
        for page_number in sorted(page_numbers):
            page = document[page_number - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(128 / 72, 128 / 72), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            quality = 85
            buffer = io.BytesIO()
            while quality >= 20:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                if buffer.tell() <= max_size_bytes:
                    break
                quality -= 10
            images[page_number] = base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        document.close()
    return images


def create_page_plan(
    payload: bytes,
    pdf_payload: bytes,
    session_id: str,
    schema_directory: Path,
    max_image_size_mb: float = 4.0,
) -> tuple[dict[str, Any], dict[int, str]]:
    """Create schema-validated native outputs plus visual artifacts where required."""
    blocks, metadata = inspect_docx(payload)
    layouts = index_pdf_pages(pdf_payload)
    align_blocks_to_pages(blocks, layouts)
    native_relationships = metadata["native_relationships"]
    pages: list[dict[str, Any]] = []
    selected_pages: set[int] = set()
    for layout in layouts:
        page_blocks = [block for block in blocks if layout.page in block.page_numbers]
        score, route, reasons = decide_page_route(
            layout,
            page_blocks,
            max(
                (block.column_count for block in page_blocks),
                default=int(metadata["columns"]),
            ),
            native_relationships,
        )
        page_relationships = [
            relationship
            for relationship in native_relationships
            if relationship.get("source_id") in {block.source_id for block in page_blocks}
            or relationship.get("target_id") in {block.source_id for block in page_blocks}
        ]
        native_xml = build_content_tree_xml(
            layout.page,
            page_blocks,
            session_id,
            relationships=page_relationships,
        )
        schema_valid, schema_errors = validate_content_tree(native_xml, schema_directory)
        if not schema_valid:
            reasons.append("content_tree_schema_validation_failed")
            route = "hybrid" if route == "native_docx" else route
        if route != "native_docx":
            selected_pages.add(layout.page)
        pages.append(
            {
                "page": layout.page,
                "route": route,
                "complexity_score": score,
                "decision_reasons": list(dict.fromkeys(reasons)),
                "alignment_confidence": round(
                    min((block.alignment_confidence for block in page_blocks), default=1.0),
                    2,
                ),
                "content_tree_schema_valid": schema_valid,
                "content_tree_schema_errors": schema_errors,
                "native_output_xml": native_xml.decode("utf-8"),
                "native_blocks": [asdict(block) for block in page_blocks],
                "relationships": page_relationships,
                "rendered_image_count": layout.image_count,
                "rendered_drawing_count": layout.drawing_count,
                "native_column_count": max(
                    (block.column_count for block in page_blocks), default=1
                ),
            }
        )
    images = rasterize_selected_pages(pdf_payload, selected_pages, max_image_size_mb)
    return {
        "source_format": "docx",
        "page_count": len(pages),
        "native_metadata": metadata,
        "pages": pages,
    }, images


def serializable_page_plan(page_plan: dict[str, Any]) -> str:
    """Return stable JSON for S3 audit artifacts."""
    return json.dumps(page_plan, sort_keys=True, ensure_ascii=False, indent=2)
