# PDF/UA Remediation Analyzer

A BADGERS analyzer that transforms inaccessible PDFs into PDF/UA-compliant documents with full structure trees, invisible text overlays, and pre/post accessibility auditing.

## What It Does

Takes a PDF (often image-only or untagged) and a BADGERS correlation XML (content spine), then produces a tagged PDF that passes PDF/UA accessibility validation. The output includes a structure tree with proper reading order, invisible text overlays for screen readers, alt text on figures, and a JSON audit report documenting compliance before and after remediation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    lambda_handler.py                      │
│                                                          │
│  Event In ──► Download PDF ──► process_pdf() ──► S3 Out │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   Correlation XML            No Correlation
   provided?                  (fallback path)
          │                         │
          ▼                         ▼
   Parse content spine       Full vision model
   from BADGERS XML          analysis via
          │                  AnalyzerFoundation
          │                  (uses 6 prompt XMLs
          │                   from CONFIG_BUCKET)
          │
          ▼
   ┌──────────────────────────────────────┐
   │   Three-Tier Bbox Resolution         │
   │                                      │
   │   1. PyMuPDF text search             │
   │      (free, instant, exact)          │
   │              │                       │
   │         unresolved?                  │
   │              │                       │
   │   2. Cell Grid Resolver              │
   │      (one vision call per page)      │
   │              │                       │
   │         still failed?                │
   │              │                       │
   │   3. Fallback stacked strips         │
   │      (no-cost last resort)           │
   └──────────────┬───────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────┐
   │   pdf_accessibility_tagger.py        │
   │                                      │
   │   Pre-audit ──► Tag ──► Post-audit   │
   │                                      │
   │   • Structure tree (Document/Sect)   │
   │   • MCR/MCID linkage per page        │
   │   • Invisible text overlays          │
   │   • MarkInfo, Lang, Title, Tab order │
   │   • PDF/UA XMP identifier            │
   └──────────────────────────────────────┘
```

## Pipeline Stages

### 1. Input Parsing

The handler accepts a PDF path and an optional BADGERS correlation URI. Both can be local paths or S3 URIs. The correlation XML is a content spine produced by BADGERS analysis — it contains the reading-order element list with types, text, alt text, and captions, but no bounding box coordinates.

### 2. Bbox Resolution (Correlation-Guided Path)

When correlation data is provided, the handler uses a three-tier strategy to resolve spatial coordinates for each element:

**Tier 1 — PyMuPDF Text Search.** For every text element in the content spine, `page.search_for(text)` is called against the PDF's native text layer. If the text exists in the PDF (i.e., the page isn't image-only), this resolves the bbox for free with perfect accuracy. Source tag: `pymupdf_text_search` or `pymupdf_partial_search`.

**Tier 2 — Cell Grid Resolver.** Any elements that text search can't locate (image-only pages, OCR gaps) plus all figure elements are batched into a single vision model call. The resolver renders the page, overlays a labeled cell grid (A1, B2, C3, ...), sends the gridded image alongside the element descriptions, and asks the model which cells each element occupies. Cell references are converted back to normalized bounding boxes. This replaces raw-coordinate vision calls, which vision models are imprecise at. Source tag: `cell_grid_high`, `cell_grid_medium`, or `cell_grid_low` (reflecting the model's self-reported confidence).

**Tier 3 — Fallback Stacked.** If the vision call itself fails (timeout, service error), elements get evenly-spaced horizontal strip bboxes. Structurally valid for reading order but spatially incorrect. Source tag: `fallback_stacked`.

### 3. Bbox Resolution (Fallback Path)

When no correlation XML is provided, the handler falls back to the full AnalyzerFoundation pipeline. This sends the raw page image to a vision model with a six-prompt chain that instructs it to identify all document elements and return structured JSON with bounding boxes. This path exists for standalone use but is not the primary BADGERS flow.

### 4. PDF/UA Tagging

The `PDFAccessibilityTagger` receives the resolved elements and builds the tagged PDF:

- Creates a structure tree rooted at Document → Sect, with child elements (H1–H6, P, Figure, etc.) linked to page content via marked content references (MCR/MCID).
- Inserts invisible text overlays on image-only pages so screen readers can read the content even when there's no native text layer.
- Sets MarkInfo, document language, title, tab order, and PDF/UA XMP metadata.
- Runs a pre-remediation audit before any changes and a post-remediation audit after, producing a compliance report.

### 5. Accessibility Audit

Eight checks run both before and after tagging:

| Check             | Severity | What It Validates                            |
| ----------------- | -------- | -------------------------------------------- |
| mark_info         | critical | /MarkInfo dictionary with /Marked = true     |
| structure_tree    | critical | StructTreeRoot exists with child elements    |
| language          | critical | /Lang set on document catalog                |
| figure_alt_text   | critical | All Figure elements have alt text            |
| title             | major    | Document title in metadata + DisplayDocTitle |
| tab_order         | major    | Tab order set to /S (structure) on all pages |
| text_layer        | major    | All pages have extractable text              |
| pdf_ua_identifier | major    | PDF/UA identifier in XMP metadata            |

Compliance levels: `pass` (all checks pass), `pass_with_warnings` (only info-level issues), `fail` (critical or major failures), `not_assessed`.

### 6. Output

The tagged PDF and audit report JSON are uploaded to the output bucket. The response includes a top-level `compliance` field for downstream routing (e.g., flag documents that still fail after remediation).

## File Inventory

```
remediation_analyzer/
├── Dockerfile
├── requirements.txt
├── lambda_handler.py              # Entry point, orchestration, S3 I/O
├── pdf_accessibility_tagger.py    # Structure tree builder, auditor, overlay engine
├── cell_grid_resolver.py          # Grid-based vision bbox resolution
└── foundation/                    # Shared analyzer foundation (BedrockClient, etc.)
```

## S3 Layout

**Config bucket** (`CONFIG_BUCKET`):
```
{ANALYZER_NAME}/
├── manifest.json
└── prompts/
    ├── prompt_locate_elements.xml          # Cell grid resolver prompt
    ├── remediation_job_role.xml            # Fallback path prompts
    ├── remediation_context.xml             #   (only used when no
    ├── remediation_coordinate_system.xml   #    correlation XML is
    ├── remediation_element_types.xml       #    provided)
    ├── remediation_rules.xml               #
    └── remediation_output_format.xml       #
```

**Output bucket** (`OUTPUT_BUCKET`):
```
{ANALYZER_NAME}/
└── results/
    └── {session_id}/
        ├── {document_name}_{timestamp}.pdf          # Tagged PDF
        └── {document_name}_report_{timestamp}.json   # Audit report
```

## Environment Variables

| Variable      | Required     | Default              | Purpose                               |
| ------------- | ------------ | -------------------- | ------------------------------------- |
| CONFIG_BUCKET | Yes (Lambda) | —                    | S3 bucket for prompts and manifest    |
| OUTPUT_BUCKET | Yes (Lambda) | —                    | S3 bucket for tagged PDFs and reports |
| ANALYZER_NAME | No           | remediation_analyzer | Directory prefix in both buckets      |
| LOGGING_LEVEL | No           | INFO                 | Python log level                      |
| MAX_TOKENS    | No           | 8000                 | Bedrock max_tokens for vision calls   |
| TEMPERATURE   | No           | 0.1                  | Bedrock temperature                   |
| AWS_REGION    | No           | us-west-2            | Bedrock region                        |

## Event Schema

```json
{
  "pdf_path": "s3://bucket/path/to/document.pdf",
  "correlation_uri": "s3://bucket/path/to/correlation.xml",
  "session_id": "session-abc123",
  "title": "Accessible Document Title",
  "lang": "en-US",
  "dpi": 150
}
```

`pdf_path` is required. All other fields are optional. When `correlation_uri` is omitted, the fallback vision analysis path runs.

## Response Schema

```json
{
  "statusCode": 200,
  "body": {
    "success": true,
    "session_id": "session-abc123",
    "compliance": "pass",
    "result": {
      "output_pdf": "/tmp/.../tagged_output.pdf",
      "s3_output_uri": "s3://output-bucket/remediation_analyzer/results/session-abc123/document_20260211_123456.pdf",
      "s3_report_uri": "s3://output-bucket/remediation_analyzer/results/session-abc123/document_report_20260211_123456.json",
      "pages_processed": 1,
      "correlation_used": true,
      "accessibility_report": {
        "pre_remediation": { "compliance_level": "fail", "checks": [...] },
        "post_remediation": { "compliance_level": "pass", "checks": [...] },
        "summary": {
          "pages_processed": 1,
          "total_elements_tagged": 20,
          "total_figures_with_alt": 3,
          "invisible_text_overlays_added": 17
        },
        "page_audits": [...]
      }
    }
  }
}
```

## Local Testing

```python
from lambda_handler import lambda_handler

event = {
    "pdf_path": "/path/to/document.pdf",
    "correlation_uri": "/path/to/correlation.xml",
    "session_id": "test-local-001",
    "title": "Test Document",
    "lang": "en-US",
    "dpi": 150,
}

result = lambda_handler(event, None)
```

When running locally, `CONFIG_BUCKET` and `OUTPUT_BUCKET` are unset, so prompts load from the local filesystem (or inline fallback) and outputs stay in the local temp directory. The notebook in the test folder copies them to a permanent location.

## Cell Grid Resolver — How It Works

The core innovation that solves the spatial positioning problem on image-only PDFs.

Vision models are excellent at reading text and understanding spatial relationships, but poor at reporting precise pixel coordinates. The cell grid resolver exploits this asymmetry:

1. Render the PDF page as an image.
2. Overlay a labeled grid (12×12 for square pages, auto-sized for portrait/landscape). Each cell gets a visible label: A1, A2, ... L12.
3. Send the gridded image to the vision model alongside the list of elements to locate, each with its text content or description.
4. The model reports which cells each element occupies: `{"id": "elem_003", "cells": ["D4", "E4", "F4"]}`.
5. Cell references are converted to normalized bounding boxes, then Y-flipped from image coordinates (top-left origin) to PDF coordinates (bottom-left origin).

The prompt is loaded from S3 at `{ANALYZER_NAME}/prompts/prompt_locate_elements.xml`. On first call per Lambda invocation, the resolved prompt is cached at module level so subsequent pages don't re-download.

Grid precision: a 12×12 grid gives ~8% spatial resolution per cell — sufficient for structure tree bboxes where the goal is "which region of the page does this element occupy" rather than pixel-perfect bounds.