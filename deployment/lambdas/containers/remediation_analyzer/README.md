# PDF/UA Remediation Analyzer (v2.1)

**Correlation-guided PDF accessibility remediation with cell grid vision resolution and CJK support.**

## What's New in v2.1

### Credential Threading
- `aws_profile` and `aws_region` are now properly threaded from the Lambda handler through to `BedrockClient` and the cell grid resolver. Previously these were silently dropped, causing the resolver to fall back to default/ambient credentials.

### Pre-Processed Image Support (`page_b64_uris`)
- The pipeline now accepts pre-processed base64 image files from S3 via the `page_b64_uris` event parameter. When provided, these are used directly instead of re-rendering the PDF page, avoiding redundant processing and ensuring images are already sized for the Bedrock API.
- Falls back to rendering from the PDF + `ImageProcessor.optimize_image()` compression when no b64 URI is available for a page.

### Cell Grid Resolver Image Sizing
- The grid overlay image is now resized to max 2048px and compressed as JPEG with iterative quality reduction to stay under Bedrock's 5MB base64 limit. Previously the gridded image was sent as full-resolution RGBA PNG, which exceeded the limit on most documents.

### Cross-Region Model ID
- Default model ID updated to `us.anthropic.claude-sonnet-4-6` (cross-region inference profile), matching the image enhancer's pattern.

### CJK Font Encoding Fix
- Invisible text overlays now use a Type0 composite font with Identity-H encoding and a ToUnicode CMap. This allows screen readers and PDF validators to correctly resolve Chinese, Japanese, Korean, and all other Unicode characters. Previously, Helvetica with WinAnsiEncoding was used, which silently corrupted non-Latin text.

### Dockerfile Fix
- Removed `yum install` for system mupdf packages (no longer needed — PyMuPDF ships as a self-contained wheel, and the base image moved from AL2 to AL2023 which uses `dnf`).

## Screen Reader Verification

VoiceOver on macOS reading the remediated PDF structure tree, including figure alt text descriptions:

[<video src="remediation-screen-reader-mac-02-24-2026.mov" width="320" height="240" controls></video>](https://github.com/user-attachments/assets/8d9c14a4-5c68-4ae8-a148-7147a32c6eae)

*Adobe Acrobat Accessibility Checker confirms 29 passed checks, 0 failures, including character encoding for CJK text. The only items flagged are "Logical Reading Order" and "Color contrast" which always require manual review. See the [full Acrobat accessibility report](1_test_chinese_book_20260224_161932.pdf.accreport.html).*

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

## API

### Lambda Event Schema

```json
{
  "pdf_path": "s3://bucket/path/to/document.pdf",
  "correlation_uri": "s3://bucket/path/to/correlation.xml",
  "page_b64_uris": {
    "0": "s3://bucket/temp/abc123/page_001.b64"
  },
  "session_id": "session-abc123",
  "title": "Accessible Document Title",
  "lang": "en-US",
  "dpi": 150
}
```

| Parameter         | Required | Default                 | Description                                                                       |
| ----------------- | -------- | ----------------------- | --------------------------------------------------------------------------------- |
| `pdf_path`        | Yes      | —                       | S3 URI to the source PDF                                                          |
| `correlation_uri` | No       | —                       | S3 URI to correlation XML from the analysis pipeline                              |
| `page_b64_uris`   | No       | —                       | Dict mapping page number (string, 0-indexed) to S3 URI of pre-processed b64 image |
| `session_id`      | No       | `"no_session"`          | Tracking identifier                                                               |
| `title`           | No       | `"Accessible Document"` | PDF metadata title                                                                |
| `lang`            | No       | `"en-US"`               | BCP-47 language code                                                              |
| `dpi`             | No       | `150`                   | Rendering resolution for fallback path                                            |

## Compliance Checks

| Check             | Severity | Validates                           |
| ----------------- | -------- | ----------------------------------- |
| mark_info         | critical | /MarkInfo with /Marked = true       |
| structure_tree    | critical | StructTreeRoot with child elements  |
| language          | critical | /Lang on document catalog           |
| figure_alt_text   | critical | All Figure elements have alt text   |
| title             | major    | Title in metadata + DisplayDocTitle |
| tab_order         | major    | /S (structure) on all pages         |
| text_layer        | major    | All pages have extractable text     |
| pdf_ua_identifier | major    | PDF/UA identifier in XMP metadata   |

## File Inventory

```
remediation_analyzer/
├── lambda_handler.py              # Entry point, orchestration, S3 I/O
├── pdf_accessibility_tagger.py    # Structure tree builder, overlay engine
├── pdf_accessibility_auditor.py   # Pre/post compliance checks
├── pdf_accessibility_models.py    # Data models and constants
├── cell_grid_resolver.py          # Grid-based vision bbox resolution
├── Dockerfile                     # Container image definition
├── requirements.txt               # Python dependencies
├── DEPLOYMENT.md                  # Deployment guide
└── README.md                      # This file
```
