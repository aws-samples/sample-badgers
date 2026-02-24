# PDF/UA Remediation Analyzer (v2.2)

**Correlation-guided PDF accessibility remediation with cell grid vision resolution, CJK support, and diagnostic visualization.**

## What's New in v2.2

### Diagnostic Visualizer
- New `diagnostic_visualizer.py` module captures the full pipeline output at the page level: correlation analyzer output, grid resolver output, and a color-coded bbox overlay image showing all detected elements.
- Controlled via the `ENABLE_DIAGNOSTICS` environment variable (`true`/`1`/`yes` to enable).
- Saves per-page JSON diagnostics and overlay PNGs to S3 (`{analyzer_name}/diagnostics/{session_id}/{pdf_stem}/`) and/or local disk.
- Each element record merges identity (from correlation) with position (from resolver), including pixel bboxes, resolution tier classification, and summary stats by type and tier.

### Cell Grid Resolver v3 (Corner-Point + Hierarchical Refinement)
- Corner-point sub-positions: model returns TL/BR cell + sub-position (top/middle/bottom × left/center/right) for ~9× effective resolution over the v2 cell-union approach.
- Resolved anchors: already-located elements (from PyMuPDF text search) are included in the prompt as spatial landmarks to help the model orient.
- Hierarchical refinement: low-confidence or oversized results trigger a second pass on a cropped region with a finer grid (~11× vertical resolution gain).
- Auto grid sizing based on page dimensions.
- TrueType font for grid labels when available (covers Lambda AL2023, Ubuntu, macOS).

### Build Script
- New `build.sh` for building, pushing to ECR, and updating the Lambda function. Supports `--build-only`, `--push`, and `--update-lambda` modes with `--region` and `--profile` options.

## Previous Changes (v2.1)

### Credential Threading
- `aws_profile` and `aws_region` properly threaded from Lambda handler through to `BedrockClient` and cell grid resolver.

### Pre-Processed Image Support (`page_b64_uris`)
- Pipeline accepts pre-processed base64 image files from S3 via the `page_b64_uris` event parameter, avoiding redundant rendering.

### Cell Grid Resolver Image Sizing
- Grid overlay image resized to max 2048px and compressed as JPEG with iterative quality reduction to stay under Bedrock's 5MB base64 limit.

### Cross-Region Model ID
- Default model ID updated to `us.anthropic.claude-sonnet-4-6` (cross-region inference profile).

### CJK Font Encoding Fix
- Invisible text overlays use a Type0 composite font with Identity-H encoding and a ToUnicode CMap for correct CJK and Unicode character resolution.

### Dockerfile Fix
- Removed `yum install` for system mupdf packages (PyMuPDF ships self-contained; base image moved from AL2 to AL2023 which uses `dnf`).

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
   │   2. Cell Grid Resolver v3           │
   │      (corner-point + refinement,     │
   │       one vision call per page)      │
   │              │                       │
   │         still failed?                │
   │              │                       │
   │   3. Fallback stacked strips         │
   │      (no-cost last resort)           │
   │                                      │
   │   ──► Diagnostic Visualizer          │
   │       (optional, per-page overlays)  │
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

### Environment Variables

| Variable             | Default                | Description                                         |
| -------------------- | ---------------------- | --------------------------------------------------- |
| `ENABLE_DIAGNOSTICS` | (disabled)             | Set to `true`/`1`/`yes` to enable diagnostic output |
| `OUTPUT_BUCKET`      | —                      | S3 bucket for tagged PDF and diagnostics output     |
| `ANALYZER_NAME`      | `remediation_analyzer` | Analyzer name for S3 path prefix                    |
| `CONFIG_BUCKET`      | —                      | S3 bucket for analyzer configuration                |
| `LOGGING_LEVEL`      | `INFO`                 | Python logging level                                |

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
├── cell_grid_resolver.py          # Grid-based vision bbox resolution (v3 corner-point)
├── diagnostic_visualizer.py       # Per-page diagnostic JSON + bbox overlay images
├── Dockerfile                     # Container image definition
├── requirements.txt               # Python dependencies
├── build.sh                       # Build, push to ECR, update Lambda
├── DEPLOYMENT.md                  # Deployment guide
└── REMEDIATION_README.md          # This file
```
