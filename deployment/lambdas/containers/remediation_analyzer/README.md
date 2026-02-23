# PDF/UA Remediation Analyzer (v2.0)

**Replacement package for the original remediation_analyzer — modernized core modules with improved functionality.**

## What's New in v2.0

This version maintains full backward compatibility with the original Lambda handler interface while incorporating significant improvements to the core PDF accessibility modules:

### Core Improvements

1. **Enhanced PDF Accessibility Tagger** (`pdf_accessibility_tagger.py`)
   - Unified audit + remediation engine
   - Better handling of mixed content (text layer + image-only)
   - Improved invisible text overlay insertion
   - More robust content stream wrapping
   - Context manager support (`with` statement)

2. **Comprehensive Auditor** (`pdf_accessibility_auditor.py`)
   - Pre and post-remediation compliance checks
   - Eight distinct PDF/UA validation checks
   - Detailed compliance reporting
   - Per-page audit trails

3. **Cell Grid Resolver** (`cell_grid_resolver.py`)
   - Vision model bbox detection using labeled cell grids
   - More accurate than raw coordinate requests
   - Automatic grid sizing based on page aspect ratio
   - Fallback to stacked regions when vision calls fail

4. **Modern Data Models** (`pdf_accessibility_models.py`)
   - Type-safe dataclasses for all models
   - Enum-based compliance levels
   - Clean serialization to JSON
   - Comprehensive tag validation

## Deployment Architecture

This function uses a **container-based Lambda** deployment with dependencies split across two layers:

1. **Docker Container**: Large binary dependencies (pymupdf, pikepdf, system libraries)
2. **Lambda Layer**: Foundation library (pure Python, shared across functions)

The container image is built from [Dockerfile](Dockerfile) and pushed to ECR. The foundation layer is attached at deployment time, making the foundation library available at `/opt/python/foundation/`.

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

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

## File Inventory

```
deployment/lambdas/containers/remediation_analyzer/
├── lambda_handler.py              # Entry point, orchestration, S3 I/O
├── pdf_accessibility_tagger.py    # Structure tree builder, auditor, overlay engine (v2.0)
├── pdf_accessibility_auditor.py   # Pre/post compliance checks (v2.0)
├── pdf_accessibility_models.py    # Data models and constants (v2.0)
├── cell_grid_resolver.py          # Grid-based vision bbox resolution (v2.0)
├── Dockerfile                     # Container image definition
├── requirements.txt               # Python dependencies
├── build.sh                       # Build and deployment script
├── README.md                      # This file
└── DEPLOYMENT.md                  # Detailed deployment guide
```

## Installation

### Lambda Deployment (Container-based)

This function is deployed as a **container-based Lambda** function. For complete deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

**Quick Start**:
```bash
cd deployment/lambdas/containers/remediation_analyzer

# Build and push to ECR
./build.sh --push --region us-west-2

# Deploy via CDK (from project root)
cd ../../../..
cdk deploy
```

**Dependencies**:
- **Foundation library**: Provided via Lambda Layer at `/opt/python/foundation/` (deployed separately by CDK)
- **Container dependencies** (in requirements.txt):
  - boto3/botocore (provided by Lambda runtime)
  - pymupdf>=1.24.0
  - pikepdf>=8.0.0
  - lxml>=5.0.0
  - Pillow>=10.0.0

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure foundation library is in Python path**:
   ```python
   import sys
   # For local development, add foundation to path
   sys.path.insert(0, "/path/to/badgers/foundation")
   ```

   **Note**: In Lambda, the foundation library is automatically available at `/opt/python/foundation/` via the attached Lambda Layer. No path modification is needed in deployed code.

## API

### Lambda Handler

**Event Schema**:
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

**Response Schema**:
```json
{
  "statusCode": 200,
  "body": {
    "success": true,
    "session_id": "session-abc123",
    "compliance": "pass",
    "result": {
      "output_pdf": "/tmp/.../tagged_output.pdf",
      "s3_output_uri": "s3://output-bucket/remediation_analyzer/results/...",
      "s3_report_uri": "s3://output-bucket/remediation_analyzer/results/...",
      "pages_processed": 1,
      "correlation_used": true,
      "accessibility_report": { ... }
    }
  }
}
```

### Direct Python Usage

```python
from pdf_accessibility_tagger import PDFAccessibilityTagger

with PDFAccessibilityTagger("input.pdf") as tagger:
    # Check pre-remediation compliance
    print(f"Pre-compliance: {tagger.report.pre_level.value}")

    # Add tagged regions
    tagger.add_region_normalized(
        page=0,
        bbox_normalized=(0.1, 0.1, 0.9, 0.2),
        tag="H1",
        text_content="Document Title",
        order=0
    )

    # Save with metadata
    output_path, report = tagger.save(
        "output.pdf",
        title="Accessible Document",
        lang="en-US"
    )

    print(f"Post-compliance: {report.post_level.value}")
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

## Differences from Original

### What's the Same
- Lambda handler interface (fully backward compatible)
- Event and response schemas
- S3 integration patterns
- Foundation library integration
- Environment variable configuration
- Three-tier bbox resolution strategy

### What's New
- **v2.0 core modules**: Modernized tagger, auditor, and data models
- **Better error handling**: More graceful degradation when vision calls fail
- **Enhanced auditing**: Pre/post compliance checks with detailed reporting
- **Context manager support**: Safer resource management with `with` statements
- **Type safety**: dataclasses and type hints throughout
- **Improved documentation**: Comprehensive inline comments and docstrings

### Migration from Original

The v2.0 replacement is **API-compatible** with the original:

1. **Deployment changes**: Now uses container-based Lambda + Layer architecture (see [DEPLOYMENT.md](DEPLOYMENT.md))
2. **No code changes** required in calling code
3. **Lambda events** work identically
4. **Response format** unchanged
5. **Foundation dependency**: Now provided via Lambda Layer instead of bundled in package

## Testing

### Unit Testing

```bash
python -m pytest tests/
```

### Integration Testing (Local)

```python
from lambda_handler import lambda_handler

event = {
    "pdf_path": "/path/to/document.pdf",
    "correlation_uri": "/path/to/correlation.xml",
    "session_id": "test-001",
    "title": "Test Document",
    "lang": "en-US",
    "dpi": 150,
}

result = lambda_handler(event, None)
print(result)
```

### Jupyter Notebook Testing

See the included `remediation_analyzer.ipynb` for interactive testing and development.

## Compliance Checks

The auditor runs these eight checks pre and post-remediation:

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

**Compliance Levels**:
- `pass`: All checks pass
- `pass_with_warnings`: Only info-level issues
- `fail`: Critical or major failures
- `not_assessed`: Audit didn't run

## Cell Grid Resolver

The core innovation for accurate bbox detection on image-only PDFs:

1. Overlays a labeled grid (A1, B2, ...) on the page image
2. Sends gridded image + element descriptions to vision model
3. Model reports which cells each element occupies
4. Converts cell references to normalized bboxes

**Advantages over raw coordinates**:
- Vision models excel at reading grid labels
- More accurate spatial understanding
- Graceful degradation with confidence levels
- Auto-sized grids for different page aspect ratios

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'foundation'**
   - **For Lambda**: Verify the foundation Lambda Layer is attached to the function
   - Check that the layer is compatible with Python 3.12 runtime
   - Foundation should be available at `/opt/python/foundation/` in the Lambda environment
   - **For local development**: Ensure foundation is in your Python path (see Local Development section)

2. **Vision model timeouts**
   - Check AWS region and Bedrock model availability
   - Verify IAM permissions for Bedrock access
   - Consider increasing Lambda timeout

3. **Missing correlation XML**
   - System falls back to full vision analysis
   - Check S3 paths and permissions
   - Verify XML format matches expected schema

4. **Post-remediation compliance still fails**
   - Review the audit report details
   - Check that all regions have proper tags
   - Verify figures have alt text

## Support

For issues specific to this replacement package, refer to the main project documentation or contact the development team.

## License

Same as the parent BADGERS project.
