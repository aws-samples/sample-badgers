<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [Frontend](../../frontend/FRONTEND_README.md) | [Deployment](../DEPLOYMENT_README.md) | [CDK Stacks](../stacks/STACKS_README.md) | [Runtime](../runtime/RUNTIME_README.md) | [S3 Files](../s3_files/S3_FILES_README.md) | 🔵 **Lambda Analyzers** | [Prompting System](../s3_files/prompts/PROMPTING_SYSTEM_README.md) | [Analyzer Wizard](../../frontend/ANALYZER_CREATION_WIZARD.md)</sub>

---

# ⚡ Lambda Analyzers

This document explains how Lambda analyzers work in BADGERS—their anatomy, required layers, environment variables, and code patterns.

---

## 🏗️ Lambda Types

BADGERS uses four types of Lambda functions:

| Type                    | Purpose                                     | Example                                      |
| ----------------------- | ------------------------------------------- | -------------------------------------------- |
| 🔍 **Vision Analyzers**  | Analyze images using Bedrock vision models  | `full_text_analyzer`, `table_analyzer`, etc. |
| 🐳 **Container Lambdas** | ML-based processing requiring large deps    | `image_enhancer`, `remediation_analyzer`     |
| 🔧 **Utilities**         | Transform or prepare data                   | `pdf_to_images_converter`                    |
| 📊 **Aggregators**       | Combine results from multiple analyses      | `aggregate_analysis_results`                 |
| 🔗 **Correlators**       | Correlate outputs across analyzers per page | `correlation_analyzer`                       |

---

## 📦 Required Layers

### All Lambdas

| Layer                  | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| 🧠 **Foundation Layer** | Core framework, boto3, Pillow, pdf2image, shared utilities |

### PDF Converter Only

| Layer               | Purpose                                                 |
| ------------------- | ------------------------------------------------------- |
| 📄 **Poppler Layer** | `pdf2image` requires Poppler binaries for PDF rendering |

---

## ⚙️ Environment Variables

### Vision Analyzers

| Variable        | Required | Default     | Description                                      |
| --------------- | -------- | ----------- | ------------------------------------------------ |
| `CONFIG_BUCKET` | ✅        | -           | S3 bucket containing analyzer configs            |
| `OUTPUT_BUCKET` | ✅        | -           | S3 bucket for saving results                     |
| `ANALYZER_NAME` | ✅        | -           | Analyzer identifier (e.g., `full_text_analyzer`) |
| `LOGGING_LEVEL` | ❌        | `INFO`      | Log verbosity                                    |
| `MAX_TOKENS`    | ❌        | `8000`      | Max response tokens from Bedrock                 |
| `TEMPERATURE`   | ❌        | `0.1`       | Model temperature (lower = more deterministic)   |
| `AWS_REGION`    | ❌        | `us-west-2` | Region for Bedrock calls                         |

### Input Parameters

| Parameter     | Required | Description                                      |
| ------------- | -------- | ------------------------------------------------ |
| `session_id`  | ✅        | Runtime session ID for tracing and S3 output     |
| `image_path`  | ✅*       | S3 URL or file path (*or `image_data`)           |
| `image_data`  | ✅*       | Base64-encoded image (*or `image_path`)          |
| `aws_profile` | ❌        | Optional AWS profile for local testing           |
| `audit_mode`  | ❌        | Enable confidence scoring and human review flags |

### Utilities

| Variable        | Required | Description                            |
| --------------- | -------- | -------------------------------------- |
| `OUTPUT_BUCKET` | ✅        | S3 bucket for storing converted images |

---

## 🔬 Anatomy of a Vision Analyzer

Every vision analyzer follows the same pattern:

```python
def lambda_handler(event, context):
    # 1️⃣ Log Gateway context (for AgentCore tracing)
    # 2️⃣ Detect config source (S3 vs local)
    # 3️⃣ Parse input and extract session_id
    # 4️⃣ Get image data (S3 or base64)
    # 5️⃣ Load configuration
    # 6️⃣ Initialize analyzer with foundation
    # 7️⃣ Run analysis
    # 8️⃣ Save result to S3
    # 9️⃣ Return response
```

### 1️⃣ Gateway Context Logging

```python
if hasattr(context, "client_context") and context.client_context:
    gateway_id = context.client_context.custom.get("bedrockAgentCoreGatewayId", "unknown")
    tool_name = context.client_context.custom.get("bedrockAgentCoreToolName", "unknown")
    logger.info("Gateway invocation - Gateway: %s, Tool: %s", gateway_id, tool_name)
```

When invoked via AgentCore Gateway, the Lambda receives metadata about which gateway and tool triggered it.

### 2️⃣ Config Source Detection

```python
config_bucket = os.environ.get("CONFIG_BUCKET")
if os.environ.get("AWS_EXECUTION_ENV") and config_bucket:
    config_source = "s3"  # Running in Lambda with S3 config
else:
    config_source = "local"  # Local testing with manifest.json
```

Analyzers auto-detect whether to load config from S3 (production) or local filesystem (testing).

### 3️⃣ Session Tracking

```python
session_id = body.get("session_id", "no_session")
logger.info("Processing request for runtime session_id: %s", session_id)
```

AgentCore Runtime passes a `session_id` that links all tool invocations in a conversation. This enables:
- 📊 Tracing requests across multiple Lambda calls
- 📁 Organizing outputs by session in S3
- 🔍 Debugging multi-step workflows

### 4️⃣ Image Data Handling

```python
def _get_image_data(body: dict) -> bytes:
    if "image_data" in body:
        return base64.b64decode(body["image_data"])  # Direct base64

    if "image_path" in body:
        if image_path.startswith("s3://"):
            # Download from S3
            s3.get_object(Bucket=bucket, Key=key)
            # Handle .b64 files (pre-encoded from pdf_to_images)
            if key.endswith(".b64"):
                return base64.b64decode(data.decode("utf-8"))
            return bytes(data)
```

Supports three input modes:
- 📤 **Direct base64** - `image_data` field
- ☁️ **S3 path** - `s3://bucket/key` format
- 📄 **Pre-encoded .b64** - From PDF converter output

### 5️⃣ Foundation Initialization

```python
from foundation.analyzer_foundation import AnalyzerFoundation

analyzer = AnalyzerFoundation("full_text_analyzer")
result = analyzer.analyze(image_data, aws_profile)
```

The Foundation Layer handles all complexity:
- 📝 Prompt loading and composition
- 🖼️ Image optimization
- 🤖 Bedrock invocation with retries
- 📤 Response processing

### 6️⃣ Result Persistence

```python
from foundation.s3_result_saver import save_result_to_s3

s3_uri = save_result_to_s3(
    result=result,
    analyzer_name=analyzer_name,
    output_bucket=output_bucket,
    session_id=session_id,
    image_path=body.get("image_path"),
)
```

Results are saved to S3 with path: `results/{session_id}/{analyzer_name}/{timestamp}.txt`

---

## 🔧 Utility Lambda: PDF Converter

The PDF converter transforms PDFs into analyzable images:

```python
def lambda_handler(event, context):
    # 1️⃣ Get PDF from S3 or local path
    pdf_data = _get_pdf_data(pdf_path)

    # 2️⃣ Convert to images using pdf2image + Poppler
    base64_images = _convert_pdf_to_images(pdf_data, dpi, max_size_mb)

    # 3️⃣ Store as .b64 files in S3 temp location
    s3_paths = _store_images_to_s3(base64_images, session_id)

    # 4️⃣ Return S3 paths for downstream analyzers
    return {"images": s3_paths, "page_count": len(s3_paths)}
```

### Image Compression

```python
quality = 85
while quality > 20:
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    if buffer.tell() <= max_size_bytes:
        break
    quality -= 10  # Reduce quality until under size limit
```

Iteratively compresses images to meet Bedrock's size limits (default 4MB).

---

## 📊 Aggregator Lambda

Combines results from multiple analyzer invocations:

```python
def _aggregate_by_page(execution_results: list, pdf_name: str) -> dict:
    pages = {}
    for result in execution_results:
        page_num = result.get("page", 0)
        tool_name = result.get("tool", "unknown")

        if page_num not in pages:
            pages[page_num] = {"page": page_num, "analyses": []}

        pages[page_num]["analyses"].append({
            "tool": tool_name,
            "result": result.get("result"),
            "success": result.get("success")
        })

    return {"pdf_name": pdf_name, "pages": sorted(pages.values())}
```

Output structure:
```json
{
  "pdf_name": "document.pdf",
  "total_pages": 3,
  "pages": [
    {"page": 1, "analyses": [{"tool": "full_text", "result": "..."}]},
    {"page": 2, "analyses": [{"tool": "full_text", "result": "..."}]}
  ]
}
```

---

## ❌ Error Handling

All Lambdas use standard try/except patterns:

```python
try:
    # ... processing ...
except Exception as e:
    logger.error("Error: %s", e, exc_info=True)
    return {
        "statusCode": 500,
        "body": json.dumps({"result": str(e), "success": False}),
    }
```

---

## 📥 Input/Output Format

### Request

```json
{
  "session_id": "abc123",
  "image_path": "s3://bucket/image.png",
  "aws_profile": null
}
```

Or with direct base64:
```json
{
  "session_id": "abc123",
  "image_data": "base64-encoded-image-bytes"
}
```

### Response

```json
{
  "statusCode": 200,
  "body": {
    "result": "Extracted text or analysis...",
    "success": true,
    "session_id": "abc123"
  }
}
```

---

## 🚀 Adding a New Analyzer

**Option 1: Use the Wizard (Recommended)**

```bash
cd frontend
uv run python main.py
```

The [Analyzer Creation Wizard](../../frontend/ANALYZER_CREATION_WIZARD.md) is available as a tab in the multi-page Gradio app.

**Option 2: Manual Creation**

1. Create directory: `deployment/lambdas/code/{analyzer_name}/`
2. Copy `lambda_handler.py` from an existing analyzer
3. Update `ANALYZER_NAME` references
4. Create manifest in `deployment/s3_files/manifests/{analyzer_name}.json`
5. Create prompts in `deployment/s3_files/prompts/{analyzer_name}/`
6. Add to CDK stack in `deployment/stacks/`

The Foundation Layer handles everything else automatically.
