<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [UI](../../ui/UI_README.md) | [Deployment](../DEPLOYMENT_README.md) | [CDK Stacks](../stacks/STACKS_README.md) | [Runtime](../runtime/RUNTIME_README.md) | [S3 Files](../s3_files/S3_FILES_README.md) | 🔵 **Lambda Specialists** | [Prompting System](../s3_files/prompts/PROMPTING_SYSTEM_README.md)</sub>

---

# ⚡ Lambda Specialists

This document explains how Lambda specialists work in BADGERS—their anatomy, required layers, environment variables, and code patterns.

---

## 🏗️ Lambda Types

BADGERS uses four types of Lambda functions:

| Type                     | Purpose                                             | Example                                          |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------ |
| 🔍 **Vision Specialists** | Analyze images using Bedrock vision models          | `full_text_specialist`, `table_specialist`, etc. |
| 🐳 **Container Lambdas**  | Processing requiring large deps or custom pipelines | `image_enhancer`, `remediation_specialist`       |
| 🔧 **Utilities**          | Transform or prepare data                           | `pdf_to_images_converter`                        |
| 🔗 **Correlators**        | Correlate outputs across specialists per page       | `correlation_specialist`                         |

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

### Vision Specialists

| Variable                 | Required | Default     | Description                                          |
| ------------------------ | -------- | ----------- | ---------------------------------------------------- |
| `CONFIG_BUCKET`          | ✅        | -           | S3 bucket containing specialist configs              |
| `OUTPUT_BUCKET`          | ✅        | -           | S3 bucket for saving results                         |
| `SPECIALIST_NAME`        | ✅        | -           | Specialist identifier (e.g., `full_text_specialist`) |
| `LOGGING_LEVEL`          | ❌        | `INFO`      | Log verbosity                                        |
| `MAX_TOKENS`             | ❌        | `8000`      | Max response tokens from Bedrock                     |
| `TEMPERATURE`            | ❌        | `0.1`       | Model temperature (lower = more deterministic)       |
| `AWS_REGION`             | ❌        | `us-west-2` | Region for Bedrock calls                             |
| `DYNAMIC_TOKENS_ENABLED` | ❌        | `false`     | Enable complexity-based dynamic token estimation     |
| `JOBS_TABLE_NAME`        | ❌        | -           | DynamoDB jobs table. Unset disables job tracking     |

### Input Parameters

| Parameter                | Required | Description                                                                               |
| ------------------------ | -------- | ----------------------------------------------------------------------------------------- |
| `session_id`             | ✅        | Runtime session ID for tracing and S3 output                                              |
| `image_path`             | ✅*       | S3 URL or file path (*or `image_data`)                                                    |
| `image_data`             | ✅*       | Base64-encoded image (*or `image_path`)                                                   |
| `aws_profile`            | ❌        | Optional AWS profile for local testing                                                    |
| `audit_mode`             | ❌        | Enable confidence scoring and human review flags                                          |
| `dynamic_tokens_enabled` | ❌        | Enable dynamic max_tokens based on image complexity (default: false)                      |
| `job_id`                 | ❌        | Job this invocation belongs to. Stamped by the agent — see [Job Tracking](#-job-tracking) |
| `doc_id`                 | ❌        | Document the job belongs to. Stamped by the agent                                         |

> `job_id` and `doc_id` are declared on the tool schemas so the model can see them, but
> both are optional. A turn that invokes a specialist outside any tracked job passes
> neither, and tracking simply no-ops.

### Utilities

| Variable        | Required | Description                            |
| --------------- | -------- | -------------------------------------- |
| `OUTPUT_BUCKET` | ✅        | S3 bucket for storing converted images |

---

## 🔬 Anatomy of a Vision Specialist

Every vision specialist follows the same pattern:

```python
def lambda_handler(event, context):
    # 1️⃣ Log Gateway context (for AgentCore tracing)
    # 2️⃣ Detect config source (S3 vs local)
    # 3️⃣ Parse input and extract session_id
    # 4️⃣ Mark the subtask RUNNING (job tracking)
    # 5️⃣ Get image data (S3 or base64)
    # 6️⃣ Load configuration
    # 7️⃣ Initialize specialist with foundation
    # 8️⃣ Run analysis
    # 9️⃣ Save result to S3 (raise if it cannot be persisted)
    # 🔟 Mark the subtask COMPLETE, or FAILED from the except block
    # 1️⃣1️⃣ Return response
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

Specialists auto-detect whether to load config from S3 (production) or local filesystem (testing).

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
from foundation.specialist_foundation import SpecialistFoundation

specialist = SpecialistFoundation("full_text_specialist")
result = specialist.analyze(image_data, aws_profile)
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
    specialist_name=specialist_name,
    output_bucket=output_bucket,
    session_id=session_id,
    image_path=body.get("image_path"),
)
```

Results are saved to S3 with path:
`{session_id}/{specialist_name}/{specialist}_{image_identifier}_{timestamp}`

Persisting the result is part of succeeding. If `OUTPUT_BUCKET` is unset, or the artifact
cannot be written, the specialist raises rather than returning a result the caller cannot
retrieve. A false success would also encourage the orchestrator to keep calling a tool
that cannot deliver.

---

## 📋 Job Tracking

Specialist work is recorded in a three-level hierarchy:

```
doc_id  ->  job_id  ->  subtask_id
```

| Level        | Meaning                               | Minted by                                         |
| ------------ | ------------------------------------- | ------------------------------------------------- |
| `doc_id`     | One uploaded document                 | UI server at upload (`POST /api/upload`)          |
| `job_id`     | One analysis run over that document   | The agent, on the first specialist call of a turn |
| `subtask_id` | One specialist against one page/image | Each specialist Lambda, deterministically         |

Why three levels: BADGERS invokes the same specialist repeatedly within one run, once per
page. Keying on the specialist name alone would collapse every page into a single row and
retain only the last one.

The subtask sort key is `{specialist}#{image_identifier}` — deterministic rather than a
UUID. That gives uniqueness across the page fan-out while making retries idempotent:
re-running the same specialist against the same page upserts the same row.

### How identifiers arrive

The agent stamps `job_id` and `doc_id` into the tool input via a Strands
`BeforeToolCallEvent` hook, for tools whose schema declares `job_id`. The model is not
asked to supply them — it would have to invent and then remember a UUID across turns,
which is not something to depend on for the integrity of a tracking record.

### What a specialist writes

```python
from foundation import job_state

job_id = body.get("job_id") or ""
doc_id = body.get("doc_id") or ""
subtask = job_state.subtask_id(specialist_name, body.get("image_path"))

job_state.mark_running(job_id, subtask, doc_id=doc_id,
                       specialist=specialist_name,
                       image_id=job_state.image_identifier(body.get("image_path")),
                       session_id=session_id)
# ... analysis ...
job_state.mark_complete(job_id, subtask, s3_uri)     # success, records the S3 key
# ... on exception ...
job_state.mark_failed(job_id, subtask, str(e))       # failure, records the reason
```

Status moves `PENDING → RUNNING → COMPLETE | FAILED`. On failure the exception message is
stored in the `error` attribute (truncated to 1024 characters), so a failed page carries
its own reason.

### Two properties worth relying on

**Tracking never breaks analysis.** Every `job_state` write is wrapped so it logs a
warning instead of raising. A tracking outage degrades observability, not throughput.

**Tracking is opt-in per deployment.** Every write is a no-op when `JOBS_TABLE_NAME` is
unset, so specialists work unchanged in deployments or local runs without a jobs table.

### Reading job state

Job status is computed at read time by aggregating subtasks, not stored on the job row.
See the [UI](../../ui/UI_README.md#job-tracking-endpoints) endpoints, or use the read
helpers in `job_state` directly:

```python
job_state.get_job_records(job_id)   # job row plus every subtask
job_state.get_record(job_id, subtask)
```

> **Custom specialists skip tracking.** The wizard generator (`/api/wizard/generate`) is
> still a stub and does not emit `job_id`/`doc_id` on generated schemas, so wizard-created
> specialists are not stamped and record nothing.

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

    # 4️⃣ Return S3 paths for downstream specialists
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

Combines results from multiple specialist invocations:

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

All Lambdas use standard try/except patterns, recording the failure before returning:

```python
job_id = ""      # declared outside the try so the handler below can still
subtask = ""     # mark the subtask failed if parsing itself blew up
try:
    # ... processing ...
except Exception as e:
    logger.error("Error: %s", e, exc_info=True)
    job_state.mark_failed(job_id, subtask, str(e))
    return {
        "statusCode": 500,
        "body": json.dumps({"result": str(e), "success": False}),
    }
```

A failed specialist returns 500 to the orchestrator and records its own reason. It does
not abort the run: other pages continue, and the job ends up `PARTIAL` rather than
`FAILED`.

---

## 📥 Input/Output Format

### Request

```json
{
  "session_id": "abc123",
  "image_path": "s3://bucket/image.png",
  "aws_profile": null,
  "job_id": "9f2c1ab84e7d4c0f8b1a",
  "doc_id": "5d7e3f10-2c44-4b8e-9a10-77c1f0e2b3d9"
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

## 🚀 Adding a New Specialist

**Option 1: Use the Wizard (Recommended)**

```bash
cd frontend
uv run python main.py
```

The Specialist Creation Wizard is available as the 🧙 Create Specialist tab in the [UI](../../ui/UI_README.md).

**Option 2: Manual Creation**

1. Create directory: `deployment/lambdas/code/{specialist_name}/`
2. Copy `lambda_handler.py` from an existing specialist
3. Update `SPECIALIST_NAME` references
4. Create manifest in `deployment/s3_files/manifests/{specialist_name}.json`
5. Create prompts in `deployment/s3_files/prompts/{specialist_name}/`
6. Add to CDK stack in `deployment/stacks/`

The Foundation Layer handles everything else automatically.
