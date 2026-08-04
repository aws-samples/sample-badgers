<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [UI](../../ui/UI_README.md) | [Deployment](../DEPLOYMENT_README.md) | [CDK Stacks](../stacks/STACKS_README.md) | [Runtime](../runtime/RUNTIME_README.md) | [S3 Files](../s3_files/S3_FILES_README.md) | [Lambda Specialists](LAMBDA_SPECIALISTS.md) | [Prompting System](../s3_files/prompts/PROMPTING_SYSTEM_README.md)</sub>

# 🧠 Foundation Lambda Layer

The Foundation Layer is a reusable AWS Lambda layer containing the core framework, dependencies, and shared utilities for all specialist tools. It implements the "foundation + specialization" pattern where common functionality lives in the layer while specialist-specific logic remains in individual Lambda functions.

## 🏗️ Architecture

```
layer/python/
├── foundation/                    # 🧠 Core framework modules
│   ├── specialist_foundation.py     # 🎯 Main orchestrator class
│   ├── bedrock_client.py          # 🤖 AWS Bedrock integration
│   ├── configuration_manager.py   # ⚙️ Config loading and validation
│   ├── image_processor.py         # 🖼️ Image optimization and encoding
│   ├── message_chain_builder.py   # 🔗 Few-shot message construction
│   ├── prompt_loader.py           # 📝 Prompt file loading and composition
│   ├── response_processor.py      # 📤 Response extraction and validation
│   ├── s3_config_loader.py        # ☁️ S3-based configuration loading
│   ├── s3_result_saver.py         # 💾 Result persistence to S3
│   ├── job_state.py               # 📋 DynamoDB job/subtask state writer
│   └── lambda_error_handler.py    # ❌ Standardized error handling
├── config/                        # ⚙️ Configuration utilities
├── prompts/core_system_prompts/   # 📝 Shared prompt components
├── boto3/                         # ☁️ AWS SDK
├── botocore/                      # ☁️ AWS core library
├── PIL/                           # 🖼️ Pillow image processing
├── pdf2image/                     # 📄 PDF to image conversion
└── [other dependencies]
```

---

## 🧩 Core Modules

### 🎯 SpecialistFoundation (`specialist_foundation.py`)

The main orchestrator class that all specialists use. Coordinates the complete analysis workflow:

1. **⚙️ Configuration Loading** - Loads specialist config from local manifest or central config
2. **🖼️ Image Processing** - Optimizes and encodes target images
3. **📝 Prompt Composition** - Loads and combines system prompts with placeholders
4. **🎓 Example Loading** - Loads few-shot example images if configured
5. **🔗 Message Building** - Constructs the message chain for Bedrock
6. **🤖 Model Invocation** - Calls Bedrock with fallback support
7. **📤 Response Processing** - Extracts and validates results

```python
from foundation import SpecialistFoundation

class MySpecialist:
    def __init__(self):
        self.foundation = SpecialistFoundation("my_specialist")

    def analyze(self, image_path, aws_profile=None):
        return self.foundation.analyze(image_path, aws_profile)
```

### 🤖 BedrockClient (`bedrock_client.py`)

Manages AWS Bedrock interactions with:
- **🔄 Multi-model support** - Claude and Nova model families
- **🔀 Automatic payload conversion** - Converts between Claude and Nova formats
- **🛡️ Fallback chains** - Tries alternate models on failure
- **⏱️ Throttling handling** - Exponential backoff on rate limits
- **📊 Response normalization** - Consistent output format regardless of model

### ⚙️ ConfigurationManager (`configuration_manager.py`)

Handles configuration loading and validation:
- 📂 Loads from JSON config files or S3
- ✅ Validates required fields and types
- 🔄 Supports both central config and per-specialist manifests
- 💾 Caches loaded configurations

### 🖼️ ImageProcessor (`image_processor.py`)

Image optimization for Bedrock vision models:
- 📏 Resizes images exceeding max dimensions (default 2048px)
- 🎨 Converts to RGB JPEG format
- 🏳️ Handles transparency with white background
- 🔤 Base64 encoding for API transmission
- 📐 Dimension extraction for prompt placeholders

### 📝 PromptLoader (`prompt_loader.py`)

Composes system prompts from multiple files:
- 📂 Loads core system files (rules, error handlers)
- 📄 Loads specialist-specific prompt files
- 🎁 Injects content into wrapper template
- 🔄 Supports placeholder replacement (e.g., `[[PIXEL_WIDTH]]`)
- ☁️ Works with both local filesystem and S3

### 🔗 MessageChainBuilder (`message_chain_builder.py`)

Constructs message chains for Bedrock:
- 🎓 Builds few-shot examples from image directories
- 💬 Creates user/assistant message pairs
- 🎯 Adds target image with analysis request
- ✅ Validates message structure

### 📤 ResponseProcessor (`response_processor.py`)

Processes Bedrock responses:
- 📝 Extracts text content from response
- 🧹 Strips markdown code fences
- ✅ Validates response quality
- ❌ Handles empty/error responses
- 🔍 Extracts structured data (JSON/XML)

### 📋 job_state (`job_state.py`)

The single writer for DynamoDB job records. Every writer goes through this module — the
orchestrator runtime, the built-in specialist Lambdas, and generated custom specialists —
so the record shape stays defined in one place.

Key derivation:
```python
job_state.image_identifier("s3://b/pages/page_1.png")   # -> "page_1"
job_state.subtask_id("table_specialist", "s3://b/pages/page_1.png")
# -> "table_specialist#page_1"
```

Writes:
| Function                                         | Effect                                                               |
| ------------------------------------------------ | -------------------------------------------------------------------- |
| `create_job(job_id, doc_id, session_id, reason)` | Conditional put of the job-level row (`subtask_id` = `orchestrator`) |
| `mark_running(job_id, subtask, ...)`             | Upserts the subtask and sets `RUNNING`                               |
| `mark_complete(job_id, subtask, s3_key)`         | Sets `COMPLETE` and records the output key                           |
| `mark_failed(job_id, subtask, error)`            | Sets `FAILED` and records the reason                                 |

Reads: `get_record(job_id, subtask)`, `get_job_records(job_id)`.

Two behaviors this module guarantees:
- **Never raises.** Failures are logged as warnings. A tracking outage does not fail an
  analysis.
- **No-ops when unconfigured.** Every call returns early when `JOBS_TABLE_NAME` is unset,
  read at call time rather than import time because the Lambda environment may be
  populated after the module is first imported.

Because `mark_running` uses `update_item` (an upsert), no `PENDING` row needs to exist
first. See [Lambda Specialists](LAMBDA_SPECIALISTS.md#-job-tracking) for the hierarchy and
the calling pattern.

> The AgentCore Runtime image is not built on this layer, but it imports this same module.
> `build_and_push_websocket.sh` copies `badgers-foundation/foundation/` into the runtime
> build context so `create_job` comes from one implementation.

---

## ✨ What the Layer Enables

### 1. 📦 Minimal Specialist Code

Individual specialists only need ~50 lines of code:

```python
from foundation import SpecialistFoundation

class FullTextSpecialist:
    def __init__(self):
        self.foundation = SpecialistFoundation("full_text")

    def analyze_full_text(self, image_path, aws_profile=None):
        return self.foundation.analyze(image_path, aws_profile)
```

### 2. 🎯 Consistent Behavior

All specialists automatically get:
- 🔄 Retry logic with exponential backoff
- 🛡️ Model fallback chains
- 🖼️ Image optimization
- ❌ Error handling
- ✅ Response validation
- 📊 Logging

### 3. ⚙️ Configuration-Driven

New specialists require only:
- 📋 A manifest.json with model and prompt configuration
- 📝 Prompt XML files defining the analysis task
- 🔌 A thin wrapper calling the foundation

### 4. 🚀 Efficient Deployment

- 📦 Layer deployed once (~50MB compressed)
- 🪶 Individual specialists are tiny (~10-20KB)
- ⚡ Fast cold starts
- 🔄 Independent specialist updates

---

## 📚 Dependencies Included

| Package         | Version | Purpose                             |
| --------------- | ------- | ----------------------------------- |
| boto3           | 1.42.5  | ☁️ AWS SDK for Bedrock, S3           |
| botocore        | 1.42.5  | ☁️ AWS core functionality            |
| pillow          | 12.0.0  | 🖼️ Image processing and optimization |
| pdf2image       | 1.17.0  | 📄 PDF to image conversion           |
| jmespath        | 1.0.1   | 🔍 JSON query (boto3 dependency)     |
| urllib3         | 2.6.1   | 🌐 HTTP client (boto3 dependency)    |
| python-dateutil | 2.9.0   | 📅 Date utilities                    |

---

> **Note:** The container Lambdas do not attach the Foundation Layer — they bundle their
> own dependencies in their Docker images. `image_enhancer` uses the full foundation
> framework; `remediation_specialist` is otherwise self-contained (pikepdf, pymupdf, etc.)
> but both import `foundation.job_state` to record their subtask state, so
> `build_container_lambdas.sh` copies the `foundation/` module into each build context.

## 🛠️ Build and Deploy

```bash
# Run from deployment/lambdas directory
cd deployment/lambdas

# Build the layer
./build_foundation_layer.sh

# Deploy to AWS
./deploy_foundation_layer.sh

# Layer ARN saved to layer_arn.txt
```

### ⚠️ Build order matters

`layer/` is a generated, gitignored directory. `build_foundation_layer.sh` populates
`layer/python/foundation/` by copying `../badgers-foundation/foundation/*.py`, which is the
tracked source of truth.

`build_container_lambdas.sh` copies the module from `layer/python/foundation/`, so a stale
or missing `layer/` produces container images without the current foundation code. If a
container Lambda fails on `from foundation import job_state`, rebuild the layer first:

```bash
./build_foundation_layer.sh && ./build_container_lambdas.sh
```

The layer is compatible with:
- **🐍 Runtime**: Python 3.12
- **💻 Architectures**: x86_64, arm64
- **🌍 Regions**: Any region with Bedrock access

---

## 🔗 Attaching to Lambda Functions

```bash
aws lambda update-function-configuration \
    --function-name my-specialist \
    --layers $(cat layer_arn.txt)
```

Or in CDK/CloudFormation, reference the layer ARN from `layer_arn.txt`.
