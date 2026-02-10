<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [Frontend](../../frontend/FRONTEND_README.md) | [Deployment](../DEPLOYMENT_README.md) | [CDK Stacks](../stacks/STACKS_README.md) | [Runtime](../runtime/RUNTIME_README.md) | [S3 Files](../s3_files/S3_FILES_README.md) | [Lambda Analyzers](LAMBDA_ANALYZERS.md) | [Prompting System](../s3_files/prompts/PROMPTING_SYSTEM_README.md)</sub>

# 🧠 Foundation Lambda Layer

The Foundation Layer is a reusable AWS Lambda layer containing the core framework, dependencies, and shared utilities for all analyzer tools. It implements the "foundation + specialization" pattern where common functionality lives in the layer while analyzer-specific logic remains in individual Lambda functions.

## 🏗️ Architecture

```
layer/python/
├── foundation/                    # 🧠 Core framework modules
│   ├── analyzer_foundation.py     # 🎯 Main orchestrator class
│   ├── bedrock_client.py          # 🤖 AWS Bedrock integration
│   ├── configuration_manager.py   # ⚙️ Config loading and validation
│   ├── image_processor.py         # 🖼️ Image optimization and encoding
│   ├── message_chain_builder.py   # 🔗 Few-shot message construction
│   ├── prompt_loader.py           # 📝 Prompt file loading and composition
│   ├── response_processor.py      # 📤 Response extraction and validation
│   ├── s3_config_loader.py        # ☁️ S3-based configuration loading
│   ├── s3_result_saver.py         # 💾 Result persistence to S3
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

### 🎯 AnalyzerFoundation (`analyzer_foundation.py`)

The main orchestrator class that all analyzers use. Coordinates the complete analysis workflow:

1. **⚙️ Configuration Loading** - Loads analyzer config from local manifest or central config
2. **🖼️ Image Processing** - Optimizes and encodes target images
3. **📝 Prompt Composition** - Loads and combines system prompts with placeholders
4. **🎓 Example Loading** - Loads few-shot example images if configured
5. **🔗 Message Building** - Constructs the message chain for Bedrock
6. **🤖 Model Invocation** - Calls Bedrock with fallback support
7. **📤 Response Processing** - Extracts and validates results

```python
from foundation import AnalyzerFoundation

class MyAnalyzer:
    def __init__(self):
        self.foundation = AnalyzerFoundation("my_analyzer")

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
- 🔄 Supports both central config and per-analyzer manifests
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
- 📄 Loads analyzer-specific prompt files
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

---

## ✨ What the Layer Enables

### 1. 📦 Minimal Analyzer Code

Individual analyzers only need ~50 lines of code:

```python
from foundation import AnalyzerFoundation

class FullTextAnalyzer:
    def __init__(self):
        self.foundation = AnalyzerFoundation("full_text")

    def analyze_full_text(self, image_path, aws_profile=None):
        return self.foundation.analyze(image_path, aws_profile)
```

### 2. 🎯 Consistent Behavior

All analyzers automatically get:
- 🔄 Retry logic with exponential backoff
- 🛡️ Model fallback chains
- 🖼️ Image optimization
- ❌ Error handling
- ✅ Response validation
- 📊 Logging

### 3. ⚙️ Configuration-Driven

New analyzers require only:
- 📋 A manifest.json with model and prompt configuration
- 📝 Prompt XML files defining the analysis task
- 🔌 A thin wrapper calling the foundation

### 4. 🚀 Efficient Deployment

- 📦 Layer deployed once (~50MB compressed)
- 🪶 Individual analyzers are tiny (~10-20KB)
- ⚡ Fast cold starts
- 🔄 Independent analyzer updates

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

The layer is compatible with:
- **🐍 Runtime**: Python 3.12
- **💻 Architectures**: x86_64, arm64
- **🌍 Regions**: Any region with Bedrock access

---

## 🔗 Attaching to Lambda Functions

```bash
aws lambda update-function-configuration \
    --function-name my-analyzer \
    --layers $(cat layer_arn.txt)
```

Or in CDK/CloudFormation, reference the layer ARN from `layer_arn.txt`.
