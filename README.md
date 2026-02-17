<sub>🧭 **Navigation:**</sub><br>
<sub>🔵 **Home** | [Vision LLM Theory](VISION_LLM_THEORY_README.md) | [Frontend](frontend/FRONTEND_README.md) | [Deployment](deployment/DEPLOYMENT_README.md) | [CDK Stacks](deployment/stacks/STACKS_README.md) | [Runtime](deployment/runtime/RUNTIME_README.md) | [S3 Files](deployment/s3_files/S3_FILES_README.md) | [Lambda Analyzers](deployment/lambdas/LAMBDA_ANALYZERS.md) | [Prompting System](deployment/s3_files/prompts/PROMPTING_SYSTEM_README.md) | [Analyzer Wizard](frontend/ANALYZER_CREATION_WIZARD.md) | [Pricing Calculator](frontend/PRICING_CALCULATOR.md)</sub>

---

# 🦡 BADGERS

**Broad Agentic Document Generative Extraction & Recognition System**

BADGERS transforms document processing through vision-enabled AI and deep layout analysis. Unlike traditional text extraction tools, BADGERS understands document structure and meaning by recognizing visual hierarchies, reading patterns, and contextual relationships between elements.

## 🤔 Why BADGERS?

Traditional document processing tools extract text but lose context. They can't distinguish a header from body text, understand table relationships, or recognize that a diagram explains the adjacent paragraph. BADGERS solves this by:

- 🏗️ **Preserving semantic structure** - Maintains document hierarchy and element relationships
- 👁️ **Understanding visual context** - Recognizes how layout conveys meaning
- 📚 **Processing diverse content** - Handles 21+ element types from handwriting to equations
- 🤖 **Automating complex workflows** - Orchestrates multiple specialized analyzers via an AI agent

Use cases: research acceleration, compliance automation, content management, accessibility remediation.

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AgentCore Runtime                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  PDF Analysis Agent (Strands)                                       │   │
│   │  - Claude Sonnet 4.5 with Extended Thinking                         │   │
│   │  - Session state management                                         │   │
│   │  - MCP tool orchestration                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AgentCore Gateway                                 │
│   - MCP Protocol (2025-03-26)                                               │
│   - Cognito JWT Authentication                                              │
│   - Semantic tool search                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                  │                  │
                   ▼                  ▼                  ▼
            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
            │   Lambda    │    │   Lambda    │    │   Lambda    │
            │  Analyzer   │    │  Analyzer   │    │  Analyzer   │
            │ (29 tools)  │    │             │    │             │
            └─────────────┘    └─────────────┘    └─────────────┘
                   │                  │                  │
                   └──────────────────┼──────────────────┘
                                      ▼
                               ┌─────────────┐
                               │   Bedrock   │
                               │   Claude    │
                               └─────────────┘
```

1. 📄 **User submits a document** with analysis instructions
2. 🧠 **Strands Agent** (running in AgentCore Runtime) interprets the request
3. 🔧 **Agent selects tools** from 29 specialized analyzers via MCP Gateway
4. ⚡ **Lambda analyzers** (29 functions) process document elements using Claude vision models
5. 📊 **Results aggregate** with preserved structure and semantic relationships

## 🛠️ Tech Stack

| Component          | Technology                                                         |
| ------------------ | ------------------------------------------------------------------ |
| 🤖 Agent Framework  | [Strands Agents](https://github.com/strands-agents/strands-agents) |
| 🏠 Agent Hosting    | Amazon Bedrock AgentCore Runtime                                   |
| 🚪 Tool Gateway     | Amazon Bedrock AgentCore Gateway (MCP Protocol)                    |
| 🧠 Foundation Model | Claude Sonnet 4.5 (via Amazon Bedrock)                             |
| ⚡ Compute          | AWS Lambda (29 analyzer functions)                                 |
| 📦 Storage          | Amazon S3 (configs, prompts, outputs)                              |
| 🔐 Auth             | Amazon Cognito (OAuth 2.0 client credentials)                      |
| 🏗️ IaC              | AWS CDK (Python)                                                   |
| 📈 Observability    | CloudWatch Logs, X-Ray                                             |
| 📊 Cost Tracking    | Bedrock Application Inference Profiles                             |

## 🔬 Analyzers

| Analyzer                             | Purpose                                   |
| ------------------------------------ | ----------------------------------------- |
| 📸 `pdf_to_images_converter`          | Convert PDF pages to images               |
| 🏷️ `classify_pdf_content`             | Classify document content type            |
| 📝 `full_text_analyzer`               | Extract all text content                  |
| 📊 `table_analyzer`                   | Extract and structure tables              |
| 📈 `charts_analyzer`                  | Analyze charts and graphs                 |
| 🔀 `diagram_analyzer`                 | Process diagrams and flowcharts           |
| 📐 `layout_analyzer`                  | Document structure analysis               |
| ♿ `accessibility_analyzer`           | Generate accessibility metadata           |
| 🏥 `decision_tree_analyzer`           | Medical/clinical document analysis        |
| 🔬 `scientific_analyzer`              | Scientific paper analysis                 |
| ✍️ `handwriting_analyzer`             | Handwritten text recognition              |
| 💻 `code_block_analyzer`              | Extract code snippets                     |
| 🗂️ `metadata_generic_analyzer`        | Generic metadata extraction               |
| 🗂️ `metadata_mads_analyzer`           | MADS metadata format extraction           |
| 🗂️ `metadata_mods_analyzer`           | MODS metadata format extraction           |
| 🔑 `keyword_topic_analyzer`           | Extract keywords and topics               |
| 🔧 `remediation_analyzer`             | Analyze remediation requirements          |
| 📄 `page_analyzer`                    | Single page content analysis              |
| 🧱 `elements_analyzer`                | Document element detection                |
| 🧱 `robust_elements_analyzer`         | Enhanced element detection with fallbacks |
| 👁️ `general_visual_analysis_analyzer` | General-purpose visual content analysis   |
| ✏️ `editorial_analyzer`               | Editorial content and markup analysis     |
| 🗺️ `war_map_analyzer`                 | Historical war map analysis               |
| 🎓 `edu_transcript_analyzer`          | Educational transcript analysis           |
| 🔗 `correlation_analyzer`             | Correlate multi-analyzer results per page |
| 🖼️ `image_enhancer`                   | Image enhancement and preprocessing       |

## 🚀 Deployment

### Prerequisites

- ☁️ [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with credentials
- 📦 [AWS CDK v2](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html) (`npm install -g aws-cdk`)
- 🐳 [Docker](https://docs.docker.com/get-started/get-docker/) (running)
- 🐍 [Python 3.12+](https://www.python.org/downloads/)
- ⚡ [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Quick Start

```bash
cd deployment
./deploy_from_scratch.sh
```

This deploys 10 CloudFormation stacks:
1. 📦 S3 (config + output buckets)
2. 🔐 Cognito (OAuth authentication)
3. 👤 IAM (execution roles)
4. 🐳 ECR (container registry)
5. ⚡ Lambda (29 analyzer functions)
6. 🚪 Gateway (MCP endpoint)
7. 🧠 Memory (session persistence)
8. 📊 Inference Profiles (cost tracking)
9. 🏃 Runtime (Strands agent container)
10. 🧩 Custom Analyzers (optional, wizard-created)

### Manual Steps

See [deployment/DEPLOYMENT_README.md](deployment/DEPLOYMENT_README.md) for step-by-step instructions.

### Cleanup

```bash
cd deployment
./destroy.sh
```

## 📁 Project Structure

```
├── deployment/
│   ├── app.py                 # CDK app entry point
│   ├── stacks/                # CDK stack definitions
│   ├── lambdas/code/          # Analyzer Lambda functions
│   ├── runtime/               # AgentCore Runtime container
│   ├── s3_files/              # Prompts, schemas, manifests
│   └── badgers-foundation/    # Shared analyzer framework
├── frontend/
│   ├── main.py                # Multi-page Gradio app entry point
│   └── pages/                 # UI modules (chat, wizard, editor, etc.)
└── pyproject.toml
```

---

## 🔍 Technical Deep Dive

### 📦 Lambda Layers

BADGERS uses Lambda layers shared across analyzer functions:

**🏗️ Foundation Layer** (`layer.zip`)
- Built via `deployment/lambdas/build_foundation_layer.sh`
- Contains the analyzer framework (7 Python modules)
- Includes dependencies: boto3, botocore
- Includes core system prompts used by all analyzers

```
layer/python/
├── foundation/
│   ├── analyzer_foundation.py    # 🎯 Main orchestration class
│   ├── bedrock_client.py         # 🔄 Bedrock API with retry/fallback
│   ├── configuration_manager.py  # ⚙️ Config loading/validation
│   ├── image_processor.py        # 🖼️ Image optimization
│   ├── message_chain_builder.py  # 💬 Claude message formatting
│   ├── prompt_loader.py          # 📜 Prompt file loading (local/S3)
│   └── response_processor.py     # 📤 Response extraction
├── config/
│   └── config.py
└── prompts/core_system_prompts/
    └── *.xml
```

**📄 Poppler Layer** (`poppler-layer.zip`)
- PDF rendering library for `pdf_to_images_converter`
- Built via `deployment/lambdas/build_poppler_layer.sh`

### 🔬 How an Analyzer Works

Each analyzer follows the same pattern using `AnalyzerFoundation`:

```python
# Lambda handler (simplified)
def lambda_handler(event, context):
    # 1️⃣ Load config from S3 manifest
    config = load_manifest_from_s3(bucket, "full_text_analyzer")

    # 2️⃣ Initialize foundation with S3-aware prompt loader
    analyzer = AnalyzerFoundation(...)

    # 3️⃣ Run analysis pipeline
    result = analyzer.analyze(image_data)

    # 4️⃣ Save result to S3 and return
    save_result_to_s3(result, session_id)
    return {"result": result}
```

The `analyze()` method orchestrates:
1. 🖼️ **Image processing** - Resize/optimize for Claude's vision API
2. 📜 **Prompt loading** - Combine wrapper + analyzer prompts from S3
3. 💬 **Message building** - Format for Bedrock Converse API
4. 🤖 **Model invocation** - Call Claude with retry/fallback logic
5. ✅ **Response processing** - Extract and validate result

### 📜 Prompting System

Prompts are modular XML files composed at runtime:

```
s3://config-bucket/
├── core_system_prompts/
│   ├── prompt_system_wrapper.xml   # 🎁 Main template with placeholders
│   ├── core_rules/rules.xml        # 📏 Shared rules for all analyzers
│   └── error_handling/*.xml        # ⚠️ Error response templates
├── prompts/{analyzer_name}/
│   ├── {analyzer}_job_role.xml     # 👤 Role definition
│   ├── {analyzer}_context.xml      # 🌍 Domain context
│   ├── {analyzer}_rules.xml        # 📏 Analyzer-specific rules
│   ├── {analyzer}_tasks.xml        # ✅ Task instructions
│   └── {analyzer}_format.xml       # 📋 Output format spec
└── wrappers/
    └── prompt_system_wrapper.xml
```

The `PromptLoader` composes the final system prompt:

```xml
<!-- prompt_system_wrapper.xml -->
<system_prompt>
    {core_rules}           <!-- 📏 Injected from core_rules/rules.xml -->
    {composed_prompt}      <!-- 🧩 Injected from analyzer prompt files -->
    {error_handler_general}
    {error_handler_not_found}
</system_prompt>
```

Placeholders like `[[PIXEL_WIDTH]]` and `[[PIXEL_HEIGHT]]` are replaced with actual image dimensions at runtime.

### ⚙️ Configuration System

Each analyzer has a manifest file in S3:

```json
// s3://config-bucket/manifests/full_text_analyzer.json
{
    "tool": {
        "name": "analyze_full_text_tool",
        "description": "Extracts text content maintaining reading order...",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": { "type": "string" },
                "session_id": { "type": "string" },
                "audit_mode": { "type": "boolean" }
            },
            "required": ["image_path", "session_id"]
        }
    },
    "analyzer": {
        "name": "full_text_analyzer",
        "enhancement_eligible": true,
        "model_selections": {
            "primary": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "fallback_list": [
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "us.amazon.nova-premier-v1:0"
            ]
        },
        "max_retries": 3,
        "prompt_files": [
            "full_text_job_role.xml",
            "full_text_context.xml",
            "full_text_rules.xml",
            "full_text_tasks_extraction.xml",
            "full_text_format.xml"
        ],
        "max_examples": 0,
        "analysis_text": "full text content",
        "expected_output_tokens": 6000,
        "output_extension": "xml"
    }
}
```

Key configuration features:
- 🔄 **Model fallback chain** - Primary model with ordered fallbacks
- 🔁 **Retry logic** - Configurable retry count per analyzer
- 🧩 **Prompt composition** - List of XML files to combine
- 📋 **Tool schema** - MCP-compatible input schema for Gateway
- 🖼️ **Enhancement eligible** - Flag indicating analyzer benefits from image preprocessing (used by `image_enhancer` tool)

Global settings (from environment or defaults):
```python
{
    "max_tokens": 8000,
    "temperature": 0.1,
    "max_image_size": 20971520,  # 20MB
    "max_dimension": 2048,
    "jpeg_quality": 85,
    "throttle_delay": 1.0,
    "aws_region": "us-west-2"
}
```

### 📊 Inference Profiles for Cost Tracking

BADGERS uses Application Inference Profiles to enable cost allocation and usage monitoring. The system maps model IDs to profile ARNs at runtime:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Inference Profile Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. CDK deploys InferenceProfilesStack                                      │
│     └─> Creates ApplicationInferenceProfile for each model                  │
│         • badgers-claude-sonnet-{id}  (Global)                              │
│         • badgers-claude-haiku-{id}   (Global)                              │
│         • badgers-claude-opus-{id}    (Global)                              │
│         • badgers-nova-premier-{id}   (US)                                  │
│                                                                             │
│  2. Runtime receives profile ARNs as environment variables                  │
│     └─> CLAUDE_SONNET_PROFILE_ARN, CLAUDE_HAIKU_PROFILE_ARN, etc.           │
│                                                                             │
│  3. At invocation, bedrock_client.py maps model_id → profile ARN            │
│     └─> "global.anthropic.claude-sonnet-4-5-*" → $CLAUDE_SONNET_PROFILE_ARN │
│                                                                             │
│  4. Bedrock invoked with profile ARN (enables cost tracking)                │
│     └─> Falls back to model ID if no profile configured                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Model ID to environment variable mapping:
| Model Pattern         | Environment Variable        |
| --------------------- | --------------------------- |
| `*claude-sonnet-4-5*` | `CLAUDE_SONNET_PROFILE_ARN` |
| `*claude-haiku-4-5*`  | `CLAUDE_HAIKU_PROFILE_ARN`  |
| `*claude-opus-4-6*`   | `CLAUDE_OPUS_PROFILE_ARN`   |
| `*nova-premier*`      | `NOVA_PREMIER_PROFILE_ARN`  |

### ➕ Adding a New Analyzer

**Option 1: Use the Wizard (Recommended)**

```bash
cd frontend
uv run python main.py
```

The [Analyzer Creation Wizard](frontend/ANALYZER_CREATION_WIZARD.md) is available as a tab in the multi-page Gradio app.

**Option 2: Manual Creation**

1. 📜 Create prompt files in `deployment/s3_files/prompts/{analyzer_name}/`
2. 📋 Create manifest in `deployment/s3_files/manifests/{analyzer_name}.json`
3. 📐 Create schema in `deployment/s3_files/schemas/{analyzer_name}.json`
4. ⚡ Create Lambda code in `deployment/lambdas/code/{analyzer_name}/lambda_handler.py`
5. 📝 Register in `deployment/stacks/lambda_stack.py`
6. 🚀 Redeploy: `cdk deploy badgers-lambda badgers-gateway`

---

## Notices

Customers are responsible for making their own independent assessment of the information in this Guidance. This Guidance: (a) is for informational purposes only, (b) represents AWS current product offerings and practices, which are subject to change without notice, and (c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided "as is" without warranties, representations, or conditions of any kind, whether express or implied. AWS responsibilities and liabilities to its customers are controlled by AWS agreements, and this Guidance is not part of, nor does it modify, any agreement between AWS and its customers.

---

## Authors
- Randall Potter

---

## 📖 Further Reading

### 🤖 Amazon Bedrock & Foundation Models
- [Amazon Bedrock Developer Experience](https://aws.amazon.com/bedrock/developer-experience/) - Foundation model choice and customization
- [Anthropic's Claude in Amazon Bedrock](https://aws.amazon.com/bedrock/anthropic/) - Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 hybrid reasoning models
- [Claude Sonnet 4.5 in Amazon Bedrock](https://aws.amazon.com/blogs/aws/introducing-claude-sonnet-4-5-in-amazon-bedrock-anthropics-most-intelligent-model-best-for-coding-and-complex-agents/) - Most intelligent model for coding and complex agents
- [Claude Opus 4.6 in Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/claude-opus-4-5-now-in-amazon-bedrock/) - Tool search, extended thinking, and agent capabilities
- [Amazon Nova Foundation Models](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-frontier-intelligence-and-industry-leading-price-performance/) - Nova Micro, Lite, Pro, Premier - frontier intelligence
- [Using Amazon Nova in AI Agents](https://docs.aws.amazon.com/nova/latest/userguide/agents-use-nova.html) - Nova as foundation model for agents

### 🚀 Amazon Bedrock AgentCore
- [Amazon Bedrock AgentCore Overview](https://aws.amazon.com/bedrock/agentcore/) - Build, deploy, and operate agents at scale
- [AgentCore Gateway Guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-building.html) - Set up unified tool connectivity
- [AgentCore Gateway Blog](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock-agentcore-gateway-transforming-enterprise-ai-agent-tool-development/) - Transforming enterprise AI agent tool development
- [AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html) - Secure serverless hosting for AI agents

### ⚡ AWS Lambda
- [Lambda Layers Overview](https://docs.aws.amazon.com/lambda/latest/dg/chapter-layers.html) - Managing dependencies with layers
- [Python Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html) - Working with layers for Python functions
- [Adding Layers to Functions](https://docs.aws.amazon.com/lambda/latest/dg/adding-layers.html) - Layer configuration and management

### 🔐 Amazon Cognito
- [OAuth 2.0 Grants](https://docs.aws.amazon.com/cognito/latest/developerguide/federation-endpoints-oauth-grants.html) - Authorization code, implicit, and client credentials
- [M2M Authorization](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-define-resource-servers.html) - Scopes, resource servers, and machine-to-machine auth
- [M2M Security Best Practices](https://aws.amazon.com/blogs/security/how-to-monitor-optimize-and-secure-amazon-cognito-machine-to-machine-authorization/) - Monitor, optimize, and secure M2M authorization

### 📈 Observability
- [CloudWatch + X-Ray Integration](https://docs.aws.amazon.com/xray/latest/devguide/xray-services-cloudwatch.html) - Enhanced application monitoring
- [Cross-Account Tracing](https://docs.aws.amazon.com/xray/latest/devguide/xray-console-crossaccount.html) - Distributed tracing across accounts
- [AWS Observability Best Practices](https://aws.amazon.com/blogs/publicsector/building-resilient-public-services-with-aws-observability-best-practices/) - Logs, metrics, and traces

### 📦 Amazon S3
- [S3 as Data Lake Storage](https://docs.aws.amazon.com/whitepapers/latest/building-data-lakes/amazon-s3-data-lake-storage-platform.html) - Central storage platform best practices
- [S3 Performance Optimization](https://aws.amazon.com/s3/whitepaper-best-practices-s3-performance/) - Design patterns for optimal performance

### 💻 Amazon Kiro IDE
- [Amazon Kiro Overview](https://aws.amazon.com/kiro/) - Agentic IDE for spec-driven development
- [Kiro with AWS Builder ID](https://docs.aws.amazon.com/signin/latest/userguide/builder_id-apps.html) - Sign in and get started with Kiro
- [Nova Act IDE Extension](https://aws.amazon.com/blogs/aws/accelerate-ai-agent-development-with-the-nova-act-ide-extension/) - Accelerate AI agent development in Kiro
- [Production-Ready AI Agents at Scale](https://aws.amazon.com/blogs/machine-learning/enabling-customers-to-deliver-production-ready-ai-agents-at-scale/) - Kiro as part of the agent development ecosystem
