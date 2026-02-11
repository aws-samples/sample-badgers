<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../README.md) | [Vision LLM Theory](../VISION_LLM_THEORY_README.md) | [Frontend](../frontend/FRONTEND_README.md) | 🔵 **Deployment** | [CDK Stacks](stacks/STACKS_README.md) | [Runtime](runtime/RUNTIME_README.md) | [S3 Files](s3_files/S3_FILES_README.md) | [Lambda Analyzers](lambdas/LAMBDA_ANALYZERS.md) | [Prompting System](s3_files/prompts/PROMPTING_SYSTEM_README.md) | [Analyzer Wizard](../frontend/ANALYZER_CREATION_WIZARD.md)</sub>

---

# 🚀 BADGERS Deployment Guide

Step-by-step AWS CDK deployment for BADGERS. For architecture overview and technical details, see the [main README](../README.md).

## ☁️ AWS Services

| Service                                                               | Purpose                                   |
| --------------------------------------------------------------------- | ----------------------------------------- |
| [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) | Runtime + Gateway for agent orchestration |
| [Amazon Bedrock](https://aws.amazon.com/bedrock/)                     | Claude foundation model access            |
| [AWS Lambda](https://aws.amazon.com/lambda/)                          | Serverless analyzer functions             |
| [Amazon S3](https://aws.amazon.com/s3/)                               | Configuration and output storage          |
| [Amazon Cognito](https://aws.amazon.com/cognito/)                     | OAuth 2.0 authentication                  |
| [Amazon ECR](https://aws.amazon.com/ecr/)                             | Container image registry                  |
| [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)        | Credential storage                        |
| [AWS SSM Parameter Store](https://aws.amazon.com/systems-manager/)    | Configuration parameters                  |
| [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/)               | Logging and observability                 |
| [AWS X-Ray](https://aws.amazon.com/xray/)                             | Distributed tracing                       |

## ✅ Prerequisites

Verify your environment:

```bash
aws --version        # AWS CLI
cdk --version        # AWS CDK v2
docker info          # Docker running
python --version     # Python 3.12+
uv --version         # uv package manager
```

> [!IMPORTANT]
> Docker must be running before deployment. Lambda layers and the Runtime container require Docker to build.

## ⚡ Quick Start

Deploy everything:

```bash
./deploy_from_scratch.sh
```

## 📦 CDK Stacks

10 stacks deployed in dependency order (plus 1 optional):

| Stack                       | Purpose                                           |
| --------------------------- | ------------------------------------------------- |
| `badgers-s3`                | Config bucket (manifests/prompts) + Output bucket |
| `badgers-cognito`           | User pool, app client, OAuth 2.0 credentials      |
| `badgers-iam`               | Lambda execution role with Bedrock/S3 permissions |
| `badgers-lambda`            | Base analyzer functions + foundation layer        |
| `badgers-gateway`           | AgentCore MCP Gateway with Lambda targets         |
| `badgers-ecr`               | Container registry for agent image                |
| `badgers-memory`            | AgentCore Memory for session persistence          |
| `badgers-inference`         | Application Inference Profiles for cost tracking  |
| `badgers-runtime-websocket` | AgentCore Runtime (Strands agent with WebSocket)  |
| `badgers-custom-analyzers`  | Custom analyzers (optional, wizard-created)       |

## 🔧 Manual Deployment

### 1️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2️⃣ Build Lambda Layers

All layer build scripts must be run from the `deployment/lambdas` directory.

```bash
cd lambdas
./build_foundation_layer.sh      # Core framework, boto3, pillow
./build_poppler_layer.sh         # PDF rendering (pdftoppm, pdfinfo)
./build_enhancement_layer.sh     # OpenCV, numpy for image enhancement
./build_pdf_processing_layer.sh  # pikepdf, pymupdf for PDF/A tagging
cd ..
```

#### Layer Build Scripts

| Script                          | Output                     | Purpose                                        | Used By                                  |
| ------------------------------- | -------------------------- | ---------------------------------------------- | ---------------------------------------- |
| `build_foundation_layer.sh`     | `layer.zip`                | Core analyzer framework, AWS SDK, Pillow       | All Lambda analyzers                     |
| `build_poppler_layer.sh`        | `poppler-layer.zip`        | Poppler binaries for PDF→image conversion      | `pdf_to_images_converter`                |
| `build_enhancement_layer.sh`    | `enhancement-layer.zip`    | OpenCV headless, numpy for image preprocessing | `image_enhancer`                         |
| `build_pdf_processing_layer.sh` | `pdf-processing-layer.zip` | pikepdf, pymupdf for PDF manipulation          | `remediation_analyzer`                   |
| `build_container_lambdas.sh`    | ECR images                 | Container images for complex analyzers         | `image_enhancer`, `remediation_analyzer` |

#### Container Lambda Build

Container-based Lambdas (for functions exceeding layer size limits) are built separately:

```bash
cd lambdas
./build_container_lambdas.sh <deployment_id>
cd ..
```

This builds and pushes Docker images to ECR for `image_enhancer` and `remediation_analyzer`.

#### Automated Build

`deploy_from_scratch.sh` calls all build scripts automatically in the correct order. Manual builds are only needed for:
- Partial redeployments
- Layer updates without full redeploy
- Troubleshooting build issues

### 3️⃣ Bootstrap CDK

```bash
cdk bootstrap
```

> [!TIP]
> New to CDK? See the [AWS CDK Developer Guide](https://docs.aws.amazon.com/cdk/v2/guide/home.html) for installation and concepts.
>
> This project uses alpha CDK modules:
> - [aws-bedrock-agentcore-alpha](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-bedrock-agentcore-alpha-readme.html)
> - [aws-bedrock-alpha](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-bedrock-alpha-readme.html)

### 4️⃣ Deploy S3 + Upload Config

```bash
cdk deploy badgers-s3 --require-approval never

# Sync configuration files
./sync_s3_files.sh
```

### 5️⃣ Deploy Auth & IAM

```bash
cdk deploy badgers-cognito --require-approval never
cdk deploy badgers-iam --require-approval never
```

### 6️⃣ Deploy Lambda Functions

```bash
cdk deploy badgers-lambda --require-approval never
```

### 7️⃣ Deploy Gateway

```bash
cdk deploy badgers-gateway --require-approval never
```

### 8️⃣ Deploy ECR + Build Container

```bash
cdk deploy badgers-ecr --require-approval never

cd runtime
./build_and_push_websocket.sh
cd ..
```

### 9️⃣ Deploy Memory + Runtime

```bash
cdk deploy badgers-memory --require-approval never
cdk deploy badgers-runtime-websocket --require-approval never
```

## 📤 Stack Outputs

Key outputs after deployment:

| Output                                  | Description                      |
| --------------------------------------- | -------------------------------- |
| `GatewayUrl`                            | MCP endpoint for tool invocation |
| `RuntimeEndpoint`                       | Agent HTTP endpoint              |
| `UserPoolId` / `UserPoolClientId`       | Cognito authentication           |
| `ConfigBucketName` / `OutputBucketName` | S3 buckets                       |
| `MemoryId`                              | AgentCore Memory ID              |

## 📁 Directory Structure

```
deployment/
├── app.py                    # 🎯 CDK app entry point
├── deploy_from_scratch.sh    # 🚀 Full deployment orchestrator
├── stacks/                   # 📦 CDK stack definitions
├── lambdas/
│   ├── build_foundation_layer.sh    # Core framework layer
│   ├── build_poppler_layer.sh       # PDF rendering layer
│   ├── build_enhancement_layer.sh   # Image enhancement layer
│   ├── build_pdf_processing_layer.sh # PDF manipulation layer
│   ├── build_container_lambdas.sh   # Container image builder
│   ├── deploy_foundation_layer.sh   # Manual layer deployment
│   ├── deploy_poppler_layer.sh      # Manual layer deployment
│   ├── containers/           # 🐳 Container Lambda Dockerfiles
│   └── code/                 # ⚡ 28 analyzer functions + utilities
├── runtime/                  # 🐳 AgentCore container
│   ├── Dockerfile.websocket
│   └── agent/main-websocket.py
├── s3_files/                 # ☁️ S3 configuration
│   ├── manifests/
│   ├── prompts/
│   ├── schemas/
│   └── wrappers/
└── badgers-foundation/       # 🏗️ Shared analyzer framework
```

## 📋 Analyzer Manifest Configuration

Each analyzer has a manifest file in `s3_files/manifests/` that configures its behavior. The `model_selections` section supports extended thinking (Claude's chain-of-thought reasoning):

```json
{
    "analyzer": {
        "name": "page_analyzer",
        "model_selections": {
            "primary": {
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "extended_thinking": true,
                "budget_tokens": 6400
            },
            "fallback_list": [
                {
                    "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "extended_thinking": true,
                    "budget_tokens": 4000
                },
                {
                    "model_id": "amazon.nova-pro-v1:0",
                    "extended_thinking": false
                }
            ]
        }
    }
}
```

| Field                    | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `model_id`               | Bedrock model identifier                                                  |
| `extended_thinking`      | Enable Claude's reasoning traces (Claude models only)                     |
| `budget_tokens`          | Max tokens for thinking content (required when extended_thinking is true) |
| `expected_output_tokens` | Estimated output tokens for cost calculation (in `analyzer` section)      |
| `audit_mode`             | Boolean in `inputSchema` - enables confidence scoring and review flags    |

> [!NOTE]
> Extended thinking is only supported on Claude models. When enabled, thinking content is saved to S3 alongside results: `{session_id}/{analyzer_name}/{image}_thinking_{timestamp}.txt`

Simple format (no extended thinking) is still supported for backward compatibility:
```json
"model_selections": {
    "primary": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "fallback_list": ["amazon.nova-pro-v1:0"]
}
```

## 📊 Inference Profiles for Cost Tracking

BADGERS uses Application Inference Profiles to enable cost allocation and usage monitoring per model. The `inference_profiles_stack.py` creates trackable profiles that wrap cross-region system-defined profiles.

### How It Works

1. **CDK creates profiles** for each model (Claude Sonnet, Haiku, Opus, Nova Premier)
2. **Profile ARNs are passed** to Runtime containers as environment variables
3. **At invocation time**, `bedrock_client.py` maps model IDs to profile ARNs
4. **Bedrock is invoked** using the profile ARN instead of raw model ID

### Environment Variable Mapping

| Model ID Pattern                       | Environment Variable         |
| -------------------------------------- | ---------------------------- |
| `global.anthropic.claude-sonnet-4-5-*` | `CLAUDE_SONNET_PROFILE_ARN`  |
| `us.anthropic.claude-haiku-4-5-*`      | `CLAUDE_HAIKU_PROFILE_ARN`   |
| `*claude-opus-4-6*`                    | `CLAUDE_OPUS_46_PROFILE_ARN` |
| `us.amazon.nova-premier-v1:0`          | `NOVA_PREMIER_PROFILE_ARN`   |

### Profile Naming

Profiles are named: `badgers-{model}-{deployment_id}`

Example: `badgers-claude-sonnet-abc12345`

> [!NOTE]
> If no inference profile is configured for a model ID, the system falls back to using the model ID directly. This allows local development without deployed profiles.

## 🎨 Custom Analyzers

BADGERS ships with 5 base analyzers. Organizations can create additional analyzers using the wizard UI without modifying the core deployment.

### Architecture

```
s3://{config-bucket}/
├── manifests/              # Base analyzers (deployed with badgers-lambda)
├── schemas/
├── prompts/
└── custom-analyzers/       # Wizard-created analyzers
    ├── analyzer_registry.json
    ├── manifests/
    ├── schemas/
    └── prompts/
```

### Workflow

1. **Create analyzer** via wizard UI (`frontend/analyzer_wizard.py`)
   - Wizard uploads files to S3 under `custom-analyzers/` prefix

2. **Sync to local** for CDK deployment:
   ```bash
   cd deployment
   ./sync_custom_analyzers.sh
   ```

3. **Deploy custom stack**:
   ```bash
   cdk deploy badgers-custom-analyzers
   ```

The custom stack:
- Creates Lambda functions for each custom analyzer
- Registers them as Gateway targets via Custom Resource
- Uses the same foundation layer and IAM role as base analyzers

### Editing Analyzers

| Type   | Editor Behavior                                     |
| ------ | --------------------------------------------------- |
| Base   | Read-only by default, toggle to enable with warning |
| Custom | Always editable                                     |

See [Analyzer Wizard](../frontend/ANALYZER_CREATION_WIZARD.md) for detailed usage.

## 🔄 Redeploying

Update specific components:

```bash
# Lambda code changes
cdk deploy badgers-lambda --require-approval never

# Gateway target changes
cdk deploy badgers-gateway --require-approval never

# Agent container changes
cd runtime && ./build_and_push_websocket.sh && cd ..
cdk deploy badgers-runtime-websocket --require-approval never

# Prompt/manifest changes only
./sync_s3_files.sh
```

## 🔐 Authentication

Gateway uses Cognito OAuth 2.0 client credentials:
- Credentials stored in Secrets Manager
- Runtime fetches tokens automatically
- Resource server scope: `agentcore-gateway/invoke`

## 📊 Observability

Gateway logs are automatically configured:
- 📝 **Application**: `/aws/vendedlogs/bedrock-agentcore/gateway/APPLICATION_LOGS/`
- 📈 **Usage**: `/aws/vendedlogs/bedrock-agentcore/gateway/USAGE_LOGS/`
- 🔍 **Traces**: X-Ray via CloudWatch Transaction Search

> [!WARNING]
> **Manual step required**: After deployment, enable Runtime observability in the AWS Console:
> 1. Navigate to Amazon Bedrock → AgentCore → Runtimes
> 2. Select your runtime and click "Edit"
> 3. Enable **Application logs** and **Usage logs**
> 4. Enable **Tracing** for X-Ray integration
> 5. Runtime logs will appear at `/aws/bedrock-agentcore/runtimes/`

## 🗑️ Cleanup

> [!CAUTION]
> This permanently deletes all resources including S3 buckets and their contents.

```bash
./destroy.sh
```

## 🐛 Troubleshooting

### Lambda Layer Build Fails

```bash
cd lambdas

# Foundation layer
rm -rf layer/ layer.zip
./build_foundation_layer.sh

# Poppler layer
rm -rf poppler_build/ poppler-layer.zip
./build_poppler_layer.sh

# Enhancement layer
rm -rf enhancement_build/ enhancement-layer.zip
./build_enhancement_layer.sh

# PDF processing layer
rm -rf pdf_processing_build/ pdf-processing-layer.zip
./build_pdf_processing_layer.sh
```

### Container Image Not Found

```bash
# Verify image exists
aws ecr describe-images --repository-name pdf-analysis-agent-<deployment_id>

# Rebuild
cd runtime && ./build_and_push_websocket.sh
```

### Gateway Auth Errors

```bash
# Check credentials
aws secretsmanager get-secret-value \
    --secret-id pdf-extractor/cognito-config-<deployment_id>
```

### Runtime Startup Issues

```bash
# Tail logs
aws logs tail /aws/bedrock-agentcore/runtimes/<runtime_id> --follow
```

## 🏷️ Deployment ID

Use a consistent ID for redeployments:

```bash
cdk deploy -c deployment_id=abc12345 --all
```

## 🏷️ Resource Tagging

All resources are tagged using a centralized configuration in `app.py`. Customize the `deployment_tags` dict before deployment:

```python
deployment_tags = {
    "application_name": "badgers",
    "application_description": "BADGERS (Broad Agentic Document Generative Extraction & Recognition System)",
    "environment": "dev",
    "owner": "your-team",
    "cost_center": "your-cost-center",
    "project_code": "your-project-code",
    "cdk_stack_prefix": STACK_PREFIX,
    "team": "your-team",
    "team_contact_email": "team@company.com",
}
```

These tags are applied to all resources across all stacks. Additionally, each resource gets:
- `resource_name` - Identifier for the specific resource
- `resource_description` - Description of the resource's purpose

Tagged resources include:
- AgentCore Gateway, Runtime, Memory
- ECR repositories
- S3 buckets and KMS keys
- Lambda functions and layers
- Cognito User Pool, Identity Pool, Secrets
- IAM roles
