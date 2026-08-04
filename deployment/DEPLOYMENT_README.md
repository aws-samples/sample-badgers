<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../README.md) | [Vision LLM Theory](../VISION_LLM_THEORY_README.md) | [UI](../ui/UI_README.md) | 🔵 **Deployment** | [CDK Stacks](stacks/STACKS_README.md) | [Runtime](runtime/RUNTIME_README.md) | [S3 Files](s3_files/S3_FILES_README.md) | [Lambda Specialists](lambdas/LAMBDA_SPECIALISTS.md) | [Prompting System](s3_files/prompts/PROMPTING_SYSTEM_README.md)</sub>

---

# 🚀 BADGERS Deployment Guide

Step-by-step AWS CDK deployment for BADGERS. For architecture overview and technical details, see the [main README](../README.md).

## ☁️ AWS Services

| Service                                                               | Purpose                                             |
| --------------------------------------------------------------------- | --------------------------------------------------- |
| [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) | Runtime + Gateway for agent orchestration           |
| [Amazon Bedrock](https://aws.amazon.com/bedrock/)                     | Claude foundation model access                      |
| [AWS Lambda](https://aws.amazon.com/lambda/)                          | Serverless specialist functions                     |
| [Amazon S3](https://aws.amazon.com/s3/)                               | Configuration and output storage                    |
| [Amazon DynamoDB](https://aws.amazon.com/dynamodb/)                   | Job and subtask state tracking                      |
| [Amazon Cognito](https://aws.amazon.com/cognito/)                     | OIDC/PKCE for the UI, OAuth 2.0 M2M for the Gateway |
| [Amazon ECS](https://aws.amazon.com/ecs/)                             | UI hosting on an Express Gateway service            |
| [Amazon VPC](https://aws.amazon.com/vpc/)                             | Private networking and VPC endpoints for the UI     |
| [Amazon ECR](https://aws.amazon.com/ecr/)                             | Container image registry                            |
| [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)        | Credential storage                                  |
| [AWS SSM Parameter Store](https://aws.amazon.com/systems-manager/)    | Configuration parameters                            |
| [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/)               | Logging and observability                           |
| [AWS X-Ray](https://aws.amazon.com/xray/)                             | Distributed tracing                                 |

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

> [!WARNING]
> **Academic/Research Deployments:** If your users process documents with sensitive, inflammatory, or offensive content (common in academic research), you must configure the operating environment before use. Edit `s3_files/agent_config/agent_operating_environment_config.json` and sync to S3. Without this, the model may refuse to extract or omit sensitive content. See [S3 Files → Operating Environment Configuration](s3_files/S3_FILES_README.md#%EF%B8%8F-operating-environment-configuration) for details. For production deployments, we recommend moving this value to AWS Secrets Manager for added security.

## 🏷️ Stack and Resource Naming

`DEPLOYMENT_ID` is a label you choose. `deploy.sh` generates a short random
`STACK_SUFFIX` once and persists both in `.deploy-state/{DEPLOYMENT_ID}.json`.

|                | Pattern                        | Example                            |
| -------------- | ------------------------------ | ---------------------------------- |
| Stack names    | `BADGERS-{Name}-{suffix}`      | `BADGERS-S3-a1b`                   |
| Resource names | `badgers-{kind}-{id}-{suffix}` | `badgers-config-dev-a1b`           |
| SSM prefix     | `/badgers-{id}-{suffix}`       | `/badgers-dev-a1b/jobs-table-name` |

Both are unique per deployment, so several deployments can coexist in one account and
region. Anything running `cdk` directly needs both values:

```bash
DEPLOYMENT_ID=dev STACK_SUFFIX=a1b uv run cdk deploy BADGERS-S3-a1b
# or as context
uv run cdk deploy -c deployment_id=dev -c stack_suffix=a1b BADGERS-S3-a1b
```

`app.py` fails fast with usage if either is missing, rather than inventing a suffix and
deploying a second copy of everything by accident.

> AgentCore runtime and memory names must match `[a-zA-Z][a-zA-Z0-9_]{0,47}` and cannot
> contain hyphens, so those two stacks normalise the composite id to underscores —
> `badgers_runtime_ws_dev_a1b`, `badgers_memory_dev_a1b`.

## ⚡ Quick Start

Deploy everything from the repository root:

```bash
DEPLOYMENT_ID=dev ./deploy.sh        # interactive menu, pick 9 for a full deploy
DEPLOYMENT_ID=dev ./deploy.sh 9      # or run it non-interactively
```

`deploy.sh` is resumable: it records each completed step in
`.deploy-state/{DEPLOYMENT_ID}.json`, so a re-run after a failure continues rather than
starting over. See [DEPLOYMENT_SCRIPTS.md](DEPLOYMENT_SCRIPTS.md) for every script.

## 📦 CDK Stacks

13 stacks deployed in dependency order, plus 1 optional. See
[CDK Stacks](stacks/STACKS_README.md) for details and the dependency graph.

| Stack                                | Purpose                                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| `BADGERS-S3-{suffix}`                | Config bucket (manifests/prompts) + source + output buckets    |
| `BADGERS-Cognito-{suffix}`           | User pool with UI (OIDC/PKCE) and Gateway (M2M) clients        |
| `BADGERS-DynamoDB-{suffix}`          | Jobs table for doc/job/subtask tracking                        |
| `BADGERS-IAM-{suffix}`               | Lambda execution role with Bedrock/S3/DynamoDB permissions     |
| `BADGERS-ECR-{suffix}`               | Container registry for agent and container Lambda images       |
| `BADGERS-InferenceProfiles-{suffix}` | Application Inference Profiles for cost tracking               |
| `BADGERS-Lambda-{suffix}`            | Base specialist functions + foundation layer                   |
| `BADGERS-XRay-{suffix}`              | X-Ray Transaction Search (account-level singleton)             |
| `BADGERS-Gateway-{suffix}`           | AgentCore MCP Gateway with Lambda targets                      |
| `BADGERS-Memory-{suffix}`            | AgentCore Memory for session persistence                       |
| `BADGERS-RuntimeWebSocket-{suffix}`  | AgentCore Runtime (Strands agent with WebSocket)               |
| `BADGERS-Vpc-{suffix}`               | VPC with public/private subnets, NAT, flow logs, VPC endpoints |
| `BADGERS-ECS-{suffix}`               | Unified UI on an ECS Express Gateway service                   |
| `BADGERS-CustomSpecialists-{suffix}` | Custom specialists (optional, wizard-created)                  |

### UI Stacks

The UI runs on `BADGERS-Vpc-{suffix}` + `BADGERS-ECS-{suffix}`. The ECS Express Gateway service provisions
and manages its own load balancer and HTTPS endpoint, so **no hosted zone, domain, or ACM
certificate is required** — there is nothing to configure before deploying it.

Cognito values must be baked into the Vite bundle at build time, so ordering matters.
`deploy.sh` steps 7 and 8 handle it:

```bash
DEPLOYMENT_ID=dev ./deploy.sh 7   # generate ui/.env, build the bundle, build and push the image
DEPLOYMENT_ID=dev ./deploy.sh 8   # deploy the ECS stack and wait for the rollout
```

Step 7 runs `scripts/generate_ui_env.sh` first, because a bundle built before Cognito
exists has no authority or client id compiled into it and falls back to the server's
local-dev bypass.

`BADGERS-ECS-{suffix}` reads all container configuration from SSM Parameter Store under
`/badgers-{id}-{suffix}/`, so there is no `.env` to ship into the image beyond the
build-time `VITE_*` values.

> **Note:** `update_frontend_env.sh` writes `ui/config/.env` for local development only.
> It does not write the Cognito values, which are build-time inputs to the Vite bundle —
> that is `scripts/generate_ui_env.sh`. The deployed service reads everything from SSM.

## 🔧 Manual Deployment

Prefer `./deploy.sh` — it does all of this with resumable state. The steps below are for
when you need to drive a single stack yourself.

Every command assumes the deployment identity is exported, and `{suffix}` below stands for
your actual suffix from `.deploy-state/{DEPLOYMENT_ID}.json`:

```bash
export DEPLOYMENT_ID=dev
export STACK_SUFFIX=a1b
```

### 1️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2️⃣ Build Lambda Layers

All layer build scripts must be run from the `deployment/lambdas` directory.

```bash
cd lambdas
./build_foundation_layer.sh      # Core framework, boto3, pillow
./build_poppler_qdf_layer.sh         # PDF rendering (pdftoppm, pdfinfo)
./build_pdf_processing_layer.sh  # pikepdf, pymupdf for PDF/A tagging
cd ..
```

#### Layer Build Scripts

| Script                          | Output                     | Purpose                                    | Used By                                    |
| ------------------------------- | -------------------------- | ------------------------------------------ | ------------------------------------------ |
| `build_foundation_layer.sh`     | `layer.zip`                | Core specialist framework, AWS SDK, Pillow | All Lambda specialists                     |
| `build_poppler_qdf_layer.sh`    | `poppler-qpdf-layer.zip`   | Poppler binaries for PDF→image conversion  | `pdf_to_images_converter`                  |
| `build_pdf_processing_layer.sh` | `pdf-processing-layer.zip` | pikepdf, pymupdf for PDF manipulation      | Non-container specialists needing PDF libs |
| `build_container_lambdas.sh`    | ECR images                 | Container images for complex specialists   | `image_enhancer`, `remediation_specialist` |

> **Note:** `build_enhancement_layer.sh` is retained on disk but no longer deployed. Image enhancement runs in the container-based `image_enhancer` Lambda which bundles its own dependencies.

#### Container Lambda Build

Container-based Lambdas (for functions exceeding layer size limits) are built separately:

```bash
cd lambdas
./build_container_lambdas.sh <deployment_id>
cd ..
```

This builds and pushes Docker images to ECR for `image_enhancer` and `remediation_specialist`.

#### Automated Build

`./deploy.sh 1` runs all the layer builds in the correct order. Manual builds are only needed for:
- Partial redeployments
- Layer updates without full redeploy
- Troubleshooting build issues

### 3️⃣ Bootstrap CDK

```bash
uv run cdk bootstrap
```

> [!TIP]
> New to CDK? See the [AWS CDK Developer Guide](https://docs.aws.amazon.com/cdk/v2/guide/home.html) for installation and concepts.
>
> This project uses alpha CDK modules:
> - [aws-bedrock-agentcore-alpha](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-bedrock-agentcore-alpha-readme.html)
> - [aws-bedrock-alpha](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-bedrock-alpha-readme.html)

### 4️⃣ Deploy S3 + Upload Config

```bash
uv run cdk deploy BADGERS-S3-{suffix} --require-approval never

# Sync configuration files
./sync_s3_files.sh
```

### 5️⃣ Deploy Auth & IAM

```bash
uv run cdk deploy BADGERS-Cognito-{suffix} --require-approval never
uv run cdk deploy BADGERS-IAM-{suffix} --require-approval never
```

### 6️⃣ Deploy Lambda Functions

```bash
uv run cdk deploy BADGERS-Lambda-{suffix} --require-approval never
```

### 7️⃣ Deploy Gateway

```bash
uv run cdk deploy BADGERS-Gateway-{suffix} --require-approval never
```

### 8️⃣ Deploy ECR + Build Container

```bash
uv run cdk deploy BADGERS-ECR-{suffix} --require-approval never

cd runtime
./build_and_push_websocket.sh
cd ..
```

### 9️⃣ Deploy Memory + Runtime

```bash
uv run cdk deploy BADGERS-Memory-{suffix} --require-approval never
uv run cdk deploy BADGERS-RuntimeWebSocket-{suffix} --require-approval never
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
├── scripts/common.sh             # 🔗 Shared logging, state, stack-name and CDK helpers
├── deploy_specialist.sh      # 🔬 Single specialist deployment
├── deploy_custom_specialists.sh # 🎨 Wizard-created specialist deployment
├── scripts/
│   └── generate_ui_env.sh    # 🔐 Writes ui/.env from BADGERS-Cognito-{suffix} outputs
├── stacks/                   # 📦 CDK stack definitions
├── lambdas/
│   ├── build_foundation_layer.sh    # Core framework layer
│   ├── build_poppler_qdf_layer.sh       # PDF rendering layer
│   ├── build_enhancement_layer.sh   # Image enhancement layer (UNUSED - retained for reference)
│   ├── build_pdf_processing_layer.sh # PDF manipulation layer
│   ├── build_container_lambdas.sh   # Container image builder
│   ├── deploy_foundation_layer.sh   # Manual layer deployment
│   ├── deploy_poppler_layer.sh      # Manual layer deployment
│   ├── containers/           # 🐳 Container Lambda Dockerfiles
│   └── code/                 # ⚡ 24 specialist/utility functions (+2 containers)
├── runtime/                  # 🐳 AgentCore container
│   ├── Dockerfile.websocket
│   ├── build_and_push_websocket.sh
│   └── agent/main-websocket.py
├── s3_files/                 # ☁️ S3 configuration
│   ├── manifests/
│   ├── prompts/
│   ├── schemas/
│   └── wrappers/
├── badgers-foundation/       # 🏗️ Shared specialist framework (used by non-container specialists and image_enhancer)
```

## 📋 Specialist Manifest Configuration

Each specialist has a manifest file in `s3_files/manifests/` that configures its behavior. The `model_selections` section supports extended thinking (Claude's chain-of-thought reasoning):

```json
{
    "specialist": {
        "name": "page_specialist",
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
| `expected_output_tokens` | Estimated output tokens for cost calculation (in `specialist` section)    |
| `audit_mode`             | Boolean in `inputSchema` - enables confidence scoring and review flags    |

> [!NOTE]
> Extended thinking is only supported on Claude models. When enabled, thinking content is saved to S3 alongside results: `{session_id}/{specialist_name}/{image}_thinking_{timestamp}.txt`

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

| Model ID Pattern                   | Environment Variable         |
| ---------------------------------- | ---------------------------- |
| `us.anthropic.claude-sonnet-4-5-*` | `CLAUDE_SONNET_PROFILE_ARN`  |
| `us.anthropic.claude-haiku-4-5-*`  | `CLAUDE_HAIKU_PROFILE_ARN`   |
| `*claude-opus-4-6*`                | `CLAUDE_OPUS_46_PROFILE_ARN` |
| `us.amazon.nova-premier-v1:0`      | `NOVA_PREMIER_PROFILE_ARN`   |

### Profile Naming

Profiles are named: `badgers-{model}-{deployment_id}`

Example: `badgers-claude-sonnet-abc12345`

> [!NOTE]
> If no inference profile is configured for a model ID, the system falls back to using the model ID directly. This allows local development without deployed profiles.

## 🎨 Custom Specialists

BADGERS ships with 5 base specialists. Organizations can create additional specialists using the wizard UI without modifying the core deployment.

### Architecture

```
s3://{config-bucket}/
├── manifests/              # Base specialists (deployed with BADGERS-Lambda-{suffix})
├── schemas/
├── prompts/
└── custom-specialists/       # Wizard-created specialists
    ├── specialist_registry.json
    ├── manifests/
    ├── schemas/
    └── prompts/
```

### Workflow

1. **Create specialist** via the 🧙 Create Specialist tab in the [UI](../ui/UI_README.md)
   - Wizard uploads files to S3 under `custom-specialists/` prefix

2. **Sync to local** for CDK deployment:
   ```bash
   cd deployment
   ./sync_custom_specialists.sh
   ```

3. **Deploy custom stack**:
   ```bash
   uv run cdk deploy BADGERS-CustomSpecialists-{suffix}
   ```

The custom stack:
- Creates Lambda functions for each custom specialist
- Registers them as Gateway targets via Custom Resource
- Uses the same foundation layer and IAM role as base specialists

### Editing Specialists

| Type   | Editor Behavior                                     |
| ------ | --------------------------------------------------- |
| Base   | Read-only by default, toggle to enable with warning |
| Custom | Always editable                                     |

See the 🧙 Create Specialist tab in the [UI](../ui/UI_README.md) for detailed usage.

## 🔄 Redeploying

Update specific components:

```bash
# Lambda code changes
uv run cdk deploy BADGERS-Lambda-{suffix} --require-approval never

# Gateway target changes
uv run cdk deploy BADGERS-Gateway-{suffix} --require-approval never

# Agent container changes
cd runtime && ./build_and_push_websocket.sh && cd ..
uv run cdk deploy BADGERS-RuntimeWebSocket-{suffix} --require-approval never

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
> This permanently deletes all resources including S3 buckets and their contents, and all
> job records in the DynamoDB jobs table.

From the repository root:

```bash
DEPLOYMENT_ID=dev ./destroy.sh
```

It requires you to type the `DEPLOYMENT_ID` to confirm, then handles the ordering that
makes teardown succeed:

1. Empties the config, source and output buckets, including all versions and delete markers
2. Deletes the ECS Express service and the AgentCore runtime, then polls until their ENIs
   release — CloudFormation cannot delete a VPC while an ENI is still attached, and both
   release theirs asynchronously well after they report gone
3. Destroys all 14 stacks in reverse dependency order (`BADGERS-ECS-{suffix}` before
   `BADGERS-Vpc-{suffix}`, because it imports three VPC exports)
4. Schedules the KMS key for deletion. The default 7-day window is deliberate: the alias
   `alias/badgers-s3-key-{id}-{suffix}` stays reserved until the key is gone, so a long
   window blocks redeploying under the same `DEPLOYMENT_ID`. Override with `KMS_WAIT_DAYS`.
5. Verifies every stack is gone, and if the VPC stack is `DELETE_FAILED`, retries with
   `--retain-resources` and cleans up what was retained

If a VPC is left behind by an earlier failed teardown:

```bash
DEPLOYMENT_ID=dev ./destroy.sh --vpc-cleanup-only
```

`destroy.sh` passes `--exclusively`, so it removes only the stacks it names. It does not
cover `BADGERS-ECS-{suffix}`, `BADGERS-Vpc-{suffix}`, or `BADGERS-DynamoDB-{suffix}` — hence the explicit commands
above.

## 🐛 Troubleshooting

### Lambda Layer Build Fails

```bash
cd lambdas

# Foundation layer
rm -rf layer/ layer.zip
./build_foundation_layer.sh

# Poppler layer
rm -rf poppler_build/ poppler-qpdf-layer.zip
./build_poppler_qdf_layer.sh

# Enhancement layer — no longer deployed, runs in container Lambda
# rm -rf enhancement_build/ enhancement-layer.zip
# ./build_enhancement_layer.sh

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

`deploy.sh` keeps this consistent for you: the suffix is generated once per
`DEPLOYMENT_ID` and reused on every subsequent run, so redeploying targets the same stacks.

```bash
DEPLOYMENT_ID=dev ./deploy.sh 10    # show the current suffix and step status
cat .deploy-state/dev.json          # or read it directly
```

To drive `cdk` yourself, supply both values — see
[Stack and Resource Naming](#-stack-and-resource-naming).

```bash
DEPLOYMENT_ID=dev STACK_SUFFIX=a1b uv run cdk deploy --all
```

> Deleting `.deploy-state/{DEPLOYMENT_ID}.json` loses the suffix. A later `deploy.sh` run
> would generate a new one and deploy a second, parallel set of stacks rather than updating
> the existing ones. Keep the file, or pass `STACK_SUFFIX` explicitly to recover.

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
