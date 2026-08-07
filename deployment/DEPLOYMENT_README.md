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

Two values identify a deployment:

- **`DEPLOYMENT_ID`** — a short label you choose (`dev`, `demo`, `b2`). Must match
  `^[a-z][a-z0-9-]{0,15}$`: lowercase, starting with a letter, 16 characters or fewer.
  It ends up in S3 bucket names, which is why uppercase is rejected.
- **`STACK_SUFFIX`** — three random hex characters, generated once per `DEPLOYMENT_ID`.

Both are persisted in `.deploy-state/{DEPLOYMENT_ID}.json` and appear in every name:

|                | Pattern                                   | Example                            |
| -------------- | ----------------------------------------- | ---------------------------------- |
| Stack names    | `BADGERS-{Name}-{DEPLOYMENT_ID}-{suffix}` | `BADGERS-S3-dev-a1b`               |
| Resource names | `badgers-{kind}-{id}-{suffix}`            | `badgers-config-dev-a1b`           |
| SSM prefix     | `/badgers-{id}-{suffix}`                  | `/badgers-dev-a1b/jobs-table-name` |

Stack names carry the deployment id as well as the suffix so that each stack is
self-describing. The suffix alone would be enough to keep names unique, but not enough to
establish *identity* — with suffix-only names, a mistyped `DEPLOYMENT_ID` still resolves
real stacks while every resource name derived from it points at something that does not
exist. Tooling can now read a deployment's identity straight off the stack name.

Anything running `cdk` directly needs both values:

```bash
DEPLOYMENT_ID=dev STACK_SUFFIX=a1b uv run cdk deploy BADGERS-S3-dev-a1b
# or as context
uv run cdk deploy -c deployment_id=dev -c stack_suffix=a1b BADGERS-S3-dev-a1b
```

`app.py` fails fast with usage if either is missing, rather than inventing a suffix and
deploying a second copy of everything by accident.

> AgentCore runtime and memory names must match `[a-zA-Z][a-zA-Z0-9_]{0,47}` and cannot
> contain hyphens, so those two stacks normalise the composite id to underscores —
> `badgers_runtime_ws_dev_a1b`, `badgers_memory_dev_a1b`.

## ⚡ Quick Start

From the repository root:

```bash
./deploy.sh
```

That is the whole command. **`deploy.sh` does not read `DEPLOYMENT_ID` from the
environment** — it unsets it and always asks, because a value left exported in your shell
silently targets another deployment's stacks and state file.

On start it scans `.deploy-state/` and offers what it finds:

```
Deployments found — most recent activity first:

    1) dev                8/9 steps  in progress   last activity 2026-08-04T16:14:38Z
    2) demo               9/9 steps  complete      last activity 2026-08-02T11:02:10Z

    n) Start a new deployment instead

Select which? (1-2, or n) [1]:
```

Complete and in-progress deployments are both listed, so you can re-run a single step
against a finished deployment. Choosing `n` prompts for a new id and rejects one that
already has state — pick it from the list instead. Pressing Enter takes the most recent.

Then the menu:

| Option | Step                                | What it does                                                           |
| ------ | ----------------------------------- | ---------------------------------------------------------------------- |
| `1`    | Build Lambda Layers                 | foundation, PDF processing, poppler/qdf archives                       |
| `2`    | Foundational Infra                  | S3, Cognito, DynamoDB, IAM, ECR, Inference Profiles, XRay, Memory, VPC |
| `3`    | Upload Prompts, Manifests & Schemas | syncs `s3_files/` to the config bucket                                 |
| `4`    | Specialist Lambdas                  | container images to ECR, then the Lambda stack                         |
| `5`    | Gateway                             | AgentCore MCP Gateway with Lambda targets                              |
| `6`    | Runtime — Build & Deploy            | builds/pushes the agent image, deploys the Runtime                     |
| `7`    | UI — Build & Push Image             | generates `ui/.env`, builds the Vite bundle and image, pushes          |
| `8`    | UI — Deploy ECS                     | deploys the ECS stack, forces the image rollout, waits                 |
| `9`    | Full Deployment                     | runs 1 → 8 in order, stopping at the first failure                     |
| `12`   | Resume                              | runs only the steps still outstanding, without prompting on each       |
| `10`   | Show Deployment Status              | current suffix, per-step completion with timestamps                    |
| `11`   | Reset Deployment State              | marks all steps incomplete; **deletes nothing in AWS**                 |
| `0`    | Exit                                |                                                                        |

You can also run one directly — `./deploy.sh 8` or `./deploy.sh resume` — but the
deployment is still chosen interactively first.

### Resume vs Full Deployment

Both get you to a complete deployment; they differ in friction.

**Option 9** walks all eight steps. Steps already marked complete stop and ask whether to
re-run, so resuming from step 4 costs you three prompts. Declining a prompt is treated as
success and the run continues.

**Option 12** skips completed steps *before* calling them, so those prompts never fire. It
starts at the first outstanding step. A step counts as complete only when every state key
it writes is set, so an interrupted step 6 (image pushed, runtime not deployed) is
re-entered rather than skipped.

`BADGERS_ASSUME_YES=1` answers every confirmation with yes, for unattended runs. Note it
*re-runs* completed steps rather than skipping them — it is not a quiet resume.

### Resumability

Each step records completion in `.deploy-state/{DEPLOYMENT_ID}.json` with a timestamp, so
a re-run after a failure continues rather than starting over. Every step is idempotent.

If step 8 fails, `ui_image_pushed` is cleared deliberately: an ECS rollout almost always
fails because of something inside the image, and redeploying the stack alone will not pick
up a code change. Resume therefore rebuilds and pushes in step 7 before retrying step 8.
When the cause was outside the image — a quota, a permission — step 7 is a no-op rebuild
that costs only time.

See [DEPLOYMENT_SCRIPTS.md](DEPLOYMENT_SCRIPTS.md) for every script and its flags.

## 📦 CDK Stacks

13 stacks deployed in dependency order, plus 1 optional. See
[CDK Stacks](stacks/STACKS_README.md) for details and the dependency graph.

| Stack                                     | Purpose                                                           |
| ----------------------------------------- | ----------------------------------------------------------------- |
| `BADGERS-S3-{id}-{suffix}`                | Config bucket (manifests/prompts) + source + output buckets       |
| `BADGERS-Cognito-{id}-{suffix}`           | User pool with UI (OIDC/PKCE) and Gateway (M2M) clients           |
| `BADGERS-DynamoDB-{id}-{suffix}`          | Jobs table for doc/job/subtask tracking                           |
| `BADGERS-IAM-{id}-{suffix}`               | Lambda execution role with Bedrock/S3/DynamoDB permissions        |
| `BADGERS-ECR-{id}-{suffix}`               | Container registry for agent and container Lambda images          |
| `BADGERS-InferenceProfiles-{id}-{suffix}` | Application Inference Profiles for cost tracking                  |
| `BADGERS-Lambda-{id}-{suffix}`            | Base specialist functions + foundation layer                      |
| `BADGERS-XRay-{id}-{suffix}`              | X-Ray Transaction Search (account-level singleton, often skipped) |
| `BADGERS-Gateway-{id}-{suffix}`           | AgentCore MCP Gateway with Lambda targets                         |
| `BADGERS-Memory-{id}-{suffix}`            | AgentCore Memory for session persistence                          |
| `BADGERS-RuntimeWebSocket-{id}-{suffix}`  | AgentCore Runtime (Strands agent with WebSocket)                  |
| `BADGERS-Vpc-{id}-{suffix}`               | VPC with public/private subnets, NAT, flow logs, VPC endpoints    |
| `BADGERS-ECS-{id}-{suffix}`               | Unified UI on an ECS Express Gateway service                      |
| `BADGERS-CustomSpecialists-{id}-{suffix}` | Custom specialists (optional, wizard-created)                     |

### X-Ray and the CloudWatch Logs resource policy quota

The XRay stack enables X-Ray Transaction Search, which is an **account-and-region
singleton** and needs a CloudWatch Logs resource policy. That policy competes for a hard
quota of **10 per region** (Service Quotas code `L-89892494`), which is **not
adjustable** — a support request will not raise it — and is shared with every other
project in the account.

`deploy.sh` resolves this before any `cdk deploy` runs, in all five stack-deploying steps:

- **Already ACTIVE** → the XRay stack is dropped from the app entirely. Nothing to enable,
  and skipping it avoids consuming a policy slot for a setting that is already on.
- **Not enabled, room in the quota** → the stack deploys and enables it.
- **Not enabled, quota full** → the deploy is refused *before* anything is created, and the
  existing policies are listed with what each one grants, so you can decide what to
  reclaim. Left unchecked, CloudFormation fails mid-deploy with a bare
  `ServiceLimitExceeded` after other stacks already exist.

Set `BADGERS_SKIP_XRAY=1` to skip tracing regardless. This matters because
`RuntimeWebSocket` depends on the XRay stack and `cdk deploy` includes a stack's
dependencies — so deploying the Runtime or ECS stack with the decision unresolved would
quietly pull XRay back in.

### UI Stacks

The UI runs on `BADGERS-Vpc-{id}-{suffix}` + `BADGERS-ECS-{id}-{suffix}`. The ECS Express
Gateway service provisions and manages its own load balancer and HTTPS endpoint, so **no
hosted zone, domain, or ACM certificate is required** — there is nothing to configure
before deploying it.

Cognito values must be baked into the Vite bundle at build time, so ordering matters.
`deploy.sh` steps 7 and 8 handle it:

```bash
./deploy.sh 7   # generate ui/.env, build the bundle, build and push the image
./deploy.sh 8   # deploy the ECS stack, force the rollout, wait for it
```

#### Network exposure — asked once, fixed for the VPC's lifetime

Step 8 prompts before doing anything:

```
UI network exposure — fixed for the life of this VPC.

    y) Public    internet-facing load balancer; tasks get public IPs.
    n) Internal  internal load balancer; reachable only from inside the VPC.

Make the UI publicly accessible? (y/n) [y]:
```

ECS Express Mode derives the load balancer scheme from the subnets it is given: public
subnets produce an internet-facing ALB, subnets without an internet gateway produce an
internal one. **The first Express service in a VPC establishes that scheme for the VPC**,
so answering differently later does not flip an existing load balancer — you would have to
destroy the ECS and VPC stacks and redeploy. Answer `n` and the public
`https://<id>.ecs.<region>.on.aws` URL will resolve but never respond.

The prompt is asked up front for options 9 and 12 so the rest of the run is unattended.
`UI_PUBLIC_ACCESS=true|false` in the environment is honoured without prompting.

#### Why step 8 forces a rollout

The stack references the UI image by a static tag (`:frontend`). Pushing a new image to
that tag leaves the CloudFormation template unchanged, so `cdk deploy` would report no
changes and the service would keep serving the old image. Step 8 therefore calls
`update-express-gateway-service` with the container spec after the `cdk deploy`, forcing a
new deployment regardless, then polls `rolloutState` until `COMPLETED`. That is what makes
a step 7 rebuild actually take effect.

Step 7 runs `scripts/generate_ui_env.sh` first, because a bundle built before Cognito
exists has no authority or client id compiled into it and falls back to the server's
local-dev bypass.

`BADGERS-ECS-{id}-{suffix}` reads all container configuration from SSM Parameter Store under
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
uv run cdk deploy BADGERS-S3-{id}-{suffix} --require-approval never

# Sync configuration files
./sync_s3_files.sh
```

### 5️⃣ Deploy Auth & IAM

```bash
uv run cdk deploy BADGERS-Cognito-{id}-{suffix} --require-approval never
uv run cdk deploy BADGERS-IAM-{id}-{suffix} --require-approval never
```

### 6️⃣ Deploy Lambda Functions

```bash
uv run cdk deploy BADGERS-Lambda-{id}-{suffix} --require-approval never
```

### 7️⃣ Deploy Gateway

```bash
uv run cdk deploy BADGERS-Gateway-{id}-{suffix} --require-approval never
```

### 8️⃣ Deploy ECR + Build Container

```bash
uv run cdk deploy BADGERS-ECR-{id}-{suffix} --require-approval never

cd runtime
./build_and_push_websocket.sh
cd ..
```

### 9️⃣ Deploy Memory + Runtime

```bash
uv run cdk deploy BADGERS-Memory-{id}-{suffix} --require-approval never
uv run cdk deploy BADGERS-RuntimeWebSocket-{id}-{suffix} --require-approval never
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
│   └── generate_ui_env.sh    # 🔐 Writes ui/.env from BADGERS-Cognito-{id}-{suffix} outputs
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
├── manifests/              # Base specialists (deployed with BADGERS-Lambda-{id}-{suffix})
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
   uv run cdk deploy BADGERS-CustomSpecialists-{id}-{suffix}
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
uv run cdk deploy BADGERS-Lambda-{id}-{suffix} --require-approval never

# Gateway target changes
uv run cdk deploy BADGERS-Gateway-{id}-{suffix} --require-approval never

# Agent container changes
cd runtime && ./build_and_push_websocket.sh && cd ..
uv run cdk deploy BADGERS-RuntimeWebSocket-{id}-{suffix} --require-approval never

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
./destroy.sh
```

Like `deploy.sh`, it asks what to tear down — but it discovers deployments from
**CloudFormation**, not from `.deploy-state/`. A state file can be deleted while the stacks
are still live, so the stacks are the source of truth:

```
Scanning CloudFormation for BADGERS-*-{id}-{suffix} stacks...

Deployments found in us-west-2 — newest first:

    1) dev                  suffix a1b     11 stack(s)   created 2026-08-04T19:11:24Z
    2) demo                 suffix c7f     14 stack(s)   created 2026-08-02T10:02:11Z

Destroy which? (1-2, or q to abort):
```

You are then required to type the `DEPLOYMENT_ID` to confirm.

You can still pass it explicitly — `DEPLOYMENT_ID=dev ./destroy.sh` — in which case it is
validated against the naming rules and checked against what is actually deployed. If no
stacks match `BADGERS-*-{id}-{suffix}`, it refuses rather than running a teardown that
quietly does nothing.

### What it does, in order

1. **Empties the config, source and output buckets**, including all versions and delete
   markers. The buckets also have `auto_delete_objects`, so this is belt and braces.
2. **Deletes the ECS Express service**, then polls until it is gone. This uses
   `delete-express-gateway-service` — `delete-service` rejects an Express Gateway Service
   outright with *"has ResourceManagementType=ECS use DeleteExpressGatewayService"*.
   Deleting the Express service also removes the load balancer it created, provided no
   other service is sharing it.
3. **Deletes the AgentCore runtime endpoints, then the runtime**, and waits. These calls go
   to `bedrock-agentcore-control`; the `bedrock-agentcore` service only exposes
   `InvokeAgentRuntime`, so aiming control-plane calls at it fails silently.
4. **Sweeps ENIs** in the VPC. CloudFormation cannot delete a VPC while an ENI is still
   attached, and both the ECS service and the runtime release theirs asynchronously, well
   after they report gone.
5. **Destroys all 14 stacks in reverse dependency order** — see the table below. ECS goes
   before VPC because it imports three VPC exports.
6. **Verifies every stack is gone.** If the VPC stack is `DELETE_FAILED`, it retries with
   `--retain-resources`, sweeps ENIs again, then re-verifies.
7. **Schedules the KMS key for deletion**, but *only* once the stacks are confirmed gone.
   Scheduling it after a failed teardown would mark a live deployment's in-use key for
   deletion, which is unrecoverable once the window expires. The default 7-day window is
   deliberate: the alias `alias/badgers-s3-key-{id}-{suffix}` stays reserved until the key
   is gone, so a long window blocks redeploying under the same `DEPLOYMENT_ID`. Override
   with `KMS_WAIT_DAYS` (7–30).

### If it does not finish

A teardown that leaves stacks standing **exits non-zero** and prints
`❌ Teardown incomplete` with the survivors listed. It does not print a success banner over
a failed run, and it does not schedule the KMS key — it tells you the key was left alone
and gives you the ARN.

Re-run it once the cause is fixed; it is safe to run repeatedly. Scroll up to the *first*
error: a `cdk synth` or destroy failure aborts before any stack is touched, so everything
after it is a consequence rather than the cause.

To sweep ENIs from a VPC left behind by an earlier failed teardown:

```bash
DEPLOYMENT_ID=dev STACK_SUFFIX=a1b ./destroy.sh --vpc-cleanup-only
```

That flag needs both values explicitly, since it skips the discovery step.

### What teardown does not remove

- **CloudWatch Logs resource policies** created by vended log delivery. Nothing deletes an
  `AWSLogDeliveryWrite*` policy when its delivery source disappears, so they accumulate
  against the non-adjustable quota of 10 per region across repeated deploy/destroy cycles.
  See [X-Ray and the CloudWatch Logs resource policy quota](#x-ray-and-the-cloudwatch-logs-resource-policy-quota).
- **The KMS key itself** until its pending-deletion window expires. Cancel with
  `aws kms cancel-key-deletion --key-id <arn>` if you scheduled it by mistake.
- **`.deploy-state/{DEPLOYMENT_ID}.json`.** Delete it yourself once the stacks are gone,
  or the deploy menu will keep offering a deployment that no longer exists.
- **CloudWatch log groups** with a retention policy still counting down.

## 🖱️ Manual Teardown in the Console

Use this when `destroy.sh` cannot run — no local environment, a partially renamed stack set
that discovery does not match, or a teardown wedged badly enough that you want to drive it
by hand.

> [!IMPORTANT]
> Two things must happen **before** you delete any stack. Neither is a CloudFormation
> resource, and skipping either will wedge a stack delete.

### Step 1 — Delete the ECS Express service

The ECS console can do this, or:

```bash
aws ecs delete-express-gateway-service --region us-west-2 \
  --service-arn arn:aws:ecs:us-west-2:<account>:service/default/badgers-ui-{id}-{suffix}
```

`DeleteService` will not work on it. Its ENIs block the VPC stack later, and deleting it
also removes the load balancer it created.

### Step 2 — Delete the AgentCore runtime

Bedrock AgentCore console → delete the runtime's **endpoints first**, then the runtime. Or:

```bash
aws bedrock-agentcore-control list-agent-runtimes --region us-west-2
aws bedrock-agentcore-control list-agent-runtime-endpoints --region us-west-2 --agent-runtime-id <id>
aws bedrock-agentcore-control delete-agent-runtime-endpoint --region us-west-2 \
  --agent-runtime-id <id> --endpoint-name <name>
aws bedrock-agentcore-control delete-agent-runtime --region us-west-2 --agent-runtime-id <id>
```

Note the service name: **`bedrock-agentcore-control`**, not `bedrock-agentcore`.

### Step 3 — Delete the stacks, one at a time, in this order

CloudFormation → select the stack → **Delete**. Wait for `DELETE_COMPLETE` before starting
the next one.

| #   | Stack                                     | Notes                                                         |
| --- | ----------------------------------------- | ------------------------------------------------------------- |
| 1   | `BADGERS-CustomSpecialists-{id}-{suffix}` | Only exists if you created custom specialists                 |
| 2   | `BADGERS-ECS-{id}-{suffix}`               | Step 1 above must be done first                               |
| 3   | `BADGERS-RuntimeWebSocket-{id}-{suffix}`  | Step 2 above must be done first                               |
| 4   | `BADGERS-Gateway-{id}-{suffix}`           |                                                               |
| 5   | `BADGERS-Lambda-{id}-{suffix}`            |                                                               |
| 6   | `BADGERS-Memory-{id}-{suffix}`            |                                                               |
| 7   | `BADGERS-XRay-{id}-{suffix}`              | Often absent — skipped when Transaction Search was already on |
| 8   | `BADGERS-ECR-{id}-{suffix}`               | Images are removed automatically (`empty_on_delete`)          |
| 9   | `BADGERS-InferenceProfiles-{id}-{suffix}` |                                                               |
| 10  | `BADGERS-IAM-{id}-{suffix}`               |                                                               |
| 11  | `BADGERS-DynamoDB-{id}-{suffix}`          | Deletes the jobs table and all job records                    |
| 12  | `BADGERS-S3-{id}-{suffix}`                | Buckets empty themselves (`auto_delete_objects`)              |
| 13  | `BADGERS-Cognito-{id}-{suffix}`           | Deletes the user pool and all users                           |
| 14  | `BADGERS-Vpc-{id}-{suffix}`               | **Always last.** Most likely to fail — see below              |

**Do not delete them in parallel.** Several export values that later stacks import, and
CloudFormation refuses to delete a stack whose exports are still in use — a parallel delete
fails with `Export ... cannot be deleted as it is in use`.

Not every stack will exist. Skip what is not there.

### Step 4 — If the VPC stack fails

`DELETE_FAILED` on the VPC is almost always a leftover ENI from the ECS service or the
AgentCore runtime. Find them in the EC2 console under **Network Interfaces**, filtered by
VPC, or:

```bash
aws ec2 describe-network-interfaces --region us-west-2 \
  --filters "Name=vpc-id,Values=<vpc-id>" \
  --query "NetworkInterfaces[].[NetworkInterfaceId,Status,Description]" --output table
```

Detach then delete any that remain, and retry the stack delete. An ENI in `available`
status can be deleted directly; one that is `in-use` still has an owner, which means step 1
or step 2 did not complete.

If CloudFormation still refuses, delete the stack again and choose **retain** for the
failed resources, then remove those resources by hand afterwards.

### Step 5 — Clean up what the stacks do not own

- **KMS key** — KMS console → the key behind `alias/badgers-s3-key-{id}-{suffix}` →
  *Schedule key deletion*. Nothing schedules it for you when tearing down by hand, and the
  alias stays reserved until it is gone.
- **CloudWatch Logs resource policies** — check for `AWSLogDeliveryWrite*` policies left
  behind by vended log delivery. These are invisible in the IAM console; they live under
  CloudWatch Logs and are only reachable via
  `aws logs describe-resource-policies`.
- **`.deploy-state/{DEPLOYMENT_ID}.json`** — delete it so the deploy menu stops offering a
  deployment that no longer exists.

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
./deploy.sh                 # pick the deployment, then option 10 for suffix and step status
cat .deploy-state/dev.json  # or read it directly
```

To drive `cdk` yourself, supply both values — see
[Stack and Resource Naming](#-stack-and-resource-naming).

```bash
DEPLOYMENT_ID=dev STACK_SUFFIX=a1b uv run cdk deploy --all
```

> Deleting `.deploy-state/{DEPLOYMENT_ID}.json` loses the suffix. A later `deploy.sh` run
> would generate a new one and deploy a second, parallel set of stacks rather than updating
> the existing ones.
>
> Recovering it is straightforward, because stack names contain both values. List what is
> deployed and read the id and suffix off any stack name:
>
> ```bash
> aws cloudformation list-stacks --region us-west-2 \
>   --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE \
>   --query "StackSummaries[?starts_with(StackName,'BADGERS-')].StackName" --output text
> ```
>
> `BADGERS-S3-dev-a1b` means `DEPLOYMENT_ID=dev`, `STACK_SUFFIX=a1b`. Recreate the file with
> at least those two keys — every step flag being absent simply reads as incomplete, and all
> steps are idempotent:
>
> ```bash
> printf '{\n  "deployment_id": "dev",\n  "stack_suffix": "a1b"\n}\n' > .deploy-state/dev.json
> ```
>
> `./destroy.sh` does not need the state file at all — it discovers deployments from
> CloudFormation.

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
