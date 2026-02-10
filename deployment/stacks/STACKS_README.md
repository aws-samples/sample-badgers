<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [Frontend](../../frontend/FRONTEND_README.md) | [Deployment](../DEPLOYMENT_README.md) | 🔵 **CDK Stacks** | [Runtime](../runtime/RUNTIME_README.md) | [S3 Files](../s3_files/S3_FILES_README.md) | [Lambda Analyzers](../lambdas/LAMBDA_ANALYZERS.md) | [Prompting System](../s3_files/prompts/PROMPTING_SYSTEM_README.md)</sub>

---

# 📦 CDK Stacks

10 CloudFormation stacks deployed in dependency order (plus 1 optional).

## Stack Overview

| Stack              | File                                   | Purpose                                           |
| ------------------ | -------------------------------------- | ------------------------------------------------- |
| S3                 | `s3_stack.py`                          | Config, source, and output buckets                |
| Cognito            | `cognito_stack.py`                     | OAuth 2.0 authentication for Gateway              |
| IAM                | `iam_stack.py`                         | Lambda execution role with Bedrock/S3 permissions |
| ECR                | `agentcore_ecr_stack.py`               | Container registry for agent image                |
| Lambda             | `lambda_stack.py`                      | Analyzer functions + foundation layer             |
| Gateway            | `agentcore_gateway_stack.py`           | AgentCore MCP Gateway with Lambda targets         |
| Memory             | `agentcore_memory_stack.py`            | Session persistence (90-day TTL)                  |
| Inference Profiles | `inference_profiles_stack.py`          | Application inference profiles for cost tracking  |
| Runtime WebSocket  | `agentcore_runtime_websocket_stack.py` | Strands agent container with WebSocket streaming  |
| Custom Analyzers   | `custom_analyzers_stack.py`            | *(Optional)* Wizard-created analyzers             |

## Dependency Graph

```
S3 ─────────────────┬──────────────────────────────────────────┐
                    │                                          │
Cognito ────────────┼──────────────────────────────────────────┤
                    │                                          │
IAM ────────────────┤                                          │
                    │                                          │
ECR ────────────────┼──────────────────────────────────────────┤
                    │                                          │
                    ▼                                          │
               Lambda ──────────────────┐                      │
                    │                   │                      │
                    ▼                   │                      │
               Gateway                  │                      │
                    │                   │                      │
               Memory ──────────────────┤                      │
                    │                   │                      │
       Inference Profiles ──────────────┤                      │
                    │                   │                      │
                    └───────────────────┴──────────────────────┤
                                                               ▼
                                                      Runtime WebSocket
                                                               │
                                                               ▼
                                                   Custom Analyzers (optional)
```

## Stack Details

### S3 (`s3_stack.py`)
Creates 3 buckets:
- **Config** — Manifests, prompts, schemas (versioned)
- **Source** — PDF uploads (versioned)
- **Output** — Analysis results with 1-day TTL on `temp/` prefix

### Cognito (`cognito_stack.py`)
OAuth 2.0 setup for Gateway authentication:
- User pool with client credentials flow
- Resource server with `agentcore-gateway/invoke` scope
- Credentials stored in Secrets Manager

### IAM (`iam_stack.py`)
Lambda execution role with:
- `bedrock:InvokeModel` for foundation models
- S3 read/write for config, source, output buckets
- CloudWatch Logs

### Lambda (`lambda_stack.py`)
Deploys all analyzer functions:
- Auto-discovers functions from `lambdas/code/` directory
- Attaches foundation layer + Pillow layer
- PDF converter gets additional Poppler layer
- Loads descriptions from schema files

### Gateway (`agentcore_gateway_stack.py`)
MCP Gateway configuration:
- Semantic tool search enabled
- Lambda targets for each analyzer
- Tool schemas loaded from S3
- CloudWatch + X-Ray logging via custom resources

### ECR (`agentcore_ecr_stack.py`)
Container registry:
- Keeps last 5 images
- Image scanning enabled
- Auto-delete on stack destroy

### Memory (`agentcore_memory_stack.py`)
AgentCore Memory for session state:
- 90-day event expiry
- Used by Runtime for conversation context

### Inference Profiles (`inference_profiles_stack.py`)
Application Inference Profiles for cost tracking and usage monitoring:
- Creates trackable profiles wrapping cross-region system-defined profiles
- 5 profiles: Claude Sonnet 4.5 (Global), Claude Haiku 4.5 (Global), Claude Opus 4.6 (Global), Claude Opus 4.5 (Global), Nova Premier (US)
- Naming convention: `badgers-{model}-{deployment_id}`
- Grants invoke permissions to Runtime role
- Profile ARNs passed to Runtime as environment variables

### Runtime WebSocket (`agentcore_runtime_websocket_stack.py`)
Strands agent hosting with WebSocket streaming:
- Pulls container from ECR
- Environment variables for Gateway URL, Cognito secret, Memory ID, inference profile ARNs
- IAM role with Bedrock, S3, Secrets Manager, Memory permissions
- WebSocket support for streaming responses

### Custom Analyzers (`custom_analyzers_stack.py`)
*(Optional)* Deployed only when `custom_analyzers/analyzer_registry.json` exists:
- Analyzers created via the wizard UI
- Uses CloudFormation exports from other stacks (no explicit dependencies)
- Registers new Lambda targets with the Gateway

## Deployment Commands

```bash
# Deploy all stacks
cdk deploy --all

# Deploy specific stack
cdk deploy badgers-lambda

# Deploy with specific deployment ID
cdk deploy -c deployment_id=abc12345 --all
```

## Adding a New Stack

1. Create `new_stack.py` in this directory
2. Import and instantiate in `../app.py`
3. Add dependencies with `new_stack.add_dependency(other_stack)`
