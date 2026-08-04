<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [UI](../../ui/UI_README.md) | [Deployment](../DEPLOYMENT_README.md) | 🔵 **CDK Stacks** | [Runtime](../runtime/RUNTIME_README.md) | [S3 Files](../s3_files/S3_FILES_README.md) | [Lambda Specialists](../lambdas/LAMBDA_SPECIALISTS.md) | [Prompting System](../s3_files/prompts/PROMPTING_SYSTEM_README.md)</sub>

---

# 📦 CDK Stacks

13 CloudFormation stacks, plus 1 optional, deployed in dependency order.

## Stack Overview

| Stack              | File                                   | Purpose                                                    |
| ------------------ | -------------------------------------- | ---------------------------------------------------------- |
| S3                 | `s3_stack.py`                          | Config, source, and output buckets                         |
| Cognito            | `cognito_stack.py`                     | User pool with UI (OIDC/PKCE) and Gateway (M2M) clients    |
| DynamoDB           | `dynamodb_stack.py`                    | Jobs table for doc/job/subtask tracking                    |
| IAM                | `iam_stack.py`                         | Lambda execution role with Bedrock/S3/DynamoDB permissions |
| ECR                | `agentcore_ecr_stack.py`               | Container registry for agent and container Lambda images   |
| Inference Profiles | `inference_profiles_stack.py`          | Application inference profiles for cost tracking           |
| Lambda             | `lambda_stack.py`                      | Specialist functions + foundation layer                    |
| X-Ray              | `xray_transaction_search_stack.py`     | Account-level Transaction Search for AgentCore tracing     |
| Gateway            | `agentcore_gateway_stack.py`           | AgentCore MCP Gateway with Lambda targets                  |
| Memory             | `agentcore_memory_stack.py`            | Session persistence (90-day TTL)                           |
| Runtime WebSocket  | `agentcore_runtime_websocket_stack.py` | Strands agent container with WebSocket streaming           |
| VPC                | `vpc_stack.py`                         | Public/private subnets and VPC endpoints for the UI        |
| ECS                | `ecs_stack.py`                         | Unified UI on an ECS Express Gateway service               |
| Custom Specialists | `custom_specialists_stack.py`          | *(Optional)* Wizard-created specialists                    |

Stack names follow `BADGERS-{Name}-{suffix}`, where the suffix is generated once per
`DEPLOYMENT_ID` by `deploy.sh` and persisted in `.deploy-state/{DEPLOYMENT_ID}.json`. The
ECS stack therefore deploys as, for example, `BADGERS-ECS-a1b`. Resource names inside the
stacks carry both parts (`badgers-config-dev-a1b`), so several deployments can coexist in
one account and region. See
[Stack and Resource Naming](../DEPLOYMENT_README.md#-stack-and-resource-naming).

## Dependency Graph

Arrows point from a stack to the stacks it depends on.

```
        S3   Cognito   DynamoDB   ECR   Inference Profiles   X-Ray   Memory   VPC
         │      │         │        │           │               │        │      │
         ├──────┼─────────┤        │           │               │        │      │
         ▼      │         ▼        │           │               │        │      │
        IAM ────┼──────► (jobs table grants)   │               │        │      │
         │      │         │        │           │               │        │      │
         ▼      │         ▼        ▼           ▼               │        │      │
       Lambda ──┴─────────┴────────┴───────────┘               │        │      │
         │                                                     │        │      │
         ▼                                                     │        │      │
      Gateway ◄──── Cognito                                    │        │      │
         │                                                     │        │      │
         ▼                                                     │        │      │
   Runtime WebSocket ◄────────────────────────────────────────┴────────┘      │
         │                                                                     │
         ▼                                                                     │
        ECS ◄──────────────────────────────────────────────────────────────────┘
         │
         ▼
 Custom Specialists (optional, via CloudFormation exports — no explicit dependency)
```

The UI tier (VPC, ECS) sits at the end because `ecs_stack.py` needs the Runtime ARN
and Gateway ID to inject as container config, and re-points the Cognito UI client's
callback URLs at the service endpoint once it exists.

## Stack Details

### S3 (`s3_stack.py`)
Creates 3 buckets:
- **Config** — Manifests, prompts, schemas (versioned)
- **Source** — PDF uploads (versioned)
- **Output** — Analysis results with 1-day TTL on `temp/` prefix

### Cognito (`cognito_stack.py`)
One user pool carrying two app clients, plus Managed Login v2 with explicit branding.

**UI client** — authorization code + PKCE, used by the React frontend via
`react-oidc-context`:
- Public client (no secret), scopes `openid email profile`
- Callback and logout URLs default to the local dev origin, then get re-pointed at
  the deployed ECS endpoint by the ECS stack (that endpoint does not exist at synth time)

**Gateway client** — OAuth 2.0 client credentials, machine to machine, used by the
Runtime to mint tokens for Gateway auth:
- Has a client secret, mirrored into Secrets Manager as `credentials_secret`
- Resource server with the `agentcore-gateway/invoke` scope

Also creates `admin` and `tester` groups. Group membership is what the UI server
turns into a role — see [UI](../../ui/UI_README.md).

> Managed Login v2 requires an explicit branding style. An app client with no
> branding returns "Login pages unavailable" at the hosted login URL. Custom branding
> loads from `deployment/assets/cognito-branding-definitions.json` when present.

### DynamoDB (`dynamodb_stack.py`)
Single table, `badgers-jobs-{deployment_id}`, holding the doc/job/subtask hierarchy:
- `job_id` (PK) / `subtask_id` (SK), on-demand billing, 30-day TTL via `ttl`
- Point-in-time recovery enabled, AWS-managed encryption
- `status-index` GSI for ops monitoring and UI status filters
- `doc-index` GSI to list every job and subtask belonging to one document
- Table name published to SSM at `/badgers-{deployment_id}/jobs-table-name`

See [Lambda Specialists](../lambdas/LAMBDA_SPECIALISTS.md#-job-tracking) for the record
shape and who writes it.

### IAM (`iam_stack.py`)
Lambda execution role with:
- `bedrock:InvokeModel` for foundation models
- S3 read/write for config, source, output buckets
- `dynamodb:PutItem/UpdateItem/GetItem/Query` on the jobs table, for subtask records
- CloudWatch Logs

### Lambda (`lambda_stack.py`)
Deploys all specialist functions:
- Auto-discovers functions from `lambdas/code/` directory
- Attaches foundation layer + Pillow layer
- PDF converter gets additional Poppler layer
- Loads descriptions from schema files
- Injects `JOBS_TABLE_NAME` so each specialist can record its own subtask state

### Gateway (`agentcore_gateway_stack.py`)
MCP Gateway configuration:
- Semantic tool search enabled
- Lambda targets for each specialist
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

### X-Ray (`xray_transaction_search_stack.py`)
Enables X-Ray Transaction Search, a prerequisite for AgentCore tracing:
- Indexing percentage set to 1 (the free tier)
- **This is a singleton per account and region.** If Transaction Search is already
  enabled, destroy this stack or drop it from the deploy sequence rather than
  deploying a second one.

### Runtime WebSocket (`agentcore_runtime_websocket_stack.py`)
Strands agent hosting with WebSocket streaming:
- Pulls container from ECR
- Environment variables for Gateway URL, Cognito secret, Memory ID, inference profile
  ARNs, and `JOBS_TABLE_NAME`
- IAM role with Bedrock, S3, Secrets Manager, Memory permissions, plus
  `dynamodb:PutItem` on the jobs table — the agent writes the job-level record when
  it mints a `job_id`
- WebSocket support for streaming responses

### VPC (`vpc_stack.py`)
Network for the UI ECS service:
- Public and private subnets across 2 AZs, 1 NAT gateway (cost-optimised; raise for HA)
- VPC Flow Logs enabled
- Gateway endpoints (no hourly charge): S3, DynamoDB
- Interface endpoints: Bedrock Runtime, SSM, Secrets Manager — keeps AWS API traffic
  off the NAT gateway
- Two security groups: one for the UI task, one for the interface endpoints accepting
  HTTPS from it

### ECS (`ecs_stack.py`)
The unified UI, running as an ECS Express Gateway service:
- `CfnExpressGatewayService` provisions and manages **its own load balancer and HTTPS
  endpoint** via the ECS infrastructure role. No hosted zone, domain, or ACM
  certificate is required.
- 1024 CPU / 2048 memory, container port 7860
- All container configuration is injected from SSM Parameter Store via `valueFrom`
  references — no plaintext config in the service definition or deploy scripts
- Task role scoped to what the UI server actually calls: `InvokeAgentRuntime`,
  `ListGatewayTargets`, DynamoDB Query/GetItem/DeleteItem on the jobs table and its
  GSIs, S3 on the three buckets, CloudWatch Logs Insights queries, KMS for the S3 CMK,
  and SSM reads on the deployment prefix
- After the service exists, an `AwsCustomResource` re-points the Cognito UI client's
  callback and logout URLs at the service endpoint

### Custom Specialists (`custom_specialists_stack.py`)
*(Optional)* Deployed only when `custom_specialists/specialist_registry.json` exists:
- Specialists created via the wizard UI
- Uses CloudFormation exports from other stacks (no explicit dependencies)
- Registers new Lambda targets with the Gateway

## Deployment Commands

`cdk.json` sets the app command to `python3 app.py`, so run the CDK through the
project environment or `python3` will not find `aws_cdk`:

Prefer `./deploy.sh` from the repository root. To drive the CDK directly, export the
deployment identity first — `app.py` requires both values and fails fast without them:

```bash
export DEPLOYMENT_ID=dev
export STACK_SUFFIX=a1b        # from .deploy-state/dev.json

# Deploy all stacks
uv run cdk deploy --all

# Deploy specific stack
uv run cdk deploy BADGERS-Lambda-a1b

# Synthesize one stack (note: synth takes stack ids, not --all)
uv run cdk synth BADGERS-ECS-a1b
```

Set `CDK_NAG=1` to run `AwsSolutionsChecks` at synth time.

## Adding a New Stack

1. Create `new_stack.py` in this directory
2. Import and instantiate in `../app.py`
3. Add dependencies with `new_stack.add_dependency(other_stack)`
