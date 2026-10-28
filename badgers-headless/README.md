[← BADGERS home](../README.md)

# BADGERS Headless

An API-oriented BADGERS deployment for teams that provide their own user
experience, integration layer, or workflow. It retains the agent and tool
orchestration but leaves browser application infrastructure out.

> [!IMPORTANT]
> This branch is a private, untagged 6.0 beta candidate. The Headless API is
> implemented for maintainer validation, but no 6.0 customer beta has been
> published. For source/tag, configuration-release, and support status, see the
> [Release Guide](../RELEASE_GUIDE.md).

```bash
./badgers-headless/deploy.sh
```

Choose **9** to deploy every applicable component, or choose **`r`** on a later
run to continue only unfinished steps.

## What gets deployed

- AgentCore Runtime and MCP Gateway
- Cognito machine identity, S3 storage, IAM, model profiles, and the minimum
  DynamoDB operation ledger
- Shared agent prompt core and curated specialist packs
- The `document-essentials` and `tables-and-forms` packs by default

## What is intentionally omitted

- Browser UI, ECS service, VPC, and UI image build
- Session memory, UI specialist wizard, and X-Ray Transaction Search
- WebSocket handler registration

WebSockets are not required for Headless. Invoke the AgentCore runtime through
its normal request interface from your application or service. Versioned
Headless requests authenticate, create durable operations, and submit to the
private runner. Headless retains the repository's existing AgentCore Gateway
for runtime-to-specialist calls; it is not the future customer-facing BADGERS
Gateway MCP product, and its credentials must not be distributed as an external
specialist-tool integration.

## Versioned Agent API

Headless now recognizes `badgers-agent-request/v1`. Versioned requests require
a private S3 source URI and produce an authenticated, durable operation:

```json
{
  "schema_version": "badgers-agent-request/v1",
  "session_id": "contract-review-104",
  "document_uri": "s3://<source-bucket>/tenants/<your-workload-owner>/contracts/msa.pdf",
  "instruction": "Identify renewal and termination obligations.",
  "routing_mode": "auto"
}
```

Progress events are deliberately metadata-only. The terminal event contains a
bounded `badgers-operation-result/v1` pointer manifest with the operation ID,
source URI, classifier URI, correlated URI, individual specialist artifact
URIs, configuration release ID, and trace ID only when one is available.
Native artifacts remain private in S3; they are not returned inline.

### First successful request

After deployment, use the status menu or CloudFormation outputs to record the
Runtime ARN, Headless client ID, and Headless credentials-secret ARN. Then:

1. Store a PDF, PNG, or JPEG beneath the Headless client's private source
   prefix: `s3://<source-bucket>/tenants/headless-<client-id>/...`.
2. Retrieve the approved Headless client credentials from Secrets Manager and
   obtain a Cognito client-credentials access token with the
   `badgers-headless/invoke` scope.
3. Use the AgentCore Runtime HTTP invocation path with that bearer token and a
   `badgers-agent-request/v1` payload. Do not use the AWS CLI invocation
   command for this OAuth-protected integration.
4. Treat the initial `accepted` status event as a durable submission, not a
   completed analysis. Save the returned `operation_id`, `status_manifest_uri`,
   and `plan_uri`; the private status manifest bridges a disconnected caller
   to the eventual result.

The [integration contracts](../deployment/s3_files/contracts/README.md) are
the authoritative payload definition. This guide intentionally does not expose
or print a client secret.

For a versioned request, Headless authenticates the caller at the AgentCore
Runtime using the dedicated Cognito client-credentials client and
`badgers-headless/invoke` scope provisioned by this profile. The runtime derives
the operation owner from the validated JWT `client_id`; it does not trust
request-body identity fields. Upload sources below
`tenants/headless-<client-id>/` in the deployed source bucket. The generated
client secret is retained in Secrets Manager, never printed in deployment
output, and must be used only by the approved deployment integration.

### Credential rotation

Treat the Headless client secret as a customer-managed workload credential.
Cognito supports two active app-client secrets, allowing zero-downtime
rotation: add a new secret, update the customer workload from the current
Secrets Manager value, verify it can obtain an access token and submit a
request, then remove the previous Cognito secret. Existing access tokens
remain valid until they expire, so rotate immediately if a secret is exposed.
Do not configure unattended rotation unless every calling workload reads the
current value from an approved secret provider; otherwise automatic rotation
can cause a customer outage.

See the staged [integration contracts](../deployment/s3_files/contracts/README.md)
for the exact schemas. Headless releases now also contain a validated,
declarative workflow catalog and classifier-led routing policies. The catalog
resolves only stable capability tags against the selected specialist pack; its
deterministic plan is persisted privately before the runner invokes page
specialists. Versioned requests submit to that runner and immediately receive
durable operation identifiers and status pointers. The default
`standard-document-analysis` workflow accepts PDF, PNG, and JPEG sources.
A Headless deployment can additionally select `docx_processor` or
`pptx_processor` and enable its matching native template; omitted workflow IDs
then dispatch by the source content type, while an explicit workflow ID always
wins. Native processors are opt-in because they require container image builds
and appropriate workload testing. Gateway workflow MCP tools remain a later
implementation increment.

## Before you start

Install and configure AWS CLI, CDK v2, Docker, Python 3.12+, and `uv`. Your AWS
identity needs the permissions to provision the Headless resources and access
the required Bedrock models. The menu performs local-tool, AWS identity,
region, model-access, and CDK preflight checks.

Use a new deployment ID for Headless. Deployment IDs are profile-bound, so a
Full deployment cannot be converted in place.

## Configure prompts, models, and packs

The Headless profile stages only its shared agent/prompt core plus the selected
packs' prompts, schemas, and manifests. Step 3 creates a content-addressed
configuration release; rerun downstream steps after changing prompts, manifests,
schemas, or model configuration so each consumer pins the same release.

The curated pack definitions live in
[`deployment/profiles/`](../deployment/profiles/README.md). Use the model menu
(**`m`**) to understand which deployment steps are required after a registry
change. The detailed infrastructure contract is in the
[deployment reference](../deployment/DEPLOYMENT_README.md). Step 3's
configuration SHA is private to this deployment and is separate from the Git
source release; see the [Release Guide](../RELEASE_GUIDE.md#release-terminology).

For an explicit review surface, run `./deploy.sh configure --ui`. It edits only
local desired state for one deployment ID: choose packs and direct specialists,
then select compatible PDF/image or native DOCX/PPTX workflow templates. The
normal CLI remains the only AWS executor.

## Go deeper

| Topic | Guide |
| --- | --- |
| Validate request and result payloads | [Integration contracts](../deployment/s3_files/contracts/README.md) |
| Understand the approved workflow and routing boundary | [Workflows](../deployment/s3_files/workflows/README.md) and [routing policies](../deployment/s3_files/routing_policies/README.md) |
| Operate the agent container in Headless mode | [Runtime guide](../deployment/runtime/RUNTIME_README.md#headless-request-mode) |
| Deploy individual stacks, resume a deployment, or clean up | [Deployment reference](../deployment/DEPLOYMENT_README.md) |
| Inspect product capability and selected packs | [Deployment profiles](../deployment/profiles/README.md) |
| Plan specialist and routing changes locally | [Local configuration planner](../deployment/DEPLOYMENT_README.md#configure-packs-and-routing-locally) |
| Choose a source tag and understand releases | [Release guide](../RELEASE_GUIDE.md) |
