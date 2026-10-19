# BADGERS Headless

An API-oriented BADGERS deployment for teams that provide their own user
experience, integration layer, or workflow. It retains the agent and tool
orchestration but leaves browser application infrastructure out.

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
its normal request interface from your application or service, and use the
Gateway for the agent's specialist tools.

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
[deployment reference](../deployment/DEPLOYMENT_README.md).
