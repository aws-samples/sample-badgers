# BADGERS Specialists

Deploy self-contained document analyzer packs as direct AWS Lambda tools. This
is the smallest BADGERS product: no agent, Gateway, UI, Cognito, session memory,
or operation ledger.

```bash
./badger-specialists/deploy.sh
```

The default is the **Document Essentials** pack. To compose a deployment from
additional packs, pass `--pack` before the menu option:

```bash
./badger-specialists/deploy.sh --pack tables-and-forms
./badger-specialists/deploy.sh --pack tables-and-forms 9
```

## What gets deployed

- The selected specialist Lambda functions and their shared foundation layer
- S3 configuration and artifact storage, IAM, and model inference profiles
- A content-addressed release containing only the selected packs' prompts,
  manifests, schemas, and shared prompt core

## Invoke a specialist

Your application calls the deployed Lambda function directly using AWS IAM.
Grant its calling identity `lambda:InvokeFunction` for the specialist functions
in that deployment. Stack outputs and the deployment status menu identify the
functions to call.

This product deliberately does not provide an HTTP endpoint, Gateway, MCP
adapter, user identity, or UI. Put your own API boundary and authorization in
front of it if your callers are not already AWS principals.

## Before you start

Install and configure AWS CLI, CDK v2, Docker, Python 3.12+, and `uv`. The
deployment menu verifies the local toolchain, target account/region, Bedrock
model access, and CDK readiness before it creates resources.

Specialist deployments are pack-bound. Select the pack set before choosing a
new deployment ID; change the pack set by creating a separate deployment rather
than changing an existing one in place.

## Choose or create packs

Pack definitions are declarative, ordered selections of enabled specialists in
[`deployment/profiles/`](../deployment/profiles/README.md). The provided packs
are:

- `document-essentials` — PDF conversion, classification, text, layout, and
  element analysis.
- `tables-and-forms` — table analysis, composable with Document Essentials.

To create a customer-specific pack, add a named pack definition that selects
existing enabled specialists; then deploy it under a new deployment ID. Shared
prompts and specialist configuration are released together in Step 3, so a
function has a stable configuration release rather than mutable configuration
at invocation time.

For deployment mechanics and cleanup, see the
[deployment reference](../deployment/DEPLOYMENT_README.md).
