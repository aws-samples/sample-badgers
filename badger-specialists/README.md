[← BADGERS home](../README.md)

# BADGERS Specialists

Deploy self-contained document analyzer packs as direct AWS Lambda tools. This
is the smallest BADGERS product: no agent, Gateway, UI, Cognito, session memory,
or operation ledger.

> [!IMPORTANT]
> `release/6.0.0-beta` is a private, untagged candidate, not a customer
> release. Use `v5.0.2` for a current production deployment and read the
> [Release Guide](../RELEASE_GUIDE.md) before consuming a future beta tag.

```bash
./badger-specialists/deploy.sh
```

The default is the **Document Essentials** pack. To compose a deployment from
additional packs or individual specialists, pass `--pack` / `--specialist`
before the menu option:

```bash
./badger-specialists/deploy.sh --pack tables-and-forms
./badger-specialists/deploy.sh --pack tables-and-forms 9
./badger-specialists/deploy.sh --specialist docx_processor 9
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

### First successful invocation

Use a selected vision specialist with a page image that the Lambda can read
from the deployment's source bucket:

```bash
aws lambda invoke \
  --function-name <full-text-specialist-function-name> \
  --cli-binary-format raw-in-base64-out \
  --payload '{"session_id":"example-001","image_path":"s3://<source-bucket>/pages/page-1.png"}' \
  response.json

cat response.json
```

Success returns `statusCode` `200` and a body with `success: true`. The
specialist result includes an `S3_RESULT_URI` marker identifying the retained
artifact. This is a page-level specialist call; use the PDF/image intake tools
when your application needs to create page images from a source document.

## Before you start

Install and configure AWS CLI, CDK v2, Docker, Python 3.12+, and `uv`. The
deployment menu verifies the local toolchain, target account/region, Bedrock
model access, and CDK readiness before it creates resources.

Specialist selections are stored as desired state under the chosen deployment
ID. Change them in place with `s` in the menu (Space toggles, Enter saves), or
use `--pack` / `--specialist` for automation. The next deploy reconciles the
same stack identity and shows whether its saved desired and applied selections
are synchronized.

`./deploy.sh configure --ui` offers the same selection and pack management in a
local browser planner. The Specialists profile has no workflow runner, so the
planner intentionally omits workflow routing for this product. It cannot
deploy; use its CLI handoff to apply the reviewed desired state.

## Choose or create packs

Pack definitions are declarative, ordered selections of enabled specialists in
[`deployment/profiles/`](../deployment/profiles/README.md). The provided packs
are:

- `document-essentials` — PDF conversion, classification, text, layout, and
  element analysis.
- `tables-and-forms` — table analysis, composable with Document Essentials.

Use `k` in the deployment menu to create, edit, or delete reusable packs.
Deletion is blocked while a saved deployment selects the pack, so no existing
deployment is silently left with invalid desired state. Shared prompts and
specialist configuration are released together in Step 3, so a function has a
stable configuration release rather than mutable configuration at invocation time.

Native DOCX/PPTX/XLSX processors are container-based. Select one in the
specialist picker, add it to a reusable pack, or pass `--specialist`. The
deployment workflow automatically creates or updates the ECR dependency,
builds the selected image, and updates the Lambda stack under the same
deployment ID. The optional `xlsx_processor` is a normal direct-specialist
tool: it combines safe native inspection and any eligible embedded-raster
Vision enrichment into one canonical result.

Step 3's configuration SHA is an immutable private deployment artifact, not a
Git product version. Source releases are Git tags; see the [Release
Guide](../RELEASE_GUIDE.md#release-terminology) for the lifecycle.

## Go deeper

| Topic | Guide |
| --- | --- |
| Follow specialist invocation, input, output, and error behavior | [Lambda specialists](../deployment/lambdas/LAMBDA_SPECIALISTS.md) |
| Understand the shared runtime framework | [Foundation layer](../deployment/lambdas/FOUNDATION_LAYER_README.md) |
| Select profiles and compose specialist packs | [Deployment profiles](../deployment/profiles/README.md) |
| Use the local pack and specialist planner | [Deployment reference](../deployment/DEPLOYMENT_README.md#configure-packs-and-routing-locally) |
| Author prompts and specialist manifests | [Prompting system](../deployment/s3_files/prompts/PROMPTING_SYSTEM_README.md) |
| Consume structured document output | [Schema and vocabulary](../schemas/VOCABULARY.md) |
| Deploy individual stacks, resume a deployment, or clean up | [Deployment reference](../deployment/DEPLOYMENT_README.md) |
