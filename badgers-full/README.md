[← BADGERS home](../README.md)

# BADGERS Full

The complete BADGERS application: agent, Gateway, every enabled specialist,
browser UI, session memory, operation ledger, and the specialist-creation
workflow.

> [!IMPORTANT]
> On `release/6.0.0-beta`, this is an internal beta candidate, not a published
> customer release. Use the `v5.0.2` tag for current production deployments.
> See the [Release Guide](../RELEASE_GUIDE.md) before choosing a source branch
> or release tag.

```bash
./badgers-full/deploy.sh
```

Choose **9** from the menu to deploy all Full components, or run the same
command later and choose **`r`** to resume unfinished work.

## What gets deployed

- AgentCore Runtime and MCP Gateway
- All specialists enabled in `deployment/deployment_config.json`
- Native PDF, DOCX, and PPTX intake, with selective OCR and visual fallback
- Shared prompt/configuration release, model profiles, S3 storage, IAM, Cognito,
  DynamoDB operation ledger, and ECR
- Session memory and X-Ray Transaction Search when available
- VPC, ECS service, and the browser UI
- The UI specialist wizard for creating and managing custom specialists

## Before you start

Install and configure AWS CLI, CDK v2, Docker, Python 3.12+, and `uv`. Use an
AWS identity allowed to provision the listed services and invoke the required
Bedrock models. The deployment menu validates your local tools, AWS account,
region, model access, and CDK readiness before resource creation.

The UI is internal by default. If you explicitly choose public access during
deployment, that load-balancer topology is fixed for the life of the VPC.

## After deployment

Use the deployment status option to find the UI endpoint and Gateway details.
The UI provides document testing, reports, observability, and specialist
administration. Use option **`m`** to inspect the central model registry before
changing a model selection.

### First successful result

1. Open the UI endpoint reported by deployment status and sign in with a user
   provisioned for the deployment.
2. Upload a PDF, DOCX, PPTX, or supported image from the **Home** or **Chat**
   experience.
3. Submit an analysis instruction, then use the report view to inspect the
   page-level output and retained specialist artifacts.

If the deployment completed but you cannot sign in, start with the
[UI authentication guide](../ui/UI_README.md#authentication), not the
infrastructure teardown or runtime documentation.

## Configuration and specialist changes

Prompts, manifests, schemas, and agent configuration are uploaded as one
immutable, content-addressed release. Changing one of them means running Step
3, then redeploying the affected consumers so the agent, Gateway, and
specialists advance to the new release together.

This configuration release pins one deployment's runtime behavior. It is not a
Git source release or customer version: source releases are immutable Git tags,
while Step 3 creates the private S3 configuration SHA. The [Release
Guide](../RELEASE_GUIDE.md#release-terminology) explains the distinction.

## Go deeper

| Topic | Guide |
| --- | --- |
| Deploy individual stacks, resume a deployment, or clean up | [Deployment reference](../deployment/DEPLOYMENT_README.md) |
| Run the testing workbench and operations console | [UI guide](../ui/UI_README.md) |
| Understand the agent container and Gateway connection | [Runtime guide](../deployment/runtime/RUNTIME_README.md) |
| Follow document intake, specialist execution, and results | [Lambda specialists](../deployment/lambdas/LAMBDA_SPECIALISTS.md) |
| See the Full capability definition | [Deployment profiles](../deployment/profiles/README.md) |
| Choose the correct source tag and understand releases | [Release guide](../RELEASE_GUIDE.md) |
