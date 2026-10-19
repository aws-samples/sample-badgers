# BADGERS Full

The complete BADGERS application: agent, Gateway, every enabled specialist,
browser UI, session memory, operation ledger, and the specialist-creation
workflow.

```bash
./badgers-full/deploy.sh
```

Choose **9** from the menu to deploy all Full components, or run the same
command later and choose **`r`** to resume unfinished work.

## What gets deployed

- AgentCore Runtime and MCP Gateway
- All specialists enabled in `deployment/deployment_config.json`
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

## Configuration and specialist changes

Prompts, manifests, schemas, and agent configuration are uploaded as one
immutable, content-addressed release. Changing one of them means running Step
3, then redeploying the affected consumers so the agent, Gateway, and
specialists advance to the new release together.

For individual stack operations, cleanup, and troubleshooting, see the
[deployment reference](../deployment/DEPLOYMENT_README.md). For the capability
definition behind this product, see [deployment profiles](../deployment/profiles/README.md).
