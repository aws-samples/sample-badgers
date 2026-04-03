# Deployment Scripts

## deploy_analyzer.sh

Deploys a single analyzer without redeploying the entire BADGERS stack. Interactive arrow-key picker lists all analyzers from `lambdas/code/`, shows which supporting files exist (manifest, schema, prompts), validates completeness, then runs three steps:

1. Uploads that analyzer's S3 files (prompts, manifest, schema) to the config bucket
2. Deploys `badgers-lambda` stack (creates the new Lambda function)
3. Deploys `badgers-gateway` stack (wires the Lambda as a gateway target)

Uses `--exclusively` to avoid cascading into unrelated stacks.

```bash
cd deployment
./deploy_analyzer.sh
```

## deploy_from_scratch.sh

Full deployment of all BADGERS CDK stacks from scratch. Builds layers, bootstraps CDK, deploys all stacks in dependency order, configures gateway observability, and builds the runtime container.

## resume_deploy.sh

Resumes a deployment from any step. Pass a deployment ID and optional step number.

## sync_s3_files.sh

Syncs the entire `s3_files/` directory to the deployed S3 config bucket.

## deploy_custom_analyzers.sh

Syncs custom analyzers from S3 and deploys the `badgers-custom-analyzers` CDK stack.
