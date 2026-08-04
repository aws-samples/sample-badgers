# Deployment Scripts

The two entry points live at the repository root. Everything under `deployment/` is a
helper invoked either by them or directly for targeted work.

All scripts resolve stack names through `deployment/scripts/common.sh`, which reads
`STACK_SUFFIX` from `.deploy-state/{DEPLOYMENT_ID}.json`. They therefore all need
`DEPLOYMENT_ID` set, and `deploy.sh` must have run at least once to create that file.

## deploy.sh (repo root)

Interactive deployment menu, resumable and idempotent. Step completion is recorded in
`.deploy-state/{DEPLOYMENT_ID}.json`, so re-running after a failure continues from where
it stopped rather than starting over.

```bash
DEPLOYMENT_ID=dev ./deploy.sh        # menu
DEPLOYMENT_ID=dev ./deploy.sh 9      # full deployment, non-interactive
DEPLOYMENT_ID=dev ./deploy.sh 10     # status only
DEPLOYMENT_ID=dev ./deploy.sh 11     # reset state (keeps the suffix, deletes nothing in AWS)
```

Steps: 1 layers, 2 foundational infra, 3 upload config, 4 specialist Lambdas,
5 Gateway, 6 Runtime, 7 UI image, 8 UI ECS service.

Set `BADGERS_ASSUME_YES=1` to answer every confirmation with yes. This is required when
running without a terminal — the UI's Deploy All button relies on it, because its output
stream leaves stdin closed and an interactive prompt would otherwise read EOF and skip
the step.

## destroy.sh (repo root)

Full teardown. Requires typing the `DEPLOYMENT_ID` to confirm.

```bash
DEPLOYMENT_ID=dev ./destroy.sh
DEPLOYMENT_ID=dev ./destroy.sh --vpc-cleanup-only
KMS_WAIT_DAYS=30 DEPLOYMENT_ID=dev ./destroy.sh
```

Order matters and the script enforces it: empty the buckets, delete the ECS service and
the AgentCore runtime and wait for their ENIs to release, then destroy the stacks in
reverse dependency order. CloudFormation cannot delete a VPC while an ENI is attached,
which is why the compute goes first. It then schedules the KMS key for deletion (7 days
by default, which frees the alias sooner than the 30-day default), verifies every stack
is gone, and if the VPC stack is `DELETE_FAILED` retries with `--retain-resources` and
cleans up what was retained.

`--vpc-cleanup-only` runs just the ENI sweep: deletes interface endpoints, then deletes
or force-detaches whatever ENIs remain. Use it when a previous teardown left a VPC behind.

## deploy_specialist.sh

Deploys a single specialist without redeploying everything. An arrow-key picker lists the
specialists under `lambdas/code/`, shows which supporting files exist (manifest, schema,
prompts), validates completeness, then:

1. Uploads that specialist's S3 files to the config bucket
2. Deploys the Lambda stack (creating the new function)
3. Deploys the Gateway stack (wiring the Lambda as a target)

Uses `--exclusively` to avoid cascading into unrelated stacks.

```bash
DEPLOYMENT_ID=dev ./deploy_specialist.sh
```

## deploy_specialist_container.sh

Builds and pushes a single container-based specialist image, then updates its function.

## deploy_custom_specialists.sh

Syncs custom specialists from S3 and deploys the CustomSpecialists stack. That stack only
exists when `custom_specialists/specialist_registry.json` is present.

## sync_s3_files.sh

Syncs the whole `s3_files/` directory to the deployed config bucket. Equivalent to step 3
of `deploy.sh`, useful on its own after editing a prompt.

## sync_custom_specialists.sh

Pulls wizard-created specialists down from S3 into `custom_specialists/` so the CDK app
can see them.

## scripts/generate_ui_env.sh

Writes `ui/.env` from the Cognito stack outputs. Vite only exposes `VITE_`-prefixed
variables and bakes them in at build time, so this must run after Cognito is deployed and
before the UI image is built. `deploy.sh` step 7 calls it.

## update_frontend_env.sh

Writes `ui/config/.env` for local development — bucket names, the Runtime ARN, the Gateway
ID and the jobs table name. Local convenience only: the deployed UI reads all of these
from SSM Parameter Store. It does not write the Cognito values, which are build-time
inputs (see `scripts/generate_ui_env.sh`).

## cleanup-stack.sh

Targeted recovery for a wedged AgentCore Runtime stack: deletes the runtime, then the
stack. `destroy.sh` does this as part of a full teardown; use this when only the runtime
stack is stuck.

## scripts/common.sh

Sourced by the scripts above rather than executed. Provides logging, the deployment state
file helpers, `_sn` for stack names, `resource_id` for resource names, CloudFormation
output lookups, and the CDK wrappers.
