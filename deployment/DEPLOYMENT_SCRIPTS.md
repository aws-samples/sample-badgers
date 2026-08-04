# Deployment Scripts

The two entry points live at the repository root. Everything under `deployment/` is a
helper invoked either by them or directly for targeted work.

All scripts resolve stack names through `deployment/scripts/common.sh`, which composes
`BADGERS-{Name}-{DEPLOYMENT_ID}-{suffix}` and reads `STACK_SUFFIX` from
`.deploy-state/{DEPLOYMENT_ID}.json`.

The two root entry points choose the deployment interactively. Every helper under
`deployment/` needs `DEPLOYMENT_ID` set in the environment, and `deploy.sh` must have run at
least once to create the state file.

## deploy.sh (repo root)

Interactive deployment menu, resumable and idempotent. Step completion is recorded in
`.deploy-state/{DEPLOYMENT_ID}.json`, so re-running after a failure continues from where
it stopped rather than starting over.

```bash
./deploy.sh           # choose a deployment, then the menu
./deploy.sh 9         # choose a deployment, then run option 9 (full deployment)
./deploy.sh resume    # choose a deployment, then run only outstanding steps
```

**`DEPLOYMENT_ID` is unset on startup and never read from the environment.** The script
scans `.deploy-state/` and offers every deployment it finds — complete and in progress —
newest activity first, plus `n` to start a new one. A new id must match
`^[a-z][a-z0-9-]{0,15}$` and must not already have state; pick it from the list instead.

Steps: 1 layers, 2 foundational infra, 3 upload config, 4 specialist Lambdas,
5 Gateway, 6 Runtime, 7 UI image, 8 UI ECS service. Then 9 full deployment, 12 resume,
10 status, 11 reset state (keeps the suffix, deletes nothing in AWS), 0 exit.

**9 vs 12** — both reach a complete deployment. Option 9 runs all eight steps and stops at
each completed one to ask whether to re-run. Option 12 skips completed steps before calling
them, so those prompts never fire, and starts at the first outstanding step.

Behaviour worth knowing:

- **Step 8 failure clears `ui_image_pushed`.** An ECS rollout usually fails because of
  something in the image, and redeploying the stack alone will not pick up a code change,
  so resume rebuilds in step 7 first. When the cause was external, that rebuild is a no-op.
- **Step 8 forces the image rollout** with `update-express-gateway-service` after the
  `cdk deploy`, because the stack pins a static image tag and pushing to that tag leaves
  the template unchanged.
- **The X-Ray decision is resolved before every `cdk deploy`**, not just in step 2 —
  `RuntimeWebSocket` depends on the XRay stack and `cdk deploy` includes dependencies.

Environment variables:

| Variable                                         | Effect                                                                                                                                                                                                                                                                              |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BADGERS_ASSUME_YES`                             | `1` answers every confirmation with yes. Re-runs completed steps rather than skipping them — not a quiet resume. Required without a terminal: the UI's Deploy All button relies on it, because its output stream leaves stdin closed and a prompt would read EOF and skip the step. |
| `UI_PUBLIC_ACCESS`                               | `true`/`false` answers the step 8 network-exposure prompt without asking.                                                                                                                                                                                                           |
| `BADGERS_SKIP_XRAY`                              | `1` omits the XRay stack regardless of the live state.                                                                                                                                                                                                                              |
| `UI_CONTAINER_PORT`                              | Container port sent with the forced rollout. Default `7860`; must match `CONTAINER_PORT` in `stacks/ecs_stack.py`.                                                                                                                                                                  |
| `IMAGE_TAG`, `RUNTIME_IMAGE_TAG`, `UI_IMAGE_TAG` | Image tags. Default `latest`, `websocket`, `frontend`.                                                                                                                                                                                                                              |

## destroy.sh (repo root)

Full teardown. Requires typing the `DEPLOYMENT_ID` to confirm.

```bash
./destroy.sh                                                   # choose from what is deployed
DEPLOYMENT_ID=dev ./destroy.sh                                 # explicit, validated against AWS
KMS_WAIT_DAYS=30 ./destroy.sh                                  # longer KMS window
DEPLOYMENT_ID=dev STACK_SUFFIX=a1b ./destroy.sh --vpc-cleanup-only
```

With no `DEPLOYMENT_ID` it **discovers deployments from CloudFormation**, not from
`.deploy-state/` — a state file can be deleted while the stacks are still live, so the
stacks are authoritative. Identity is parsed out of the stack names, so it works even for a
partial deployment. Passing `DEPLOYMENT_ID` explicitly is validated against the naming rules
and against what exists; if nothing matches `BADGERS-*-{id}-{suffix}` it refuses rather than
running a teardown that does nothing.

Order matters and the script enforces it: empty the buckets, delete the ECS Express service
and the AgentCore runtime and wait for their ENIs to release, then destroy the stacks in
reverse dependency order. CloudFormation cannot delete a VPC while an ENI is attached, which
is why the compute goes first.

- The ECS service is deleted with **`delete-express-gateway-service`**. `delete-service`
  rejects it with *"has ResourceManagementType=ECS use DeleteExpressGatewayService"*.
- AgentCore calls target **`bedrock-agentcore-control`**. The `bedrock-agentcore` service
  only exposes `InvokeAgentRuntime`, so control-plane calls aimed there fail silently.
- If the VPC stack is `DELETE_FAILED` it retries with `--retain-resources`, sweeps ENIs
  again, then re-verifies.
- The KMS key is scheduled **only after** the stacks are confirmed gone (7 days by default,
  which frees the alias sooner than the 30-day maximum). Scheduling it after a failed
  teardown would mark a live deployment's in-use key for deletion.
- A teardown leaving stacks standing prints `❌ Teardown incomplete`, lists them, states
  that the KMS key was left alone, and **exits non-zero**.

`--vpc-cleanup-only` runs just the ENI sweep: deletes interface endpoints, then deletes
or force-detaches whatever ENIs remain. Use it when a previous teardown left a VPC behind.
It skips discovery, so it needs both `DEPLOYMENT_ID` and `STACK_SUFFIX`.

For tearing down by hand, see
[Manual Teardown in the Console](DEPLOYMENT_README.md#️-manual-teardown-in-the-console).

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
