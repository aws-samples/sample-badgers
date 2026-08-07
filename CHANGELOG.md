# Changelog

## [4.0.0] - 2026-08-07
### Changed
- **Version numbering is realigned across the repository.** `pyproject.toml` had drifted
  to `1.2.0` while this file had reached `2.5.0`, and the two moved independently after
  that — the branch carried `pyproject` `2.0.0` against a changelog claiming `3.1.0`.
  Both now read `4.0.0`. The 2.x range was consumed between February and April 2026 and
  `[2.0.0]` already denotes the February remediation container release, so this release
  moves forward to a free major rather than reusing a number.
  - `3.0.0` and `3.1.0` were never merged to `mainline`; they ship for the first time
    alongside this entry and are left intact, since they document the specialist rename,
    job tracking, Cognito OIDC/PKCE, and the ECS Express Gateway migration in detail.
- `cdk-nag` is capped below 3.x. 3.x changed the Aspect API and fails at synth against
  aws-cdk-lib 2.x with `aspectApplication.aspect.visit is not a function`.
- Raised `mcp>=1.28.1`, `opencv-python>=4.13.0.92`, and `pydantic>=2.13.4`.

### Security
- **Merged the outstanding dependabot backlog from `mainline`.** pypdf 6.14.2,
  bedrock-agentcore 1.18.1, setuptools 83.0.0, pillow 12.3.0, cryptography 50.0.0,
  mcp 1.29.0, and the npm advisories covering vite, postcss, ip-address, body-parser,
  shell-quote, and concurrently. `pip-audit` and `npm audit` both report no known
  vulnerabilities.
- **The `torch` advisory is resolved by removing the dependency rather than pinning it.**
  `torch`, `torchvision`, `realesrgan`, `basicsr`, `spandrel`, and `super-image` are
  dropped along with the upscaling stack they supported.
- **npm fixes that landed on `mainline` under `local_testing/` are carried into `ui/`.**
  That directory was consolidated into `ui/` in 3.0.0, so the upstream edits applied to
  files this branch no longer has. Resolving the conflict as a deletion alone would have
  discarded the fixes; the equivalent floors were raised in `ui/package.json` instead, so
  a satisfied version cannot regress.

### Removed
- Duplicate `constructs` and `jpype1` entries in the `pyproject.toml` dependency list.

## [3.1.0] - 2026-08-04
### Changed
- **BREAKING: stack names now include the `DEPLOYMENT_ID`.** `BADGERS-{Name}-{suffix}`
  becomes `BADGERS-{Name}-{DEPLOYMENT_ID}-{suffix}`, e.g. `BADGERS-S3-a1b` →
  `BADGERS-S3-dev-a1b`. Stacks deployed under the previous naming cannot be adopted or
  updated in place and must be torn down and redeployed.
  - The suffix alone was enough for uniqueness but not for identity: with suffix-only
    names a mistyped `DEPLOYMENT_ID` still resolved real stacks, while every resource name
    derived from it pointed at something that did not exist. In one case that combination
    scheduled a live deployment's in-use KMS key for deletion while reporting success.
  - Tooling now reads a deployment's identity off the stack name, which removed the need
    to recover it by parsing a bucket name out of a stack output.
- **`deploy.sh` no longer reads `DEPLOYMENT_ID` from the environment.** It unsets the
  variable and always chooses interactively, listing every deployment in `.deploy-state/`
  — complete and in progress — newest activity first, with an option to start a new one.
  A value left exported in the shell could otherwise target another deployment silently.
  New ids are validated against `^[a-z][a-z0-9-]{0,15}$`, the rule S3 bucket naming imposes.
- **`destroy.sh` chooses its target interactively, discovered from CloudFormation** rather
  than from `.deploy-state/`, because a state file can be deleted while the stacks are
  still live. An explicitly passed `DEPLOYMENT_ID` is validated and checked against what
  exists, and refused if nothing matches.

### Added
- **Option 12, Resume** — runs only outstanding steps, skipping completed ones *before*
  calling them so their "already complete, re-run?" prompts never fire. A step counts as
  complete only when every state key it writes is set, so a partially finished step is
  re-entered rather than skipped.
- **A network-exposure prompt in step 8.** ECS Express Mode derives the load balancer
  scheme from the subnets it is given, and the first Express service in a VPC fixes that
  scheme for the VPC. The prompt is asked up front for options 9 and 12 so the rest of the
  run is unattended. `UI_PUBLIC_ACCESS` answers it without prompting; default is public.
  Previously the UI was always placed on private subnets, which produced an internal load
  balancer whose public URL resolved but never answered.
- **`UI_PUBLIC_ACCESS` and `vpc_stack.public_subnet_ids`**, matching the media-contracts
  toggle. Defaults to `true`.
- **A full manual console teardown procedure** in the deployment guide: the two resources
  that must be removed before any stack, the 14-stack deletion order with per-stack notes,
  why parallel deletion fails, `DELETE_FAILED` remediation, and the resources no teardown
  removes.

### Fixed
- **Step 8 now forces the image rollout.** The ECS stack pins a static image tag, so
  pushing a new image to that tag left the template unchanged, `cdk deploy` reported no
  changes, and the service kept serving the old image. Step 8 now calls
  `update-express-gateway-service` with the container spec after the `cdk deploy`, as
  media-contracts does, then polls `rolloutState`.
- **The UI task role was missing `bedrock-agentcore:InvokeAgentRuntimeWithWebSocketStream`.**
  The presigned WebSocket path authorizes against a different action than
  `InvokeAgentRuntime`, so chat failed with a 403 at the handshake — before reaching the
  container, which is why nothing appeared in its logs. The permission was inherited from
  media-contracts, which invokes over HTTP.
- **`/api/env` is exempt from the `/api/` auth middleware.** It is the ALB health check
  path and the SPA reads it for branding before login, so a blanket `requireAuth` mount
  made every health check 401 and the task was killed for failing them. Media-contracts
  applies auth per route and leaves this endpoint open; BADGERS took the health check path
  without the exemption.
- **`destroy.sh` used `delete-service` on an Express Gateway Service**, which rejects it
  outright. It now uses `delete-express-gateway-service`, which also removes the load
  balancer the service created.
- **`destroy.sh` printed a success banner over a failed teardown.** It now re-verifies
  after the VPC auto-fix and, if stacks remain, lists them, states that the KMS key was
  left alone, and exits non-zero.
- **KMS key deletion is gated on the teardown succeeding.** It was previously scheduled
  even when `cdk destroy` had failed and every stack was still standing.
- **The X-Ray decision is resolved before every `cdk deploy`,** not just in step 2.
  `RuntimeWebSocket` depends on the XRay stack and `cdk deploy` includes a stack's
  dependencies, so deploying the Runtime or ECS stack directly reintroduced the stack and
  attempted a CloudWatch Logs resource policy against a non-adjustable quota of 10.
  `preflight_xray` also no longer resets an explicitly set `BADGERS_SKIP_XRAY`.
- **`build_container_lambdas.sh` was called without its required argument** in step 4, and
  needed the composite `{id}-{suffix}` to derive the right ECR repository.
- **The menu reported `exit 0` for failed steps**, having read `$?` after an intervening
  command. It now captures the step's status directly.
- **Step 8 marked itself complete on a rollout timeout.** The poll loop could exhaust its
  window without ever reaching `COMPLETED` and still record success.
- **A completed deployment was unreachable from the deploy menu.** The chooser filtered to
  unfinished deployments, and starting a "new" one refuses an id that already has state, so
  there was no way to re-run a single step against a finished deployment.

## [3.0.0] - 2026-07-30
### Changed
- **BREAKING: stack and resource names now follow the media-contracts (MC) convention.**
  Existing `badgers-*` deployments cannot be adopted or updated in place — they must be
  torn down and redeployed under a `DEPLOYMENT_ID`.
  - Stack names are `BADGERS-{Name}-{suffix}`, e.g. `badgers-s3` → `BADGERS-S3-a1b`
  - Resource names carry `{DEPLOYMENT_ID}-{suffix}`, e.g. `badgers-config-dev-a1b`, and
    SSM parameters move under `/badgers-dev-a1b/`
  - `app.py` now requires `DEPLOYMENT_ID` and `STACK_SUFFIX` (env or CDK context) and
    fails fast with usage rather than generating a random id, which previously made it
    easy to deploy a second copy of everything by accident
  - Because both parts are unique per deployment, several deployments can now coexist in
    one account and region
  - The Gateway stack takes the Cognito stack name as a parameter instead of hardcoding
    `badgers-cognito-UserPoolId` / `-UserPoolClientId`
  - AgentCore runtime and memory names normalise the composite id to underscores, since
    their names must match `[a-zA-Z][a-zA-Z0-9_]{0,47}` and cannot contain hyphens
- **Deployment is now driven by `./deploy.sh` and `./destroy.sh` at the repository root**,
  replacing `deploy_from_scratch.sh`, `resume_deploy.sh` and `deployment/destroy.sh`
  - `deploy.sh` is an interactive menu with eight steps, also runnable non-interactively
    as `./deploy.sh N`. Step completion and timestamps are recorded in
    `.deploy-state/{DEPLOYMENT_ID}.json`, so a re-run after a failure resumes instead of
    starting over — this replaces the separate resume script, which took a step number
    and had no record of what had actually completed
  - `BADGERS_ASSUME_YES=1` answers every confirmation, for use without a terminal
  - `destroy.sh` requires typing the `DEPLOYMENT_ID`, empties the buckets, and deletes the
    ECS service and AgentCore runtime *before* the VPC, polling until their ENIs release.
    CloudFormation cannot delete a VPC while an ENI is attached, and both release theirs
    asynchronously after reporting gone — the previous script did neither and would strand
    the VPC. It also schedules the KMS key for deletion on a 7-day window so the alias is
    freed for redeployment, verifies deletion, auto-fixes a `DELETE_FAILED` VPC via
    `--retain-resources`, and offers `--vpc-cleanup-only` for a stranded VPC
  - New `deployment/scripts/common.sh` holds the shared logging, state, stack-name and CDK
    helpers that every script sources

### Fixed
- The admin Stacks tab listed a `frontend` stack that no longer exists, omitted
  `dynamodb`, `ecs` and `custom-specialists`, and described the VPC as serving an ALB and
  Fargate. It is now suffix-aware, receiving `DEPLOYMENT_ID` and `STACK_SUFFIX` from the
  ECS container environment.
- The UI's Deploy All button called `/api/deploy-test`, an endpoint that never existed, so
  it always returned 404. It now calls `/api/deploy-all`.
- `/api/stacks/:stackId/outputs` returned a 404 message that interpolated a function
  rather than the stack name, and the deploy/destroy routes accepted arbitrary stack ids.
  Both now validate against the known stack list.
- `s3_stack` published a global `/badgers/config-bucket-name` SSM parameter for the agent
  runtime to discover. With per-deployment naming a second deployment would overwrite it
  and point the first deployment's agent at the wrong bucket. The parameter is gone and
  the runtime receives `CONFIG_BUCKET_NAME` directly as a container environment variable.
- `sync_custom_specialists.sh` read `../frontend/.env`, a path removed in 2.5.0.

### Changed
- **BREAKING: "analyzer" is now "specialist" across the entire codebase.** Aligns BADGERS
  with the media-contracts (MC) reference implementation. Case-aware rename across 189
  tracked text files plus 306 path renames:
  - 22 Lambda specialist directories under `deployment/lambdas/code/`
  - The remediation container directory and its utils
  - Prompt, manifest, and schema directories and files under `s3_files/`
  - The `custom_specialists` tree, including `specialist_registry.json`
  - Manifest key `analyzer` → `specialist`; `metadata.analyzer_type` → `metadata.specialist_type`
  - Env var `ANALYZER_NAME` → `SPECIALIST_NAME`
  - `AnalyzerFoundation` → `SpecialistFoundation` (`analyzer_foundation.py` → `specialist_foundation.py`)
  - Route `/api/analyzers` → `/api/specialists`
  - Scripts `deploy_analyzer.sh` → `deploy_specialist.sh`, `deploy_custom_analyzers.sh` → `deploy_custom_specialists.sh`
  - Stack `custom_analyzers_stack.py` → `custom_specialists_stack.py`
  - Doc `LAMBDA_ANALYZERS.md` → `LAMBDA_SPECIALISTS.md`
- UI moved from `local_testing/` and `deployment/ui/` into a single `ui/` directory,
  serving both the testing workbench and the deployment console from one Express server
  (port 7860, Vite dev server on 5175)
- UI infrastructure adopts the MC pattern: the self-managed ALB + Fargate + ACM + Route 53
  `badgers-frontend` stack is replaced by `badgers-ecs`, an ECS Express Gateway service that
  provisions and manages its own load balancer and HTTPS endpoint. No hosted zone, domain,
  or ACM certificate is required.
- All UI container configuration now comes from SSM Parameter Store via `valueFrom`
  references — no plaintext configuration in the service definition or deploy scripts
- Specialists now fail loudly when an analysis result cannot be persisted, instead of
  returning a result the caller cannot retrieve
- Fixed a latent suffix-strip bug surfaced by the rename: the Gateway target builder
  stripped a hardcoded 9 characters (`_analyzer`), which would have truncated names once the
  suffix became the 11-character `_specialist`

### Added
- **Job tracking** — a `doc_id` → `job_id` → `subtask_id` hierarchy recorded in DynamoDB
  - New `badgers-dynamodb` stack: single jobs table, on-demand billing, 30-day TTL,
    point-in-time recovery, plus `status-index` and `doc-index` GSIs
  - New `foundation/job_state.py`, the single writer for job records. Never raises, and
    no-ops entirely when `JOBS_TABLE_NAME` is unset, so tracking cannot break an analysis
    and stays opt-in per deployment
  - All 26 specialist Lambdas record their own subtask state: `PENDING → RUNNING →
    COMPLETE | FAILED`, with the S3 output key on success and the failure reason on error
  - Subtask keys are `{specialist}#{image_identifier}`, deterministic so the per-page fan-out
    stays unique and retries are idempotent
  - `job_id` and `doc_id` declared as optional parameters on all 26 tool schemas
  - The Runtime agent mints `job_id` on the first specialist call of a turn and stamps
    `job_id`/`doc_id` into tool inputs via a Strands `BeforeToolCallEvent` hook, rather than
    asking the model to carry identifiers
  - `doc_id` minted per upload and threaded from the browser through `/api/chat` to the agent
  - `GET /api/jobs/:jobId` and `GET /api/jobs?doc_id=` — job status is computed at read time
    by aggregating subtasks, so a run with 9 of 10 pages succeeding reports `PARTIAL`
- Cognito OIDC + PKCE authentication for the UI, replacing ALB-injected `x-amzn-oidc-data`
  headers, which the ECS Express Gateway service does not provide
  - One user pool with two app clients: a public UI client (`openid email profile`) and the
    existing machine-to-machine Gateway client
  - Managed Login v2 with explicit branding, plus `admin` and `tester` groups
  - Tokens verified against the user pool JWKS in `ui/server/auth.js`; every `/api/*` route
    sits behind `requireAuth`
  - `deployment/scripts/generate_ui_env.sh` writes the build-time `VITE_*` values
- New `badgers-vpc` stack: private subnets, VPC Flow Logs, and S3/DynamoDB gateway plus
  Bedrock/SSM/Secrets Manager interface endpoints for the UI service
- New `badgers-xray` stack enabling X-Ray Transaction Search for AgentCore tracing

### Fixed
- Documentation updated for the rename, the new infrastructure, and job tracking. Removes
  the stale `badgers-frontend` / `frontend_config.json` deployment instructions, corrects
  the stack list from 10 to 13 (plus 1 optional), drops the nonexistent
  `accessibility_specialist` from the README specialist table, adds the missing
  `handwriting_math_specialist`, corrects the tool count from 25 to 26, and documents the
  teardown order the UI's VPC exports require

### Removed
- `deploy_frontend.sh`, `destroy_frontend.sh`, and `frontend_config.example.json` —
  leftovers from the previous self-managed ALB + Fargate design, all referencing the
  deleted `badgers-frontend` stack. `deploy_frontend.sh` failed outright;
  `destroy_frontend.sh` was worse, skipping the missing stack and then failing to delete
  `badgers-vpc` (whose exports `badgers-ecs` consumes) while swallowing the error and
  reporting success. Deploy and tear down `badgers-vpc` / `badgers-ecs` directly.
- The `deployment/frontend_config.json` gitignore entry, now that nothing reads it

### Fixed
- `update_frontend_env.sh` wrote to `../frontend/config/.env`, a directory removed back in
  2.5.0. With `set -e` it exited 1 on every run, and both `deploy_from_scratch.sh` and
  `resume_deploy.sh` swallowed the failure with a warning. Now targets `ui/config/.env`
  and also writes `JOBS_TABLE_NAME`, without which the local `/api/jobs` endpoints
  return 503.

### Known Issues
- The specialist creation wizard (`/api/wizard/generate`) is a stub and does not emit
  `job_id`/`doc_id` on generated schemas, so custom specialists skip job tracking.
- `build_container_lambdas.sh` copies the foundation module from the generated `layer/`
  directory, so `build_foundation_layer.sh` must run first or container images ship stale
  foundation code.
- `deploy_from_scratch.sh` and `destroy.sh` do not name `badgers-vpc`, `badgers-ecs`, or
  `badgers-dynamodb`. Deploy resolves the jobs table automatically as a dependency of
  `badgers-iam`, but `destroy.sh` uses `--exclusively` and leaves all three standing.

## [2.5.0] - 2026-04-03
### Added
- Local Testing UI (`local_testing/`) — React + Express app replacing the Gradio frontend
  - 8-tab interface: Home, Chat, Edit Analyzer, Create Analyzer, Evaluations, Pricing, Observability, Chat Log
  - WebSocket proxy to AgentCore Runtime for interactive chat
  - SSE streaming for long-running operations
  - Vite dev server (port 5174) + Express API (port 3457)
- Deployment UI (`deployment/ui/`) — React + Express app for CDK stack management
  - 4-tab interface: Stacks, Analyzers, S3 Configs, Deploy Tags
  - Deploy/destroy individual CDK stacks with streaming log output
  - S3 config file editor for manifests, prompts, and schemas
  - Vite dev server (port 5173) + Express API (port 3456)
- `ui/UI_README.md` — documentation for the BADGERS UI

### Removed
- Gradio-based frontend (`frontend/`) replaced by `local_testing/` and `deployment/ui/`

### Changed
- Updated navigation bars across all 10 README files to replace `frontend/` links with Local Testing and Deployment UI links
- Updated inline references to Analyzer Creation Wizard in deployment and Lambda analyzer docs
- Updated project structure section in main README to reflect `local_testing/` replacing `frontend/`

## [2.4.1] - 2026-03-28
### Changed
- Increased all timeout configurations from 300s to 900s to support 10+ minute agent runs (#42)
  - Lambda function timeouts (Duration.seconds), BEDROCK_READ_TIMEOUT env var, MCP server timeout
  - Bedrock client connect_timeout raised from 10s to 30s
  - WebSocket ping_timeout raised to 90s, close_timeout to 30s for long-running stability
  - Frontend AGENTCORE_READ_TIMEOUT default raised to 900s
  - Prompt generator boto3 read_timeout raised to 900s
- Added operating environment configuration value for agent context (#47)

## [2.4.0] - 2026-03-28
### Added
- Poppler-qpdf Lambda layer with fontconfig for improved PDF text extraction (#40)
- PDF syntax repair pre-processing step in remediation pipeline (#25)
- Configurable `RESOLVER_MAX_TOKENS` env var for remediation analyzer
- New pricing models and analyzer defaults in pricing calculator

### Fixed
- Path injection vulnerabilities in chat download functions (CodeQL py/path-injection)

### Security
- Remediated CodeQL alerts #11, #12, #18, #19 — path traversal in `agent_chat_websocket.py`

### Dependencies
- Bumped requests from 2.32.5 to 2.33.0
- Bumped pypdf from 6.8.0 to 6.9.2
- Bumped pymupdf from 1.26.6 to 1.26.7
- Bumped pyjwt from 2.10.1 to 2.12.0

## [2.3.0] - 2026-03-12
### Added
- Dynamic token estimation based on image complexity (#15)
- Complexity scorer using text ratio, entropy, edge density, color std
- Token usage vs budget logging for calibration (#15)
- Dynamic token toggle checkbox in Gradio chat UI
- Sonnet 4.6 application inference profile for image enhancer
- Dynamic token estimation docs in README and Lambda Analyzers docs

### Changed
- Container stack adjustments
- 21 Lambda handlers updated to support dynamic tokens env var from request payload

### Dependencies
- Bumped pypdf from 6.7.1 to 6.7.5
- Bumped gradio from 6.3.0 to 6.7.0

## [2.2.0] - 2026-02-24
### Added
- Cell grid resolver v3 for remediation analyzer with improved table detection
- Diagnostic visualizer for remediation analyzer output inspection
- `ENABLE_DIAGNOSTICS` environment variable for remediation analyzer Lambda
- Claude Opus 4.6 inference profile support

### Changed
- Remediation analyzer README moved to `REMEDIATION_README.md`
- Updated README analyzer count from 29 to 25 (accurate Lambda function count)
- Updated remediation analyzer description to reflect container architecture and new capabilities

### Fixed
- Increased font size in remediation analyzer for improved analysis
- CDK IAM policies and manifest schema for remediation analyzer
- Remediation analyzer credential threading, image sizing, and CJK font encoding

## [2.1.0] - 2026-02-24
### Added
- Acrobat accessibility report and screen reader video for remediation analyzer
- Updated README to v2.1

### Changed
- Image enhancement tool updates

## [2.0.0] - 2026-02-23
### Added
- Remediation analyzer v2.0 with container + layer architecture (moved from code-based to ECR container)
- PDF accessibility auditor, tagger, and models modules
- Container build script and Dockerfile for remediation analyzer

### Fixed
- Remediation analyzer container missing required Python modules and dependencies (#9)

## [1.2.0] - 2026-02-18
### Fixed
- Hard coded klayers and Pillow ARN regions now uses `Stack.of(self).region` (#8)

## [1.1.0] - 2026-02-11
### Changed
- PDF remediation adjustments
- Initial codebase clean-up

### Dependencies
- Bumped Pillow from 11.3.0 to 12.1.1

## [1.0.0] - 2026-02-03
### Added
- Initial commit with 25 Lambda analyzer functions (23 code-based + 2 container-based)
- Strands Agent with AgentCore Runtime and Gateway
- CDK deployment (10 CloudFormation stacks)
- Multi-page Gradio frontend with chat, wizard, editor
- Foundation layer shared across all analyzers
- Modular XML prompting system
- Inference profiles for cost tracking
