# Changelog

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
