# Changelog

## [Unreleased]

Everything on top of `[4.0.0]`: 15 commits, `f6ad01e` through `f352f2e`. None of it has
shipped as a version — `pyproject.toml` still reads `4.0.0`.

Two items are breaking for anyone scripting the deploy entry points or referencing the UI
env file; both are marked **BREAKING** below.

### Added

- **HTML document reports.** A new deterministic specialist, `html_report_specialist`,
  runs once per document after every page correlation and produces a durable report. It
  makes no model call and has no prompt files.
  - Tool `generate_html_report_tool`; new workflow **step 6 (REPORT)** in
    `agent_system_prompt.xml`, exempt from the per-page tool budget.
  - New S3 layout under the output bucket: `reports/{ownerKey}/{reportId}/` holding
    `report.html`, `manifest.json`, and `pages/page-{n}.jpg` / `.xml` per page.
    `ownerKey` is `local` in local development, otherwise the first 24 hex characters of
    `sha256(user_id)`.
  - Five endpoints: `GET /api/reports`, `/api/reports/:id/manifest`,
    `/api/reports/:id/pages/:n/image`, `/api/reports/:id/pages/:n/xml`, and
    `/api/reports/:id/download`.
  - New **`owner-index` GSI** on the jobs table (PK `owner_sub`, SK `started_at`,
    INCLUDE projection). Deliberately sparse: `owner_sub` is written only on the
    job-level `orchestrator` row, so subtask rows never appear.
  - New job-row attributes: `owner_sub`, `user_name`, and — once a report exists —
    `report_id`, `report_title`, `report_created_at`, `report_page_count`.
  - Identity now propagates end to end. The UI sends the caller's Cognito sub as
    `actor_id` and email as a new `user_name` payload field, replacing the hardcoded
    `local_testing_user`; `JobTrackingHook` stamps `user_id`/`user_name` into any tool
    whose schema declares them.
  - New **📚 Reports** tab with three views — Overview, Page Reader (image beside the
    correlated spine, with Rendered Spine / Raw XML / Specialists / Audit tabs and
    ← → keyboard navigation), and Audit Trail — plus a single-file HTML download that
    inlines the CSS, JS, and every page image as base64 so it works offline.
- **The specialist creation wizard works end to end.** Its three server endpoints were
  stubs returning `{prompts:{}}`, `{}`, and `{output:'Not yet wired'}`; they are now
  implemented in a new `ui/server/routes/wizard.js`, mounted by `mountWizardRoutes`.
  - `POST /api/wizard/generate` streams **SSE**, not JSON. Six sequential Bedrock calls
    (`gestalt`, `job_role`, `context`, `rules`, `tasks`, `format`) take minutes, which no
    plain POST survives behind the load balancer. The `start` frame ships the whole
    section list so the client keeps no second copy of it.
  - `POST /api/wizard/save` is **new** — it did not previously exist in any form. It
    writes every artifact under `deployment/custom_specialists/`, which
    `CustomSpecialistsStack` already reads. The built-in specialists in
    `deployment/s3_files/` are never touched.
  - `POST /api/wizard/deploy` also changed from JSON to SSE, streaming
    `deploy_custom_specialists.sh` into the shared log panel.
  - A failed section becomes an `<!-- ERROR ... -->` stub and is reported in `warnings`,
    so one bad generation does not lose the other five. All six failing is a hard error
    instead: that means credentials, region, or model access is broken, and six comment
    stubs would read as content.
  - Bedrock is reached by hand-signing an HTTPS Converse request with SigV4 rather than
    through `@aws-sdk/client-bedrock-runtime`, which is not a dependency of `ui/`. **No
    new npm dependency and no container-build change.**
  - Save and Deploy are separate buttons; Deploy stays disabled until a save succeeds.
- **Gateway observability, which previously produced nothing.** The Gateway had the IAM
  permissions and claimed full observability but had no delivery configured — the console
  showed "Log delivery (0)" and "Tracing: Not enabled". A new
  `deployment/stacks/log_delivery.py` provides `shared_delivery_policy`,
  `deliver_to_log_group`, and `deliver_traces_to_xray`. The Gateway stack now creates log
  group `/aws/bedrock-agentcore/gateways/{deployment_id}/app` (TWO_YEARS retention,
  `RETAIN`), delivers `APPLICATION_LOGS` to it, and delivers `TRACES` to X-Ray.
- **Deploy preflights.** `step_infra` now refuses to proceed unless the account/region is
  CDK-bootstrapped and service quotas have headroom: `preflight_bootstrap` and
  `preflight_service_quotas` (VPCs, internet gateways, S3 buckets, Cognito pools,
  DynamoDB tables, ECR repositories, 26 Lambda functions, 5 IAM roles, KMS keys). Steps 6
  and 7 additionally check QEMU binfmt for cross-architecture container builds via
  `preflight_docker_cross_platform`, installing `qemu-user-static` on Linux if absent.
- **A log-delivery preflight before the Gateway and Runtime steps.** A CloudWatch
  `DeliverySource` is unique per `(resourceArn, logType)`, not per name, and
  CloudFormation creates new resources before deleting removed ones — so both spellings
  coexist during an update, collide, and fail the deploy with `AlreadyExists`. Renaming
  cannot fix it. `preflight_log_delivery` reports what occupies the slot, explains that
  removing it deletes delivery configuration only, and clears it on confirmation.
- **Region is resolved rather than assumed.** `ensure_region` walks env
  (`AWS_REGION`, `AWS_DEFAULT_REGION`) → `aws configure get region` → prompt →
  `us-west-2` with a confirmation. An invalid region name is a hard failure.
- **Enhanced-image provenance in HTML reports.** Reports previously showed the page as
  scanned, because the correlation artifact names the *original* as its source — while
  most specialists had read the enhanced copy. `html_report_specialist` now persists the
  enhanced image to `{report_prefix}/pages/page-{token}-enhanced.jpg` and records
  `enhanced_image_key` per page; the Page Reader offers Original/Enhanced tabs when a
  second image exists. New endpoint
  `GET /api/reports/:reportId/pages/:pageNumber/enhanced-image`, returning 404 rather
  than an error when absent.
- **A `levels` enhancement operation** — the 13th. A Photoshop-style levels adjustment
  that auto-computes black and white points from the histogram by percentile, applies
  gamma, and blends back by intensity. Unlike `threshold` it preserves continuous tonal
  gradation, so ink stroke weight survives. It returns percentile analysis to the agent
  so intensity can be retuned on retry, and comes with a new **Pipeline F, "Historical
  Manuscript Restoration"** (`desaturate@1.0 → levels@0.5 → remove_stains@0.4`).
- **Enhancement document-type context is now operator-editable.** The enhancer reads
  `s3://{CONFIG_BUCKET}/config/document_type_contexts.json`, falling back to an in-code
  map, so enhancement behaviour can be tuned per document type without a container
  rebuild. Contexts went from a terse phrase to a paragraph of guidance each, and five
  types were added: `historical_manuscript`, `photograph`, `map`, `legal`, `newspaper`.
  New `deploy.sh` step-3 submenu option `6) Runtime Config` uploads it.
- **Live analyzer status in Chat.** A new right-hand "🧩 Analyzers" panel polls
  `/api/jobs/:jobId` every 2.5s while a turn runs and shows one pill per specialist with
  `complete/total` pages. It exists because the stream reports a tool *starting* but
  never finishing or failing, so real COMPLETE/FAILED can only come from the job rows.
  Pills report the status needing attention (FAILED before RUNNING before PENDING before
  COMPLETE) and carry the first failure's error in the tooltip.
- Live activity indicator in Chat, driven by the server's `status` events, which the UI
  previously discarded.
- Copy buttons on chat messages, extracted to a shared `CopyButton` component, plus Shiki
  syntax highlighting in the Reports Raw XML tab and the Evaluations Result Output pane.
  The Evaluations pane picks its grammar from the artifact's file extension rather than
  assuming XML.
- Attachment chips in the composer with a status line, and deferred upload — nothing
  reaches S3 until Send, so a removed or unsent attachment leaves no object behind.

### Changed

- **BREAKING — `deploy.sh` menu keys and option 9 semantics.** Resume moved from `12` to
  `r`/`R`/`resume`, and `./deploy.sh 12` is now rejected. Option `9` (Full Deployment) no
  longer re-runs completed steps or prompts on each one; it delegates to the same
  `run_remaining` path as resume, so it behaves like resume. Anyone who used option 9 to
  force a redeploy needs `11` (Reset Deployment State) first.
- **BREAKING — the UI env file moved and `update_frontend_env.sh` was deleted.**
  `ui/config/.env` → `ui/.env`, and `deployment/update_frontend_env.sh` is gone with its
  job folded into `deployment/scripts/generate_ui_env.sh`. An existing `ui/config/.env`
  is now silently ignored: the UI comes up with unset buckets and table name, and
  `/api/reports` returns 503, until `generate_ui_env.sh` is re-run.
- **Deploy behaviour: `step_gateway` and `step_runtime` now have an interactive gate.**
  Both call `preflight_log_delivery`, which prompts before deleting anything and stops
  the deploy on decline. It also stops when it cannot determine the answer, on the
  reasoning that "could not determine" is not "clear to proceed". **An unattended or CI
  invocation that previously ran clean can now block on the prompt.** New escape hatch
  `BADGERS_SKIP_LOG_DELIVERY_PREFLIGHT=1`; note `BADGERS_ASSUME_YES=1` answers the prompt
  too and therefore deletes the conflicting sources without asking.
- **`destroy.sh` now deletes local deploy state** (`.deploy-state/{id}.json` and the
  outputs cache) and resolves bucket names from stack outputs instead of hardcoding them.
  It also deletes the KMS **alias** after scheduling key deletion, so a redeploy with the
  same `DEPLOYMENT_ID` no longer waits out `KMS_WAIT_DAYS`. With nothing to destroy it
  now waits for a keypress rather than exiting, **which will hang a non-interactive run.**
- **Specialist Lambda memory raised 2048 MB → 6144 MB** via a new `SPECIALIST_MEMORY_MB`
  constant, applied to zip, container, and wizard-generated specialists. Lambda scales CPU
  with memory and specialists hold source plus processed page images. Faster pages, higher
  per-invocation cost.
- **`full_text` + `elements` are now mandatory on every page**, including visual-only and
  handwriting-only pages, so every page can be correlated into a canonical spine. More
  model invocations and more cost per document than before.
- **Default theme is now light, and the AWS Purple theme was removed.** A stored
  `badgers-theme` of `purple` is rewritten to `light` on load.
- Runtime images now carry a unique tag per deploy (`websocket-{UTC timestamp}` from
  `runtime_image_tag`, exposed as `RUNTIME_IMAGE_TAG`) alongside the floating `websocket`
  alias. See Fixed for why.
- The chat SSE stream sends a `: ping` comment every 15s. The load balancer closes any
  connection idle in either direction for 60s (`idle_timeout.timeout_seconds`, the AWS
  default, which `CfnExpressGatewayService` does not expose) and this stream writes
  nothing for the whole of a tool call — `full_text` has taken 115s on a dense page.
- `deploy_specialist.sh` no longer requires a `prompts/` directory; it reads the manifest
  and only uploads prompts when `specialist.prompt_files` is non-empty. This is what lets
  a promptless specialist such as `html_report_specialist` deploy.
- The text an attachment sends to the agent changed from `Uploaded file: <s3 uri>` to
  **`Process: <s3 uri>`**. The agent parses this, so it is a contract between UI and
  prompt.
- `foundation/__init__.py` resolves submodules lazily (PEP 562) instead of eagerly
  re-exporting them. Source-compatible for every documented name, but anything relying on
  `import foundation` pre-importing `foundation.image_processor` as a side effect breaks.
- The `aws` and `python3` CLI calls in `common.sh` are now shell-function wrappers that
  strip carriage returns, for Windows VDI / Git Bash hosts where captured values silently
  gained a trailing `\r`.
- `@assistant-ui/react` `^0.12.21` → `^0.15.8`, adding `@assistant-ui/core`, `store`, and
  `tap`. The hook `useMessage()` was replaced by `useAuiState()`.
- `markitdown` and its dependency tree (`magika`, `onnxruntime`, `mammoth`, `openpyxl`,
  `pdfminer-six`, Azure Document Intelligence clients) added to the lock file.
- `pypdf` `6.14.2` → `6.15.0`; `playwright>=1.62.0` and `pillow>=12.3.0` added to the
  `dev` dependency group. `version` remains `4.0.0`.

### Fixed

- **Job tracking was silently disabled in every deployed environment.**
  `foundation/__init__.py` eagerly re-exported every submodule, so
  `from foundation import job_state` in the orchestrator dragged in `image_processor`,
  which imports Pillow at module scope — and Pillow is deliberately absent from the agent
  container. The caller catches `ImportError` to degrade gracefully, so **no `job_id` was
  ever minted** and every specialist requiring one, including
  `html_report_specialist`, failed. The import error is now captured and named in the
  warning rather than blamed on the build script.
- **Rebuilt agent images never went live.** AgentCore runtime versions are immutable and
  `CfnRuntime` only cuts a new version when `container_uri` changes; with a fixed
  `:websocket` tag, re-pushing left the CloudFormation property identical, so no update
  happened and the runtime kept serving the previously pinned digest.
- **`No module named 'config'` cold-start failure** in container Lambdas.
  `build_container_lambdas.sh` skipped staging `layer/python/config` into the
  `remediation_specialist` build context, and neither it nor `image_enhancer` copied it in
  their Dockerfile.
- **The quota preflight aborted the deploy in any account large enough to paginate.** The
  AWS CLI applies a `--query length(...)` per page when auto-paginating, so
  `lambda list-functions` in an account with 81 functions returned `"50\n31"`. Bash
  arithmetic on a multi-line value is a syntax error, and under `set -euo pipefail` with
  an `ERR` trap that aborted the whole run with a bare arithmetic error rather than any
  quota message. Counts are now summed with `awk`. Operators in large accounts who never
  saw quota checks will now see them — and may be legitimately blocked by real headroom
  failures the crash was masking.
- **`/api/chat` crashed with `EACCES` in the container.** The server is copied to
  `/app/server` with no `ui/` prefix, so `PROJECT_ROOT` resolved to `/` and the default
  log path landed on the read-only filesystem root. New `CHAT_LOGS_DIR` override.
- **A new chat session threw `ENOENT`** — `realpathSync` ran on the log file, which does
  not exist yet for a new session. It now resolves the directory, which exists after
  `mkdir`.
- **Assistant messages stopped rendering** after the assistant-ui upgrade:
  `useAuiState()` was called without a selector, so it returned the whole state rather
  than the message. Now `useAuiState((s) => s.message)`.
- **Gateway emitted no logs and no traces at all** — see Added.
- **Markdown tables rendered as one run-together line of pipes.** react-markdown is
  CommonMark-only, where tables do not exist; `remark-gfm` is now wired in, bringing
  strikethrough and autolinks with it.
- **Attachment-only messages rendered as an empty bubble and sent an empty string.** The
  composer keeps typed text and attachment parts in separate arrays and never merges
  them; a new `collectUserText` merges both for rendering and for the request body.
- Upload failures surfaced as an unhelpful JSON parse error when a proxy or crash
  returned non-JSON. Non-2xx and missing-`s3Uri` cases are now reported distinctly.
- The activity indicator could stick on after an abort or a thrown error; it is now
  cleared in a `finally`.
- **"New Session" in Chat did not reset the session.** It swapped the session id in
  place, leaving the thread, the composer's attachments, the activity indicator, the
  analyzer pills, and `S3AttachmentAdapter.lastDocId` intact. `lastDocId` is the one that
  mattered: carrying it into a new session attributed the next job to the *previous*
  document — the `doc_id` mismatch that breaks report generation. The button now remounts
  the component, since `useLocalRuntime` owns the message store and no single call clears
  all of it.
- `job_state.get_job_records` now reads with `ConsistentRead=True`; the report Lambda
  reads job rows immediately after the specialists wrote them, and an eventually
  consistent read could miss records and produce an incomplete report.
- `job_state.set_report` is guarded by `attribute_exists(job_id)` so an `update_item`
  upsert cannot fabricate an ownerless job row, and is called only after both
  `report.html` and `manifest.json` are durable — a pointer to a half-written report
  would list and then 404 on open.
- `loadReportManifest`'s prefix-confinement check pushed `undefined` for
  `enhanced_image_key`, which would have failed manifest validation on every
  pre-existing report. It is now confined only when present.
- `image_enhancer` sets `ContentType: image/jpeg` on upload. The report side additionally
  tolerates `binary/octet-stream` and `application/octet-stream` — the S3 default when an
  uploader omits the header — and verifies JPEG identity by the SOI marker rather than
  trusting metadata, which had been silently dropping enhanced images.
- The resource-policy preflight both under-counted BADGERS' own policies and misattributed
  one of them to an unknown third party, which is the opposite of useful in a report whose
  purpose is answering "which of these can I remove?".
- `step_ui_deploy` gained the completion guard it was missing, and `mark_complete` moved
  from before the rollout poll to after it — the step previously recorded success before
  the service was confirmed healthy.

### Security

- **`/api/env` disclosed deployment configuration to unauthenticated callers.** It was the
  load balancer's health check path and therefore necessarily unauthenticated, and the
  service defaults to public subnets and an internet-facing balancer — yet it returned the
  region, gateway id, and both bucket names. Bucket names occupy a global namespace and the
  deployment id embedded in them derives every other resource name in the deployment. It is
  replaced by `GET /api/healthcheck`, which returns a constant `{"status":"ok"}`. The ECS
  stack's `health_check_path` was updated to match.
- **`/api/me` was removed.** Identity now comes from claims in the ID token the OIDC
  authorization-code + PKCE flow already produced and `oidc-client-ts` already validated
  (signature, issuer, audience, nonce). Asking the server to echo back claims the browser
  holds added a request and an endpoint without adding information. The client-side role
  still decides only which tabs render; every admin route enforces `requireAdmin`
  server-side, so a client claiming `admin` still gets 403.
- Branding moved to build-time config bundled by a Vite plugin, removing a synchronous
  file read that ran on every request to the one route reachable without a token.
- The Home "Environment" panel that displayed region, gateway id, and bucket names was
  removed. Every field was rendered as text and none was used by the browser, which never
  talks to S3, DynamoDB, or AgentCore directly.

### Known Issues

- **Wizard-created specialists still skip job tracking**, but no longer because the
  generator is a stub. The generated schema declares only `session_id`, where built-in
  schemas also declare `job_id` and `doc_id` — which is what the foundation layer stamps
  records from. This supersedes the stub-related entry under `[2.5.0]`.
- **Jobs recorded before `owner_sub` propagation have no owner and are absent from the
  sparse `owner-index`, so their reports never list.** This is the intended fail-closed
  outcome — a row whose owner cannot be established is not attributed to anybody — but it
  means historical runs surface no reports.
- The enhanced copy counts against the same 150 MiB `_MAX_AGGREGATE_BYTES` ceiling, so a
  second image per page effectively halves the page budget on enhanced runs. Exceeding it
  is caught and logged, and the page simply carries no enhanced key.
- `preflight_model_access`, `preflight_model_invocation`, and `preflight_ecs_slr` are
  defined in `common.sh` but never called from any entry point. `BADGERS_DEFAULT_MODELS`
  also names `us.anthropic.claude-sonnet-4-5-20250514-v1:0`, a date that appears nowhere
  else in the repository.
- On resume, `step_runtime`'s guard tests only `runtime_image_pushed`, so a deployment
  that pushed the runtime image but never deployed the runtime is skipped rather than
  re-entered. The previous resume table checked both keys.
- Editing `config/document_type_contexts.json` in S3 does not take effect on warm Lambda
  containers, because the map is cached in a module global for the life of the execution
  environment.
- `watermarked` is a valid `document_type` in both the enhancer schema and manifest but
  has no entry in `document_type_contexts.json` or the in-code fallback, so it silently
  resolves to empty context.
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
