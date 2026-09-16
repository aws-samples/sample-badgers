<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../README.md) | [Vision LLM Theory](../VISION_LLM_THEORY_README.md) | 🔵 **UI** | [Deployment](../deployment/DEPLOYMENT_README.md) | [CDK Stacks](../deployment/stacks/STACKS_README.md) | [Runtime](../deployment/runtime/RUNTIME_README.md) | [S3 Files](../deployment/s3_files/S3_FILES_README.md) | [Lambda Specialists](../deployment/lambdas/LAMBDA_SPECIALISTS.md) | [Prompting System](../deployment/s3_files/prompts/PROMPTING_SYSTEM_README.md) </sub>

---

# 🦡 BADGERS UI

Single React + Express application that serves as both the developer testing workbench and the deployment/ops console. Runs locally via `npm run dev`, or on AWS as a container on an ECS Express Gateway service authenticated with Cognito OIDC.

## Two Modes, One Codebase

|                   | Local Development                                      | Deployed on AWS                                    |
| ----------------- | ------------------------------------------------------ | -------------------------------------------------- |
| **Start**         | `npm run dev`                                          | Container on an ECS Express Gateway service        |
| **Auth**          | Bypassed — defaults to `admin` role (all tabs visible) | Cognito OIDC authorization code + PKCE             |
| **Role override** | `BADGERS_UI_ROLE=tester` env var                       | Cognito group membership (`admin` / `tester`)      |
| **Ports**         | Vite 5175 / Express 7860                               | Container exposes 7860                             |
| **Config**        | `ui/.env` (`deployment/scripts/generate_ui_env.sh`)    | SSM Parameter Store, injected as container secrets |

Auth is bypassed only when `COGNITO_USER_POOL_ID` is unset **and** the process is not
running on ECS. On ECS a missing user pool is treated as a misconfiguration, not a dev
shortcut, so requests fail rather than silently running unauthenticated.

## Tabs by Role

| Row     | Tabs                                                                                                          | Who sees it       |
| ------- | ------------------------------------------------------------------------------------------------------------- | ----------------- |
| Testing | 🏠 Home, 💬 Chat, ✏️ Edit Specialist, 🧙 Create Specialist, 🧪 Evaluations, 💰 Pricing, 📊 Observability, 📝 Chat Log | All users         |
| Deploy  | 📦 Stacks, 🔬 Specialists, 📄 S3 Configs, ⚙️ Deploy Tags                                                          | `admin` role only |

## Quick Start (Local)

```bash
cd ui
npm install    # first time only
npm run dev    # starts Express API (7860) + Vite (5175)
```

| Service                   | URL                   |
| ------------------------- | --------------------- |
| UI (Vite, proxies `/api`) | http://localhost:5175 |
| UI + API (Express)        | http://localhost:7860 |

Both URLs serve the app. Outside production the Express server mounts Vite as
middleware, so port 7860 serves the UI and the API from one origin; port 5175 is the
standalone Vite dev server proxying `/api` to 7860.

By default you get the `admin` role locally, so both tab rows are visible. Set
`BADGERS_UI_ROLE=tester` to test the restricted view.

## Authentication

The browser runs an OIDC authorization code + PKCE flow against the Cognito hosted UI
(`react-oidc-context`), then sends the access token as `Authorization: Bearer <token>`.
`server/auth.js` verifies it against the user pool's JWKS using `jose`. Role comes from
the token's Cognito group membership: `admin` if the `admin` group is present, otherwise
`tester`.

Every `/api/*` route is behind `requireAuth`, mounted before the route groups so no
individual handler can be reached unauthenticated.

The Vite build needs the Cognito values at build time — `deployment/scripts/generate_ui_env.sh`
writes `VITE_COGNITO_AUTHORITY`, `VITE_COGNITO_CLIENT_ID`, and `VITE_COGNITO_DOMAIN` to
`ui/.env`. A bundle built without them falls through to the server's local-dev bypass.

`ui/.env` is the only env file. Vite only exposes `VITE_`-prefixed lines to the bundle;
the remaining lines (bucket names, Runtime ARN, Gateway ID, jobs table, `AWS_PROFILE`) are
read by `server/index.js` when running locally. Nothing copies the file into the Docker
image — the deployed container gets those values from SSM.

## Docker Deployment

```bash
npm run build                    # build static assets into dist/
docker build -t badgers-ui .     # build container
docker run -p 7860:7860 badgers-ui
```

In production the container runs on an ECS Express Gateway service, which provisions and
manages its own load balancer and HTTPS endpoint. There is no ALB injecting identity
headers, which is why identity is established at the application layer instead. See
[CDK Stacks](../deployment/stacks/STACKS_README.md#ecs-ecs_stackpy).

## Architecture

```
Browser (React/Vite)
    │  Cognito hosted UI ──→ authorization code + PKCE ──→ access token
    │
    │  Identity read from the ID token's claims in the browser — no /api/me round trip
    │
    ├── /api/healthcheck ──→ {"status":"ok"}, the only unauthenticated route
    ├── /api/* ──→ Express server (port 7860), all behind requireAuth
    │                ├── Core routes (all roles)
    │                │   ├── AgentCore WebSocket proxy (chat, SSE to browser)
    │                │   ├── PDF upload to S3 (mints doc_id)
    │                │   ├── Job records (/api/jobs)
    │                │   ├── S3 file operations (manifests, prompts, schemas)
    │                │   ├── CloudWatch Logs Insights queries
    │                │   └── Evaluation and pricing endpoints
    │                └── Admin routes (admin role required)
    │                    ├── CDK deploy/destroy (SSE streaming)
    │                    ├── Stack status queries
    │                    ├── S3 config file read/write
    │                    └── Deployment tag management
    │
    └── Static assets (Vite middleware in dev, pre-built dist/ in production)
```

## Job Tracking Endpoints

The UI is the read side of the doc/job/subtask hierarchy described in
[Lambda Specialists](../deployment/lambdas/LAMBDA_SPECIALISTS.md#-job-tracking).

| Endpoint                | Returns                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `GET /api/jobs/:jobId`  | One job: computed status, counts, and every subtask with its outcome |
| `GET /api/jobs?doc_id=` | Every job recorded against one document, newest first                |

`POST /api/upload` mints a `doc_id` per upload and returns it. `POST /api/chat` accepts
that `doc_id` and forwards it to the agent, which stamps it onto each specialist call.
When the agent mints a `job_id` it emits a `job` SSE event, also written to the session
log as `[job] job_id=… doc_id=…`, so a chat transcript can be traced to its job record.

Job status is computed at read time rather than stored — see the endpoint comments in
`server/routes/core.js` for why.

## Report Endpoints

The Reports tab reads the artifacts written by `html_report_specialist`. Every endpoint is
authenticated and scoped to the caller.

| Endpoint                                       | Returns                                    |
| ---------------------------------------------- | ------------------------------------------ |
| `GET /api/reports`                             | Every report the caller owns, newest first |
| `GET /api/reports/:id/manifest`                | One report's manifest                      |
| `GET /api/reports/:id/pages/:n/image`          | The durable analysis image for one page    |
| `GET /api/reports/:id/pages/:n/enhanced-image` | The enhanced copy, when the run made one   |
| `GET /api/reports/:id/pages/:n/xml`            | The correlated page spine for one page     |
| `GET /api/reports/:id/download`                | The offline single-file HTML report        |

`enhanced-image` returns **404 rather than an error** when the page has no
`enhanced_image_key`: a clean page is never enhanced, and reports generated before that
field existed have none. The Page Reader only requests it when the manifest declares one, so
un-enhanced and legacy reports cost no extra round trip, and it offers Original/Enhanced
tabs only when a second image exists.

The image the correlation artifact names as its source is the *original*, so before this
existed a report showed the page as scanned while most specialists had read the enhanced
copy. Persisting both is what makes that provenance visible rather than implied.

The listing is a keyed DynamoDB query on the `owner-index` GSI, partitioned by the
caller's `owner_sub` and filtered to job rows carrying a `report_id`. Ownership therefore
comes from the job record rather than from the bucket layout, and no S3 listing is
involved. A job whose owner was never recorded is absent from the index, so its reports do
not list — the deliberate fail-closed case.

The remaining endpoints load the manifest and re-check its `owner_sub` against the caller,
then serve only keys the manifest itself declares under that report's prefix. Nothing
accepts a caller-supplied S3 key, and no presigned URLs are issued: the listing decides
what may be enumerated, the manifest decides what may be opened.

## Specialist Wizard Endpoints

The 🧙 Create Specialist tab's four steps map to four endpoints in `server/routes/wizard.js`,
mounted by `mountWizardRoutes(app, PROJECT_ROOT)`.

| Endpoint                    | Transport | Does                                                         |
| --------------------------- | --------- | ------------------------------------------------------------ |
| `POST /api/wizard/generate` | **SSE**   | Six sequential Bedrock calls, one per prompt section         |
| `POST /api/wizard/preview`  | JSON      | Assembles the manifest and schema for review; writes nothing |
| `POST /api/wizard/save`     | JSON      | Writes every artifact under `deployment/custom_specialists/` |
| `POST /api/wizard/deploy`   | **SSE**   | Streams `deployment/deploy_custom_specialists.sh`            |

`generate` streams because six calls take minutes, which no plain POST survives behind the
load balancer. Its frames are `start` (carrying the full section list, so the client keeps no
second copy), then `progress` and `section` per prompt, then `done` or `error`. Sections are
generated in load order: `gestalt`, `job_role`, `context`, `rules`, `tasks`, `format`.

A single failed section becomes an `<!-- ERROR ... -->` stub and is reported in `warnings`, so
one bad generation does not lose the other five. All six failing is a hard error instead —
that means credentials, region, or model access is broken, and six comment stubs would read
as content.

Bedrock is called by **hand-signing an HTTPS Converse request with SigV4**, not through
`@aws-sdk/client-bedrock-runtime`, which is deliberately not a dependency of this package.
Region comes from `AWS_REGION` (default `us-west-2`), credentials from the node provider
chain, profile from `AWS_PROFILE`. The generator model is `WIZARD_GENERATOR_MODEL_ID` when
set, else the `DEFAULT_GENERATOR_MODEL_ID` literal in `wizard.js`. In ECS the task
definition sets it from `WIZARD_GENERATOR_MODEL_ID` in `deployment/stacks/ecs_stack.py`,
the same constant the task role's `bedrock:InvokeModel` grant is built from, so the deployed
call and its grant cannot name different models; the literal is for local development. The
primary/fallback dropdowns are populated from `GET /api/models` and validated against the
same list, so there is no second model map to keep aligned.

`save` and `deploy` are separate steps, and Deploy stays disabled until a save succeeds.
Everything is written to the **local working tree**, never to S3 — see
[Custom Specialists](../deployment/DEPLOYMENT_README.md#-custom-specialists). Every write
target is resolved and confirmed to be inside `custom_specialists/` first, and the shared
`specialist_registry.json` is read-modify-written so other entries are never clobbered.

## Streaming and Timeouts

The chat SSE stream writes a `: ping` comment every 15 seconds. Two reasons, both load-bearing:

- The deployed load balancer closes any connection idle in **either** direction for 60
  seconds (`idle_timeout.timeout_seconds`, the AWS default, which `CfnExpressGatewayService`
  does not expose). This stream writes nothing for the whole of a tool call, and `full_text`
  has taken 115 seconds on a dense page — so a slow specialist dropped the connection every
  time.
- Writing is the only way to notice the client has gone. TCP does not report a closed peer
  until you write to it, so without a heartbeat an abandoned request surfaced only when the
  next real event arrived. A `[client-gone]` line is now logged when the browser leaves
  mid-run.

`POST /api/wizard/deploy` sends the same kind of heartbeat and kills the child process when
the client disconnects.

## Tech Stack

| Component         | Technology                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| Frontend          | React 19, Vite 8                                                                                               |
| Chat UI           | @assistant-ui/react                                                                                            |
| Backend           | Express 5, Node.js                                                                                             |
| Auth              | Cognito OIDC + PKCE (react-oidc-context, oidc-client-ts), JWKS via jose                                        |
| Code highlighting | react-shiki, highlight.js                                                                                      |
| Markdown          | react-markdown                                                                                                 |
| AWS SDK           | client-s3, client-cloudwatch-logs, client-dynamodb, lib-dynamodb, client-ssm, client-bedrock-agentcore-control |
| WebSocket         | ws (AgentCore Runtime connection)                                                                              |
| Streaming         | Server-Sent Events (SSE)                                                                                       |

## Project Structure

```
ui/
├── src/
│   ├── App.jsx                    # Tab router with role gating, OIDC provider
│   ├── main.jsx                   # React entry point
│   ├── authFetch.js               # Attaches the bearer token to API calls
│   ├── index.css                  # Global styles
│   ├── hooks/
│   │   └── useUser.js             # User context (role, email) from ID token claims
│   └── components/
│       ├── Home.jsx               # Dashboard
│       ├── Chat.jsx               # Agent chat interface, PDF upload, doc_id
│       ├── CopyButton.jsx         # Shared copy-to-clipboard control
│       ├── Reports.jsx            # Report index and Page Reader
│       ├── SpecialistEditor.jsx   # Manifest/prompt editor
│       ├── SpecialistWizard.jsx   # New specialist wizard (consumes the SSE stream)
│       ├── Evaluator.jsx          # Test runner
│       ├── PricingCalculator.jsx  # Cost estimator
│       ├── Observability.jsx      # CloudWatch queries
│       ├── ChatLog.jsx            # Session log viewer
│       ├── StackList.jsx          # CDK stack deploy/destroy
│       ├── SpecialistSelector.jsx # Specialist browser
│       ├── S3ConfigEditor.jsx     # S3 config file editor
│       ├── ConfigEditor.jsx       # Deployment tag editor
│       ├── JsonHighlighter.jsx    # JSON syntax highlighting
│       ├── Header.jsx             # App header with user/role badge, sign out
│       └── LogPanel.jsx           # Streaming log output
├── server/
│   ├── index.js                   # Express entry, SSM config load, static/Vite serving
│   ├── auth.js                    # Cognito JWT verification / local-dev bypass
│   └── routes/
│       ├── core.js                # Core API routes (all roles)
│       ├── wizard.js              # Specialist wizard routes (SSE generate/deploy)
│       └── admin.js               # Admin API routes (admin only)
├── config/
│   ├── .env                       # Local environment variables
│   ├── branding.json              # App name, emoji, subtitle, default theme
│   └── pricing_config.json        # Pricing presets and defaults
├── public/images/                 # favicon, logo
├── Dockerfile                     # Production container
├── package.json
└── vite.config.js
```
