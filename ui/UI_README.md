<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../README.md) | [Vision LLM Theory](../VISION_LLM_THEORY_README.md) | 🔵 **UI** | [Deployment](../deployment/DEPLOYMENT_README.md) | [CDK Stacks](../deployment/stacks/STACKS_README.md) | [Runtime](../deployment/runtime/RUNTIME_README.md) | [S3 Files](../deployment/s3_files/S3_FILES_README.md) | [Lambda Specialists](../deployment/lambdas/LAMBDA_SPECIALISTS.md) | [Prompting System](../deployment/s3_files/prompts/PROMPTING_SYSTEM_README.md) </sub>

---

# 🦡 BADGERS UI

Single React + Express application that serves as both the developer testing workbench and the deployment/ops console. Runs locally via `pnpm run dev`, or on AWS as a container on an ECS Express Gateway service authenticated with Cognito OIDC.

## Two Modes, One Codebase

|                   | Local Development                                      | Deployed on AWS                                    |
| ----------------- | ------------------------------------------------------ | -------------------------------------------------- |
| **Start**         | `pnpm run dev`                                         | Container on an ECS Express Gateway service        |
| **Auth**          | Bypassed — defaults to `admin` role (all tabs visible) | Cognito OIDC authorization code + PKCE             |
| **Role override** | `BADGERS_UI_ROLE=tester` env var                       | Cognito group membership (`admin` / `tester`)      |
| **Ports**         | Vite 5175 / Express 7860                               | Container exposes 7860                             |
| **Config**        | `config/.env`                                          | SSM Parameter Store, injected as container secrets |

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
corepack enable
pnpm install   # first time only
pnpm run dev   # starts Express API (7860) + Vite (5175)
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
writes `VITE_COGNITO_AUTHORITY`, `VITE_COGNITO_CLIENT_ID`, and `VITE_COGNITO_DOMAIN`.
A bundle built without them falls through to the server's local-dev bypass.

## Docker Deployment

```bash
pnpm run build                   # build static assets into dist/
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
│       ├── SpecialistEditor.jsx   # Manifest/prompt editor
│       ├── SpecialistWizard.jsx   # New specialist wizard
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
