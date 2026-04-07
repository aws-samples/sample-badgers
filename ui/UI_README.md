<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../README.md) | [Vision LLM Theory](../VISION_LLM_THEORY_README.md) | 🔵 **UI** | [Deployment](../deployment/DEPLOYMENT_README.md) | [CDK Stacks](../deployment/stacks/STACKS_README.md) | [Runtime](../deployment/runtime/RUNTIME_README.md) | [S3 Files](../deployment/s3_files/S3_FILES_README.md) | [Lambda Analyzers](../deployment/lambdas/LAMBDA_ANALYZERS.md) | [Prompting System](../deployment/s3_files/prompts/PROMPTING_SYSTEM_README.md) </sub>

---

# 🦡 BADGERS UI

Single React + Express application that serves as both the developer testing workbench and the deployment/ops console. Runs locally via `npm run dev` or deployed to AWS via Docker behind an ALB with OIDC authentication.

## Two Modes, One Codebase

|                   | Local Development                                  | Deployed on AWS                          |
| ----------------- | -------------------------------------------------- | ---------------------------------------- |
| **Start**         | `npm run dev`                                      | Docker container behind ALB              |
| **Auth**          | None — defaults to `admin` role (all tabs visible) | ALB OIDC → Cognito groups determine role |
| **Role override** | `BADGERS_UI_ROLE=tester` env var                   | Cognito group membership                 |
| **Ports**         | Vite 5175 / Express 7860                           | Container exposes 7860                   |

## Tabs by Role

| Row     | Tabs                                                                                                      | Who sees it       |
| ------- | --------------------------------------------------------------------------------------------------------- | ----------------- |
| Testing | 🏠 Home, 💬 Chat, ✏️ Edit Analyzer, 🧙 Create Analyzer, 🧪 Evaluations, 💰 Pricing, 📊 Observability, 📝 Chat Log | All users         |
| Deploy  | 📦 Stacks, 🔬 Analyzers, 📄 S3 Configs, ⚙️ Deploy Tags                                                        | `admin` role only |

## Quick Start (Local)

```bash
cd unified-ui
npm install    # first time only
npm run dev    # starts Vite (5175) + Express API (7860)
```

| Service | URL                   |
| ------- | --------------------- |
| UI      | http://localhost:5175 |
| API     | http://localhost:7860 |

By default you get the `admin` role locally, so both tab rows are visible. Set `BADGERS_UI_ROLE=tester` in your environment to test the restricted view.

## Docker Deployment

```bash
npm run build                    # build static assets into dist/
docker build -t badgers-ui .     # build container
docker run -p 7860:7860 badgers-ui
```

In production, the container sits behind an ALB that injects `x-amzn-oidc-data` JWT headers. The server verifies these against ALB public keys and extracts role from Cognito groups.

## Architecture

```
Browser (React/Vite)
    │
    ├── /api/me ──→ User identity (OIDC or local fallback)
    ├── /api/* ──→ Express server (port 7860)
    │                ├── Testing routes
    │                │   ├── AgentCore WebSocket proxy (chat)
    │                │   ├── S3 file operations (manifests, prompts, schemas)
    │                │   ├── CloudWatch Logs Insights queries
    │                │   └── Evaluation and pricing endpoints
    │                └── Admin routes (admin role required)
    │                    ├── CDK deploy/destroy (SSE streaming)
    │                    ├── Stack status queries
    │                    ├── S3 config file read/write
    │                    └── Deployment tag management
    │
    └── Static assets (Vite dev server or pre-built dist/)
```

## Tech Stack

| Component         | Technology                                          |
| ----------------- | --------------------------------------------------- |
| Frontend          | React 19, Vite 8                                    |
| Chat UI           | @assistant-ui/react                                 |
| Backend           | Express 5, Node.js                                  |
| Auth              | ALB OIDC / JWT (jsonwebtoken)                       |
| Code highlighting | react-shiki, highlight.js                           |
| Markdown          | react-markdown                                      |
| AWS SDK           | @aws-sdk/client-s3, @aws-sdk/client-cloudwatch-logs |
| WebSocket         | ws (AgentCore Runtime connection)                   |
| Streaming         | Server-Sent Events (SSE)                            |

## Project Structure

```
ui/
├── src/
│   ├── App.jsx                    # Tab router with role gating
│   ├── main.jsx                   # React entry point
│   ├── index.css                  # Global styles
│   ├── hooks/
│   │   └── useUser.js             # User context (role, email)
│   └── components/
│       ├── Home.jsx               # Dashboard
│       ├── Chat.jsx               # Agent chat interface
│       ├── AnalyzerEditor.jsx     # Manifest/prompt editor
│       ├── AnalyzerWizard.jsx     # New analyzer wizard
│       ├── Evaluator.jsx          # Test runner
│       ├── PricingCalculator.jsx  # Cost estimator
│       ├── Observability.jsx      # CloudWatch queries
│       ├── ChatLog.jsx            # Session log viewer
│       ├── StackList.jsx          # CDK stack deploy/destroy
│       ├── AnalyzerSelector.jsx   # Analyzer browser
│       ├── S3ConfigEditor.jsx     # S3 config file editor
│       ├── ConfigEditor.jsx       # Deployment tag editor
│       ├── JsonHighlighter.jsx    # JSON syntax highlighting
│       ├── Header.jsx             # App header with user/role badge
│       └── LogPanel.jsx           # Streaming log output
├── server/
│   ├── index.js                   # Express server entry
│   ├── middleware/
│   │   └── auth.js                # OIDC JWT verification / local fallback
│   └── routes/
│       ├── testing.js             # Testing API routes (all users)
│       └── admin.js               # Admin API routes (admin only)
├── config/
│   ├── .env                       # Environment variables
│   └── pricing_config.json        # Pricing presets and defaults
├── Dockerfile                     # Production container
├── package.json
└── vite.config.js
```
