<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [UI](../../ui/UI_README.md) | [Deployment](../DEPLOYMENT_README.md) | [CDK Stacks](../stacks/STACKS_README.md) | 🔵 **Runtime** | [S3 Files](../s3_files/S3_FILES_README.md) | [Lambda Analyzers](../lambdas/LAMBDA_ANALYZERS.md) | [Prompting System](../s3_files/prompts/PROMPTING_SYSTEM_README.md)</sub>

---

# 🐳 AgentCore Runtime Container

Docker container running the Strands agent for PDF analysis orchestration with WebSocket streaming.

## Contents

```
runtime/
├── Dockerfile.websocket      # Container definition (WebSocket streaming)
├── build_and_push_websocket.sh # Build and push to ECR
├── requirements.txt          # Python dependencies
└── agent/
    ├── __init__.py
    └── main-websocket.py     # Strands agent entry point (WebSocket)
```

## Quick Start

> [!IMPORTANT]
> The ECR stack must be deployed before building the container.

```bash
# Build and push (requires ECR stack deployed first)
./build_and_push_websocket.sh

# Then deploy Runtime stack
cd .. && cdk deploy badgers-runtime-websocket
```

## Container Details

| Property     | Value                          |
| ------------ | ------------------------------ |
| Base Image   | `python:3.11-slim`             |
| Architecture | `linux/arm64`                  |
| Port         | 8080                           |
| User         | `bedrock_agentcore` (non-root) |
| Health Check | `GET /ping` every 30s          |

## Dependencies

- `strands-agents` — Agent framework
- `bedrock-agentcore[strands-agents]` — AgentCore SDK
- `mcp` — Model Context Protocol client
- `boto3` — AWS SDK
- `httpx` — HTTP client
- `aws-opentelemetry-distro` — X-Ray tracing

## Environment Variables

Set by Runtime stack:

| Variable                         | Description                               |
| -------------------------------- | ----------------------------------------- |
| `AWS_DEFAULT_REGION`             | AWS region                                |
| `GATEWAY_URL`                    | MCP Gateway endpoint                      |
| `COGNITO_CREDENTIALS_SECRET_ARN` | Secrets Manager ARN for OAuth credentials |
| `AGENTCORE_MEMORY_ID`            | Memory ID for session persistence         |

## Build Options

```bash
# Build with custom tag
./build_and_push_websocket.sh v1.0.0

# Manual build
docker build -f Dockerfile.websocket --platform linux/arm64 -t pdf-analysis-agent .
```

## Customization

### Modify Agent Behavior
Edit `agent/main-websocket.py` to change:
- System prompt
- Model selection
- Tool handling
- Session management

### Add Dependencies
1. Add to `requirements.txt`
2. Rebuild and push: `./build_and_push_websocket.sh`
3. Redeploy Runtime stack

## Troubleshooting

```bash
# View Runtime logs
aws logs tail /aws/bedrock-agentcore/runtimes/<runtime_id> --follow

# Check container locally
docker run -p 8080:8080 pdf-analysis-agent

# Verify ECR image
aws ecr describe-images --repository-name pdf-analysis-agent-<deployment_id>
```
