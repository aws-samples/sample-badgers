# Changelog

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
- `local_testing/LOCAL_TESTING_README.md` — documentation for the Local Testing UI
- `deployment/ui/DEPLOYMENT_UI_README.md` — documentation for the Deployment UI

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
