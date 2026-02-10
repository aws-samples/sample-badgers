# Analyzer Wizard & Editor

Gradio-based tools for creating and editing document analyzers.

## Tools

| Tool                  | Purpose                                       |
| --------------------- | --------------------------------------------- |
| Analyzer Wizard (tab) | Create new analyzers via guided 4-step wizard |
| Analyzer Editor (tab) | Edit existing wizard-managed analyzers        |

## Quick Start

```bash
cd frontend

# Launch the multi-page Gradio app
uv run python main.py
```

Then open `http://localhost:7860` and navigate to the Analyzer Creator or Analyzer Editor tab.

## Creation Wizard

The wizard automates analyzer creation by:

1. Collecting analyzer name, description, and model preferences
2. Using Claude to generate XML prompt files (job_role, rules, context, tasks, format)
3. Allowing you to review/edit the generated prompts
4. Optionally uploading few-shot example images
5. Generating all required files

### Files Created

For an analyzer named "invoice":

```
deployment/
├── lambdas/code/invoice_analyzer/
│   └── lambda_handler.py
├── s3_files/
│   ├── manifests/invoice_analyzer.json
│   ├── schemas/invoice_analyzer.json
│   └── prompts/invoice_analyzer/
│       ├── invoice_job_role.xml
│       ├── invoice_rules.xml
│       ├── invoice_context.xml
│       ├── invoice_tasks.xml
│       ├── invoice_format.xml
│       └── few-shot-images/  (if examples provided)
```

## Editor

The editor only shows analyzers with `"wizard_managed": true` in their manifest metadata. This prevents editing analyzers that may have custom Lambda code.

You can edit:
- Description
- Model selections (primary + 2 fallbacks)
- All 5 XML prompt files
- `audit_mode` support (confidence scoring)
- `expected_output_tokens` (cost estimation)

## After Creating/Editing

### For Custom Analyzers (created via wizard)

Custom analyzers are stored in S3 and deployed separately:

```bash
cd deployment

# 1. Sync custom analyzers from S3 to local
./sync_custom_analyzers.sh

# 2. Deploy custom analyzers stack (creates Lambda + gateway targets)
cdk deploy badgers-custom-analyzers
```

### For Base Analyzers (editing existing)

If you edited a base analyzer's prompts:

```bash
cd deployment
./sync_s3_files.sh      # Sync configs/prompts to S3
```

> [!NOTE]
> Base analyzer Lambda code is deployed with the main stack. Editing prompts only requires syncing to S3.

## Available Models

| Display Name        | Model ID                                           |
| ------------------- | -------------------------------------------------- |
| Claude Sonnet 4.5   | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Claude Haiku 4.5    | `us.anthropic.claude-haiku-4-5-20251001-v1:0`      |
| Claude Opus 4.6     | `global.anthropic.claude-opus-4-6-v1`              |
| Amazon Nova Premier | `us.amazon.nova-premier-v1:0`                      |

## Requirements

- AWS credentials configured (for Bedrock access during prompt generation)
- `gradio` and `boto3` packages (included in project dependencies)
