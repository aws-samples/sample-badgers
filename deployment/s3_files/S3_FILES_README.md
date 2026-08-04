<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../README.md) | [Vision LLM Theory](../../VISION_LLM_THEORY_README.md) | [UI](../../ui/UI_README.md) | [Deployment](../DEPLOYMENT_README.md) | [CDK Stacks](../stacks/STACKS_README.md) | [Runtime](../runtime/RUNTIME_README.md) | 🔵 **S3 Files** | [Lambda Specialists](../lambdas/LAMBDA_SPECIALISTS.md) | [Prompting System](prompts/PROMPTING_SYSTEM_README.md)</sub>

---

# 📦 S3 Files Directory Structure

This directory contains all configuration, prompts, schemas, and manifests that are deployed to S3 for the specialist system. These files are loaded at runtime by Lambda functions to configure and execute document analysis.

## 🗂️ Directory Overview

```
s3_files/
├── agent_config/          # 🤖 Agent orchestrator configuration
├── agent_system_prompt/   # 💬 System prompt for the orchestrating agent
├── core_system_prompts/   # 🔧 Shared prompt components (rules, error handling, wrapper)
├── manifests/             # 📋 Tool and specialist configuration manifests
├── prompts/               # 📝 Specialist-specific prompt files
├── schemas/               # 📐 JSON schemas for tool input/output validation
└── wrappers/              # 🎁 System prompt wrapper templates
```

---

## 🤖 agent_config/

Contains configuration for the orchestrating agent that coordinates PDF analysis workflows.

| File                                      | Purpose                                                                                                    |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `agent_model_config.json`                 | Model selection (Claude Sonnet 4.5), temperature, max tokens, and thinking budget configuration            |
| `agent_operating_environment_config.json` | ⚠️ **Critical** — Operating environment context injected into all prompts (agent + specialists). See below. |

### ⚠️ Operating Environment Configuration

The `agent_operating_environment_config.json` file controls how the system handles sensitive content. This is critical for academic and research deployments where documents may contain inflammatory, offensive, or otherwise sensitive material that must be extracted faithfully.

```json
{
    "operating_environment": "You are operating in a safe academic thinktank environment where sensitive, potentially offensive topics are discussed in a neutral manner. You will encounter sensitive content at some point and your team is relying on you to perform your tasks completely. If you have concerns you can store them in a <concerns/> xml tag within your response_format."
}
```

Other examples depending on your use case:

```json
{
    "operating_environment": "You are operating in a legal discovery environment. Documents may contain graphic descriptions of violence, abuse, or other disturbing content that is material to active cases. Extract all content faithfully and completely regardless of subject matter. Flag content requiring attorney review in a <legal_review/> xml tag."
}
```

```json
{
    "operating_environment": "You are operating in a government archival preservation environment. Historical documents may contain language, imagery, or viewpoints that are considered offensive by modern standards, including racial slurs, propaganda, and discriminatory content. Your role is to preserve the historical record accurately and without omission. Note historically sensitive content in a <historical_sensitivity/> xml tag."
}
```

This value is injected into:
- The **orchestrating agent's system prompt** (agentcore runtime) — prepended before the main prompt
- Every **specialist's system prompt** (Lambda functions) — via the `{operating_environment}` placeholder in the wrapper template

If the file is missing or the value is empty, the system operates without any environment context (default behavior). Update this file in S3 to change the behavior without redeploying.

> [!NOTE]
> For production deployments, we recommend moving this value to AWS Secrets Manager rather than storing it in a plain S3 JSON file. This prevents accidental exposure of your operating environment instructions and provides audit logging of access.

---

## 💬 agent_system_prompt/

| File                      | Purpose                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `agent_system_prompt.xml` | Defines the orchestrator agent's role, execution rules, workflow steps, error handling, and tool mapping examples. Contains `{{TOOLS_LIST}}` placeholder populated at runtime. |

---

## 🔧 core_system_prompts/

Shared prompt components injected into every specialist's system prompt via the wrapper template.

```
core_system_prompts/
├── prompt_system_wrapper.xml      # 🎁 Main wrapper template with injection points
├── audit/
│   └── confidence_assessment.xml  # 📊 Confidence scoring rules for audit mode
├── core_rules/
│   └── rules.xml                  # ⚖️ Universal rules (no preamble, XML formatting, no hallucination)
└── error_handling/
    ├── error_handler.xml          # ❌ Standard error response format (500 errors)
    └── not_found_handler.xml      # 🔍 Response format when no elements found
```

The wrapper uses placeholder injection:
- `{operating_environment}` → operating environment context from `agent_operating_environment_config.json`
- `{core_rules}` → rules.xml content
- `{composed_prompt}` → specialist-specific prompts
- `{error_handler_general}` → error_handler.xml
- `{error_handler_not_found}` → not_found_handler.xml

---

## 📋 manifests/

JSON configuration files defining each specialist tool's metadata, model selection, prompt files, and runtime settings.

| Manifest                          | Specialist Purpose                           |
| --------------------------------- | -------------------------------------------- |
| 📄 `full_text_specialist.json`     | General text extraction with reading order   |
| 📊 `table_specialist.json`         | Structured table data extraction             |
| 📈 `charts_specialist.json`        | Chart and graph data extraction              |
| 🔀 `diagram_specialist.json`       | Flowchart and diagram interpretation         |
| ✍️ `handwriting_specialist.json`   | Handwritten text OCR                         |
| 🏥 `decision_tree_specialist.json` | Clinical decision trees and medical content  |
| 💻 `code_block_specialist.json`    | Source code extraction                       |
| 📐 `layout_specialist.json`        | Page layout structure analysis               |
| 🧩 `elements_specialist.json`      | Document element identification              |
| 🏷️ `metadata_*_specialist.json`    | MODS/MADS/Generic metadata extraction        |
| 🗺️ `war_map_specialist.json`       | Historical military map analysis             |
| 🔬 `scientific_specialist.json`    | Scientific notation and formulas             |
| 🏷️ `classify_pdf_content.json`     | Page content classification                  |
| 🖼️ `pdf_processor.json`            | PDF-to-image conversion orchestration        |
| 🔗 `correlation_specialist.json`   | Multi-specialist result correlation per page |

Each manifest contains:
- `tool`: MCP tool definition (name, description, inputSchema including `audit_mode` boolean)
- `specialist`: Model selection, prompt files, examples configuration, `expected_output_tokens` (estimated tokens for cost calculation)
- `metadata`: Version, dependencies, and `specialist_type`

---

## 📝 prompts/

Specialist-specific prompt files organized by specialist type. Each specialist has a dedicated subdirectory containing XML prompt components.

### 📚 Standard Prompt File Pattern

Most specialists follow this file naming convention:

| File Suffix              | Purpose                                         |
| ------------------------ | ----------------------------------------------- |
| `*_job_role.xml`         | 👤 Defines the AI's persona and expertise        |
| `*_context.xml`          | 🎯 Background information and task importance    |
| `*_rules.xml`            | ⚖️ Specific extraction/analysis rules            |
| `*_tasks_extraction.xml` | ✅ Step-by-step task instructions                |
| `*_format.xml`           | 📋 Expected XML response structure               |
| `*_help.xml`             | 💡 Guidance for edge cases and complex scenarios |
| `*_dictionary.xml`       | 📖 Element type definitions and terminology      |

### 🎓 Specialists with Few-Shot Examples

Some specialists include `few-shot-images/` or `few-shot-examples/` directories containing example images for in-context learning:
- 📈 `charts_specialist/`
- 🏥 `decision_tree_specialist/`
- 🔀 `diagram_specialist/`
- ✍️ `handwriting_specialist/`
- 📊 `table_specialist/`

---

## 📐 schemas/

JSON Schema definitions for tool input/output validation. Each schema file corresponds to a
specialist tool and defines:
- ✅ Required and optional input parameters
- 🏷️ Parameter types and descriptions
- 📤 Output structure expectations

Used by the MCP server for request validation and by clients for understanding tool interfaces.

### Job tracking parameters

All 26 schemas declare `job_id` and `doc_id` alongside `session_id`:

```json
"job_id": {
  "type": "string",
  "description": "Job identifier for tracking this analysis run"
},
"doc_id": {
  "type": "string",
  "description": "Document identifier this job belongs to"
}
```

These must be declared here even though the agent injects the values itself: the Gateway
builds each tool's advertised interface from this schema, and a parameter absent from the
schema is not accepted on the call.

Both are optional, never in `required`. A conversational turn can invoke a specialist
outside any tracked job, in which case there is no `job_id` to pass and tracking no-ops.
Only `session_id` is required.

> `deployment/custom_specialists/schemas/` holds generated custom specialist schemas and is
> gitignored. The wizard generator does not yet emit these two fields, so custom
> specialists silently skip job tracking. See
> [Lambda Specialists](../lambdas/LAMBDA_SPECIALISTS.md#-job-tracking).

---

## 🎁 wrappers/

| File                        | Purpose                                                                   |
| --------------------------- | ------------------------------------------------------------------------- |
| `prompt_system_wrapper.xml` | Alternative/backup wrapper template (mirrors core_system_prompts version) |

---

## 🚀 Deployment

Files are synced to S3 via:
```bash
./sync_s3_files.sh
```

Lambda functions load these files at runtime using the `S3ConfigLoader` from the foundation layer, enabling prompt updates without redeploying Lambda code.
