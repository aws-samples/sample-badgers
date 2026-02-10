<sub>🧭 **Navigation:**</sub><br>
<sub>[Home](../../../README.md) | [Vision LLM Theory](../../../VISION_LLM_THEORY_README.md) | [Frontend](../../../frontend/FRONTEND_README.md) | [Deployment](../../DEPLOYMENT_README.md) | [CDK Stacks](../../stacks/STACKS_README.md) | [Runtime](../../runtime/RUNTIME_README.md) | [S3 Files](../S3_FILES_README.md) | [Lambda Analyzers](../../lambdas/LAMBDA_ANALYZERS.md) | 🔵 **Prompting System**</sub>

---

# 📝 Prompting System Architecture

This document explains the design principles, structure, and rationale behind the modular prompting system used by all analyzers.

## 🎯 Core Design Principles

### 1. 🧩 Separation of Concerns

Prompts are decomposed into discrete, single-purpose files rather than monolithic prompts. This enables:
- 🔄 Independent iteration on specific aspects
- ♻️ Reuse of common components across analyzers
- 🧪 Easier testing and debugging
- 👥 Clear ownership of prompt sections

### 2. 🏗️ Composition Over Inheritance

The system composes final prompts from multiple files at runtime:
```
Final System Prompt = Wrapper + Core Rules + Analyzer Prompts + Error Handlers
```

### 3. 📋 XML as Prompt Structure

XML provides:
- 🏷️ Clear semantic boundaries between sections
- 📖 Self-documenting tag names
- 🪆 Nested structure for complex instructions
- ✅ Easy parsing and validation
- 🎯 Explicit section markers the model can reference

### 4. ⚙️ Configuration-Driven Assembly

Manifests declare which prompt files to load, allowing:
- 🔀 Different analyzers to share common files
- 🧪 Easy A/B testing of prompt variations
- 🔄 Runtime prompt updates without code changes

---

## 🔄 Prompt Composition Flow

```
+-----------------------------------------------------------------+
|                    prompt_system_wrapper.xml                    |
|  +-----------------------------------------------------------+  |
|  | {core_rules}           <- core_rules/rules.xml            |  |
|  | {composed_prompt}      <- [analyzer prompt files...]      |  |
|  | {error_handler_general}<- error_handling/error_handler.xml|  |
|  | {error_handler_not_found}<- error_handling/not_found_handler.xml |
|  +-----------------------------------------------------------+  |
+-----------------------------------------------------------------+
```

The `PromptLoader` class:
1. 📂 Loads the wrapper template
2. 🔧 Loads core system files (rules, error handlers)
3. 📝 Loads and concatenates analyzer-specific prompt files
4. 💉 Injects all content into wrapper placeholders
5. 🔄 Replaces dynamic placeholders (e.g., `[[PIXEL_WIDTH]]`)

---

## 📚 Standard Prompt File Types

Each analyzer typically includes these file types, each serving a specific purpose:

### 👤 `*_job_role.xml` - Identity and Expertise

Establishes the AI's persona and professional context.

**Purpose**: Primes the model with domain expertise and professional standards.

**Example structure**:
```xml
<job_role>
    <role>You are a 'Full Text Extraction Specialist.'...</role>
    <job_description>
        <job_description_title>...</job_description_title>
        <job_description_summary>...</job_description_summary>
        <job_description_responsibilities>
            <responsibility>...</responsibility>
        </job_description_responsibilities>
        <job_description_skills>
            <skill>...</skill>
        </job_description_skills>
    </job_description>
</job_role>
```

**Why it matters**: 🎭 Role-playing improves task adherence. The model performs better when given a clear professional identity with specific responsibilities.

---

### 🎯 `*_context.xml` - Background and Importance

Provides situational awareness and explains why the task matters.

**Purpose**: Helps the model understand the downstream use of its output.

**Example content**:
```xml
<context>
    <item>Your structured output will be used for PDF accessibility tagging (PDF/UA compliance)</item>
    <item>Each element you extract will be assigned a unique ID for joining with bounding box data</item>
    <item>Heading hierarchy is critical for accessibility - screen reader users navigate by headings</item>
</context>
```

**Why it matters**: 🧠 When the model understands *why* something matters (accessibility, downstream processing), it makes better decisions in ambiguous situations.

---

### ⚖️ `*_rules.xml` - Constraints and Requirements

Hard constraints the model must follow.

**Purpose**: Defines non-negotiable behaviors and output requirements.

**Example content**:
```xml
<full_text_rules>
    <rule>Extract ALL text from the page, regardless of perceived importance</rule>
    <rule>Assign a unique sequential ID to every element (elem_001, elem_002, etc.)</rule>
    <rule>Do not attempt to interpret or summarize the content; extract it verbatim</rule>
    <rule>Decorative elements should be typed as "artifact" or omitted entirely</rule>
</full_text_rules>
```

**Why it matters**: 🚫 Explicit rules prevent common failure modes. Rules are easier to audit and update than implicit expectations buried in prose.

---

### ✅ `*_tasks_extraction.xml` - Step-by-Step Instructions

Procedural workflow the model should follow.

**Purpose**: Guides the model through a structured analysis process.

**Example structure**:
```xml
<tasks>
    <task>Take a deep breath and carefully review the information provided</task>
    <task>Examine the given page, identifying all textual elements</task>
    <task>For each identified text element:
        <sub_task>Assign a unique sequential ID</sub_task>
        <sub_task>Determine its type (h1, h2, paragraph, etc.)</sub_task>
        <sub_task>Assign an order number reflecting reading sequence</sub_task>
    </task>
    <task>Review your extracted text to ensure completeness</task>
</tasks>
```

**Why it matters**: 📋 Breaking complex tasks into steps improves accuracy. The model can focus on one aspect at a time rather than juggling everything simultaneously.

---

### 📋 `*_format.xml` - Output Structure

Defines the exact XML structure expected in responses.

**Purpose**: Provides a template the model should follow for output.

**Standard Structure**:
All format files follow a unified structure:

```xml
<response_format>
    <response extraction_type="{TYPE}">
        <metadata>
            <page_number>{PAGE_NUMBER_VISIBLE_ON_PAGE_OR_INFERRED}</page_number>
            <examples_count>{NUMBER_OF_EXAMPLE_IMAGES_YOU_WERE_GIVEN}</examples_count>
            <element_count>{NUMBER_OF_ELEMENTS_DETECTED}</element_count>
        </metadata>
        <!-- If element_count is 0, omit <elements> section entirely -->
        <elements>
            <element type="header" id="elem_001" order="1">
                <text>Page header text here</text>
            </element>
            <element type="h1" id="elem_002" order="2">
                <text>Main Title</text>
            </element>
        </elements>
    </response>
</response_format>
```

**Key components**:
- `extraction_type`: Identifies the analyzer type (e.g., "full_text", "table", "diagram")
- `<metadata>`: Contains page number, example count, and element count
- `<elements>`: Container for all extracted elements (omitted when element_count is 0)
- `<element>`: Individual extracted items with type-specific attributes

**Why it matters**: 🎯 Explicit format examples dramatically reduce structural errors. The model can pattern-match rather than invent structure. The unified structure enables consistent parsing across all analyzers.

**Special Case Formats**:

Some analyzers use non-standard formats due to external requirements:

| Format File                     | Output Type        | Reason                              |
| ------------------------------- | ------------------ | ----------------------------------- |
| `accessibility_format.xml`      | PDF Structure Tree | PDF/UA specification compliance     |
| `mads_format.xml`               | MADS XML           | Library of Congress schema          |
| `mods_format.xml`               | MODS XML           | Library of Congress schema          |
| `keyword_topic_format.xml`      | JSON               | SEO/topic analysis structure        |
| `robust_elements_format.xml`    | JSON               | Pixel bounding box coordinates      |
| `remediation_output_format.xml` | JSON               | Normalized bounding box coordinates |
| `classification_format.xml`     | JSON               | Tool routing/classification         |

These output JSON (for coordinate data or structured classification) or external schema-compliant XML where the format is dictated by external specifications.

---

### 💡 `*_help.xml` - Edge Case Guidance

Advice for handling complex or ambiguous situations.

**Purpose**: Provides decision-making guidance without being prescriptive rules.

**Example content**:
```xml
<help>
    <guidance>For multi-column layouts, extract text left to right, then top to bottom</guidance>
    <guidance>If multiple languages are present, note transitions between languages</guidance>
    <guidance>h1: The main title. Usually only one per page.</guidance>
    <guidance>Do not skip heading levels (e.g., h1 directly to h3)</guidance>
</help>
```

**Why it matters**: 🤔 Help content addresses the "what if" scenarios that rules can't anticipate. It's advisory rather than mandatory.

---

### 📖 `*_dictionary.xml` - Terminology Definitions

Defines element types and domain-specific terms.

**Purpose**: Ensures consistent interpretation of terminology.

**Example**:
```xml
<full_text_dictionary>
    <element type="header">
        <description>Text at the top of the page, often containing titles or page numbers.
        Not to be confused with heading elements (h1-h6).</description>
    </element>
    <element type="h1">
        <description>Primary heading or document title. The most prominent heading.</description>
    </element>
    <element type="artifact">
        <description>Decorative or non-meaningful content that should be ignored by screen readers</description>
    </element>
</full_text_dictionary>
```

**Why it matters**: 📚 Explicit definitions prevent ambiguity. "Header" vs "heading" confusion is eliminated by defining both.

---

## 🔬 Full Text Analyzer Deep Dive

The `full_text_analyzer` demonstrates the complete pattern. Here's how each file contributes:

### 📂 File Loading Order (from manifest)

```json
{
    "analyzer": {
        "name": "full_text_analyzer",
        "prompt_files": [
            "full_text_job_role.xml",
            "full_text_context.xml",
            "full_text_rules.xml",
            "full_text_tasks_extraction.xml",
            "full_text_format.xml",
            "full_text_help.xml",
            "full_text_dictionary.xml"
        ],
        "expected_output_tokens": 6000
    }
}
```

### 🔗 How Files Work Together

1. **👤 Job Role** tells the model it's a "Full Text Extraction Specialist" with expertise in document layouts, reading order, and accessibility requirements.

2. **🎯 Context** explains that output will be used for PDF/UA accessibility tagging, that IDs enable joining with bounding box data, and that heading hierarchy is critical for screen readers.

3. **⚖️ Rules** mandate extracting ALL text, using sequential IDs, maintaining reading order, and not summarizing content.

4. **✅ Tasks** walk through: examine page → identify elements → assign IDs and types → determine order → handle special cases → format output → review.

5. **📋 Format** shows the exact XML structure with `<elements>`, `<element>` tags with id/type/order attributes, and nested `<text>` content.

6. **💡 Help** provides guidance on multi-column layouts, heading hierarchy decisions, figure alt-text, and footnote handling.

7. **� Dictionary** defines each element type (header vs h1, paragraph, sidebar, footnote, figure, artifact) so the model classifies correctly.

### 🎯 Why This Structure Works

- **� Redundancy is intentional**: Key concepts appear in multiple files (e.g., "unique IDs" in rules, tasks, and help). This reinforcement improves adherence.

- **� Order matters**: Identity → Context → Rules → Tasks → Format → Help → Dictionary follows a logical progression from "who you are" to "what terms mean."

- **🔗 Cross-references**: Files reference each other (`refer to the <full_text_dictionary/>`) creating an interconnected instruction set.

- **😮‍💨 Breathing room**: "Take a deep breath" prompts encourage careful processing rather than rushing.

---

## 🔧 Core System Prompts

Located at `core_system_prompts/`:

### ⚖️ `core_rules/rules.xml` - Universal Rules

Applied to ALL analyzers via the wrapper:

```xml
<core_rules>
    <rule>NEVER GIVE A PREAMBLE OR INTRODUCE YOUR FINAL RESPONSE. DO NOT SAY THINGS LIKE,
        'Here is my analysis of the provided image in the specified XML format:'</rule>
    <rule>For non-English languages and non-Latin right to left scripts and pictographic
        style languages such as Chinese, Japanese, Korean, Arabic, Persian, or Hindi, ensure
        special characters are properly used in your response.</rule>
    <rule>Your response must correctly capture diacritics in Latin and non-Latin scripts.</rule>
    <rule>When responding or outputting your task results, skip the preamble. Go straight to the
        response_format.</rule>
    <rule>Do not surround any of your output in backticks (`).</rule>
    <rule>Take a deep breath and double-check that you have generated properly formatted, valid
        XML in your response.</rule>
    <rule>If you encounter an error or cannot complete your task, you must reply in the format
        provided in the <error_response_format></error_response_format> XML tags</rule>
    <rule>For any text you generate, do not use "Unicode Direction Formatting Codes"</rule>
    <rule>Do not hallucinate or invent information not present in the provided content.</rule>
    <rule>You must not include any <notes /> XML tags in your response.</rule>
    <rule>You must not produce any markdown style code blocks in your response.</rule>
</core_rules>
```

**Why universal rules exist**: 🛡️ These prevent common failure modes across all analyzers (preambles, markdown formatting, hallucination) without repeating them in every analyzer's prompts.

### ❌ Error Handlers

Located at `core_system_prompts/error_handling/`:

- `error_handler.xml`: Standard 500-style error with message, type, location
- `not_found_handler.xml`: Response when no elements are found

---

## 🔄 Placeholder System

Dynamic values can be injected at runtime:

```xml
<guidance>The image dimensions are [[PIXEL_WIDTH]] x [[PIXEL_HEIGHT]] pixels</guidance>
```

The `PromptLoader.replace_placeholders()` method substitutes actual values:
```python
placeholders = {
    "PIXEL_WIDTH": "1920",
    "PIXEL_HEIGHT": "1080"
}
```

**Use cases**:
- 📐 Image dimensions for spatial reasoning
- 🔑 Session IDs for tracking
- ⚙️ Dynamic configuration values

---

## 📁 Directory Structure

```
s3_files/
├── core_system_prompts/
│   ├── prompt_system_wrapper.xml      # Main wrapper template
│   ├── core_rules/
│   │   └── rules.xml                  # Universal rules for all analyzers
│   └── error_handling/
│       ├── error_handler.xml          # Standard error format
│       └── not_found_handler.xml      # No elements found format
├── manifests/
│   ├── full_text_analyzer.json        # Analyzer configurations
│   ├── table_analyzer.json
│   └── ...
├── prompts/
│   ├── full_text_analyzer/
│   │   ├── full_text_job_role.xml
│   │   ├── full_text_context.xml
│   │   ├── full_text_rules.xml
│   │   ├── full_text_tasks_extraction.xml
│   │   ├── full_text_format.xml
│   │   ├── full_text_help.xml
│   │   └── full_text_dictionary.xml
│   ├── table_analyzer/
│   └── ...
└── schemas/
    └── [analyzer_name].json           # Output validation schemas
```

---

## 📋 Manifest Structure

Each analyzer has a manifest file in `manifests/` with this structure:

```json
{
    "tool": {
        "name": "analyze_full_text_tool",
        "description": "Tool description for MCP registration",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": { "type": "string", "description": "..." },
                "aws_profile": { "type": "string", "description": "..." },
                "session_id": { "type": "string", "description": "..." },
                "audit_mode": { "type": "boolean", "description": "..." }
            },
            "required": ["image_path", "session_id"]
        }
    },
    "analyzer": {
        "name": "full_text_analyzer",
        "description": "Analyzer description",
        "enhancement_eligible": true,
        "model_selections": {
            "primary": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "fallback_list": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"]
        },
        "max_retries": 3,
        "prompt_analyzer_prompt_base_path": "prompts",
        "prompt_files": [
            "full_text_job_role.xml",
            "full_text_context.xml",
            "full_text_rules.xml",
            "full_text_tasks_extraction.xml",
            "full_text_format.xml",
            "full_text_help.xml",
            "full_text_dictionary.xml"
        ],
        "examples_path": "prompts",
        "max_examples": 0,
        "analysis_text": "full text content",
        "expected_output_tokens": 6000,
        "output_extension": "xml"
    },
    "metadata": {
        "version": "1.0.0",
        "dependencies": ["boto3"],
        "wizard_managed": true,
        "last_modified": "2025-12-22T23:50:59.081758+00:00",
        "analyzer_type": "standard"
    },
    "evaluation": {
        "likert_labels": ["Failed", "Major errors", "Partial success", "Minor issues", "Perfect"],
        "pass_threshold": 3,
        "questions": {
            "core": [...],
            "tool_specific": [...]
        }
    }
}
```

---

## ✨ Benefits of This Architecture

1. **🔧 Maintainability**: Change one aspect (e.g., output format) without touching other files
2. **🧪 Testability**: Test individual prompt components in isolation
3. **♻️ Reusability**: Share dictionaries or help files across similar analyzers
4. **📜 Versioning**: Track changes to specific prompt aspects over time
5. **🧪 A/B Testing**: Swap individual files to test variations
6. **🐛 Debugging**: Identify which prompt section causes issues
7. **👥 Collaboration**: Different team members can own different prompt aspects
