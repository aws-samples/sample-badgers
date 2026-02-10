# Pricing Calculator

Gradio-based tool for estimating Bedrock model costs for document/image analysis.

## Quick Start

```bash
cd frontend
uv run python main.py
```

Then open `http://localhost:7860` and navigate to the Pricing Calculator tab.

## Features

| Section            | Purpose                                                |
| ------------------ | ------------------------------------------------------ |
| Model Selection    | Choose from configured Bedrock models                  |
| Document Presets   | Pre-configured settings for common document types      |
| Analyzer Selection | Pick specific analyzers to include in cost calculation |
| Advanced Mode      | Calculate costs based on actual prompt token counts    |
| Cost Outputs       | Total input/output tokens and costs                    |

## Document Presets

Pre-configured settings for common document types:

| Preset               | Recommended Model   | Use Case                                   |
| -------------------- | ------------------- | ------------------------------------------ |
| Default              | Claude Sonnet 4.5   | General purpose documents                  |
| Insurance Claims     | Claude Haiku 4.5    | High-volume claim forms                    |
| Legal Contracts      | Claude Sonnet 4.5   | Complex contracts requiring interpretation |
| Medical Records      | Claude Sonnet 4.5   | Clinical notes, lab results, imaging       |
| Financial Documents  | Claude Sonnet 4.5   | Statements, trading docs, audit materials  |
| Invoices & Receipts  | Claude Haiku 4.5    | High-volume transactional documents        |
| Engineering Diagrams | Claude Sonnet 4.5   | Technical drawings, schematics             |
| Regulatory Filings   | Amazon Nova Premier | SEC filings, compliance documents          |

## Configuration

Pricing and models are configured in `config/pricing_config.json`:

```json
{
    "ingestion": {
        "characters_per_token": 4.5,
        "avg_characters_per_word": 5,
        "avg_words_per_page": 500,
        "avg_pages_per_document": 15,
        "avg_tokens_per_image": 1600
    },
    "presets": {
        "insurance_claims": {
            "name": "Insurance Claims",
            "recommended_model": "Claude Haiku 4.5",
            "words_per_page": 350,
            "pages_per_document": 5,
            "output_ratio": 0.3
        }
    },
    "models": {
        "model-id": {
            "name": "Display Name",
            "input_cost_per_million": 3.00,
            "output_cost_per_million": 15.00
        }
    }
}
```

## Available Models

| Display Name        | Model ID                                           | Input $/M | Output $/M |
| ------------------- | -------------------------------------------------- | --------- | ---------- |
| Claude Sonnet 4.5   | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | $3.00     | $15.00     |
| Claude Haiku 4.5    | `us.anthropic.claude-haiku-4-5-20251001-v1:0`      | $1.00     | $5.00      |
| Amazon Nova Premier | `us.amazon.nova-premier-v1:0`                      | $2.50     | $12.50     |

## Updating Prices

Edit `config/pricing_config.json` directly. Prices are not fetched from AWS Pricing API since newer models (Claude 4.5, Nova Premier) aren't yet available there.

## Requirements

- `gradio` package (included in project dependencies)
