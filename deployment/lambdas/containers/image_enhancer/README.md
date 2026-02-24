# Agentic Image Enhancer

**Iterative image enhancement using Claude Sonnet 4.6 vision model with Strands Agents framework.**

## Overview

The agentic image enhancer uses an intelligent, iterative approach to document image enhancement. Unlike traditional fixed-pipeline enhancers, this system:

- **Analyzes** each image with a vision LLM
- **Decides** which enhancement operations are needed
- **Applies** targeted operations with appropriate intensities
- **Compares** results against quality metrics
- **Iterates** until reaching optimal enhancement or MAX_ITERATIONS

The agent can choose to skip enhancement if the original is already good, or retry with different approaches if the first attempt degrades quality.

## Architecture

```
┌────────────────────────────────────┐
│  Vision LLM (Claude Sonnet 4.6)    │  ← Decision maker
│  via Strands Agent framework       │     Analyzes & decides
└──────────────┬─────────────────────┘
               │ tool_use calls
               ▼
┌────────────────────────────────────┐
│  4 Agent Tools                     │
│  • enhance_image(ops)              │  ← Executes LLM decisions
│  • compare_with_original()         │
│  • reset_to_original()             │
│  • finish_enhancement(winner)      │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│  10 Enhancement Operations         │  ← Image processing
│  contrast, brightness, sharpen,    │     OpenCV operations
│  denoise, deskew, white_balance,   │
│  equalize, crop, invert, stains    │
└────────────────────────────────────┘
```

## Available Operations

| Operation | Purpose | Use Case |
|-----------|---------|----------|
| **contrast** (CLAHE) | Adaptive contrast enhancement | Uneven lighting, fading |
| **brightness** | Overall brightness adjustment | Dark/light documents |
| **sharpen** | Unsharp mask edge enhancement | Blurry text/diagrams |
| **denoise** | Non-local means noise removal | Scanner noise, grain |
| **deskew** | Rotation correction via Hough lines | Skewed scans |
| **white_balance** | Gray-world color correction | Yellowing/aging |
| **equalize_histogram** | Global tonal range spread | Severely faded docs |
| **auto_crop** | Document boundary detection | Remove borders |
| **invert** | Negative inversion | Dark-background docs |
| **remove_stains** | Morphological background removal | Foxing, age spots |

Each operation supports:
- **Intensity control** (0.0 to 1.0)
- **Regional application** (normalized 0-1 coordinates)
- **Sequential ordering** (operations apply in specified order)

## Input Parameters

### Lambda Event Format

```json
{
  "body": "{\"image_path\": \"s3://bucket/path/image.jpg\", \"document_type\": \"manuscript\", \"enhancement_level\": \"moderate\", \"session_id\": \"session_001\", \"output_quality\": 85, \"skip_upscale\": true}"
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_path` | string | Yes* | - | S3 URI (e.g., `s3://bucket/key`) |
| `image_data` | string | Yes* | - | Base64-encoded image (alternative to image_path) |
| `document_type` | string | No | "auto" | Document type hint for LLM context |
| `enhancement_level` | string | No | "moderate" | Enhancement aggressiveness |
| `session_id` | string | Yes | "no_session" | Session identifier for S3 organization |
| `output_quality` | integer | No | 85 | JPEG quality (1-100) |
| `skip_upscale` | boolean | No | true | Skip pre-processing upscale |

*Either `image_path` or `image_data` is required.

### Document Types

Maps to LLM context for better enhancement decisions:

- `"manuscript"` → "18th century handwritten manuscript"
- `"annotated"` → "historical document with handwritten annotations"
- `"sheet_music"` → "musical score with performance annotations"
- `"diagram"` → "technical diagram or chart"
- `"printed"` → "printed historical document"
- `"mixed"` → "mixed media document with multiple content types"
- `"auto"` → No context provided, LLM assesses independently

### Enhancement Levels

Maps to MAX_ITERATIONS for the agent:

- `"minimal"` → 1 iteration (quick pass)
- `"moderate"` → 2 iterations (balanced, default)
- `"aggressive"` → 3 iterations (thorough)

## Output Format

### Success Response

```json
{
  "statusCode": 200,
  "body": {
    "result": {
      "s3_output_uri": "s3://bucket/session_001/enhanced/image_enhanced_20250223_123456.jpg",
      "operations_applied": ["deskew", "contrast", "sharpen"],
      "original_shape": [1200, 900, 3],
      "final_shape": [1200, 900, 3],
      "winner": "enhanced",
      "iterations": 2,
      "reasoning": "Enhanced contrast and sharpness improved text readability significantly.",
      "history": [
        {
          "iteration": 0,
          "operations": [...],
          "comparison": {...}
        }
      ]
    },
    "success": true
  }
}
```

### Response Fields

#### Backward Compatible Fields
- `s3_output_uri`: S3 location of enhanced image (if OUTPUT_BUCKET configured)
- `enhanced_image_base64`: Base64-encoded enhanced image (if no OUTPUT_BUCKET)
- `operations_applied`: List of operation names used
- `original_shape`: [height, width, channels]
- `final_shape`: [height, width, channels]

#### New Agentic Fields
- `winner`: "original" or "enhanced" (agent's choice)
- `iterations`: Number of agent iterations performed
- `reasoning`: Agent's explanation for winner selection
- `history`: Detailed iteration log with operations and metrics

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID |
| `MAX_ITERATIONS` | `2` | Max agent iterations (overridden by enhancement_level at runtime) |
| `MAX_IMAGE_DIMENSION` | `4000` | Max dimension for LLM submission |
| `JPEG_QUALITY` | `85` | Quality for LLM image encoding |
| `OUTPUT_QUALITY` | `95` | Quality for final output |
| `OUTPUT_BUCKET` | - | S3 bucket for enhanced images (if not set, returns base64) |
| `AWS_REGION` | `us-west-2` | AWS region for Bedrock |
| `LOGGING_LEVEL` | `INFO` | Python log level |

## Comparison: Old vs. New

| Feature | Old (Fixed Pipeline) | New (Agentic) |
|---------|---------------------|---------------|
| **Operations** | 6 fixed (upscale, deskew, denoise, contrast, balance, sharpen) | 10 available, agent selects |
| **Decision Making** | Hardcoded sequence | LLM vision analysis per image |
| **Iterations** | 1 (single pass) | 1-3 (with feedback loop) |
| **Regional Operations** | No | Yes (normalized 0-1 coords) |
| **Quality Metrics** | Basic (shape, skew) | Comprehensive (contrast, sharpness, brightness, saturation, edges, yellowing) |
| **Winner Selection** | Always enhanced | Agent chooses original or enhanced based on metrics |
| **Adaptability** | Same for all images | Tailored to document type and condition |
| **Skip Option** | Must always process | Can skip if original is already good |
| **Failure Recovery** | N/A | Resets and retries with different approach |

### Key Advantages

1. **Intelligence**: Vision LLM understands document type, degradation, and enhancement needs
2. **Efficiency**: Avoids unnecessary operations; skips enhancement for already-good images
3. **Quality**: Metrics-driven decisions prevent over-processing
4. **Flexibility**: Regional operations target specific problem areas
5. **Transparency**: Clear reasoning for each enhancement decision

## Dependencies

### Container (requirements.txt)
- `opencv-python-headless==4.9.0.80` (image processing)
- `numpy>=1.26.0,<2.0` (numerical operations)
- `strands-agents>=0.1.0` (agentic framework)
- `anthropic>=0.40.0` (Claude API with Bedrock support)
- `boto3>=1.34.0` (AWS SDK)
- `Pillow>=10.0.0` (image I/O)

### System (Dockerfile)
- `mesa-libGL` (OpenCV headless rendering)

## Testing

### Sample Test Event

See [test-event.json](test-event.json) for a complete example.

### Via AWS Console
1. Open Lambda function `badgers_image_enhancer`
2. Configure test event with test-event.json content
3. Execute and review logs

### Via AWS CLI
```bash
aws lambda invoke \
  --function-name badgers_image_enhancer \
  --payload file://test-event.json \
  --region us-west-2 \
  response.json

cat response.json | jq .
```

### Expected Output
- CloudWatch logs show agent tool use (enhance_image, compare_with_original, finish_enhancement)
- Response includes winner selection and reasoning
- S3 output at `s3://<bucket>/<session_id>/enhanced/<name>_enhanced_<timestamp>.jpg`

## Monitoring

### CloudWatch Logs
Log group: `/aws/lambda/badgers_image_enhancer`

Key log entries:
- `"Running Strands agent..."` - Agent invocation start
- `"Winner: ENHANCED (after N iteration(s))"` - Enhancement result
- `"Reasoning: ..."` - Agent's decision rationale
- Tool use logs with operation details and metrics

### Key Metrics
- **Duration**: 30-180 seconds depending on iterations
- **Memory**: 512-1536MB (2048MB allocated)
- **Bedrock Calls**: 2-6 per image (2-3 iterations × 2 calls per iteration)
- **Cost**: ~$0.015-0.045 per image

## Troubleshooting

### Common Issues

1. **Timeout (300s exceeded)**
   - Reduce MAX_ITERATIONS or use "minimal" enhancement_level
   - Check Bedrock latency (retries adding delay?)
   - Consider increasing Lambda timeout to 600s

2. **Memory Issues**
   - Large images may require more memory
   - Increase Lambda memory allocation (currently 2048MB)

3. **Agent Doesn't Finish**
   - Automatic fallback after MAX_ITERATIONS
   - Check CloudWatch logs for tool use sequence
   - Agent may be stuck in loop (should not happen with current prompt)

4. **Import Error: strands**
   - Verify container was built with updated requirements.txt
   - Rebuild and push: `./build_container_lambdas.sh <deployment_id>`

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions.

**Quick Start**:
```bash
cd deployment/lambdas
./build_container_lambdas.sh <deployment_id>
cd ../..
cdk deploy
```

## License

Same as the parent BADGERS project.
