# Deployment Guide: Agentic Image Enhancer

This guide covers building, deploying, and testing the agentic image enhancer Lambda function.

## Prerequisites

1. **Docker** installed and running
2. **AWS CLI** configured with appropriate credentials
3. **AWS Account** with permissions for:
   - ECR (Elastic Container Registry)
   - Lambda
   - S3 (for input/output buckets)
   - Bedrock (for Claude Sonnet 4.6 access)
4. **CDK** installed (for infrastructure deployment)

## Build Process

### Step 1: Build Container Image

The image enhancer uses a container-based Lambda deployment. Build and push the container:

```bash
cd deployment/lambdas
./build_container_lambdas.sh <deployment_id>
```

This script:
1. Authenticates with ECR
2. Builds the Docker image for linux/amd64 platform
3. Tags as `badgers-<deployment_id>:image_enhancer`
4. Pushes to ECR repository

**Manual Build** (if needed):
```bash
cd deployment/lambdas/containers/image_enhancer

# Build image
docker build --platform linux/amd64 -t image-enhancer:latest .

# Tag for ECR
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/badgers-<deployment_id>"
docker tag image-enhancer:latest ${ECR_REPO}:image_enhancer

# Authenticate and push
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com

docker push ${ECR_REPO}:image_enhancer
```

### Step 2: Deploy via CDK

From the project root:

```bash
cdk deploy
```

Or deploy specific stack:
```bash
cdk deploy LambdaAnalyzerStack
```

CDK will:
- Create Lambda function from ECR image
- Configure environment variables
- Attach execution role with S3 + Bedrock permissions
- Set timeout (300s), memory (2048MB), concurrency (5)

## Configuration

### Environment Variables

Set in `lambda_stack.py` or override in AWS console:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VISION_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID |
| `MAX_ITERATIONS` | `2` | Max agent iterations |
| `MAX_IMAGE_DIMENSION` | `4000` | Max dimension for LLM |
| `JPEG_QUALITY` | `85` | LLM image encoding quality |
| `OUTPUT_QUALITY` | `95` | Final output quality |
| `OUTPUT_BUCKET` | (from CDK) | S3 bucket for enhanced images |
| `AWS_REGION` | `us-west-2` | Bedrock region |
| `LOGGING_LEVEL` | `INFO` | Log level |

### IAM Permissions Required

The Lambda execution role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::input-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::output-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/lambda/badgers_image_enhancer:*"
    }
  ]
}
```

## Testing

### Test via AWS Console

1. Navigate to Lambda function `badgers_image_enhancer`
2. Click "Test" tab
3. Create test event with content from `test-event.json`:

```json
{
  "body": "{\"image_path\": \"s3://test-bucket/test-images/manuscript.jpg\", \"document_type\": \"manuscript\", \"enhancement_level\": \"moderate\", \"session_id\": \"test_session_001\", \"output_quality\": 85, \"skip_upscale\": true}"
}
```

4. Execute test
5. Review execution results and CloudWatch logs

### Test via AWS CLI

```bash
aws lambda invoke \
  --function-name badgers_image_enhancer \
  --payload file://test-event.json \
  --region us-west-2 \
  response.json

# View response
cat response.json | jq .
```

### Verify Output

Check S3 bucket for enhanced image:
```bash
aws s3 ls s3://<output-bucket>/test_session_001/enhanced/
```

Expected file:
```
manuscript_enhanced_20250223_123456.jpg
```

## Monitoring

### CloudWatch Logs

Log group: `/aws/lambda/badgers_image_enhancer`

**View logs in real-time**:
```bash
aws logs tail /aws/lambda/badgers_image_enhancer --follow
```

**Filter by session**:
```bash
aws logs filter-log-events \
  --log-group-name /aws/lambda/badgers_image_enhancer \
  --filter-pattern "test_session_001"
```

### Key Log Entries

```
INFO | ============================================================
INFO |   STRANDS AGENTIC IMAGE ENHANCEMENT
INFO |   Source: manuscript
INFO |   Dimensions: 2400x1800
INFO |   Model: us.anthropic.claude-sonnet-4-6
INFO |   Max iterations: 2
INFO | ============================================================
INFO | Running Strands agent...
INFO | enhance_image called: Document appears faded, applying gentle contrast enhancement
INFO | Finished: winner=enhanced — Enhanced contrast and sharpness improved text readability
INFO | Winner: ENHANCED (after 2 iteration(s))
INFO | Reasoning: Enhanced contrast and sharpness improved text readability
```

### CloudWatch Metrics

Monitor these metrics:
- **Invocations**: Total invocation count
- **Duration**: Execution time (expect 30-180 seconds)
- **Errors**: Invocation errors
- **Throttles**: Concurrent execution limits hit
- **Memory Usage**: Peak memory (expect 512-1536MB)

### Cost Tracking

Each enhancement makes 2-6 Bedrock API calls (Claude Sonnet 4.6 vision):
- **Minimal** (1 iteration): ~2-3 calls
- **Moderate** (2 iterations): ~4-5 calls
- **Aggressive** (3 iterations): ~6-7 calls

**Estimated cost per image**: $0.015-0.045 (based on Claude Sonnet 4.6 pricing)

Use inference profiles for detailed cost tracking (already configured in CDK).

## Troubleshooting

### Issue: Container Image Not Found in ECR

**Solution**: Build and push container
```bash
cd deployment/lambdas
./build_container_lambdas.sh <deployment_id>
```

### Issue: Lambda Timeout (300s)

**Symptoms**: Task timed out after 300.00 seconds

**Solutions**:
1. Use "minimal" or "moderate" enhancement_level (reduce iterations)
2. Check Bedrock latency (region, throttling?)
3. Increase Lambda timeout:
   ```bash
   aws lambda update-function-configuration \
     --function-name badgers_image_enhancer \
     --timeout 600
   ```

### Issue: Memory Issues (137 Exit Code)

**Symptoms**: Lambda exits with code 137 (out of memory)

**Solutions**:
1. Increase Lambda memory allocation:
   ```bash
   aws lambda update-function-configuration \
     --function-name badgers_image_enhancer \
     --memory-size 4096
   ```
2. Check image size (very large images may need preprocessing)

### Issue: Bedrock Throttling

**Symptoms**: `ThrottlingException` or `ServiceQuotaExceededException`

**Solutions**:
1. Check concurrent executions (currently limited to 5)
2. Request quota increase for Bedrock
3. Add retry logic with exponential backoff
4. Use inference profiles for priority access

### Issue: Import Error - strands

**Symptoms**: `ModuleNotFoundError: No module named 'strands'`

**Solution**: Rebuild container with updated requirements.txt
```bash
cd deployment/lambdas
./build_container_lambdas.sh <deployment_id>
```

Verify requirements.txt includes:
```
strands-agents>=0.1.0
anthropic>=0.40.0
```

### Issue: Agent Doesn't Finish

**Symptoms**: Response shows `winner: "original"` with reasoning `"Timeout: Agent exceeded MAX_ITERATIONS without finishing"`

**Cause**: Agent didn't call `finish_enhancement` tool after MAX_ITERATIONS

**Solution**: This is handled automatically by the code (forces finish with original). If persistent:
1. Check CloudWatch logs for tool use sequence
2. Verify system prompt is correct in agentic_enhancer.py
3. May indicate LLM behavior issue (unusual)

### Issue: Poor Enhancement Quality

**Symptoms**: Enhanced image looks worse than original

**Observation**: Agent should detect this and choose `winner: "original"`

**If agent chose "enhanced" incorrectly**:
1. Check quality metrics in response history
2. Review agent reasoning
3. Try different document_type for better context
4. May need to adjust system prompt for specific document types

## Updating the Function

### Code Changes

After modifying Python code:

1. **Rebuild and push**:
   ```bash
   cd deployment/lambdas
   ./build_container_lambdas.sh <deployment_id>
   ```

2. **Update Lambda**:
   ```bash
   aws lambda update-function-code \
     --function-name badgers_image_enhancer \
     --image-uri <ecr-uri>:image_enhancer
   ```

3. **Wait for update**:
   ```bash
   aws lambda wait function-updated \
     --function-name badgers_image_enhancer
   ```

### Environment Variable Changes

```bash
aws lambda update-function-configuration \
  --function-name badgers_image_enhancer \
  --environment Variables="{
    VISION_MODEL=us.anthropic.claude-sonnet-4-6,
    MAX_ITERATIONS=3,
    MAX_IMAGE_DIMENSION=4000
  }"
```

## Production Considerations

### Scaling

- **Concurrent Executions**: Currently 5 reserved, adjust based on volume
- **Provisioned Concurrency**: Consider for low-latency requirements
- **Batch Processing**: Use SQS + Lambda for high volume

### Cost Optimization

- Use "minimal" enhancement_level when possible (fewer Bedrock calls)
- Monitor usage via inference profiles
- Consider image downsampling before enhancement for large images
- Set appropriate Lambda memory (right-size based on actual usage)

### Security

- **VPC**: Deploy in VPC if accessing private resources
- **Secrets**: Use AWS Secrets Manager for sensitive configuration
- **S3 Encryption**: Enable encryption at rest (SSE-S3 or SSE-KMS)
- **IAM**: Follow principle of least privilege

### High Availability

- **Multi-Region**: Deploy to multiple regions for disaster recovery
- **S3 Replication**: Replicate input/output buckets
- **Monitoring**: Set up CloudWatch alarms for failures and throttling

## Rollback

If issues arise after deployment:

1. **Revert to previous image**:
   ```bash
   # List images
   aws ecr describe-images \
     --repository-name badgers-<deployment_id> \
     --query 'sort_by(imageDetails,& imagePushedAt)[-2:]'
   
   # Update Lambda to previous image
   aws lambda update-function-code \
     --function-name badgers_image_enhancer \
     --image-uri <previous-image-uri>
   ```

2. **Or redeploy previous CDK commit**:
   ```bash
   git checkout <previous-commit>
   cdk deploy
   ```

## Support

For issues with deployment or operation:
1. Check CloudWatch logs for detailed error messages
2. Review IAM permissions (S3, Bedrock, CloudWatch)
3. Verify container image is available in ECR
4. Test locally with Docker if possible

For code issues, see [README.md](README.md) and the main project documentation.
