# Deployment Guide - Remediation Analyzer Container Lambda

This document describes how to deploy the remediation analyzer as a container-based AWS Lambda function.

## Prerequisites

1. **Docker** installed and running
2. **AWS CLI** configured with appropriate credentials
3. **AWS Account** with permissions for:
   - ECR (Elastic Container Registry)
   - Lambda
   - S3 (for input/output buckets)
   - Bedrock (for vision model access)

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Container Lambda                     │
│                                                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  Lambda Runtime (Python 3.12)               │    │
│  │                                             │    │
│  │  ┌────────────────────────────────────┐    │    │
│  │  │  Remediation Analyzer Code         │    │    │
│  │  │  - lambda_handler.py               │    │    │
│  │  │  - pdf_accessibility_tagger.py     │    │    │
│  │  │  - cell_grid_resolver.py           │    │    │
│  │  │  - etc.                            │    │    │
│  │  └────────────────────────────────────┘    │    │
│  │                                             │    │
│  │  ┌────────────────────────────────────┐    │    │
│  │  │  Foundation Library                 │    │    │
│  │  │  (via Lambda Layer or bundled)     │    │    │
│  │  └────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
│                                                       │
└─────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
    S3 Input           Bedrock Vision        S3 Output
   (PDFs/XML)          (Claude models)    (Tagged PDFs)
```

## Foundation Library Options

The remediation analyzer depends on the BADGERS foundation library. You have three options:

### Option 1: Lambda Layer (Recommended)

1. **Create foundation layer**:
   ```bash
   # In your foundation library directory
   mkdir -p python
   cp -r foundation/ python/
   zip -r foundation-layer.zip python/

   # Upload to Lambda
   aws lambda publish-layer-version \
     --layer-name badgers-foundation \
     --zip-file fileb://foundation-layer.zip \
     --compatible-runtimes python3.12
   ```

2. **Attach layer to Lambda function** (see Step 3 below)

### Option 2: Bundle in Docker Image

1. **Copy foundation into build directory**:
   ```bash
   cp -r /path/to/foundation/ ./foundation/
   ```

2. **Uncomment in Dockerfile**:
   ```dockerfile
   # Change this line:
   # COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/
   # To:
   COPY foundation/ ${LAMBDA_TASK_ROOT}/foundation/
   ```

3. **Build image** (includes foundation)

### Option 3: Install from Package Repository

If foundation is available as a pip package:

1. **Uncomment in Dockerfile**:
   ```dockerfile
   RUN pip install --no-cache-dir strands-agents>=0.1.0
   ```

2. **Build image**

## Step-by-Step Deployment

### Step 1: Prepare the Build Directory

```bash
cd /path/to/remediation_analyzer_NEW

# Verify all files are present
ls -l
# Should show:
#   Dockerfile
#   lambda_handler.py
#   pdf_accessibility_tagger.py
#   pdf_accessibility_auditor.py
#   pdf_accessibility_models.py
#   cell_grid_resolver.py
#   requirements.txt
#   build.sh
```

### Step 2: Build and Push Container Image

#### Using the build script (easiest):

```bash
# Build only (for testing)
./build.sh --build-only

# Build and push to ECR
./build.sh --push --region us-west-2 --profile my-profile

# Build, push, and update existing Lambda
./build.sh --update-lambda --region us-west-2
```

#### Manual build:

```bash
# Set variables
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_NAME="remediation-analyzer"
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"

# Build image
docker build --platform linux/amd64 -t ${IMAGE_NAME}:latest .

# Create ECR repository (if not exists)
aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${AWS_REGION}

# Authenticate Docker to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Tag and push
docker tag ${IMAGE_NAME}:latest ${ECR_REPO}:latest
docker push ${ECR_REPO}:latest
```

### Step 3: Create Lambda Function

#### Using AWS Console:

1. **Navigate to Lambda** in AWS Console
2. **Create Function** → Container image
3. **Settings**:
   - Function name: `remediation-analyzer`
   - Container image URI: `<your-ecr-uri>:latest`
   - Architecture: x86_64

4. **Configuration**:
   - Memory: 3008 MB (recommended)
   - Timeout: 5 minutes (300 seconds)
   - Environment variables:
     ```
     CONFIG_BUCKET=your-config-bucket
     OUTPUT_BUCKET=your-output-bucket
     ANALYZER_NAME=remediation_analyzer
     LOGGING_LEVEL=INFO
     AWS_REGION=us-west-2
     ```

5. **Attach Lambda Layer** (if using Option 1 for foundation):
   - Configuration → Layers → Add a layer
   - Select: Custom layers
   - Choose: badgers-foundation
   - Version: latest

6. **Execution Role** - Ensure it has:
   - S3 read/write access to CONFIG_BUCKET and OUTPUT_BUCKET
   - Bedrock invoke model access
   - CloudWatch Logs write access

#### Using AWS CLI:

```bash
# Create execution role first (if needed)
aws iam create-role \
  --role-name remediation-analyzer-role \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name remediation-analyzer-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy \
  --role-name remediation-analyzer-role \
  --policy-name remediation-analyzer-policy \
  --policy-document file://permissions-policy.json

# Create Lambda function
aws lambda create-function \
  --function-name remediation-analyzer \
  --package-type Image \
  --code ImageUri=${ECR_REPO}:latest \
  --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/remediation-analyzer-role \
  --timeout 300 \
  --memory-size 3008 \
  --environment Variables="{
    CONFIG_BUCKET=your-config-bucket,
    OUTPUT_BUCKET=your-output-bucket,
    ANALYZER_NAME=remediation_analyzer,
    LOGGING_LEVEL=INFO,
    AWS_REGION=us-west-2
  }"

# Attach foundation layer (if using)
aws lambda update-function-configuration \
  --function-name remediation-analyzer \
  --layers arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:layer:badgers-foundation:1
```

### Step 4: Test the Function

#### Local testing:

```bash
# Run container locally
docker run -p 9000:8080 \
  -e CONFIG_BUCKET=your-config-bucket \
  -e OUTPUT_BUCKET=your-output-bucket \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
  remediation-analyzer:latest

# In another terminal, invoke
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{
    "pdf_path": "/path/to/test.pdf",
    "session_id": "test-001",
    "title": "Test Document",
    "lang": "en-US"
  }'
```

#### Lambda testing:

```bash
# Create test event
cat > test-event.json <<EOF
{
  "pdf_path": "s3://your-bucket/test.pdf",
  "correlation_uri": "s3://your-bucket/test.xml",
  "session_id": "test-001",
  "title": "Test Document",
  "lang": "en-US",
  "dpi": 150
}
EOF

# Invoke
aws lambda invoke \
  --function-name remediation-analyzer \
  --payload file://test-event.json \
  --cli-binary-format raw-in-base64-out \
  response.json

# Check response
cat response.json | jq .
```

## Updating the Function

When you make code changes:

```bash
# Rebuild and push
./build.sh --push --region us-west-2

# Update Lambda function code
aws lambda update-function-code \
  --function-name remediation-analyzer \
  --image-uri ${ECR_REPO}:latest

# Wait for update to complete
aws lambda wait function-updated --function-name remediation-analyzer
```

## Monitoring

### CloudWatch Logs

```bash
# Tail logs in real-time
aws logs tail /aws/lambda/remediation-analyzer --follow

# Filter by session ID
aws logs filter-log-events \
  --log-group-name /aws/lambda/remediation-analyzer \
  --filter-pattern "test-001"
```

### CloudWatch Metrics

Key metrics to monitor:
- **Duration**: Should be < 300s (timeout)
- **Memory**: Peak usage (adjust if consistently high)
- **Errors**: Any invocation errors
- **Throttles**: Concurrent execution limits

### X-Ray Tracing (Optional)

Enable for detailed performance insights:

```bash
aws lambda update-function-configuration \
  --function-name remediation-analyzer \
  --tracing-config Mode=Active
```

## Troubleshooting

### Common Issues

1. **Container fails to start**
   ```
   Error: Runtime exited with error: exit status 1
   ```
   - Check CloudWatch logs for Python import errors
   - Verify foundation library is available (layer or bundled)
   - Ensure all dependencies in requirements.txt are installed

2. **Timeout errors**
   ```
   Task timed out after 300.00 seconds
   ```
   - Increase Lambda timeout (max 15 minutes)
   - Increase memory (faster CPU with more memory)
   - Check Bedrock vision model latency

3. **Out of memory**
   ```
   MemoryError or Lambda exits with code 137
   ```
   - Increase Lambda memory allocation
   - Large PDFs may require 5-10 GB

4. **Permission errors**
   ```
   AccessDeniedException: User is not authorized
   ```
   - Check S3 bucket permissions in IAM role
   - Verify Bedrock model access in IAM role
   - Ensure KMS key permissions if S3 uses encryption

5. **Foundation import errors**
   ```
   ModuleNotFoundError: No module named 'foundation'
   ```
   - Verify Lambda layer is attached
   - Check layer is compatible with python3.12
   - Confirm foundation is in /opt/python/ (layer) or ${LAMBDA_TASK_ROOT} (bundled)

### Debug Mode

Enable detailed logging:

```bash
aws lambda update-function-configuration \
  --function-name remediation-analyzer \
  --environment Variables="{
    CONFIG_BUCKET=your-config-bucket,
    OUTPUT_BUCKET=your-output-bucket,
    LOGGING_LEVEL=DEBUG
  }"
```

## Production Considerations

### Scaling

- **Concurrency**: Set reserved concurrency if needed
- **Provisioned Concurrency**: For low-latency requirements
- **Batch Processing**: Use SQS + Lambda for high volume

### Cost Optimization

- **Memory**: Right-size based on actual usage
- **Timeout**: Set as low as possible for your workload
- **Bedrock Caching**: Enable prompt caching if supported
- **S3 Lifecycle**: Archive old outputs to Glacier

### Security

- **VPC**: Deploy in VPC if accessing private resources
- **Secrets**: Use AWS Secrets Manager for sensitive config
- **Encryption**: Enable S3 encryption at rest (SSE-S3 or SSE-KMS)
- **IAM**: Follow principle of least privilege

### High Availability

- **Multi-Region**: Deploy to multiple regions
- **S3 Replication**: Replicate input/output buckets
- **Monitoring**: Set up CloudWatch alarms for failures

## IAM Policy Examples

### trust-policy.json

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### permissions-policy.json

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-config-bucket/*",
        "arn:aws:s3:::your-config-bucket",
        "arn:aws:s3:::your-input-bucket/*",
        "arn:aws:s3:::your-input-bucket"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-output-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": [
        "arn:aws:logs:*:*:log-group:/aws/lambda/remediation-analyzer:*"
      ]
    }
  ]
}
```

## Support

For deployment issues:
1. Check CloudWatch Logs
2. Review IAM permissions
3. Verify foundation library availability
4. Test locally with Docker first

For code issues, see [README.md](README.md) and the main project documentation.
