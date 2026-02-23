# CDK Deployment Guide - Remediation Analyzer Container

This guide covers deploying the remediation_analyzer as a container-based Lambda function using AWS CDK with the foundation library as a Lambda Layer.

---

## Architecture: Container + Layer

```
┌──────────────────────────────────────────────────────────┐
│                    AWS Lambda Function                    │
│              badgers_remediation_analyzer                 │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │   Container Image from ECR                          │ │
│  │   Tag: remediation_analyzer                         │ │
│  │                                                      │ │
│  │   Contains:                                          │ │
│  │   • System libraries (mupdf, qpdf, libxml2)         │ │
│  │   • Binary Python packages (pymupdf, pikepdf, lxml) │ │
│  │   • Analyzer Python code (lambda_handler.py, etc.)  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │   Lambda Layer (mounted at /opt/python/)            │ │
│  │   analyzer-foundation layer                          │ │
│  │                                                      │ │
│  │   Contains:                                          │ │
│  │   • foundation/ (pure Python)                        │ │
│  │     - analyzer_foundation.py                         │ │
│  │     - bedrock_client.py                              │ │
│  │     - All other foundation modules                   │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## CDK Stack Configuration

The remediation_analyzer is already configured in the CDK stack:

### [deployment/stacks/lambda_stack.py](../../stacks/lambda_stack.py)

```python
# Line 25: Container functions list
CONTAINER_FUNCTIONS = ["image_enhancer", "remediation_analyzer"]

# Lines 302-318: create_container_functions()
# Automatically creates Lambda functions for each container in the list

# Lines 319-370: _create_ecr_container_function()
# Updated to attach foundation layer to container functions:
layers = [self.foundation_layer]  # Foundation layer mounted at /opt/python/

function = lambda_.Function(
    self,
    f"ContainerFunction-{func_name}",
    function_name=f"badgers_{func_name}",
    code=lambda_.Code.from_ecr_image(
        repository=self.ecr_repository,
        tag_or_digest=func_name,  # Must match: "remediation_analyzer"
    ),
    handler=lambda_.Handler.FROM_IMAGE,
    runtime=lambda_.Runtime.FROM_IMAGE,
    role=self.execution_role,
    layers=layers,  # ✅ Foundation layer attached
    timeout=Duration.seconds(300),
    memory_size=2048,
    ...
)
```

---

## Pre-Deployment Steps

### 1. Build the Foundation Layer

The foundation layer must be built first:

```bash
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment/lambdas

# Build the foundation layer
./build_foundation_layer.sh

# Verify the layer zip exists
ls -lh layer.zip
# Should be ~5-10 MB
```

### 2. Build the Container Image

Build the container WITHOUT bundling foundation (it comes from the layer):

```bash
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment/lambdas/containers/remediation_analyzer

# Build with layer mode (foundation NOT bundled in container)
./deploy-with-foundation.sh --use-layer --build-only

# Verify the image was built
docker images remediation-analyzer:latest
```

### 3. Push Container Image to ECR

The CDK stack expects the image in ECR with a specific tag. You have two options:

#### Option A: Manual Push (for testing)

```bash
# Get your AWS account ID and region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-west-2

# ECR repository name (from CDK stack)
ECR_REPO_NAME="badgers-ecr-repository"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# Authenticate Docker to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_URI}

# Tag the image with function name (MUST be "remediation_analyzer")
docker tag remediation-analyzer:latest ${ECR_URI}:remediation_analyzer

# Push to ECR
docker push ${ECR_URI}:remediation_analyzer
```

#### Option B: Automated Push with deploy script

```bash
# Get the ECR repository URI from CDK outputs
ECR_REPO_URI=$(aws cloudformation describe-stacks \
  --stack-name badgers-ecr \
  --query 'Stacks[0].Outputs[?OutputKey==`RepositoryUri`].OutputValue' \
  --output text)

# Set AWS profile and region
export AWS_PROFILE=your-profile-name
export AWS_REGION=us-west-2

# Push to ECR (script will handle authentication and tagging)
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment/lambdas/containers/remediation_analyzer

# Modify build.sh to use correct ECR repo and tag
# Then run:
./build.sh --push --region $AWS_REGION --profile $AWS_PROFILE
```

**CRITICAL**: The image tag in ECR **MUST be exactly `remediation_analyzer`** to match the CDK configuration (line 352 in lambda_stack.py).

---

## Deployment with CDK

### 1. Deploy the Full Stack

From the deployment directory:

```bash
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment

# Set AWS credentials
export AWS_PROFILE=your-profile-name
export AWS_REGION=us-west-2

# Deploy all stacks (or just the lambda stack)
cdk deploy badgers-lambda --require-approval never

# Or deploy all stacks
cdk deploy --all --require-approval never
```

### 2. Verify Deployment

```bash
# Check that the Lambda function was created
aws lambda get-function --function-name badgers_remediation_analyzer

# Check that the foundation layer is attached
aws lambda get-function-configuration --function-name badgers_remediation_analyzer \
  --query 'Layers[*].Arn'
# Should show: ["arn:aws:lambda:us-west-2:ACCOUNT:layer:analyzer-foundation:VERSION"]

# Check environment variables
aws lambda get-function-configuration --function-name badgers_remediation_analyzer \
  --query 'Environment.Variables'
# Should show CONFIG_BUCKET, OUTPUT_BUCKET, etc.
```

---

## Testing

### Test the Deployed Function

```bash
# Create a test event
cat > /tmp/test-remediation-event.json <<EOF
{
  "pdf_path": "s3://your-config-bucket/samples/test.pdf",
  "session_id": "test-cdk-001",
  "title": "Test Document",
  "lang": "en-US",
  "dpi": 150
}
EOF

# Invoke the function
aws lambda invoke \
  --function-name badgers_remediation_analyzer \
  --payload file:///tmp/test-remediation-event.json \
  --cli-binary-format raw-in-base64-out \
  /tmp/response.json

# Check the response
cat /tmp/response.json | jq .

# Check CloudWatch logs
aws logs tail /aws/lambda/badgers_remediation_analyzer --follow
```

---

## Updating the Function

### Update Container Code

When you make changes to the analyzer code:

```bash
# 1. Rebuild the container
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment/lambdas/containers/remediation_analyzer
./deploy-with-foundation.sh --use-layer --build-only

# 2. Push new image to ECR with the same tag
docker tag remediation-analyzer:latest ${ECR_URI}:remediation_analyzer
docker push ${ECR_URI}:remediation_analyzer

# 3. Update Lambda function to use new image
aws lambda update-function-code \
  --function-name badgers_remediation_analyzer \
  --image-uri ${ECR_URI}:remediation_analyzer

# 4. Wait for update to complete
aws lambda wait function-updated --function-name badgers_remediation_analyzer

# 5. Test
aws lambda invoke --function-name badgers_remediation_analyzer \
  --payload file:///tmp/test-remediation-event.json /tmp/response.json
```

### Update Foundation Layer

When you update the foundation library:

```bash
# 1. Rebuild the foundation layer
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment/lambdas
./build_foundation_layer.sh

# 2. Redeploy the lambda stack
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment
cdk deploy badgers-lambda

# The new layer version is automatically attached to all Lambda functions
```

---

## CDK Stack Dependencies

The Lambda stack depends on these other stacks:

```python
# From deployment/app.py
lambda_stack.add_dependency(ecr_stack)        # ECR repository must exist
lambda_stack.add_dependency(iam_stack)        # IAM role must exist
lambda_stack.add_dependency(s3_stack)         # S3 buckets must exist
lambda_stack.add_dependency(inference_profiles_stack)  # Inference profiles must exist
```

**Initial deployment order:**
1. `badgers-s3` - S3 buckets
2. `badgers-iam` - IAM roles
3. `badgers-ecr` - ECR repository
4. `badgers-inference-profiles` - Bedrock inference profiles
5. **Build and push container image to ECR** ← Manual step
6. `badgers-lambda` - Lambda functions (including remediation_analyzer)

---

## Troubleshooting

### Error: "The image with imageUri ... does not exist"

**Cause**: Container image not pushed to ECR before deploying Lambda stack

**Solution**:
```bash
# Push the image first
cd deployment/lambdas/containers/remediation_analyzer
docker tag remediation-analyzer:latest ${ECR_URI}:remediation_analyzer
docker push ${ECR_URI}:remediation_analyzer

# Then deploy
cd ../../..
cdk deploy badgers-lambda
```

### Error: "ImportError: No module named 'foundation'"

**Cause**: Foundation layer not attached or not built

**Solution**:
```bash
# Check if layer is attached
aws lambda get-function-configuration \
  --function-name badgers_remediation_analyzer \
  --query 'Layers'

# If empty, redeploy with updated CDK stack
cd deployment
cdk deploy badgers-lambda

# If layer exists but imports fail, rebuild the layer
cd lambdas
./build_foundation_layer.sh
cd ..
cdk deploy badgers-lambda
```

### Error: "Task timed out after 300 seconds"

**Cause**: Large PDFs or slow Bedrock responses

**Solution**: Increase timeout in [lambda_stack.py](../../stacks/lambda_stack.py):
```python
# Line 357
timeout=Duration.seconds(600),  # Changed from 300 to 600
```

Then redeploy:
```bash
cdk deploy badgers-lambda
```

### Error: "Out of memory" (exit code 137)

**Cause**: Large PDF processing exceeds 2GB memory

**Solution**: Increase memory in [lambda_stack.py](../../stacks/lambda_stack.py):
```python
# Line 358
memory_size=5120,  # Changed from 2048 to 5120 (5GB)
```

Then redeploy:
```bash
cdk deploy badgers-lambda
```

---

## Cost Considerations

### Container + Layer Approach

**Storage**:
- Container image: ~290 MB × $0.10/GB-month = $0.029/month
- Foundation layer: ~5 MB × $0.10/GB-month = $0.0005/month
- **Total**: ~$0.03/month for storage

**Execution**:
- Memory: 2 GB (default)
- Typical duration: 30-60 seconds per page
- Cost: ~$0.0004 per invocation (2 GB × 60 sec)

**Benefits of Layer Approach**:
- Foundation layer shared across ALL analyzers → Saves storage costs
- Smaller container images → Faster cold starts
- Independent updates → Update foundation without rebuilding containers

---

## Summary

✅ **CDK Stack Configuration**: remediation_analyzer is configured as a container function
✅ **Foundation Layer**: Attached automatically by the updated lambda_stack.py
✅ **Container Image**: Must be pushed to ECR with tag `remediation_analyzer`
✅ **Deployment**: Use `cdk deploy badgers-lambda` after pushing the image
✅ **Updates**: Push new image → `aws lambda update-function-code`

This setup provides:
- **No layer size limits** for heavy dependencies (container handles pymupdf, pikepdf, lxml)
- **Shared foundation** across all analyzers (layer provides foundation library)
- **Independent updates** for code and foundation
- **CDK-managed infrastructure** for consistent deployments

---

## Next Steps

1. **Build foundation layer**: `cd lambdas && ./build_foundation_layer.sh`
2. **Build container**: `cd containers/remediation_analyzer && ./deploy-with-foundation.sh --use-layer --build-only`
3. **Deploy ECR stack** (if not done): `cdk deploy badgers-ecr`
4. **Push container to ECR**: Follow Option A or B above
5. **Deploy Lambda stack**: `cdk deploy badgers-lambda`
6. **Test**: Use the test event above

For questions or issues, refer to:
- [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md) - Compatibility analysis
- [DEPLOYMENT.md](DEPLOYMENT.md) - General deployment guide
- [README.md](README.md) - Analyzer documentation
