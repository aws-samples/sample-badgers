# Remediation Analyzer Deployment

## Issue Fixed

The remediation analyzer Lambda was failing with `"No module named 'foundation'"` because the container-based Lambda deployment wasn't including the foundation module.

## Changes Made

1. **Dockerfile**: Added foundation module to the Docker image
2. **build_container_lambdas.sh**: Updated to copy foundation module to build context before building
3. **.dockerignore**: Created to optimize Docker build performance

## Deployment Steps

### 1. Rebuild the Container Image

From the `deployment/lambdas` directory:

```bash
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment/lambdas

# Build and push to ECR (replace with your deployment ID)
./build_container_lambdas.sh <your-deployment-id>

# Example:
# ./build_container_lambdas.sh 89a27522
```

### 2. Update the Lambda Function

After pushing the new image, update your Lambda function to use the new image:

```bash
# Option A: Redeploy the entire stack with CDK
cd /Users/rbpotter/Documents/SourceControl/sample-badgers/deployment
cdk deploy BadgersLambdaStack --context deployment_id=<your-deployment-id>

# Option B: Force Lambda to pull the latest image
aws lambda update-function-code \
  --function-name badgers_remediation_analyzer \
  --image-uri <account-id>.dkr.ecr.us-west-2.amazonaws.com/badgers-<deployment-id>:remediation_analyzer
```

### 3. Test the Lambda

Use this test payload:

```json
{
  "pdf_path": "s3://badgers-source-89a27522/1_test_chinese_book.pdf",
  "session_id": "test_session_001",
  "title": "Chinese Book Test Document",
  "lang": "zh-CN",
  "dpi": 150
}
```

### 4. Verify Success

The Lambda should now successfully:
- Import the foundation module
- Download the PDF from S3
- Analyze pages for document structure
- Apply PDF/UA accessibility tags
- Return a 200 status code with tagged PDF S3 URI

## Test Event Parameters

### Required
- `pdf_path`: S3 URI to the PDF file

### Optional
- `session_id`: Tracking identifier (default: "no_session")
- `title`: PDF metadata title (default: "Accessible Document")
- `lang`: Language code (default: "en-US", use "zh-CN" for Chinese)
- `dpi`: Rendering resolution (default: 150)
- `correlation_uri`: S3 path to correlation XML for guided remediation
- `page_b64_uris`: Dict mapping page number (string) to S3 URI of pre-processed b64 image
- `aws_profile`: AWS profile for local testing

## Build Context

The foundation module is copied from `deployment/lambdas/layer/python/foundation` during the build process and included in the container at `${LAMBDA_TASK_ROOT}/foundation/`.

## Dependencies

The container includes:
- Python 3.12 base image
- Foundation module (BedrockClient, AnalyzerFoundation, etc.)
- PDF processing libraries (pymupdf, pikepdf)
- Accessibility tagging components
- Cell grid resolver for vision-based coordinate detection
