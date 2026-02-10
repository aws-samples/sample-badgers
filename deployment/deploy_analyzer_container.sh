cd lambdas

# Set your values
DEPLOYMENT_ID="25f3d830"  # e.g., 07a6cffb
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

ECR_REPO="badgers-${DEPLOYMENT_ID}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

# ECR login
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build and push just image_enhancer
docker build --platform linux/amd64 --provenance=false -t "${ECR_URI}:image_enhancer" ./containers/image_enhancer
docker push "${ECR_URI}:image_enhancer"
