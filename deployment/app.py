#!/usr/bin/env python3
"""CDK app for BADGERS deployment."""

import os
import uuid
import json
import warnings
from pathlib import Path

import aws_cdk as cdk
from stacks import (
    S3Stack,
    IAMStack,
    CognitoStack,
    LambdaSpecialistStack,
    AgentCoreECRStack,
    AgentCoreGatewayStack,
    AgentCoreRuntimeWebSocketStack,
    AgentCoreMemoryStack,
    InferenceProfilesStack,
    CustomSpecialistsStack,
    XRayTransactionSearchStack,
    VpcStack,
    DynamoDBStack,
    ECSStack,
)

warnings.filterwarnings("ignore", module="typeguard")

# Suppress typeguard non-runtime protocol warnings from AWS CDK
warnings.filterwarnings(
    "ignore",
    message="Typeguard cannot check the .* protocol because it is a non-runtime protocol",
    category=UserWarning,
)

app = cdk.App()

# Configuration - uses current AWS credentials
env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
)

# Generate unique deployment ID (or use existing from context)
# Use: cdk deploy -c deployment_id=abc12345 to reuse an existing deployment
deployment_id = app.node.try_get_context("deployment_id") or uuid.uuid4().hex[:8]
print(f"Deployment ID: {deployment_id}")

# Stack name prefix
STACK_PREFIX = "badgers"

# Deployment tags - customize these for your deployment
# These tags are applied to all resources across all stacks
# resource_name and resource_description are set per-resource in each stack
deployment_tags = {
    "application_name": "badgers",
    "application_description": "BADGERS - Broad Agentic Document Generative Extraction and Recognition System",
    "environment": "dev",
    "owner": "your-team",
    "cost_center": "your-cost-center",
    "project_code": "your-project-code",
    "cdk_stack_prefix": STACK_PREFIX,
    "team": "your-team",
    "team_contact_email": "team@company.com",
}

# S3 buckets for configs and outputs
s3_stack = S3Stack(
    app,
    f"{STACK_PREFIX}-s3",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="S3 buckets for BADGERS",
)

# Cognito for AgentCore Gateway authentication
cognito_stack = CognitoStack(
    app,
    f"{STACK_PREFIX}-cognito",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="Cognito authentication for AgentCore Gateway",
)

# DynamoDB jobs table for specialist job tracking
dynamodb_stack = DynamoDBStack(
    app,
    f"{STACK_PREFIX}-dynamodb",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="DynamoDB jobs table for BADGERS specialist job tracking",
)

# IAM roles and policies
iam_stack = IAMStack(
    app,
    f"{STACK_PREFIX}-iam",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    config_bucket=s3_stack.config_bucket,
    source_bucket=s3_stack.source_bucket,
    output_bucket=s3_stack.output_bucket,
    env=env,
    description="IAM roles for BADGERS",
)

# ECR repository for AgentCore Runtime container (and container Lambdas)
ecr_stack = AgentCoreECRStack(
    app,
    f"{STACK_PREFIX}-ecr",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="ECR repository for AgentCore Runtime agent container",
)

# Inference Profiles for cost tracking
inference_profiles_stack = InferenceProfilesStack(
    app,
    f"{STACK_PREFIX}-inference-profiles",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="Application Inference Profiles for cost tracking and usage monitoring",
)

# Load deployment config for selective specialist deployment
enabled_specialists = None
deployment_config_path = Path("./deployment_config.json")
if deployment_config_path.exists():
    with open(deployment_config_path, encoding="utf-8") as f:
        deployment_config = json.load(f)
    specialists_config = deployment_config.get("specialists", {})
    enabled_specialists = {
        name for name, cfg in specialists_config.items() if cfg.get("enabled", True)
    }
    disabled = {
        name for name, cfg in specialists_config.items() if not cfg.get("enabled", True)
    }
    if disabled:
        print(f"Disabled specialists: {', '.join(sorted(disabled))}")
    print(
        f"Enabled specialists: {len(enabled_specialists)} of {len(specialists_config)}"
    )

# Lambda functions and layer
lambda_stack = LambdaSpecialistStack(
    app,
    f"{STACK_PREFIX}-lambda",
    deployment_tags=deployment_tags,
    execution_role=iam_stack.lambda_role,
    config_bucket=s3_stack.config_bucket,
    output_bucket=s3_stack.output_bucket,
    ecr_repository=ecr_stack.repository,
    inference_profiles_stack=inference_profiles_stack,
    enabled_specialists=enabled_specialists,
    env=env,
    description="Lambda specialists for BADGERS",
)
lambda_stack.add_dependency(ecr_stack)
lambda_stack.add_dependency(inference_profiles_stack)

# X-Ray Transaction Search (account-level prerequisite for AgentCore tracing)
# This is a singleton per account/region — if already enabled, destroy this stack
# or remove it from the deploy sequence.
xray_stack = XRayTransactionSearchStack(
    app,
    f"{STACK_PREFIX}-xray",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    indexing_percentage=1,  # 1% is free tier
    env=env,
    description="X-Ray Transaction Search for AgentCore tracing",
)

# AgentCore Gateway with Lambda targets
gateway_stack = AgentCoreGatewayStack(
    app,
    f"{STACK_PREFIX}-gateway",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    lambda_functions=lambda_stack.functions,
    config_bucket=s3_stack.config_bucket,
    env=env,
    description="AgentCore Gateway with Lambda tool targets",
)
gateway_stack.add_dependency(lambda_stack)
gateway_stack.add_dependency(cognito_stack)

# AgentCore Memory for session persistence
memory_stack = AgentCoreMemoryStack(
    app,
    f"{STACK_PREFIX}-memory",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="AgentCore Memory for BADGERS session persistence",
)


# AgentCore Runtime WebSocket
runtime_websocket_stack = AgentCoreRuntimeWebSocketStack(
    app,
    f"{STACK_PREFIX}-runtime-websocket",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    ecr_repository_uri=ecr_stack.repository.repository_uri,
    gateway_url=gateway_stack.gateway.gateway_url or "",
    cognito_credentials_secret_arn=cognito_stack.credentials_secret.secret_arn,
    output_bucket_name=s3_stack.output_bucket.bucket_name,
    config_bucket_name=s3_stack.config_bucket.bucket_name,
    source_bucket_name=s3_stack.source_bucket.bucket_name,
    memory_id=memory_stack.memory.attr_memory_id,
    s3_kms_key_arn=s3_stack.s3_kms_key.key_arn,
    inference_profiles_stack=inference_profiles_stack,
    image_tag="websocket",
    env=env,
    description="AgentCore Runtime for BADGERS agent with WebSocket streaming",
)
runtime_websocket_stack.add_dependency(ecr_stack)
runtime_websocket_stack.add_dependency(gateway_stack)
runtime_websocket_stack.add_dependency(cognito_stack)
runtime_websocket_stack.add_dependency(memory_stack)
runtime_websocket_stack.add_dependency(inference_profiles_stack)
runtime_websocket_stack.add_dependency(xray_stack)

# Add dependencies
iam_stack.add_dependency(s3_stack)  # IAM needs S3 buckets for grant permissions
lambda_stack.add_dependency(iam_stack)  # Lambda needs IAM role
lambda_stack.add_dependency(s3_stack)  # Lambda needs S3 bucket names

# Note: Gateway authentication with Cognito is configured separately
# The Gateway stack creates the MCP endpoint
# Runtime automatically authenticates via AgentCore Identity

# Custom Specialists Stack (optional - only deployed if custom specialists exist)
# Custom specialists are created via the wizard UI and saved locally
custom_specialists_registry = Path("./custom_specialists/specialist_registry.json")
if custom_specialists_registry.exists():
    # Use Fn.import_value to reference exports - no explicit dependencies needed
    custom_specialists_stack = CustomSpecialistsStack(
        app,
        f"{STACK_PREFIX}-custom-specialists",
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        config_bucket_name=cdk.Fn.import_value(f"{STACK_PREFIX}-s3-ConfigBucketName"),
        output_bucket_name=cdk.Fn.import_value(f"{STACK_PREFIX}-s3-OutputBucketName"),
        foundation_layer_arn=cdk.Fn.import_value(f"{STACK_PREFIX}-lambda-LayerArn"),
        lambda_role_arn=cdk.Fn.import_value(f"{STACK_PREFIX}-iam-LambdaRoleArn"),
        gateway_id=cdk.Fn.import_value(f"{STACK_PREFIX}-gateway-GatewayId"),
        gateway_role_arn=cdk.Fn.import_value(f"{STACK_PREFIX}-gateway-GatewayRoleArn"),
        kms_key_arn=cdk.Fn.import_value(f"{STACK_PREFIX}-s3-S3KmsKeyArn"),
        claude_sonnet_profile_arn=cdk.Fn.import_value(
            f"{STACK_PREFIX}-inference-profiles-ClaudeSonnetProfileArn"
        ),
        claude_haiku_profile_arn=cdk.Fn.import_value(
            f"{STACK_PREFIX}-inference-profiles-ClaudeHaikuProfileArn"
        ),
        nova_premier_profile_arn=cdk.Fn.import_value(
            f"{STACK_PREFIX}-inference-profiles-NovaPremierProfileArn"
        ),
        claude_opus_46_profile_arn=cdk.Fn.import_value(
            f"{STACK_PREFIX}-inference-profiles-ClaudeOpus46ProfileArn"
        ),
        claude_opus_45_profile_arn=cdk.Fn.import_value(
            f"{STACK_PREFIX}-inference-profiles-ClaudeOpus45ProfileArn"
        ),
        env=env,
        description="Custom specialists created via the wizard UI",
    )

# --- UI infrastructure (ECS Express Gateway) ---
# No hosted zone, domain, or ACM certificate required: the ECS Express Gateway
# service provisions and manages its own load balancer and HTTPS endpoint.

# VPC with private subnets and AWS service VPC endpoints
vpc_stack = VpcStack(
    app,
    f"{STACK_PREFIX}-vpc",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="VPC for the BADGERS UI ECS service",
)

# Unified UI on ECS Express Gateway, authenticated via Cognito OIDC/PKCE
ecs_stack = ECSStack(
    app,
    f"{STACK_PREFIX}-ecs",
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    config_bucket_arn=s3_stack.config_bucket.bucket_arn,
    output_bucket_arn=s3_stack.output_bucket.bucket_arn,
    source_bucket_arn=s3_stack.source_bucket.bucket_arn,
    config_bucket_name=s3_stack.config_bucket.bucket_name,
    output_bucket_name=s3_stack.output_bucket.bucket_name,
    source_bucket_name=s3_stack.source_bucket.bucket_name,
    jobs_table_arn=dynamodb_stack.jobs_table.table_arn,
    kms_key_arn=s3_stack.s3_kms_key.key_arn,
    private_subnet_ids=vpc_stack.private_subnet_ids,
    ui_task_sg_id=vpc_stack.ui_task_sg.security_group_id,
    cognito_user_pool_id=cognito_stack.user_pool.user_pool_id,
    cognito_ui_client_id=cognito_stack.ui_client.user_pool_client_id,
    ecr_repository_uri=ecr_stack.repository.repository_uri,
    agentcore_runtime_websocket_arn=runtime_websocket_stack.runtime.attr_agent_runtime_arn,
    agentcore_gateway_id=gateway_stack.gateway.gateway_id or "",
    image_tag="frontend",
    env=env,
    description="BADGERS unified UI — ECS Express Gateway + Cognito OIDC auth",
)
ecs_stack.add_dependency(vpc_stack)
ecs_stack.add_dependency(cognito_stack)
ecs_stack.add_dependency(ecr_stack)
ecs_stack.add_dependency(s3_stack)
ecs_stack.add_dependency(dynamodb_stack)
ecs_stack.add_dependency(gateway_stack)
ecs_stack.add_dependency(runtime_websocket_stack)

# ── cdk-nag (opt-in via CDK_NAG=1 env var) ─────────────────────────────────
if os.environ.get("CDK_NAG", "").strip() in ("1", "true", "yes"):
    from cdk_nag import AwsSolutionsChecks

    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    print("cdk-nag: AwsSolutionsChecks enabled")

app.synth()
