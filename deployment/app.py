#!/usr/bin/env python3
"""CDK app for BADGERS deployment."""

import os
import uuid
import json
import warnings
from pathlib import Path

import aws_cdk as cdk
from profile_resolver import ProfileResolutionError, resolve_profile
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
    WorkflowRunnerStack,
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

# ── Deployment identity ────────────────────────────────────────────────────
# DEPLOYMENT_ID is a human-chosen label ("dev", "demo"). STACK_SUFFIX is a short
# random token that makes stack and resource names unique, so several deployments
# can coexist in one account and region.
#
# deploy.sh generates the suffix once and persists both in
# .deploy-state/{DEPLOYMENT_ID}.json, then exports them. Context values are
# accepted as a fallback for running cdk directly.
DEPLOYMENT_ID = os.environ.get("DEPLOYMENT_ID") or app.node.try_get_context(
    "deployment_id"
)
STACK_SUFFIX = os.environ.get("STACK_SUFFIX") or app.node.try_get_context(
    "stack_suffix"
)

if not DEPLOYMENT_ID:
    raise SystemExit(
        "DEPLOYMENT_ID is required. Use ./deploy.sh, or pass it explicitly:\n"
        "  DEPLOYMENT_ID=dev STACK_SUFFIX=a1b cdk deploy ...\n"
        "  cdk deploy -c deployment_id=dev -c stack_suffix=a1b ..."
    )
if not STACK_SUFFIX:
    raise SystemExit(
        "STACK_SUFFIX is required. ./deploy.sh generates and persists one; pass it "
        "explicitly when running cdk directly (see .deploy-state/)."
    )

# Resource names carry both parts: badgers-config-dev-a1b, /badgers-dev-a1b/...
# Every stack already derives its resource names from this single value.
deployment_id = f"{DEPLOYMENT_ID}-{STACK_SUFFIX}"

# ECS Express Mode derives the load balancer scheme from the subnets it is given:
# public subnets produce an internet-facing ALB (and enable public IPs on the tasks),
# while subnets without an internet gateway produce an internal one. Default is public,
# because otherwise the UI's *.ecs.<region>.on.aws URL resolves but never answers.
# Set UI_PUBLIC_ACCESS=true only when the deployment operator intentionally wants an
# internet-facing UI.  Headless never creates the UI or VPC stacks.
PROFILE_NAME = os.environ.get("BADGERS_PROFILE") or app.node.try_get_context(
    "badgers_profile"
) or "full"
SPECIALIST_PACKS = tuple(
    pack.strip()
    for pack in os.environ.get("BADGERS_SPECIALIST_PACKS", "").split(",")
    if pack.strip()
)
DIRECT_SPECIALISTS = tuple(
    specialist.strip()
    for specialist in os.environ.get("BADGERS_DIRECT_SPECIALISTS", "").split(",")
    if specialist.strip()
)
SELECTION_OVERRIDE = os.environ.get("BADGERS_SELECTION_OVERRIDE", "").lower() == "true"
try:
    profile = resolve_profile(
        PROFILE_NAME,
        specialist_packs=SPECIALIST_PACKS or None,
        direct_specialists=DIRECT_SPECIALISTS or None,
        selection_override=SELECTION_OVERRIDE,
    )
except ProfileResolutionError as error:
    raise SystemExit(f"Invalid BADGERS profile '{PROFILE_NAME}': {error}") from error

CAPABILITIES = profile.capabilities
CONFIG_RELEASE_ID = os.environ.get("BADGERS_CONFIG_RELEASE_ID", "").strip()
CONFIG_PREFIX = f"releases/{CONFIG_RELEASE_ID}" if CONFIG_RELEASE_ID else ""
ROUTING_WORKFLOWS_BY_TYPE: dict[str, str] = {}
_routing_state = os.environ.get("BADGERS_ROUTING_STATE", "").strip()
if _routing_state:
    try:
        _routing = json.loads(Path(_routing_state).read_text(encoding="utf-8")).get(
            "desired_routing_configuration", {}
        )
        for workflow_id, workflow in _routing.get("workflows", {}).items():
            for content_type in workflow.get("input_contract", {}).get(
                "accepted_source_types", []
            ):
                ROUTING_WORKFLOWS_BY_TYPE[str(content_type)] = str(workflow_id)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError) as error:
        raise SystemExit(f"Could not load saved workflow routing configuration: {error}") from error
UI_PUBLIC_ACCESS = os.environ.get("UI_PUBLIC_ACCESS", "false").lower() == "true"
print(
    f"Deployment ID: {DEPLOYMENT_ID}   Stack suffix: {STACK_SUFFIX}   "
    f"Profile: {profile.display_name}"
)

# Stack name prefix. Stack names are BADGERS-{Name}-{DEPLOYMENT_ID}-{suffix}, carrying
# the same composite identity as resource names.
#
# The suffix alone would be enough for uniqueness, but it is not enough for identity: a
# suffix-only name means a wrong DEPLOYMENT_ID still resolves real stacks while every
# resource name derived from it points somewhere that does not exist. Including the ID
# makes each stack self-describing, so tooling can read a deployment's identity off the
# stack name instead of parsing it back out of a bucket name.
STACK_PREFIX = "BADGERS"


def _sn(name: str) -> str:
    """Build a stack name: BADGERS-{Name}-{DEPLOYMENT_ID}-{suffix}."""
    return f"{STACK_PREFIX}-{name}-{DEPLOYMENT_ID}-{STACK_SUFFIX}"


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
    _sn("S3"),
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="S3 buckets for BADGERS",
)

# Cognito is an agent/Gateway concern. Direct-use specialist packs expose no
# network endpoint and are invoked through customer-provided AWS IAM grants.
cognito_stack = None
if CAPABILITIES.get("gateway", False):
    cognito_stack = CognitoStack(
        app,
        _sn("Cognito"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        enable_ui=CAPABILITIES.get("ui_identity", False),
        enable_headless_runtime_auth=(profile.name == "headless"),
        env=env,
        description="Cognito authentication for BADGERS Gateway and Headless",
    )

# Full and Headless use the operation ledger. Direct-use specialists are
# deliberately stateless and leave JOBS_TABLE_NAME unset.
dynamodb_stack = None
if CAPABILITIES.get("minimal_operation_ledger", False):
    dynamodb_stack = DynamoDBStack(
        app,
        _sn("DynamoDB"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        env=env,
        description="DynamoDB jobs table for BADGERS specialist job tracking",
    )

# IAM roles and policies
iam_stack = IAMStack(
    app,
    _sn("IAM"),
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    config_bucket=s3_stack.config_bucket,
    source_bucket=s3_stack.source_bucket,
    output_bucket=s3_stack.output_bucket,
    jobs_table=dynamodb_stack.jobs_table if dynamodb_stack else None,
    env=env,
    description="IAM roles for BADGERS",
)
if dynamodb_stack is not None:
    iam_stack.add_stack_dependency(dynamodb_stack)

# Inference Profiles for cost tracking
inference_profiles_stack = InferenceProfilesStack(
    app,
    _sn("InferenceProfiles"),
    deployment_id=deployment_id,
    deployment_tags=deployment_tags,
    env=env,
    description="Application Inference Profiles for cost tracking and usage monitoring",
)

# Resolve a profile to an explicit specialist list before CDK creates resources.  The
# resolver validates packs against deployment_config.json, including disabled entries.
enabled_specialists = set(profile.specialists)
deployment_config_path = Path("./deployment_config.json")
with open(deployment_config_path, encoding="utf-8") as f:
    deployment_config = json.load(f)
specialists_config = deployment_config.get("specialists", {})
disabled = {
    name for name, cfg in specialists_config.items() if not cfg.get("enabled", True)
}
if disabled:
    print(f"Disabled specialists: {', '.join(sorted(disabled))}")
print(
    f"Profile specialists: {len(enabled_specialists)} of {len(specialists_config)} "
    f"({', '.join(sorted(enabled_specialists))})"
)

# Full preserves the legacy names for backwards compatibility.  A Headless
# deployment gets a deployment-scoped prefix so both products can coexist in an
# account/Region without Lambda name collisions.
specialist_resource_prefix = (
    "badgers"
    if profile.name == "full"
    else f"badgers_{deployment_id.replace('-', '_')}"
)

# An ECR repository is assembled when the resolved product or its selected
# specialists require container images. profile_resolver derives this capability
# from container-specialist selection, so direct DOCX/PPTX selections receive
# the same supported ECR path as the Full product.
ecr_stack = None
if CAPABILITIES.get("container_images", False):
    ecr_stack = AgentCoreECRStack(
        app,
        _sn("ECR"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        env=env,
        description="ECR repository for BADGERS runtime and container specialists",
    )

# Lambda functions and layer
lambda_stack = LambdaSpecialistStack(
    app,
    _sn("Lambda"),
    deployment_tags=deployment_tags,
    execution_role=iam_stack.lambda_role,
    config_bucket=s3_stack.config_bucket,
    output_bucket=s3_stack.output_bucket,
    jobs_table=dynamodb_stack.jobs_table if dynamodb_stack else None,
    ecr_repository=ecr_stack.repository if ecr_stack else None,
    inference_profiles_stack=inference_profiles_stack,
    enabled_specialists=enabled_specialists,
    resource_name_prefix=specialist_resource_prefix,
    config_prefix=CONFIG_PREFIX,
    env=env,
    description="Lambda specialists for BADGERS",
)
if ecr_stack is not None:
    lambda_stack.add_stack_dependency(ecr_stack)
lambda_stack.add_stack_dependency(inference_profiles_stack)
if dynamodb_stack is not None:
    lambda_stack.add_stack_dependency(dynamodb_stack)

# The durable runner is a private control-plane component. It has no
# customer-facing endpoint in this increment; a later adapter submits
# authenticated operations to this State Machine.
workflow_runner_stack = None
if CAPABILITIES.get("workflow_runner", False):
    assert dynamodb_stack is not None
    if not CONFIG_PREFIX:
        raise SystemExit(
            "BADGERS_CONFIG_RELEASE_ID is required when workflow_runner is enabled"
        )
    workflow_runner_stack = WorkflowRunnerStack(
        app,
        _sn("WorkflowRunner"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        config_bucket=s3_stack.config_bucket,
        source_bucket=s3_stack.source_bucket,
        output_bucket=s3_stack.output_bucket,
        s3_kms_key=s3_stack.s3_kms_key,
        jobs_table=dynamodb_stack.jobs_table,
        foundation_layer=lambda_stack.foundation_layer,
        specialist_functions=lambda_stack.functions,
        config_prefix=CONFIG_PREFIX,
        env=env,
        description="Private durable BADGERS workflow runner",
    )
    workflow_runner_stack.add_stack_dependency(s3_stack)
    workflow_runner_stack.add_stack_dependency(dynamodb_stack)
    workflow_runner_stack.add_stack_dependency(lambda_stack)

# X-Ray Transaction Search (account-level prerequisite for AgentCore tracing).
#
# Transaction Search is a singleton per account and region, and enabling it requires a
# CloudWatch Logs resource policy drawn from a hard, non-adjustable quota of 10 per
# region that is shared with every other project in the account.
#
# deploy.sh checks the live state before deploying and sets BADGERS_SKIP_XRAY=1 when
# Transaction Search is already active. The stack is then left out of the app entirely
# rather than merely dropped from the deploy list — the Runtime stack depends on it, so
# cdk would otherwise pull it back in and fail on the quota anyway.
SKIP_XRAY = (
    os.environ.get("BADGERS_SKIP_XRAY", "").strip() == "1"
    or not CAPABILITIES.get("advanced_tracing", False)
)

xray_stack = None
if SKIP_XRAY:
    print("X-Ray Transaction Search stack omitted by profile or BADGERS_SKIP_XRAY=1")
else:
    xray_stack = XRayTransactionSearchStack(
        app,
        _sn("XRay"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        indexing_percentage=1,  # 1% is free tier
        env=env,
        description="X-Ray Transaction Search for AgentCore tracing",
    )

# AgentCore Gateway with Lambda targets
gateway_stack = None
if CAPABILITIES.get("gateway", False):
    assert cognito_stack is not None
    gateway_stack = AgentCoreGatewayStack(
        app,
        _sn("Gateway"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        lambda_functions=lambda_stack.functions,
        config_bucket=s3_stack.config_bucket,
        cognito_stack_name=_sn("Cognito"),
        specialist_function_name_prefix=specialist_resource_prefix,
        config_prefix=CONFIG_PREFIX,
        env=env,
        description="AgentCore Gateway with Lambda tool targets",
    )
    gateway_stack.add_stack_dependency(lambda_stack)
    gateway_stack.add_stack_dependency(cognito_stack)

# AgentCore Memory is a Full-product feature.  The runtime remains usable without it:
# its entrypoint simply runs statelessly when AGENTCORE_MEMORY_ID is absent.
memory_stack = None
if CAPABILITIES.get("session_memory", False):
    memory_stack = AgentCoreMemoryStack(
        app,
        _sn("Memory"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        env=env,
        description="AgentCore Memory for BADGERS session persistence",
    )


# AgentCore Runtime WebSocket
runtime_websocket_stack = None
if CAPABILITIES.get("agent_runtime", False):
    assert ecr_stack is not None
    assert gateway_stack is not None
    assert cognito_stack is not None
    assert dynamodb_stack is not None
    runtime_websocket_stack = AgentCoreRuntimeWebSocketStack(
        app,
        _sn("RuntimeWebSocket"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        ecr_repository_uri=ecr_stack.repository.repository_uri,
        gateway_url=gateway_stack.gateway.gateway_url or "",
        cognito_credentials_secret_arn=cognito_stack.credentials_secret.secret_arn,
        output_bucket_name=s3_stack.output_bucket.bucket_name,
        config_bucket_name=s3_stack.config_bucket.bucket_name,
        source_bucket_name=s3_stack.source_bucket.bucket_name,
        memory_id=memory_stack.memory.attr_memory_id if memory_stack else None,
        s3_kms_key_arn=s3_stack.s3_kms_key.key_arn,
        inference_profiles_stack=inference_profiles_stack,
        jobs_table=dynamodb_stack.jobs_table,
        image_tag=os.environ.get("RUNTIME_IMAGE_TAG") or "websocket",
        websocket_enabled=CAPABILITIES.get("websocket_transport", False),
        config_prefix=CONFIG_PREFIX,
        routing_workflows_by_type=ROUTING_WORKFLOWS_BY_TYPE,
        workflow_state_machine_arn=(
            workflow_runner_stack.state_machine.state_machine_arn
            if workflow_runner_stack is not None
            else None
        ),
        headless_jwt_discovery_url=(
            cognito_stack.headless_discovery_url
            if profile.name == "headless"
            else None
        ),
        headless_jwt_client_id=(
            cognito_stack.headless_client.user_pool_client_id
            if profile.name == "headless" and cognito_stack.headless_client is not None
            else None
        ),
        env=env,
        description="AgentCore Runtime for BADGERS agent",
    )
    runtime_websocket_stack.add_stack_dependency(ecr_stack)
    runtime_websocket_stack.add_stack_dependency(gateway_stack)
    runtime_websocket_stack.add_stack_dependency(cognito_stack)
    if memory_stack is not None:
        runtime_websocket_stack.add_stack_dependency(memory_stack)
    runtime_websocket_stack.add_stack_dependency(inference_profiles_stack)
    if xray_stack is not None:
        runtime_websocket_stack.add_stack_dependency(xray_stack)
    runtime_websocket_stack.add_stack_dependency(dynamodb_stack)
    if workflow_runner_stack is not None:
        runtime_websocket_stack.add_stack_dependency(workflow_runner_stack)

# Add dependencies
iam_stack.add_stack_dependency(s3_stack)  # IAM needs S3 buckets for grant permissions
lambda_stack.add_stack_dependency(iam_stack)  # Lambda needs IAM role
lambda_stack.add_stack_dependency(s3_stack)  # Lambda needs S3 bucket names

# Note: Gateway authentication with Cognito is configured separately
# The Gateway stack creates the MCP endpoint
# Runtime automatically authenticates via AgentCore Identity

# Custom Specialists Stack (optional - only deployed if custom specialists exist)
# Custom specialists are created via the wizard UI and saved locally
custom_specialists_registry = Path("./custom_specialists/specialist_registry.json")
if CAPABILITIES.get("specialist_wizard", False) and custom_specialists_registry.exists():
    # Use Fn.import_value to reference exports - no explicit dependencies needed
    custom_specialists_stack = CustomSpecialistsStack(
        app,
        _sn("CustomSpecialists"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        config_bucket_name=cdk.Fn.import_value(f"{_sn('S3')}-ConfigBucketName"),
        output_bucket_name=cdk.Fn.import_value(f"{_sn('S3')}-OutputBucketName"),
        foundation_layer_arn=cdk.Fn.import_value(f"{_sn('Lambda')}-LayerArn"),
        lambda_role_arn=cdk.Fn.import_value(f"{_sn('IAM')}-LambdaRoleArn"),
        gateway_id=cdk.Fn.import_value(f"{_sn('Gateway')}-GatewayId"),
        gateway_role_arn=cdk.Fn.import_value(f"{_sn('Gateway')}-GatewayRoleArn"),
        kms_key_arn=cdk.Fn.import_value(f"{_sn('S3')}-S3KmsKeyArn"),
        env=env,
        description="Custom specialists created via the wizard UI",
    )

if CAPABILITIES.get("ui", False):
    # --- UI infrastructure (ECS Express Gateway) ---
    # No hosted zone, domain, or ACM certificate required: the ECS Express Gateway
    # service provisions and manages its own load balancer and HTTPS endpoint.
    vpc_stack = VpcStack(
        app,
        _sn("Vpc"),
        deployment_id=deployment_id,
        deployment_tags=deployment_tags,
        env=env,
        description="VPC for the BADGERS UI ECS service",
    )

    # Unified UI on ECS Express Gateway, authenticated via Cognito OIDC/PKCE
    ecs_stack = ECSStack(
        app,
        _sn("ECS"),
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
        private_subnet_ids=(
            vpc_stack.public_subnet_ids
            if UI_PUBLIC_ACCESS
            else vpc_stack.private_subnet_ids
        ),
        ui_task_sg_id=vpc_stack.ui_task_sg.security_group_id,
        cognito_user_pool_id=cognito_stack.user_pool.user_pool_id,
        cognito_ui_client_id=cognito_stack.ui_client.user_pool_client_id,
        ecr_repository_uri=ecr_stack.repository.repository_uri,
        agentcore_runtime_websocket_arn=runtime_websocket_stack.runtime.attr_agent_runtime_arn,
        agentcore_gateway_id=gateway_stack.gateway.gateway_id or "",
        # The Create Specialist wizard calls Bedrock from the UI container; the task role's
        # grant for that one model is generated from the registry, like every other grant.
        inference_profiles_stack=inference_profiles_stack,
        stack_suffix=STACK_SUFFIX,
        image_tag="frontend",
        env=env,
        description="BADGERS unified UI — ECS Express Gateway + Cognito OIDC auth",
    )
    ecs_stack.add_stack_dependency(vpc_stack)
    ecs_stack.add_stack_dependency(cognito_stack)
    ecs_stack.add_stack_dependency(ecr_stack)
    ecs_stack.add_stack_dependency(s3_stack)
    ecs_stack.add_stack_dependency(dynamodb_stack)
    ecs_stack.add_stack_dependency(gateway_stack)
    ecs_stack.add_stack_dependency(runtime_websocket_stack)
    ecs_stack.add_stack_dependency(inference_profiles_stack)
else:
    print("UI, VPC, and wizard stacks omitted by profile")

# ── cdk-nag (opt-in via CDK_NAG=1 env var) ─────────────────────────────────
if os.environ.get("CDK_NAG", "").strip() in ("1", "true", "yes"):
    from cdk_nag import AwsSolutionsChecks

    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    print("cdk-nag: AwsSolutionsChecks enabled")

app.synth()
