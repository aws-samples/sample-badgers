"""AgentCore Gateway Stack for BADGERS."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Fn,
    RemovalPolicy,
    Tags,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct

from .log_delivery import (
    deliver_to_log_group,
    deliver_traces_to_xray,
    shared_delivery_policy,
)
from .nag_arn_renderings import account_renderings, region_renderings

try:
    import aws_cdk.aws_bedrock_agentcore_alpha as agentcore
except ImportError:
    import aws_cdk_aws_bedrock_agentcore_alpha as agentcore

try:  # cdk-nag is an optional synth-time aspect (enabled via CDK_NAG=1 in app.py)
    from cdk_nag import NagSuppressions

    _HAVE_CDK_NAG = True
except ImportError:  # pragma: no cover - cdk-nag present in the deploy venv
    _HAVE_CDK_NAG = False


class AgentCoreGatewayStack(Stack):
    """Stack for AgentCore Gateway with Lambda tool targets, logs and traces.

    Also owns the deployment's single CloudWatch Logs delivery resource policy
    (see log_delivery), which the Runtime stack's deliveries rely on. Every stack
    that delivers logs already depends on this one.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        lambda_functions: dict[str, lambda_.Function],
        config_bucket: s3.Bucket,
        cognito_stack_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags
        self.lambda_functions = lambda_functions
        self.config_bucket = config_bucket
        # Stack names carry a per-deployment suffix, so the Cognito export names
        # cannot be hardcoded here.
        self.cognito_stack_name = cognito_stack_name

        # Apply common tags to all resources
        self._apply_common_tags()

        # Create IAM role for gateway
        self.gateway_role = self.create_gateway_role()

        # Create gateway
        self.gateway = self.create_gateway()

        # Add Lambda targets
        self.add_lambda_targets()

        # Observability. Owns the deployment's single log-delivery resource policy
        # because every other stack that delivers logs already depends on this one.
        self.create_log_delivery()

        # Apply resource-specific tags
        self._apply_resource_tags(
            self.gateway_role,
            "gateway-execution-role",
            "IAM execution role for AgentCore Gateway",
        )
        self._apply_resource_tags(
            self.gateway,
            "agentcore-gateway",
            "MCP Gateway for BADGERS tools",
        )

        self._add_nag_suppressions()

        # Outputs
        CfnOutput(
            self,
            "GatewayUrl",
            value=self.gateway.gateway_url or "",
            description="AgentCore Gateway MCP endpoint URL",
            export_name=f"{Stack.of(self).stack_name}-GatewayUrl",
        )

        CfnOutput(
            self,
            "GatewayId",
            value=self.gateway.gateway_id,
            description="AgentCore Gateway ID",
            export_name=f"{Stack.of(self).stack_name}-GatewayId",
        )

        CfnOutput(
            self,
            "GatewayArn",
            value=self.gateway.gateway_arn,
            description="AgentCore Gateway ARN",
        )

        CfnOutput(
            self,
            "TargetCount",
            value=str(len(self.lambda_functions)),
            description="Number of Lambda targets added",
        )

        CfnOutput(
            self,
            "GatewayRoleArn",
            value=self.gateway_role.role_arn,
            description="Gateway execution role ARN",
            export_name=f"{Stack.of(self).stack_name}-GatewayRoleArn",
        )

    def _apply_common_tags(self) -> None:
        """Apply common deployment tags to all resources in this stack."""
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _apply_resource_tags(
        self, resource: Construct, name: str, description: str
    ) -> None:
        """Apply resource-specific name and description tags."""
        Tags.of(resource).add("resource_name", name)
        Tags.of(resource).add("resource_description", description)

    def _add_nag_suppressions(self) -> None:
        """Document the wildcard permissions AwsSolutions-IAM5 flags on the
        gateway execution role.

        AwsSolutions-IAM5 requires suppressions carry *evidence*, so each entry
        names the exact resource it applies to and why the wildcard is needed.
        """
        if not _HAVE_CDK_NAG:
            return

        NagSuppressions.add_resource_suppressions(
            self.gateway_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "The AgentCore Gateway construct grants lambda:InvokeFunction "
                        "with a :* version suffix on each target function so the "
                        "Gateway can invoke any published version or alias. The "
                        "function ARNs themselves are resolved references to the "
                        "specific specialist Lambdas -- the wildcard covers only the "
                        "version qualifier."
                    ),
                    # Derived from the functions actually granted, not transcribed. The
                    # hand-written list held 11 entries and never gained
                    # html_report_specialist when that specialist was added, so its
                    # finding sat unsuppressed. cdk-nag renders an Fn::GetAtt as
                    # `<LogicalId.Arn>`, and the logical ID carries a CDK-computed hash
                    # that cannot be spelled from the specialist name -- so ask the
                    # producing stack for it rather than guessing.
                    "appliesTo": [
                        f"Resource::<{Stack.of(fn).get_logical_id(fn.node.default_child)}"
                        f".Arn>:*"
                        for fn in self.lambda_functions.values()
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "The Gateway reads tool schemas from the config bucket. The /* "
                        "suffix is required because schema object keys are per-"
                        "specialist and are resolved at runtime; the bucket ARN is a "
                        "resolved reference, not a wildcard."
                    ),
                    "appliesTo": [
                        "Resource::<ConfigBucket2112C5EC.Arn>/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Action wildcards emitted by Bucket.grant_read, a CDK L2 grant "
                        "rather than hand-written policy. Each expands to a fixed set of "
                        "same-family read actions, scoped to the config bucket the Gateway "
                        "reads tool schemas from. These were previously unsuppressed."
                    ),
                    "appliesTo": [
                        "Action::s3:GetBucket*",
                        "Action::s3:GetObject*",
                        "Action::s3:List*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "lambda:InvokeFunction on function:badgers_* in this account and "
                        "Region. The name-prefix wildcard is deliberate: the grant must "
                        "exist before the Gateway creates its targets, target creation "
                        "happens in the same deploy, and the CustomSpecialists stack adds "
                        "further targets against this role in a later deploy, so the full "
                        "set of function ARNs is not knowable here. Every specialist "
                        "function, base and custom, is named badgers_<specialist>. Region "
                        "and account are pinned."
                    ),
                    "appliesTo": [
                        f"Resource::arn:aws:lambda:{region}:{account}:function:badgers_*"
                        for region in region_renderings(self.region)
                        for account in account_renderings(self.account)
                    ],
                },
            ],
            apply_to_children=True,
        )

    def create_gateway_role(self) -> iam.Role:
        """Create IAM role for gateway execution."""
        role = iam.Role(
            self,
            "GatewayExecutionRole",
            role_name=f"gateway-role-{self.deployment_id}",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            description="Execution role for AgentCore Gateway",
        )

        # Lambda invoke permission, by name prefix, so it exists BEFORE targets are created
        # -- target creation and the per-function grant_invoke() calls in
        # add_lambda_targets() land in the same deploy, and the CustomSpecialists stack
        # adds its own targets against this role later. Region and account are pinned;
        # only the function-name suffix is wildcarded. The name prefix `badgers_` is what
        # every specialist function is named with, base and custom alike.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeSpecialistsByPrefix",
                actions=["lambda:InvokeFunction"],
                resources=[
                    f"arn:aws:lambda:{self.region}:{self.account}:function:badgers_*"
                ],
            )
        )

        # S3 read for schemas
        self.config_bucket.grant_read(role)

        # No CloudWatch Logs or X-Ray statements. Both used to be here on Resource "*",
        # and neither had a caller: the Gateway's logs and traces are *vended* --
        # add_gateway_logging() creates a CloudWatch Logs resource policy for
        # delivery.logs.amazonaws.com and delivery sources/destinations, and that service
        # principal writes the data. The execution role never calls PutLogEvents or
        # PutTraceSegments. The AgentCore Gateway permissions guide lists only the trust
        # policy, lambda:InvokeFunction on each target, and S3 read for schemas.
        # https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-prerequisites-permissions.html

        return role

    def create_gateway(self) -> agentcore.Gateway:
        """Create AgentCore Gateway with Cognito authentication."""
        # Import Cognito outputs
        user_pool_id = Fn.import_value(f"{self.cognito_stack_name}-UserPoolId")
        user_pool_client_id = Fn.import_value(
            f"{self.cognito_stack_name}-UserPoolClientId"
        )

        # Construct OIDC discovery URL
        discovery_url = f"https://cognito-idp.{self.region}.amazonaws.com/{user_pool_id}/.well-known/openid-configuration"

        gateway = agentcore.Gateway(
            self,
            "BadgersGateway",
            gateway_name=f"badgers-gtwy-{self.deployment_id}",
            description="MCP Gateway for BADGERS specialists with full observability",
            role=self.gateway_role,
            protocol_configuration=agentcore.McpProtocolConfiguration(
                instructions="Use these tools to analyze PDF documents, extract content, and process images.",
                search_type=agentcore.McpGatewaySearchType.SEMANTIC,
                supported_versions=[agentcore.MCPProtocolVersion.MCP_2025_03_26],
            ),
            authorizer_configuration=agentcore.GatewayAuthorizer.using_custom_jwt(
                discovery_url=discovery_url,
                allowed_clients=[user_pool_client_id],
                # Note: Don't set allowed_audience - Cognito client credentials tokens don't include aud claim
            ),
        )

        return gateway

    def create_log_delivery(self) -> None:
        """Deliver Gateway application logs to CloudWatch and traces to X-Ray.

        Without this the Gateway emits nothing: the stack previously granted the
        IAM permissions to write logs and claimed "full observability" in the
        Gateway description, but never configured a delivery, so the console
        showed "Log delivery (0)" and "Tracing: Not enabled". A Gateway that
        accepts an MCP request and never answers left no trace anywhere.

        Uses the L1 delivery chain rather than CfnGatewayLogsMixin so the grant can
        be the deployment-wide policy created here instead of a second stack
        singleton — see log_delivery for why that quota matters.

        Identity (workload identity directory) logs are deliberately absent. The
        directory is named "default" and appears to be account-and-region scoped
        rather than per-deployment, so creating that delivery here would have every
        deployment in the account contend for the same one. It needs to be owned
        outside the per-deployment stacks, or gated so exactly one deployment owns
        it, and that scoping is unconfirmed.
        """
        self.log_delivery_policy = shared_delivery_policy(
            self,
            "BadgersLogDeliveryPolicy",
            self.deployment_id,
        )

        self.gateway_log_group = logs.LogGroup(
            self,
            "GatewayAppLogs",
            log_group_name=f"/aws/bedrock-agentcore/gateways/{self.deployment_id}/app",
            retention=logs.RetentionDays.TWO_YEARS,
            removal_policy=RemovalPolicy.RETAIN,
        )

        app_logs = deliver_to_log_group(
            self,
            "GatewayApplicationLogs",
            deployment_id=self.deployment_id,
            source_resource_arn=self.gateway.gateway_arn,
            log_type="APPLICATION_LOGS",
            log_group=self.gateway_log_group,
        )
        # The grant has to exist before the delivery that relies on it.
        app_logs.node.add_dependency(self.log_delivery_policy)

        # Free of the resource-policy quota: the destination is X-Ray, not a log
        # group. This is the signal that shows whether the Gateway received an MCP
        # request and dropped it.
        deliver_traces_to_xray(
            self,
            "GatewayTraces",
            deployment_id=self.deployment_id,
            source_resource_arn=self.gateway.gateway_arn,
        )

    def add_lambda_targets(self) -> None:
        """Add all Lambda functions as gateway targets."""
        # First, grant all invoke permissions to build up the role policy
        for lambda_function in self.lambda_functions.values():
            lambda_function.grant_invoke(self.gateway_role)

        # Collect all policy nodes (default + overflow policies) to add as dependencies
        # This ensures targets are created AFTER all policies are fully created
        policy_dependencies = []
        for child in self.gateway_role.node.children:
            child_id = child.node.id
            if child_id == "DefaultPolicy" or child_id.startswith("OverflowPolicy"):
                policy_dependencies.append(child)

        for specialist_name, lambda_function in self.lambda_functions.items():
            # Create short target name by stripping analyze_ prefix and _tool suffix
            # This keeps MCP tool names shorter: ${target_name}__${tool_name}
            short_name = specialist_name
            if short_name.startswith("analyze_"):
                short_name = short_name[8:]  # Remove "analyze_"
            if short_name.endswith("_tool"):
                short_name = short_name[:-5]  # Remove "_tool"

            # Target name must match pattern: ([0-9a-zA-Z][-]?){1,100}
            target_name = short_name.replace("_", "-")[:50]

            target = self.gateway.add_lambda_target(
                f"Target-{specialist_name}",
                gateway_target_name=target_name,
                description=f"Lambda target for {specialist_name}",
                lambda_function=lambda_function,
                tool_schema=agentcore.ToolSchema.from_s3_file(
                    bucket=self.config_bucket,
                    object_key=f"schemas/{specialist_name}.json",
                ),
            )

            # Add explicit dependency on all role policies to prevent race condition
            for policy in policy_dependencies:
                target.node.add_dependency(policy)
