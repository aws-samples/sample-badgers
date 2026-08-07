"""AgentCore Gateway Stack for BADGERS."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Fn,
    Tags,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct

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
    """Stack for AgentCore Gateway with Lambda tool targets and logging."""

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
                    "appliesTo": [
                        "Resource::<ContainerFunctionimageenhancer2AFB4983.Arn>:*",
                        "Resource::<ContainerFunctionremediationspecialistB67CD8CB.Arn>:*",
                        "Resource::<FunctionchartsspecialistA6AA9F82.Arn>:*",
                        "Resource::<FunctionclassifypdfcontentEC92FE26.Arn>:*",
                        "Resource::<FunctioncorrelationspecialistE250ACCF.Arn>:*",
                        "Resource::<FunctiondiagramspecialistB1D3C809.Arn>:*",
                        "Resource::<Functionelementsspecialist2A88D0C1.Arn>:*",
                        "Resource::<FunctionfulltextspecialistB95E9182.Arn>:*",
                        "Resource::<Functionpdftoimagesconverter2E10A5D5.Arn>:*",
                        "Resource::<FunctionscientificspecialistCB1826B1.Arn>:*",
                        "Resource::<Functiontablespecialist6D08F0B2.Arn>:*",
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

        # Lambda invoke permissions - wildcard for all badgers functions
        # This ensures permission exists BEFORE targets are created (no race condition)
        # Individual grant_invoke() calls in add_lambda_targets() are redundant but harmless
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=["arn:aws:lambda:*:*:function:badgers_*"],
            )
        )

        # S3 read for schemas
        self.config_bucket.grant_read(role)

        # CloudWatch logging permissions
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=["*"],
            )
        )

        # X-Ray tracing permissions
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                ],
                resources=["*"],
            )
        )

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
