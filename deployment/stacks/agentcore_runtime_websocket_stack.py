"""AgentCore Runtime WebSocket Stack for PDF Extractor POC.

Separate runtime stack for WebSocket streaming support.
"""

from typing import TYPE_CHECKING

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_bedrockagentcore as agentcore,
    aws_iam as iam,
    aws_logs as logs,
)
from aws_cdk.mixins_preview.aws_bedrockagentcore import mixins as agentcore_mixins
from constructs import Construct

if TYPE_CHECKING:
    from .inference_profiles_stack import InferenceProfilesStack


class AgentCoreRuntimeWebSocketStack(Stack):
    """Stack for AgentCore Runtime agent with WebSocket support."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        ecr_repository_uri: str,
        gateway_url: str,
        cognito_credentials_secret_arn: str,
        output_bucket_name: str,
        config_bucket_name: str,
        source_bucket_name: str,
        memory_id: str,
        s3_kms_key_arn: str,
        inference_profiles_stack: "InferenceProfilesStack",
        image_tag: str = "websocket",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags
        self.inference_profiles_stack = inference_profiles_stack
        ecr_image_uri = f"{ecr_repository_uri}:{image_tag}"
        self.gateway_url = gateway_url
        self.ecr_repository_uri = ecr_repository_uri
        self.cognito_credentials_secret_arn = cognito_credentials_secret_arn
        self.output_bucket_name = output_bucket_name
        self.config_bucket_name = config_bucket_name
        self.source_bucket_name = source_bucket_name
        self.memory_id = memory_id
        self.s3_kms_key_arn = s3_kms_key_arn

        # Apply common tags to all resources
        self._apply_common_tags()

        self.agent_role = self.create_agent_role()

        # Grant inference profile permissions via CDK grants
        self.inference_profiles_stack.grant_invoke_to_role(self.agent_role)

        self.runtime = self.create_runtime(ecr_image_uri)

        # Apply resource-specific tags
        self._apply_resource_tags(
            self.agent_role,
            "runtime-ws-execution-role",
            "IAM execution role for AgentCore Runtime WebSocket",
        )
        self._apply_resource_tags(
            self.runtime,
            "agentcore-runtime-websocket",
            "AgentCore Runtime for PDF analysis with WebSocket streaming",
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

    def create_agent_role(self) -> iam.Role:
        """Create IAM role for AgentCore Runtime WebSocket."""
        role = iam.Role(
            self,
            "AgentExecutionRole",
            role_name=f"badgers-agent-ws-role-{self.deployment_id}",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            description="Execution role for BADGERS agent WebSocket in AgentCore Runtime",
        )

        ecr_repo_name = self.ecr_repository_uri.split("/")[-1]
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRImageAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=[
                    f"arn:aws:ecr:{self.region}:{self.account}:repository/{ecr_repo_name}"
                ],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRTokenAccess",
                effect=iam.Effect.ALLOW,
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )

        # Note: Bedrock permissions are granted via inference_profiles_stack.grant_invoke_to_role()

        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3OutputAccess",
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"arn:aws:s3:::{self.output_bucket_name}/*"],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3SourceAccess",
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject", "s3:ListBucket"],
                resources=[
                    f"arn:aws:s3:::{self.source_bucket_name}",
                    f"arn:aws:s3:::{self.source_bucket_name}/*",
                ],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ConfigAccess",
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject", "s3:ListBucket"],
                resources=[
                    f"arn:aws:s3:::{self.config_bucket_name}",
                    f"arn:aws:s3:::{self.config_bucket_name}/*",
                ],
            )
        )

        # KMS permissions for S3 bucket encryption
        role.add_to_policy(
            iam.PolicyStatement(
                sid="KMSDecryptForS3",
                effect=iam.Effect.ALLOW,
                actions=[
                    "kms:Decrypt",
                    "kms:GenerateDataKey",
                ],
                resources=[self.s3_kms_key_arn],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="SSMParameterAccess",
                effect=iam.Effect.ALLOW,
                actions=["ssm:GetParameter"],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/badgers/*",
                ],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:DescribeLogStreams",
                    "logs:CreateLogGroup",
                    "logs:DescribeLogGroups",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/bedrock-agentcore/runtimes/*"
                ],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                resources=["*"],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={
                    "StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}
                },
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="GetAgentAccessToken",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:workload-identity-directory/default/workload-identity/*",
                ],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="SecretsManagerAccess",
                effect=iam.Effect.ALLOW,
                actions=["secretsmanager:GetSecretValue"],
                resources=[self.cognito_credentials_secret_arn],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="AgentCoreMemoryAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:GetEvent",
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:GetMemory",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:memory/{self.memory_id}",
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:memory/{self.memory_id}/*",
                ],
            )
        )

        return role

    def create_runtime(self, ecr_image_uri: str) -> agentcore.CfnRuntime:
        """Create AgentCore Runtime with WebSocket support."""
        # Log groups for application and usage logs
        app_log_group = logs.LogGroup(
            self,
            "RuntimeAppLogs",
            log_group_name=f"/aws/bedrock-agentcore/runtimes/{self.deployment_id}-ws/app",
        )
        usage_log_group = logs.LogGroup(
            self,
            "RuntimeUsageLogs",
            log_group_name=f"/aws/bedrock-agentcore/runtimes/{self.deployment_id}-ws/usage",
        )

        runtime = agentcore.CfnRuntime(
            self,
            "BadgersRuntimeWebSocket",
            agent_runtime_name=f"badgers_runtime_ws_{self.deployment_id}",
            agent_runtime_artifact=agentcore.CfnRuntime.AgentRuntimeArtifactProperty(
                container_configuration=agentcore.CfnRuntime.ContainerConfigurationProperty(
                    container_uri=ecr_image_uri
                )
            ),
            network_configuration=agentcore.CfnRuntime.NetworkConfigurationProperty(
                network_mode="PUBLIC"
            ),
            protocol_configuration="HTTP",
            role_arn=self.agent_role.role_arn,
            description="PDF analysis agent runtime with WebSocket streaming",
            environment_variables={
                "AWS_DEFAULT_REGION": self.region,
                "GATEWAY_URL": self.gateway_url,
                "COGNITO_CREDENTIALS_SECRET_ARN": self.cognito_credentials_secret_arn,
                "AGENTCORE_MEMORY_ID": self.memory_id,
                "OUTPUT_BUCKET_NAME": self.output_bucket_name,
                # Inference profile ARNs for cost tracking
                "CLAUDE_SONNET_PROFILE_ARN": self.inference_profiles_stack.claude_sonnet_profile_arn,
                "CLAUDE_HAIKU_PROFILE_ARN": self.inference_profiles_stack.claude_haiku_profile_arn,
                "NOVA_PREMIER_PROFILE_ARN": self.inference_profiles_stack.nova_premier_profile_arn,
                "CLAUDE_OPUS_46_PROFILE_ARN": self.inference_profiles_stack.claude_opus_46_profile_arn,
                "CLAUDE_OPUS_45_PROFILE_ARN": self.inference_profiles_stack.claude_opus_45_profile_arn,
            },
        )

        runtime.node.add_dependency(self.agent_role)

        # Apply logging and tracing mixins
        agentcore_mixins.CfnRuntimeLogsMixin.APPLICATION_LOGS.to_log_group(
            app_log_group
        ).apply_to(runtime)
        agentcore_mixins.CfnRuntimeLogsMixin.USAGE_LOGS.to_log_group(
            usage_log_group
        ).apply_to(runtime)
        agentcore_mixins.CfnRuntimeLogsMixin.TRACES.to_x_ray().apply_to(runtime)

        CfnOutput(
            self,
            "RuntimeId",
            value=runtime.attr_agent_runtime_id,
            description="AgentCore Runtime WebSocket ID",
            export_name=f"{self.stack_name}-RuntimeId",
        )

        CfnOutput(
            self,
            "RuntimeArn",
            value=runtime.attr_agent_runtime_arn,
            description="AgentCore Runtime WebSocket ARN",
            export_name=f"{self.stack_name}-RuntimeArn",
        )

        CfnOutput(
            self,
            "RuntimeRoleArn",
            value=self.agent_role.role_arn,
            description="Runtime WebSocket execution role ARN",
        )

        return runtime
