"""AgentCore Runtime WebSocket Stack for BADGERS.

Separate runtime stack for WebSocket streaming support.
"""

from typing import TYPE_CHECKING

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_bedrockagentcore as agentcore,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_logs as logs,
)
from aws_cdk.mixins_preview.aws_bedrockagentcore import mixins as agentcore_mixins
from constructs import Construct

try:  # cdk-nag is an optional synth-time aspect (enabled via CDK_NAG=1 in app.py)
    from cdk_nag import NagSuppressions

    _HAVE_CDK_NAG = True
except ImportError:  # pragma: no cover - cdk-nag present in the deploy venv
    _HAVE_CDK_NAG = False

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
        jobs_table: dynamodb.ITable,
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
        # The agent mints job_id on the first specialist tool call of a turn and
        # writes the job-level row itself, so the runtime needs table access of its
        # own — the specialist Lambdas only write their own subtask rows.
        self.jobs_table = jobs_table

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
            "AgentCore Runtime for BADGERS with WebSocket streaming",
        )

        self._add_nag_suppressions()

    def _add_nag_suppressions(self) -> None:
        """Document the wildcard permissions AwsSolutions-IAM5 flags on the
        AgentCore Runtime execution role.

        AwsSolutions-IAM5 requires suppressions carry *evidence*, so each entry
        names the exact resource it applies to and why the wildcard is needed.
        """
        if not _HAVE_CDK_NAG:
            return

        NagSuppressions.add_resource_suppressions(
            self.agent_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Cross-Region inference requires bedrock:InvokeModel on the "
                        "foundation model in every destination Region the inference "
                        "profile can route to, so the Region field is wildcarded. The "
                        "model ID itself is pinned exactly -- no model wildcard. See "
                        "https://docs.aws.amazon.com/bedrock/latest/userguide/"
                        "geographic-cross-region-inference.html"
                    ),
                    "appliesTo": [
                        "Resource::arn:aws:bedrock:*::foundation-model/amazon.nova-premier-v1:0",
                        "Resource::arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
                        "Resource::arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-5-20251101-v1:0",
                        "Resource::arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-6-v1",
                        "Resource::arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Cross-Region inference profiles are resolved per Region, so "
                        "the Region field is wildcarded while the profile ID stays "
                        "pinned. Scoped to this account."
                    ),
                    "appliesTo": [
                        "Resource::arn:aws:bedrock:*:<AWS::AccountId>:inference-profile/us.amazon.nova-premier-v1:0",
                        "Resource::arn:aws:bedrock:*:<AWS::AccountId>:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "Resource::arn:aws:bedrock:*:<AWS::AccountId>:inference-profile/us.anthropic.claude-opus-4-5-20251101-v1:0",
                        "Resource::arn:aws:bedrock:*:<AWS::AccountId>:inference-profile/us.anthropic.claude-opus-4-6-v1",
                        "Resource::arn:aws:bedrock:*:<AWS::AccountId>:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "AgentCore Runtime log group names embed a runtime ID that is "
                        "generated at deploy time, so the target is prefix-scoped to "
                        "/aws/bedrock-agentcore/runtimes/ within this account and "
                        "Region."
                    ),
                    "appliesTo": [
                        "Resource::arn:aws:logs:us-east-1:<AWS::AccountId>:log-group:/aws/bedrock-agentcore/runtimes/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Each target is prefix-scoped to a specific resource whose "
                        "child keys are created at runtime and cannot be enumerated at "
                        "deploy time: AgentCore memory records under one memory ID, "
                        "workload identities under the default directory, S3 object "
                        "keys in the three named buckets, and SSM parameters under the "
                        "/badgers/ prefix. All are scoped to this account and Region."
                    ),
                    "appliesTo": [
                        "Resource::arn:aws:bedrock-agentcore:us-east-1:<AWS::AccountId>:memory/<badgersmemory.MemoryId>/*",
                        "Resource::arn:aws:bedrock-agentcore:us-east-1:<AWS::AccountId>:workload-identity-directory/default/workload-identity/*",
                        "Resource::arn:aws:s3:::<ConfigBucket2112C5EC>/*",
                        "Resource::arn:aws:s3:::<OutputBucket7114EB27>/*",
                        "Resource::arn:aws:s3:::<SourceBucketDDD2130A>/*",
                        "Resource::arn:aws:ssm:us-east-1:<AWS::AccountId>:parameter/badgers/*",
                    ],
                },
            ],
            apply_to_children=True,
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

        # Scoped to PutItem alone: the agent's only write is job_state.create_job,
        # a conditional put of the job-level row. Subtask rows are written by the
        # specialist Lambdas under their own role (see iam_stack.py).
        role.add_to_policy(
            iam.PolicyStatement(
                sid="DynamoDBJobState",
                effect=iam.Effect.ALLOW,
                actions=["dynamodb:PutItem"],
                resources=[self.jobs_table.table_arn],
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
            # AgentCore runtime names must match [a-zA-Z][a-zA-Z0-9_]{0,47} — no
            # hyphens. deployment_id is "{DEPLOYMENT_ID}-{STACK_SUFFIX}", so the
            # separator has to be normalised here.
            agent_runtime_name=(
                f"badgers_runtime_ws_{self.deployment_id.replace('-', '_')}"
            ),
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
            description="BADGERS agent runtime with WebSocket streaming",
            environment_variables={
                "AWS_DEFAULT_REGION": self.region,
                "GATEWAY_URL": self.gateway_url,
                "COGNITO_CREDENTIALS_SECRET_ARN": self.cognito_credentials_secret_arn,
                "AGENTCORE_MEMORY_ID": self.memory_id,
                "OUTPUT_BUCKET_NAME": self.output_bucket_name,
                # Injected directly rather than discovered from a well-known SSM
                # path, which was not deployment-scoped.
                "CONFIG_BUCKET_NAME": self.config_bucket_name,
                # foundation.job_state reads this at call time; when it is absent
                # every write becomes a no-op and tracking is simply off.
                "JOBS_TABLE_NAME": self.jobs_table.table_name,
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
