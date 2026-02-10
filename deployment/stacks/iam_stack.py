"""IAM Stack for PDF Extractor POC."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


class IAMStack(Stack):
    """Stack for IAM roles and policies."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        config_bucket: s3.Bucket,
        source_bucket: s3.Bucket,
        output_bucket: s3.Bucket,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags

        # Apply common tags to all resources
        self._apply_common_tags()

        # Lambda execution role
        self.lambda_role = iam.Role(
            self,
            "LambdaAnalyzerExecutionRole",
            role_name=f"lambda-analyzer-role-{deployment_id}",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for Lambda analyzer functions with Bedrock and S3 access",
        )

        # Apply resource-specific tags
        self._apply_resource_tags(
            self.lambda_role,
            "lambda-execution-role",
            "IAM execution role for Lambda analyzer functions",
        )

        # Bedrock permissions - scoped to specific models used by analyzers
        # For inference profiles, we need permissions on BOTH the inference profile
        # AND the underlying foundation models that requests can be routed to
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvokeInferenceProfiles",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    # Primary model (global inference profile)
                    "arn:aws:bedrock:*:*:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    # Fallback models (inference profiles)
                    "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "arn:aws:bedrock:*:*:inference-profile/us.amazon.nova-premier-v1:0",
                ],
            )
        )

        # Application inference profiles - created by InferenceProfilesStack for cost tracking
        # These wrap the system-defined profiles above and are passed to analyzers via env vars
        # when running in AgentCore Runtime
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvokeApplicationInferenceProfiles",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    # Wildcard for all application inference profiles in this account
                    # Specific profiles are created in InferenceProfilesStack
                    f"arn:aws:bedrock:*:{self.account}:application-inference-profile/*",
                ],
            )
        )

        # Foundation model permissions - required when using inference profiles
        # The inference profile routes to these underlying foundation models
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvokeFoundationModels",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    # Claude Sonnet 4.5 foundation model (global profile routes here)
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
                    # Claude Haiku 4.5 foundation model
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
                    # Nova Premier foundation model
                    "arn:aws:bedrock:*::foundation-model/amazon.nova-premier-v1:0",
                    # Claude Opus 4.6 foundation model (with and without version suffix)
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-6-v1",
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-6-v1:0",
                ],
            )
        )

        # S3 config bucket read access
        config_bucket.grant_read(self.lambda_role)

        # S3 source bucket read access (for PDF uploads)
        source_bucket.grant_read(self.lambda_role)

        # S3 output bucket read/write access
        output_bucket.grant_read_write(self.lambda_role)

        # S3 access for specific buckets (config and output only)
        # Additional bucket access should be granted explicitly

        # CloudWatch Logs - scoped to Lambda log groups for this deployment
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/lambda/badgers-*",
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/lambda/badgers-*:*",
                ],
            )
        )

        # Outputs
        CfnOutput(
            self,
            "LambdaRoleArn",
            value=self.lambda_role.role_arn,
            description="Lambda execution role ARN",
            export_name=f"{Stack.of(self).stack_name}-LambdaRoleArn",
        )

        CfnOutput(
            self,
            "LambdaRoleName",
            value=self.lambda_role.role_name,
            description="Lambda execution role name",
            export_name=f"{Stack.of(self).stack_name}-LambdaRoleName",
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
