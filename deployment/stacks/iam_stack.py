"""IAM Stack for BADGERS."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct

try:  # cdk-nag is an optional synth-time aspect (enabled via CDK_NAG=1 in app.py)
    from cdk_nag import NagSuppressions

    _HAVE_CDK_NAG = True
except ImportError:  # pragma: no cover - cdk-nag present in the deploy venv
    _HAVE_CDK_NAG = False


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
        jobs_table: dynamodb.ITable,
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
            "LambdaSpecialistExecutionRole",
            role_name=f"lambda-specialist-role-{deployment_id}",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for Lambda specialist functions with Bedrock and S3 access",
        )

        # Apply resource-specific tags
        self._apply_resource_tags(
            self.lambda_role,
            "lambda-execution-role",
            "IAM execution role for Lambda specialist functions",
        )

        # Bedrock permissions - scoped to specific models used by specialists
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
                    # Primary model (regional inference profile)
                    "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    # Fallback models (inference profiles)
                    "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "arn:aws:bedrock:*:*:inference-profile/us.amazon.nova-premier-v1:0",
                    # Claude Opus 4.6 (regional inference profile for vision)
                    "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-opus-4-6-v1",
                    # Cell grid resolver (cross-region Sonnet)
                    "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-sonnet-4-6",
                ],
            )
        )

        # Application inference profiles - created by InferenceProfilesStack for cost tracking
        # These wrap the system-defined profiles above and are passed to specialists via env vars
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
                    # Claude Sonnet 4.5 foundation model (regional profile routes here)
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
                    # Claude Haiku 4.5 foundation model
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
                    # Nova Premier foundation model
                    "arn:aws:bedrock:*::foundation-model/amazon.nova-premier-v1:0",
                    # Claude Opus 4.6 foundation model (with and without version suffix)
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-6-v1",
                    # Claude Sonnet 4 foundation model (cell grid resolver)
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-20250514-v1:0",
                    # Qwen multimodal model (direct regional invocation)
                    f"arn:aws:bedrock:{self.region}::foundation-model/qwen.qwen3-vl-235b-a22b",
                ],
            )
        )

        # AWS Marketplace permissions - required for automatic model subscription
        # When Bedrock models are first invoked, AWS automatically subscribes the
        # account via Marketplace. Without these permissions, the first invocation
        # fails with AccessDeniedException. See: github.com/aws-samples/sample-badgers/issues/33
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="MarketplaceModelSubscription",
                effect=iam.Effect.ALLOW,
                actions=[
                    "aws-marketplace:ViewSubscriptions",
                    "aws-marketplace:Subscribe",
                ],
                resources=["*"],
            )
        )

        # S3 config bucket read access
        config_bucket.grant_read(self.lambda_role)

        # S3 source bucket read access (for PDF uploads)
        source_bucket.grant_read(self.lambda_role)

        # S3 output bucket read/write access
        output_bucket.grant_read_write(self.lambda_role)

        # DynamoDB jobs table access for job state tracking.
        # Specialists upsert their own subtask row (RUNNING -> COMPLETE/FAILED)
        # via foundation.job_state. Without this grant those calls fail at the
        # API and job tracking is silently lost.
        #
        # Scoped to exactly the four operations job_state performs rather than
        # using grant_read_write_data, which would also allow DeleteItem, Scan
        # and the Batch* operations. Specialists never delete or scan; the base
        # table alone is enough because job_state.query filters on the job_id
        # partition key and never reads a GSI.
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="DynamoDBJobState",
                effect=iam.Effect.ALLOW,
                actions=[
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                ],
                resources=[jobs_table.table_arn],
            )
        )

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
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/lambda/badgers_*",
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/lambda/badgers_*:*",
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

        self._add_nag_suppressions()

    def _add_nag_suppressions(self) -> None:
        """Document the wildcard permissions AwsSolutions-IAM5 flags on the
        specialist execution role.

        AwsSolutions-IAM5 requires suppressions carry *evidence*, so each entry
        names the exact resource it applies to and why the wildcard is needed.
        """
        if not _HAVE_CDK_NAG:
            return

        NagSuppressions.add_resource_suppressions(
            self.lambda_role,
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
                        "Resource::arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-6-v1",
                        "Resource::arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-20250514-v1:0",
                        "Resource::arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Cross-Region inference profiles are resolved per Region, so "
                        "the Region field is wildcarded while the profile ID stays "
                        "pinned. The application-inference-profile/* entry is scoped to "
                        "this account -- profile IDs are generated at runtime and "
                        "cannot be enumerated at deploy time."
                    ),
                    "appliesTo": [
                        "Resource::arn:aws:bedrock:*:*:inference-profile/us.amazon.nova-premier-v1:0",
                        "Resource::arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "Resource::arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-opus-4-6-v1",
                        "Resource::arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "Resource::arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-sonnet-4-6",
                        "Resource::arn:aws:bedrock:*:<AWS::AccountId>:application-inference-profile/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "CloudWatch Logs targets are prefix-scoped to this "
                        "deployment's specialist log groups (badgers-* / badgers_*). "
                        "Log group names embed the specialist name and log stream names "
                        "are generated at runtime, so neither can be enumerated at "
                        "deploy time. Scoped to this account and Region."
                    ),
                    "appliesTo": [
                        "Resource::arn:aws:logs:us-east-1:<AWS::AccountId>:log-group:/aws/lambda/badgers-*",
                        "Resource::arn:aws:logs:us-east-1:<AWS::AccountId>:log-group:/aws/lambda/badgers-*:*",
                        "Resource::arn:aws:logs:us-east-1:<AWS::AccountId>:log-group:/aws/lambda/badgers_*",
                        "Resource::arn:aws:logs:us-east-1:<AWS::AccountId>:log-group:/aws/lambda/badgers_*:*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "S3 object access is scoped to these three specific buckets. "
                        "The /* suffix is required because object keys are per-document "
                        "and per-job values created at runtime and cannot be enumerated "
                        "at deploy time. The bucket ARNs are resolved references, not "
                        "wildcards."
                    ),
                    "appliesTo": [
                        "Resource::<ConfigBucket2112C5EC.Arn>/*",
                        "Resource::<OutputBucket7114EB27.Arn>/*",
                        "Resource::<SourceBucketDDD2130A.Arn>/*",
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
