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

from .model_registry import (
    foundation_model_id,
    has_provider,
    load_registry,
    provisioned_models,
)
from .nag_arn_renderings import account_renderings, region_renderings

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

        # The model set is read from the registry at synth, not transcribed here. This was
        # previously a third and fourth hand-maintained copy of the list, and both had
        # drifted from InferenceProfilesStack. `disabled` entries are excluded, so
        # declining a model removes its grants as well as its profile.
        self.models = provisioned_models(load_registry())

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
                    f"arn:aws:bedrock:*:*:inference-profile/{model_id}"
                    for model_id in self.models
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
                    f"arn:aws:bedrock:*::foundation-model/{foundation_model_id(model_id)}"
                    for model_id in self.models
                ],
            )
        )

        # OpenAI models additionally require bedrock:InvokeModel on the account's default
        # project. Without it every OpenAI invocation returns AccessDenied even with a
        # correct profile grant, and `_should_fallback` does not retry AccessDenied — so a
        # GPT primary would not fall back to a Claude secondary, it would just fail.
        # Source: the GPT-5.6 Terra model card, Programmatic Access section.
        if has_provider(self.models, "openai"):
            self.lambda_role.add_to_policy(
                iam.PolicyStatement(
                    sid="BedrockInvokeDefaultProject",
                    effect=iam.Effect.ALLOW,
                    actions=["bedrock:InvokeModel"],
                    resources=[
                        f"arn:aws:bedrock:{self.region}:{self.account}:project/default"
                    ],
                )
            )

        # Specialists resolve model ID -> application inference profile ARN by reading the
        # parameter InferenceProfilesStack writes. This replaces the per-model
        # *_PROFILE_ARN environment variables, which arrived for free; a network read does
        # not. Scoped to the single parameter, no wildcard.
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadModelProfilesParameter",
                effect=iam.Effect.ALLOW,
                actions=["ssm:GetParameter"],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}"
                    f":parameter/badgers-{deployment_id}/model-profiles"
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

        Every `appliesTo` entry that names a model is generated from the registry, in the
        same iteration order as the policy statement it justifies. Hand-listing them meant
        a suppression could claim to cover ARNs the policy no longer contained while the
        ARNs it did contain went unsuppressed -- which is how this drifted to five stale
        strings and 28 unsuppressed findings. A generated list cannot disagree with the
        policy it describes.
        """
        if not _HAVE_CDK_NAG:
            return

        NagSuppressions.add_resource_suppressions(
            self.lambda_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Geo cross-Region inference requires bedrock:InvokeModel on the "
                        "foundation model in the source Region AND in every destination "
                        "Region the geo profile can route to. The model ID is pinned "
                        "exactly -- only the Region field is wildcarded, and no action "
                        "or account is wildcarded. Wildcarding the Region is a "
                        "deliberate choice, not a requirement: the destination set could "
                        "be enumerated instead. It is not, for two reasons. (1) The "
                        "destination set is a function of (model, source Region), and "
                        "BADGERS' source Region is chosen by the operator at deploy "
                        "time. (2) Only 3 of the 8 models in model_registry.json publish "
                        "a destination-Region table on their model card; the other 5 "
                        "document a us.* profile with no destination list, so a "
                        "hardcoded list would cover under half the set. Enumerating all "
                        "eight is possible via bedrock:GetInferenceProfile, whose "
                        "models[].modelArn field returns the Region-qualified foundation "
                        "model ARNs, but that is a deploy-time API call and would need a "
                        "custom resource. Switching to global.* profiles would also "
                        "remove the wildcard -- the global foundation-model ARN form is "
                        "arn:aws:bedrock:::foundation-model/<model>, with empty Region "
                        "and account fields -- at the cost of routing outside the US "
                        "geo. See "
                        "https://docs.aws.amazon.com/bedrock/latest/userguide/"
                        "geographic-cross-region-inference.html"
                    ),
                    "appliesTo": [
                        f"Resource::arn:aws:bedrock:*::foundation-model/"
                        f"{foundation_model_id(model_id)}"
                        for model_id in self.models
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Cross-Region inference profiles are resolved per Region, so "
                        "the Region field is wildcarded while the profile ID stays "
                        "pinned. The application-inference-profile/* entry is scoped to "
                        "this account -- profile IDs are generated at runtime and "
                        "cannot be enumerated at deploy time. Same Region-wildcard "
                        "rationale as BedrockInvokeFoundationModels above."
                    ),
                    "appliesTo": [
                        f"Resource::arn:aws:bedrock:*:*:inference-profile/{model_id}"
                        for model_id in self.models
                    ]
                    # Hand-written: not a model ARN, and deliberately account-scoped
                    # rather than per-profile. Application inference profile IDs are
                    # generated by Bedrock at create time.
                    + [
                        f"Resource::arn:aws:bedrock:*:{account}"
                        f":application-inference-profile/*"
                        for account in account_renderings(self.account)
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
                    # Built from self.region / self.account. These were hardcoded to
                    # us-east-1 and <AWS::AccountId>, so they matched nothing on the
                    # deploy path in any Region.
                    "appliesTo": [
                        f"Resource::arn:aws:logs:{region}:{account}"
                        f":log-group:/aws/lambda/{prefix}{suffix}"
                        for region in region_renderings(self.region)
                        for account in account_renderings(self.account)
                        for prefix in ("badgers-*", "badgers_*")
                        for suffix in ("", ":*")
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
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Action wildcards emitted by the CDK L2 grants "
                        "Bucket.grant_read / grant_read_write on the three named buckets, "
                        "not written by this stack. Each expands to a fixed, documented "
                        "set of same-family read or write actions -- s3:GetObject* covers "
                        "GetObject, GetObjectVersion and GetObjectTagging, for example -- "
                        "and the KMS pair is required to read and write objects encrypted "
                        "with the customer-managed key. Resources stay scoped to the "
                        "three bucket ARNs and the one key. Replacing the grants with "
                        "hand-written statements would pin the actions but silently break "
                        "whenever a bucket feature needs an action the grant would have "
                        "added. These were previously unsuppressed and are 8 of the "
                        "findings this pass closes."
                    ),
                    "appliesTo": [
                        "Action::s3:Abort*",
                        "Action::s3:DeleteObject*",
                        "Action::s3:GetBucket*",
                        "Action::s3:GetObject*",
                        "Action::s3:List*",
                        "Action::kms:GenerateDataKey*",
                        "Action::kms:ReEncrypt*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "aws-marketplace:ViewSubscriptions and aws-marketplace:Subscribe "
                        "do not support resource-level permissions, so Resource must be "
                        '"*" -- a narrower ARN makes the statement match nothing. The '
                        "grant exists because Bedrock subscribes the account through "
                        "Marketplace on a model's first invocation; without it that "
                        "invocation fails with AccessDeniedException. See "
                        "github.com/aws-samples/sample-badgers/issues/33 and the "
                        "AWS Marketplace actions in the Service Authorization Reference."
                    ),
                    "appliesTo": ["Resource::*"],
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
