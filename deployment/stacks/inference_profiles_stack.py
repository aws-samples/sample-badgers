"""Inference Profiles Stack for cost tracking and usage monitoring.

Creates Application Inference Profiles wrapping cross-region system-defined
profiles for each foundation model used by the application.
"""

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
)
from aws_cdk.aws_bedrock import CfnApplicationInferenceProfile
from constructs import Construct


# System-defined cross-region inference profile IDs
# These are the actual AWS profile IDs from:
# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
CROSS_REGION_PROFILES = {
    "claude_sonnet_4_5": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude_haiku_4_5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude_opus_4_6": "global.anthropic.claude-opus-4-6-v1",
    "claude_opus_4_5": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "nova_premier": "us.amazon.nova-premier-v1:0",
}


class InferenceProfilesStack(Stack):
    """Stack for Application Inference Profiles.

    Creates trackable inference profiles for cost allocation and usage monitoring.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags

        # Apply common tags to all resources
        self._apply_common_tags()

        # Convert tags dict to CfnTag format
        cfn_tags = [{"key": k, "value": v} for k, v in deployment_tags.items()]

        # Claude Sonnet 4.5 - Global cross-region profile
        self.claude_sonnet_cfn = CfnApplicationInferenceProfile(
            self,
            "ClaudeSonnetProfile",
            inference_profile_name=f"badgers-claude-sonnet-{deployment_id}",
            model_source=CfnApplicationInferenceProfile.InferenceProfileModelSourceProperty(
                copy_from=f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_sonnet_4_5']}"
            ),
            description="Claude Sonnet 4.5 for BADGERS document analysis",
            tags=cfn_tags,
        )

        # Claude Haiku 4.5 - Global cross-region profile
        self.claude_haiku_cfn = CfnApplicationInferenceProfile(
            self,
            "ClaudeHaikuProfile",
            inference_profile_name=f"badgers-claude-haiku-{deployment_id}",
            model_source=CfnApplicationInferenceProfile.InferenceProfileModelSourceProperty(
                copy_from=f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_haiku_4_5']}"
            ),
            description="Claude Haiku 4.5 for BADGERS fallback analysis",
            tags=cfn_tags,
        )

        # Amazon Nova Premier - US cross-region profile
        self.nova_premier_cfn = CfnApplicationInferenceProfile(
            self,
            "NovaPremierProfile",
            inference_profile_name=f"badgers-nova-premier-{deployment_id}",
            model_source=CfnApplicationInferenceProfile.InferenceProfileModelSourceProperty(
                copy_from=f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{CROSS_REGION_PROFILES['nova_premier']}"
            ),
            description="Amazon Nova Premier for BADGERS fallback analysis",
            tags=cfn_tags,
        )

        # Claude Opus 4.6 - Global cross-region profile
        self.claude_opus_cfn = CfnApplicationInferenceProfile(
            self,
            "ClaudeOpusProfile",
            inference_profile_name=f"badgers-claude-opus-{deployment_id}",
            model_source=CfnApplicationInferenceProfile.InferenceProfileModelSourceProperty(
                copy_from=f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_opus_4_6']}"
            ),
            description="Claude Opus 4.6 for BADGERS complex analysis",
            tags=cfn_tags,
        )

        # Claude Opus 4.5 - Global cross-region profile (fallback)
        self.claude_opus_45_cfn = CfnApplicationInferenceProfile(
            self,
            "ClaudeOpus45Profile",
            inference_profile_name=f"badgers-claude-opus-45-{deployment_id}",
            model_source=CfnApplicationInferenceProfile.InferenceProfileModelSourceProperty(
                copy_from=f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_opus_4_5']}"
            ),
            description="Claude Opus 4.5 for BADGERS fallback analysis",
            tags=cfn_tags,
        )

        # Outputs - use attr_inference_profile_arn for CfnApplicationInferenceProfile
        CfnOutput(
            self,
            "ClaudeSonnetProfileArn",
            value=self.claude_sonnet_cfn.attr_inference_profile_arn,
            description="Claude Sonnet inference profile ARN",
            export_name=f"{self.stack_name}-ClaudeSonnetProfileArn",
        )
        CfnOutput(
            self,
            "ClaudeHaikuProfileArn",
            value=self.claude_haiku_cfn.attr_inference_profile_arn,
            description="Claude Haiku inference profile ARN",
            export_name=f"{self.stack_name}-ClaudeHaikuProfileArn",
        )
        CfnOutput(
            self,
            "NovaPremierProfileArn",
            value=self.nova_premier_cfn.attr_inference_profile_arn,
            description="Nova Premier inference profile ARN",
            export_name=f"{self.stack_name}-NovaPremierProfileArn",
        )
        CfnOutput(
            self,
            "ClaudeOpus46ProfileArn",
            value=self.claude_opus_cfn.attr_inference_profile_arn,
            description="Claude Opus 4.6 inference profile ARN",
            export_name=f"{self.stack_name}-ClaudeOpus46ProfileArn",
        )
        CfnOutput(
            self,
            "ClaudeOpus45ProfileArn",
            value=self.claude_opus_45_cfn.attr_inference_profile_arn,
            description="Claude Opus 4.5 inference profile ARN",
            export_name=f"{self.stack_name}-ClaudeOpus45ProfileArn",
        )

    def _apply_common_tags(self) -> None:
        """Apply common deployment tags to all resources in this stack."""
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    @property
    def claude_sonnet_profile_arn(self) -> str:
        """Get Claude Sonnet profile ARN."""
        return self.claude_sonnet_cfn.attr_inference_profile_arn

    @property
    def claude_haiku_profile_arn(self) -> str:
        """Get Claude Haiku profile ARN."""
        return self.claude_haiku_cfn.attr_inference_profile_arn

    @property
    def nova_premier_profile_arn(self) -> str:
        """Get Nova Premier profile ARN."""
        return self.nova_premier_cfn.attr_inference_profile_arn

    @property
    def claude_opus_46_profile_arn(self) -> str:
        """Get Claude Opus 4.6 profile ARN."""
        return self.claude_opus_cfn.attr_inference_profile_arn

    @property
    def claude_opus_45_profile_arn(self) -> str:
        """Get Claude Opus 4.5 profile ARN."""
        return self.claude_opus_45_cfn.attr_inference_profile_arn

    def grant_invoke_to_role(self, role) -> None:
        """Grant invoke permissions on all profiles to the given role.

        Since we're using CfnApplicationInferenceProfile, we need to add
        IAM policies manually instead of using L2 grant methods.

        Per AWS docs, when using inference profiles you need permissions on BOTH:
        1. The inference profile itself
        2. The underlying foundation models (can be conditioned to only work via profile)
        """
        from aws_cdk import aws_iam as iam

        # Grant bedrock:InvokeModel on all application inference profiles
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeApplicationInferenceProfiles",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    self.claude_sonnet_cfn.attr_inference_profile_arn,
                    self.claude_haiku_cfn.attr_inference_profile_arn,
                    self.nova_premier_cfn.attr_inference_profile_arn,
                    self.claude_opus_cfn.attr_inference_profile_arn,
                    self.claude_opus_45_cfn.attr_inference_profile_arn,
                ],
            )
        )

        # Also grant access to the underlying cross-region system profiles
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeSystemInferenceProfiles",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_sonnet_4_5']}",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_haiku_4_5']}",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/{CROSS_REGION_PROFILES['nova_premier']}",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_opus_4_6']}",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/{CROSS_REGION_PROFILES['claude_opus_4_5']}",
                ],
            )
        )

        # Grant access to the underlying foundation models
        # Required per AWS docs: "When you specify an inference profile in the Resource field,
        # you must also specify the foundation model in each Region associated with it."
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeFoundationModels",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    # Claude Sonnet 4.5
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
                    # Claude Haiku 4.5
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
                    # Claude Opus 4.6
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-6-v1",
                    # Claude Opus 4.5
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-5-20251101-v1:0",
                    # Amazon Nova Premier
                    "arn:aws:bedrock:*::foundation-model/amazon.nova-premier-v1:0",
                ],
            )
        )
