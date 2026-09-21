"""Inference Profiles Stack for cost tracking and usage monitoring.

Creates one Application Inference Profile per model in the registry, wrapping the
cross-region system-defined profile for each. Every profile, IAM statement, and output in
this stack is derived from ``s3_files/config/model_registry.json`` — adding or retiring a
model is a registry edit plus a deploy, not a code change here.

Two things are generated together on purpose:

* the profiles, and
* the three ``grant_invoke_to_role`` statements,

in the same iteration. Before this, they were maintained by hand and had drifted: Claude
Sonnet 4.6 had a profile but appeared in **none** of the three statements, so it had never
been invocable. Generating both from one source makes that divergence unrepresentable
rather than merely fixed.

The SSM parameter ``/badgers-{deployment_id}/model-profiles`` is written in the same loop
and maps model ID to profile ARN. Because one loop writes both, the parameter cannot name
a model that has no profile.

Profiles are created for **every** registry entry, including ``retiring`` ones. ``status``
gates what the wizard offers, not what exists in AWS — that is what keeps already-deployed
manifests working through the release that stops offering their model.
"""

import json
from typing import Any, Iterable

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_ssm as ssm,
)
from aws_cdk.aws_bedrock import CfnApplicationInferenceProfile
from constructs import Construct

from .model_registry import (
    construct_id_for,
    disabled_models,
    foundation_model_id,
    has_provider,
    is_cross_region,
    load_registry,
    model_profiles_param_name,
    output_id_for,
    profile_name_for,
    provisioned_models,
)
from .nag_arn_renderings import account_renderings

# System-defined profiles are US **geo** cross-Region (``us.*``) rather than global
# (``global.*``). Both are cross-Region: ``us.*`` bounds routing to Regions inside the US
# geography, while ``global.*`` can route to any commercial Region. A previous version of
# this comment claimed ``us.*`` avoided cross-Region routing, which is false -- it narrows
# the destination set. The reason to prefer it is data residency: an organisation whose SCPs
# allow only US Regions can work with ``us.*`` and cannot work with ``global.*``.
# See https://docs.aws.amazon.com/bedrock/latest/userguide/geographic-cross-region-inference.html
# and DEPLOYMENT_README.md -> Inference Profiles and Regions.


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

        # Read at synth. A malformed registry fails here rather than in a Lambda later.
        self.registry = load_registry()

        # Only `active` and `retiring` models produce AWS resources. A `disabled` entry is
        # documentation of a model this deployment has chosen not to adopt, so it must
        # create no profile, no grant, and no SSM entry — otherwise "declining" a model
        # would still leave it provisioned and payable.
        self.models = provisioned_models(self.registry)

        for model_id in disabled_models(self.registry):
            print(f"  Model disabled (no profile, no grant): {model_id}")

        # Apply common tags to all resources
        self._apply_common_tags()

        # Convert tags dict to CfnTag format
        cfn_tags = [{"key": k, "value": v} for k, v in deployment_tags.items()]

        # model_id -> CfnApplicationInferenceProfile
        self._profiles: dict[str, CfnApplicationInferenceProfile] = {}

        for model_id, spec in self.models.items():
            profile = CfnApplicationInferenceProfile(
                self,
                construct_id_for(model_id),
                inference_profile_name=profile_name_for(model_id, deployment_id),
                model_source=CfnApplicationInferenceProfile.InferenceProfileModelSourceProperty(
                    copy_from=(
                        # Default: wrap the `us.*` cross-Region system-defined profile.
                        f"arn:aws:bedrock:{self.region}:{self.account}"
                        f":inference-profile/{model_id}"
                        if is_cross_region(spec)
                        # In-Region only: no system profile exists, so wrap the foundation
                        # model directly. A foundation-model ARN has an empty account field.
                        else (
                            f"arn:aws:bedrock:{self.region}::"
                            f"foundation-model/{foundation_model_id(model_id)}"
                        )
                    )
                ),
                # description is validated against ^([0-9a-zA-Z:.][ _-]?)+$ — parentheses
                # are rejected, and so are two adjacent separators, which rules out
                # "BADGERS - active" too.
                description=f"{spec['display_name']} for BADGERS status {spec['status']}",
                tags=cfn_tags,
            )
            self._profiles[model_id] = profile

            # Output, but deliberately NOT an export. A CloudFormation export is an
            # account-and-region-scoped object with a global uniqueness constraint, and one
            # that cannot be deleted while any stack imports it. Nothing imports these any
            # more — consumers read the SSM parameter — so exporting them would create 9
            # account-level objects and 9 future deletion hazards for no reader.
            CfnOutput(
                self,
                output_id_for(model_id),
                value=profile.attr_inference_profile_arn,
                description=f"{spec['display_name']} inference profile ARN",
            )

        # ── Deployed reality: model ID -> profile ARN ──────────────────────────────
        # Written in the same loop iteration set that created the profiles above, so
        # "present in SSM" implies "has a profile and is granted". Consumers read this one
        # parameter instead of a per-model environment variable.
        #
        # The ARNs are CloudFormation tokens at synth time; CDK resolves the markers inside
        # this JSON string into an Fn::Join when the template is produced.
        self.model_profiles_param_name = model_profiles_param_name(deployment_id)

        ssm.StringParameter(
            self,
            "ModelProfilesParam",
            parameter_name=self.model_profiles_param_name,
            string_value=json.dumps(
                {
                    model_id: profile.attr_inference_profile_arn
                    for model_id, profile in self._profiles.items()
                }
            ),
            description="Model ID to application inference profile ARN map",
        )

    def _apply_common_tags(self) -> None:
        """Apply common deployment tags to all resources in this stack."""
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def profile_arn(self, model_id: str) -> str:
        """Get the application inference profile ARN for a model ID."""
        try:
            return str(self._profiles[model_id].attr_inference_profile_arn)
        except KeyError:
            raise KeyError(
                f"No inference profile for {model_id!r}. "
                f"Known models: {sorted(self._profiles)}"
            ) from None

    # The six named `*_profile_arn` properties that used to live here are gone. They existed
    # only to feed per-model environment variables into three other stacks; those consumers
    # now read the SSM parameter, so `profile_arn(model_id)` is the whole interface. The one
    # remaining direct consumer is the image enhancer's VISION_MODEL, which calls it by ID.

    def _select(self, model_ids: Iterable[str] | None) -> dict[str, Any]:
        """Registry subset for an explicit model list, or every provisioned model.

        An ID that is not provisioned raises at synth. A consumer that names a model the
        registry has disabled or dropped should fail here, not with AccessDeniedException
        on its first call in the deployed account.
        """
        if model_ids is None:
            return self.models
        requested = list(model_ids)
        unknown = [m for m in requested if m not in self.models]
        if unknown:
            raise KeyError(
                f"Not provisioned by the registry: {unknown}. "
                f"Provisioned models: {sorted(self.models)}"
            )
        return {m: self.models[m] for m in requested}

    def grant_invoke_to_role(self, role, models: Iterable[str] | None = None) -> None:
        """Grant invoke permissions on registry models to the given role.

        By default every provisioned model is granted -- that is what the specialists and
        the agent need. ``models`` narrows the grant to an explicit subset for a role that
        invokes only one model, such as the UI task role behind the Create Specialist
        wizard. The subset must be provisioned by the registry; anything else raises.

        `CfnApplicationInferenceProfile` has no L2 grant method, so the statements are
        added by hand — but generated from the registry, not transcribed.

        Per AWS docs, invoking through an inference profile requires permissions on
        **both** the profile and the underlying foundation model in each Region associated
        with it. Hence three statements rather than one.

        No model wildcards. Every wildcard below is on the Region field, which cross-Region
        inference genuinely requires; the model IDs are all knowable at synth, so
        `AwsSolutions-IAM5` would have no evidence to justify widening them.
        """
        from aws_cdk import aws_iam as iam

        selected = self._select(models)

        # 1. The application inference profiles created by this stack.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeApplicationInferenceProfiles",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    self._profiles[model_id].attr_inference_profile_arn
                    for model_id in selected
                ],
            )
        )

        # 2. The underlying cross-Region system-defined profiles. Only cross-Region models
        #    have one — an In-Region-only model (cross_region=false) has no `us.*` system
        #    profile, so granting one would be a dangling permission. Its foundation model
        #    is covered by statement 3.
        system_profile_models = [
            model_id for model_id, spec in selected.items() if is_cross_region(spec)
        ]
        if system_profile_models:
            role.add_to_policy(
                iam.PolicyStatement(
                    sid="InvokeSystemInferenceProfiles",
                    actions=[
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream",
                    ],
                    resources=[
                        f"arn:aws:bedrock:*:{self.account}:inference-profile/{model_id}"
                        for model_id in system_profile_models
                    ],
                )
            )

        # 3. The foundation models behind those profiles.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeFoundationModels",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:*::foundation-model/{foundation_model_id(model_id)}"
                    for model_id in selected
                ],
            )
        )

        # 4. OpenAI models additionally require InvokeModel on the account's default
        #    project. Without it every OpenAI call returns AccessDenied even with a correct
        #    profile grant — and `_should_fallback` refuses to retry AccessDenied, so a GPT
        #    primary would not fall back to a Claude secondary. It would simply fail.
        #    Source: the GPT-5.6 Terra model card, Programmatic Access.
        #    This is one account-wide ARN, not per-model, so it sits outside the loops.
        if has_provider(selected, "openai"):
            role.add_to_policy(
                iam.PolicyStatement(
                    sid="InvokeDefaultProject",
                    actions=["bedrock:InvokeModel"],
                    resources=[
                        f"arn:aws:bedrock:{self.region}:{self.account}:project/default"
                    ],
                )
            )

    def invoke_nag_applies_to(
        self, models: Iterable[str] | None = None
    ) -> dict[str, list[str]]:
        """cdk-nag `appliesTo` strings for the Region wildcards `grant_invoke_to_role`
        introduces, keyed by the statement they belong to.

        Pass the same ``models`` the grant was given so the suppression names exactly the
        ARNs that statement produced and nothing more.

        Lives here, next to the statements, so a stack that suppresses these findings
        cannot list ARNs the grant no longer produces. Statement 1 needs no entry --
        application inference profile ARNs are `Fn::GetAtt` references with no wildcard --
        and statement 4 is a fully pinned ARN. The foundation-model list carries no account
        field, so only the profile list needs `account_renderings`.
        """
        selected = self._select(models)
        return {
            "system_inference_profiles": [
                f"Resource::arn:aws:bedrock:*:{account}:inference-profile/{model_id}"
                for model_id, spec in selected.items()
                if is_cross_region(spec)
                for account in account_renderings(self.account)
            ],
            "foundation_models": [
                f"Resource::arn:aws:bedrock:*::foundation-model/"
                f"{foundation_model_id(model_id)}"
                for model_id in selected
            ],
        }
