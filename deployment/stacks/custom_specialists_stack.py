"""Custom Specialists Stack - Deploys wizard-created specialists."""

import json
import logging
from pathlib import Path
from typing import Any
from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    Tags,
    CustomResource,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_kms as kms,
    custom_resources as cr,
)
from constructs import Construct

# Wizard-generated specialists run the same handler shape as the built-in ones, so
# they get the same memory. Imported rather than repeated so the two cannot diverge.
from .lambda_stack import SPECIALIST_MEMORY_MB
from .model_registry import model_profiles_param_name
from .nag_arn_renderings import nag_resource_string

try:  # cdk-nag is an optional synth-time aspect (enabled via CDK_NAG=1 in app.py)
    from cdk_nag import NagSuppressions

    _HAVE_CDK_NAG = True
except ImportError:  # pragma: no cover - cdk-nag present in the deploy venv
    _HAVE_CDK_NAG = False

logger = logging.getLogger(__name__)


class CustomSpecialistsStack(Stack):
    """Stack for custom specialists created via the wizard UI."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        config_bucket_name: str,
        output_bucket_name: str,
        foundation_layer_arn: str,
        lambda_role_arn: str,
        gateway_id: str,
        gateway_role_arn: str,
        kms_key_arn: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags
        self.config_bucket_name = config_bucket_name
        self.output_bucket_name = output_bucket_name
        self.gateway_id = gateway_id

        # Store inference profile ARNs

        self._apply_common_tags()

        # Import resources from base stacks
        self.foundation_layer = lambda_.LayerVersion.from_layer_version_arn(
            self, "ImportedFoundationLayer", foundation_layer_arn
        )
        self.lambda_role = iam.Role.from_role_arn(
            self, "ImportedLambdaRole", lambda_role_arn
        )
        self.config_bucket = s3.Bucket.from_bucket_name(
            self, "ImportedConfigBucket", config_bucket_name
        )

        # Import gateway role and grant KMS permissions for custom-specialists prefix
        self.gateway_role = iam.Role.from_role_arn(
            self, "ImportedGatewayRole", gateway_role_arn, mutable=True
        )
        self.kms_key = kms.Key.from_key_arn(self, "ImportedKmsKey", kms_key_arn)
        self.kms_key.grant_decrypt(self.gateway_role)
        self.pillow_layer = lambda_.LayerVersion.from_layer_version_arn(
            self,
            "PillowLayer",
            f"arn:aws:lambda:{Stack.of(self).region}:770693421928:layer:Klayers-p312-pillow:2",
        )

        self.specialists = self._load_specialist_registry()
        if not self.specialists:
            logger.warning("No custom specialists found in registry")
            return

        # Upload runtime files (prompts, manifests, schemas) to S3
        self.s3_deployment = self._upload_runtime_files()

        self.functions: dict[str, lambda_.Function] = {}
        for specialist in self.specialists:
            specialist_name = specialist.get("name")
            if specialist_name:
                self.functions[specialist_name] = self._create_specialist_function(
                    specialist
                )

        self._add_gateway_targets()
        self._create_outputs()

    def _apply_common_tags(self) -> None:
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _load_specialist_registry(self) -> list[dict[str, Any]]:
        registry_path = Path("./custom_specialists/specialist_registry.json")
        if not registry_path.exists():
            return []
        try:
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
            return list(registry.get("specialists", []))
        except Exception as e:
            logger.error("Failed to load specialist registry: %s", e)
            return []

    def _upload_runtime_files(self) -> s3deploy.BucketDeployment | None:
        """Upload prompts, manifests, and schemas to S3 for Lambda runtime."""
        custom_specialists_dir = Path("./custom_specialists")

        # Check if we have files to upload
        has_files = False
        for subdir in ["manifests", "schemas", "prompts"]:
            if (custom_specialists_dir / subdir).exists():
                has_files = True
                break

        if not has_files:
            logger.warning("No runtime files found to upload")
            self._s3_deployment_role = None
            return None

        # Create a role for the deployment with explicit S3 permissions
        deployment_role = iam.Role(
            self,
            "S3DeploymentRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        # Grant full access to the config bucket for deployment
        deployment_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject*",
                    "s3:PutObject*",
                    "s3:DeleteObject*",
                    "s3:ListBucket",
                ],
                resources=[
                    f"arn:aws:s3:::{self.config_bucket_name}",
                    f"arn:aws:s3:::{self.config_bucket_name}/*",
                ],
            )
        )

        # Grant KMS permissions for encrypted bucket
        deployment_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:GenerateDataKey*",
                ],
                resources=["*"],
                conditions={
                    "StringLike": {
                        "kms:ViaService": f"s3.{Stack.of(self).region}.amazonaws.com"
                    }
                },
            )
        )

        self._s3_deployment_role = deployment_role

        # Upload all runtime files to S3 under custom-specialists/ prefix
        return s3deploy.BucketDeployment(
            self,
            "CustomSpecialistRuntimeFiles",
            sources=[s3deploy.Source.asset(str(custom_specialists_dir))],
            destination_bucket=self.config_bucket,
            destination_key_prefix="custom-specialists",
            prune=False,  # Don't delete existing files
            retain_on_delete=False,
            role=deployment_role,
        )

    def _create_specialist_function(
        self, specialist: dict[str, Any]
    ) -> lambda_.Function:
        specialist_name = specialist["name"]
        description = specialist.get(
            "description", f"Custom specialist: {specialist_name}"
        )
        if len(description) > 256:
            description = description[:253] + "..."

        code_dir = self._generate_lambda_code(specialist_name)

        environment = {
            "SPECIALIST_NAME": specialist_name,
            "BEDROCK_READ_TIMEOUT": "900",
            "CACHE_ENABLED": "True",
            "FAIL_AFTER_ERROR": "False",
            "CONFIG_BUCKET": self.config_bucket_name,
            "OUTPUT_BUCKET": self.output_bucket_name,
            "JPEG_QUALITY": "85",
            "MAX_DIMENSIONS": "2048",
            "MAX_IMAGE_SIZE": "20971520",
            "MAX_TOKENS": "16000",
            "TEMPERATURE": "0.1",
            "THROTTLE_DELAY": "1.0",
            "CUSTOM_SPECIALIST": "true",
            # Model ID -> profile ARN map, for cost attribution. Replaces five per-model
            # ARNs that had to be imported across stack boundaries to get here.
            "MODEL_PROFILES_PARAM": model_profiles_param_name(self.deployment_id),
        }

        function = lambda_.Function(
            self,
            f"CustomFunction-{specialist_name}",
            function_name=f"badgers_{specialist_name}",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda_handler.lambda_handler",
            code=lambda_.Code.from_asset(str(code_dir)),
            role=self.lambda_role,
            layers=[self.foundation_layer, self.pillow_layer],
            timeout=Duration.seconds(900),
            memory_size=SPECIALIST_MEMORY_MB,
            reserved_concurrent_executions=5,
            description=description,
            environment=environment,
        )
        Tags.of(function).add("resource_name", f"custom-lambda-{specialist_name}")
        Tags.of(function).add("specialist_type", "custom")

        if _HAVE_CDK_NAG:
            # Same justification as the base specialists in lambda_stack.py, which has
            # carried this suppression since before the model migration. Custom
            # specialists attach the same foundation layer, so the runtime constraint is
            # identical -- this stack simply had no suppressions of any kind.
            NagSuppressions.add_resource_suppressions(
                function,
                [
                    {
                        "id": "AwsSolutions-L1",
                        "reason": (
                            "Runtime is pinned to Python 3.12 to match the shared Lambda "
                            "layers this function attaches (foundation and pillow), which "
                            "declare compatible_runtimes=[PYTHON_3_12] and ship "
                            "runtime-specific native artifacts. Bumping this function "
                            "alone would break layer compatibility -- the runtime and the "
                            "layers must be rebuilt and revalidated together. Python 3.12 "
                            "is a supported runtime and is not deprecated. Identical to "
                            "the suppression on the base specialists in lambda_stack.py; "
                            "the two are pinned by the same layers."
                        ),
                    }
                ],
            )

        return function

    def _generate_lambda_code(self, specialist_name: str) -> Path:
        code_dir = Path(f"./custom_specialists/code/{specialist_name}")
        code_dir.mkdir(parents=True, exist_ok=True)
        handler_path = code_dir / "lambda_handler.py"
        handler_code = self._get_handler_template(specialist_name)
        with open(handler_path, "w", encoding="utf-8") as f:
            f.write(handler_code)
        return code_dir

    def _get_handler_template(self, specialist_name: str) -> str:
        return (
            '''"""Custom Specialist Lambda - '''
            + specialist_name
            + '''."""
import json
import logging
import base64
import os
from pathlib import Path

from foundation.lambda_error_handler import (
    create_error_response, ValidationError, ResourceNotFoundError, handle_s3_error,
)
from foundation.s3_result_saver import save_result_to_s3

logger = logging.getLogger()
logger.setLevel(getattr(logging, os.environ.get("LOGGING_LEVEL", "INFO").upper(), logging.INFO))

SPECIALIST_NAME = "'''
            + specialist_name
            + """"

def lambda_handler(event, context):
    try:
        config_bucket = os.environ.get("CONFIG_BUCKET")
        specialist_name = os.environ.get("SPECIALIST_NAME", SPECIALIST_NAME)
        is_custom = os.environ.get("CUSTOM_SPECIALIST", "false").lower() == "true"
        body = json.loads(event["body"]) if "body" in event else event
        session_id = body.get("session_id", "no_session")
        audit_mode = body.get("audit_mode", False)

        image_data = _get_image_data(body)
        config = _load_config_from_s3(config_bucket, specialist_name, is_custom)
        specialist = _initialize_specialist(config, config_bucket, specialist_name, is_custom)
        result = specialist.analyze(image_data, body.get("aws_profile"), audit_mode)

        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket:
            try:
                s3_uri = save_result_to_s3(result=result, specialist_name=specialist_name,
                    output_bucket=output_bucket, session_id=session_id, image_path=body.get("image_path"))
                result = f"{result}\\n<!-- S3_RESULT_URI: {s3_uri} -->"
            except Exception as e:
                logger.error("Failed to save result to S3: %s", e)

        return {"statusCode": 200, "body": json.dumps({"result": result, "success": True, "session_id": session_id})}
    except Exception as e:
        return create_error_response(e)

def _get_image_data(body: dict) -> bytes:
    if "image_data" in body:
        return base64.b64decode(body["image_data"])
    if "image_path" in body:
        image_path = body["image_path"]
        if image_path.startswith("s3://"):
            import boto3
            s3 = boto3.client("s3")
            parts = image_path.replace("s3://", "").split("/", 1)
            bucket, key = parts
            response = s3.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()
            return base64.b64decode(data.decode("utf-8")) if key.endswith(".b64") else bytes(data)
        file_path = Path("/var/task") / image_path
        if file_path.exists():
            with open(file_path, "rb") as f:
                return f.read()
    raise ValidationError(message="Missing image_data or image_path", details={})

def _load_config_from_s3(bucket: str, specialist_name: str, is_custom: bool) -> dict:
    from foundation.s3_config_loader import load_manifest_from_s3
    manifest = load_manifest_from_s3(bucket, specialist_name, custom=is_custom)
    return manifest.get("specialist", manifest)

def _initialize_specialist(config: dict, s3_bucket: str, specialist_name: str, is_custom: bool):
    from foundation.specialist_foundation import SpecialistFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor

    specialist = object.__new__(SpecialistFoundation)
    specialist.specialist_type = specialist_name
    specialist.s3_bucket = s3_bucket
    specialist.logger = logging.getLogger(f"foundation.{specialist_name}")
    specialist.config = config
    specialist.global_settings = {
        "max_tokens": int(os.environ.get("MAX_TOKENS", "8000")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.1")),
        "max_image_size": int(os.environ.get("MAX_IMAGE_SIZE", "20971520")),
        "max_dimension": int(os.environ.get("MAX_DIMENSION", "2048")),
        "jpeg_quality": int(os.environ.get("JPEG_QUALITY", "85")),
        "cache_enabled": os.environ.get("CACHE_ENABLED", "True") == "True",
        "throttle_delay": float(os.environ.get("THROTTLE_DELAY", "1.0")),
        "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
    }
    specialist.config_manager = ConfigurationManager()
    specialist.prompt_loader = PromptLoader(config_source="s3", s3_bucket=s3_bucket, specialist_name=specialist_name, custom=is_custom)
    specialist.image_processor = ImageProcessor()
    specialist.bedrock_client = BedrockClient()
    specialist.message_builder = MessageChainBuilder()
    specialist.response_processor = ResponseProcessor()
    specialist._configure_components()
    return specialist
"""
        )

    def _add_gateway_targets(self) -> None:
        """Add custom specialist Lambda functions as gateway targets using Custom Resource."""
        # Grant invoke permissions to gateway role for all custom specialist functions
        for lambda_function in self.functions.values():
            lambda_function.grant_invoke(self.gateway_role)

        provider_fn = lambda_.Function(
            self,
            "GatewayTargetProvider",
            function_name=f"badgers-gw-target-provider-{self.deployment_id}",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_inline(self._get_provider_code()),
            timeout=Duration.seconds(300),
            memory_size=256,
        )

        provider_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agentcore:CreateGatewayTarget",
                    "bedrock-agentcore:DeleteGatewayTarget",
                    "bedrock-agentcore:GetGatewayTarget",
                    "bedrock-agentcore:ListGatewayTargets",
                    "bedrock-agentcore:SynchronizeGatewayTargets",
                ],
                resources=["*"],
            )
        )
        self.config_bucket.grant_read(provider_fn)
        # Grant KMS decrypt for encrypted S3 bucket
        self.kms_key.grant_decrypt(provider_fn)

        provider = cr.Provider(
            self, "GatewayTargetCustomProvider", on_event_handler=provider_fn
        )

        if _HAVE_CDK_NAG:
            self._add_nag_suppressions(provider_fn, provider)

        for specialist_name, lambda_function in self.functions.items():
            short_name = specialist_name
            if short_name.endswith("_specialist"):
                short_name = short_name[: -len("_specialist")]
            target_name = f"custom-{short_name.replace('_', '-')[:40]}"

            gateway_target = CustomResource(
                self,
                f"GatewayTarget-{specialist_name}",
                service_token=provider.service_token,
                properties={
                    "GatewayId": self.gateway_id,
                    "TargetName": target_name,
                    "Description": f"Custom specialist: {specialist_name}",
                    "LambdaArn": lambda_function.function_arn,
                    "SchemaS3Uri": f"s3://{self.config_bucket_name}/custom-specialists/schemas/{specialist_name}.json",
                },
            )
            # Ensure S3 files are uploaded before creating gateway target
            if self.s3_deployment:
                gateway_target.node.add_dependency(self.s3_deployment)

    def _add_nag_suppressions(
        self, provider_fn: lambda_.Function, provider: cr.Provider
    ) -> None:
        """Suppress, with evidence, every AwsSolutions finding this stack produces.

        This stack previously had no suppressions at all -- no ``NagSuppressions`` import --
        so every finding in it was unsuppressed. It only synthesizes when
        ``custom_specialists/specialist_registry.json`` exists, which is why the gap went
        unnoticed: a clean clone has no custom specialists and the stack contributes
        nothing.

        Every suppression is scoped to a construct and, for IAM5, to the exact finding
        string. Nothing is suppressed at stack scope. The one entry that cannot be a fixed
        string -- the CDK assets bucket, whose name embeds the bootstrap qualifier, account
        and Region -- uses cdk-nag's regex form rather than widening to the whole stack.
        Logical IDs that carry a CDK-computed hash are obtained from the construct with
        ``get_logical_id`` instead of being transcribed.
        """
        managed_policy = (
            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role"
            "/AWSLambdaBasicExecutionRole"
        )
        basic_execution_reason = (
            "AWSLambdaBasicExecutionRole grants only CloudWatch Logs "
            "CreateLogGroup/CreateLogStream/PutLogEvents on *, which is what a Lambda "
            "needs to emit logs at all. Replacing it with a scoped statement would "
            "require the log group ARN, which for a CDK-managed function is created at "
            "first invocation and not knowable at synth."
        )

        def version_qualifier(fn: lambda_.Function) -> str:
            # cdk-nag renders an Fn::GetAtt as <LogicalId.Arn>; the logical ID embeds a
            # CDK hash, so ask the construct rather than spelling it.
            return f"Resource::<{Stack.of(fn).get_logical_id(fn.node.default_child)}.Arn>:*"

        # ── The gateway-target provider handler (ours, inline boto3) ─────────────────
        NagSuppressions.add_resource_suppressions(
            provider_fn,
            [
                {
                    "id": "AwsSolutions-L1",
                    "reason": (
                        "Runtime is pinned to Python 3.12 to match the specialist "
                        "functions and shared layers this stack deploys alongside, so the "
                        "whole stack moves runtime together or not at all. The handler is "
                        "inline boto3 with no native dependencies. Python 3.12 is "
                        "supported and not deprecated."
                    ),
                },
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": basic_execution_reason,
                    "appliesTo": [managed_policy],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Two wildcard classes here, both from CDK rather than hand-written "
                        "policy. (1) Action wildcards emitted by Bucket.grant_read and "
                        "Key.grant_decrypt, each expanding to a fixed set of same-family "
                        'read actions on one bucket and one key. (2) Resource "*" on the '
                        "bedrock-agentcore gateway-target actions: CreateGatewayTarget and "
                        "its siblings are called against a gateway whose ID arrives as a "
                        "custom-resource property at deploy time, and the target ARN does "
                        "not exist until Create succeeds, so neither can be enumerated at "
                        "synth. The S3 object prefix is scoped to the config bucket, whose "
                        "keys are per-specialist values written by the wizard."
                    ),
                    "appliesTo": [
                        "Action::s3:GetBucket*",
                        "Action::s3:GetObject*",
                        "Action::s3:List*",
                        "Resource::*",
                        "Resource::" + nag_resource_string(self, f"arn:aws:s3:::{self.config_bucket_name}/*"),
                    ],
                },
            ],
            apply_to_children=True,
        )

        # ── The custom-resource Provider framework (CDK-owned function + role) ────────
        NagSuppressions.add_resource_suppressions(
            provider,
            [
                {
                    "id": "AwsSolutions-L1",
                    "reason": (
                        "The Provider framework's onEvent function is created by "
                        "aws-cdk-lib and its runtime is pinned by the library version, "
                        "not by this stack."
                    ),
                },
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": basic_execution_reason,
                    "appliesTo": [managed_policy],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "The framework grants itself lambda:InvokeFunction on the onEvent "
                        "handler with the :* version qualifier CDK appends to every Lambda "
                        "invoke grant. The function ARN is a resolved reference; only the "
                        "version qualifier is wildcarded."
                    ),
                    "appliesTo": [version_qualifier(provider_fn)],
                },
            ],
            apply_to_children=True,
        )

        # ── The gateway role (imported, mutable) invoking each custom specialist ──────
        NagSuppressions.add_resource_suppressions(
            self.gateway_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "lambda:InvokeFunction granted per custom specialist with the :* "
                        "version qualifier CDK appends to every Lambda invoke grant. Each "
                        "function ARN is a resolved reference to one specialist; only the "
                        "version qualifier is wildcarded. Generated from self.functions so "
                        "the list cannot drift from the grants."
                    ),
                    "appliesTo": [version_qualifier(fn) for fn in self.functions.values()],
                },
            ],
            apply_to_children=True,
        )

        # ── The BucketDeployment role (ours) and its CDK singleton function ───────────
        if self._s3_deployment_role is not None:
            NagSuppressions.add_resource_suppressions(
                self._s3_deployment_role,
                [
                    {
                        "id": "AwsSolutions-IAM4",
                        "reason": basic_execution_reason,
                        "appliesTo": [managed_policy],
                    },
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "This role is authored here for the BucketDeployment that "
                            "copies custom-specialist prompts, manifests and schemas into "
                            "the config bucket. Hand-written: s3:GetObject*/PutObject*/"
                            "DeleteObject* on the config bucket, and kms:GenerateDataKey* "
                            'with Resource "*" constrained by a kms:ViaService condition '
                            "to S3 in this Region -- the bucket key ARN is imported and "
                            "the deployment needs the same grant for any key S3 may use. "
                            "Added by BucketDeployment itself: the read family on the CDK "
                            "assets bucket it stages from, and the write family on the "
                            "destination. Object keys are per-specialist and created at "
                            "deploy time, hence the /* prefixes. The assets bucket name "
                            "embeds the bootstrap qualifier, account and Region, so it is "
                            "matched by regex rather than widened to stack scope."
                        ),
                        "appliesTo": [
                            "Action::s3:Abort*",
                            "Action::s3:DeleteObject*",
                            "Action::s3:GetBucket*",
                            "Action::s3:GetObject*",
                            "Action::s3:List*",
                            "Action::s3:PutObject*",
                            "Action::kms:GenerateDataKey*",
                            "Resource::*",
                            "Resource::" + nag_resource_string(self, f"arn:aws:s3:::{self.config_bucket_name}/*"),
                            {
                                "regex": (
                                    r"/^Resource::arn:aws:s3:::cdk-[a-z0-9]+-assets-"
                                    r"(\d{12}|<AWS::AccountId>)-[a-z0-9-]+\/\*$/g"
                                )
                            },
                        ],
                    },
                ],
                apply_to_children=True,
            )

            # BucketDeployment installs one stack-scoped singleton Lambda whose construct
            # id starts with "Custom::CDKBucketDeployment" followed by a CDK hash. Find it
            # by prefix rather than hardcoding the hash.
            for child in self.node.children:
                if child.node.id.startswith("Custom::CDKBucketDeployment"):
                    NagSuppressions.add_resource_suppressions(
                        child,
                        [
                            {
                                "id": "AwsSolutions-L1",
                                "reason": (
                                    "The BucketDeployment singleton function is created "
                                    "by aws-cdk-lib and its runtime is pinned by the "
                                    "library version, not by this stack."
                                ),
                            },
                        ],
                        apply_to_children=True,
                    )

    def _get_provider_code(self) -> str:
        return """
import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    logger.info("Event: %s", json.dumps(event))
    request_type = event["RequestType"]
    props = event["ResourceProperties"]
    gateway_id = props["GatewayId"]
    target_name = props["TargetName"]
    client = boto3.client("bedrock-agentcore-control")

    try:
        if request_type == "Create":
            response = client.create_gateway_target(
                gatewayIdentifier=gateway_id,
                name=target_name,
                description=props.get("Description", ""),
                credentialProviderConfigurations=[
                    {"credentialProviderType": "GATEWAY_IAM_ROLE"}
                ],
                targetConfiguration={
                    "mcp": {
                        "lambda": {
                            "lambdaArn": props["LambdaArn"],
                            "toolSchema": {"s3": {"uri": props["SchemaS3Uri"]}}
                        }
                    }
                }
            )
            return {"PhysicalResourceId": response["targetId"]}

        elif request_type == "Update":
            try:
                client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=event["PhysicalResourceId"])
            except Exception:
                pass
            response = client.create_gateway_target(
                gatewayIdentifier=gateway_id,
                name=target_name,
                description=props.get("Description", ""),
                credentialProviderConfigurations=[
                    {"credentialProviderType": "GATEWAY_IAM_ROLE"}
                ],
                targetConfiguration={
                    "mcp": {
                        "lambda": {
                            "lambdaArn": props["LambdaArn"],
                            "toolSchema": {"s3": {"uri": props["SchemaS3Uri"]}}
                        }
                    }
                }
            )
            return {"PhysicalResourceId": response["targetId"]}

        elif request_type == "Delete":
            try:
                client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=event["PhysicalResourceId"])
            except Exception as e:
                logger.warning("Delete failed (may already be deleted): %s", e)
            return {"PhysicalResourceId": event["PhysicalResourceId"]}

    except Exception as e:
        logger.error("Error: %s", e)
        raise
"""

    def _create_outputs(self) -> None:
        CfnOutput(
            self,
            "CustomSpecialistCount",
            value=str(len(self.functions)),
            description="Number of custom specialists deployed",
        )
        if self.functions:
            CfnOutput(
                self,
                "CustomSpecialistNames",
                value=",".join(list(self.functions.keys())),
                description="Names of deployed custom specialists",
            )
