"""Custom Analyzers Stack - Deploys wizard-created analyzers."""

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

logger = logging.getLogger(__name__)


class CustomAnalyzersStack(Stack):
    """Stack for custom analyzers created via the wizard UI."""

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
        claude_sonnet_profile_arn: str,
        claude_haiku_profile_arn: str,
        nova_premier_profile_arn: str,
        claude_opus_46_profile_arn: str,
        claude_opus_45_profile_arn: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags
        self.config_bucket_name = config_bucket_name
        self.output_bucket_name = output_bucket_name
        self.gateway_id = gateway_id

        # Store inference profile ARNs
        self.claude_sonnet_profile_arn = claude_sonnet_profile_arn
        self.claude_haiku_profile_arn = claude_haiku_profile_arn
        self.nova_premier_profile_arn = nova_premier_profile_arn
        self.claude_opus_46_profile_arn = claude_opus_46_profile_arn
        self.claude_opus_45_profile_arn = claude_opus_45_profile_arn

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

        # Import gateway role and grant KMS permissions for custom-analyzers prefix
        self.gateway_role = iam.Role.from_role_arn(
            self, "ImportedGatewayRole", gateway_role_arn, mutable=True
        )
        self.kms_key = kms.Key.from_key_arn(self, "ImportedKmsKey", kms_key_arn)
        self.kms_key.grant_decrypt(self.gateway_role)
        self.pillow_layer = lambda_.LayerVersion.from_layer_version_arn(
            self,
            "PillowLayer",
            "arn:aws:lambda:us-west-2:770693421928:layer:Klayers-p312-pillow:2",
        )

        self.analyzers = self._load_analyzer_registry()
        if not self.analyzers:
            logger.warning("No custom analyzers found in registry")
            return

        # Upload runtime files (prompts, manifests, schemas) to S3
        self.s3_deployment = self._upload_runtime_files()

        self.functions: dict[str, lambda_.Function] = {}
        for analyzer in self.analyzers:
            analyzer_name = analyzer.get("name")
            if analyzer_name:
                self.functions[analyzer_name] = self._create_analyzer_function(analyzer)

        self._add_gateway_targets()
        self._create_outputs()

    def _apply_common_tags(self) -> None:
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _load_analyzer_registry(self) -> list[dict[str, Any]]:
        registry_path = Path("./custom_analyzers/analyzer_registry.json")
        if not registry_path.exists():
            return []
        try:
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
            return list(registry.get("analyzers", []))
        except Exception as e:
            logger.error("Failed to load analyzer registry: %s", e)
            return []

    def _upload_runtime_files(self) -> s3deploy.BucketDeployment | None:
        """Upload prompts, manifests, and schemas to S3 for Lambda runtime."""
        custom_analyzers_dir = Path("./custom_analyzers")

        # Check if we have files to upload
        has_files = False
        for subdir in ["manifests", "schemas", "prompts"]:
            if (custom_analyzers_dir / subdir).exists():
                has_files = True
                break

        if not has_files:
            logger.warning("No runtime files found to upload")
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

        # Upload all runtime files to S3 under custom-analyzers/ prefix
        return s3deploy.BucketDeployment(
            self,
            "CustomAnalyzerRuntimeFiles",
            sources=[s3deploy.Source.asset(str(custom_analyzers_dir))],
            destination_bucket=self.config_bucket,
            destination_key_prefix="custom-analyzers",
            prune=False,  # Don't delete existing files
            retain_on_delete=False,
            role=deployment_role,
        )

    def _create_analyzer_function(self, analyzer: dict[str, Any]) -> lambda_.Function:
        analyzer_name = analyzer["name"]
        description = analyzer.get("description", f"Custom analyzer: {analyzer_name}")
        if len(description) > 256:
            description = description[:253] + "..."

        code_dir = self._generate_lambda_code(analyzer_name)

        environment = {
            "ANALYZER_NAME": analyzer_name,
            "BEDROCK_READ_TIMEOUT": "300",
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
            "CUSTOM_ANALYZER": "true",
            # Inference profile ARNs for cost tracking
            "CLAUDE_SONNET_PROFILE_ARN": self.claude_sonnet_profile_arn,
            "CLAUDE_HAIKU_PROFILE_ARN": self.claude_haiku_profile_arn,
            "NOVA_PREMIER_PROFILE_ARN": self.nova_premier_profile_arn,
            "CLAUDE_OPUS_46_PROFILE_ARN": self.claude_opus_46_profile_arn,
            "CLAUDE_OPUS_45_PROFILE_ARN": self.claude_opus_45_profile_arn,
        }

        function = lambda_.Function(
            self,
            f"CustomFunction-{analyzer_name}",
            function_name=f"badgers_{analyzer_name}",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda_handler.lambda_handler",
            code=lambda_.Code.from_asset(str(code_dir)),
            role=self.lambda_role,
            layers=[self.foundation_layer, self.pillow_layer],
            timeout=Duration.seconds(300),
            memory_size=2048,
            reserved_concurrent_executions=5,
            description=description,
            environment=environment,
        )
        Tags.of(function).add("resource_name", f"custom-lambda-{analyzer_name}")
        Tags.of(function).add("analyzer_type", "custom")
        return function

    def _generate_lambda_code(self, analyzer_name: str) -> Path:
        code_dir = Path(f"./custom_analyzers/code/{analyzer_name}")
        code_dir.mkdir(parents=True, exist_ok=True)
        handler_path = code_dir / "lambda_handler.py"
        handler_code = self._get_handler_template(analyzer_name)
        with open(handler_path, "w", encoding="utf-8") as f:
            f.write(handler_code)
        return code_dir

    def _get_handler_template(self, analyzer_name: str) -> str:
        return (
            '''"""Custom Analyzer Lambda - '''
            + analyzer_name
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

ANALYZER_NAME = "'''
            + analyzer_name
            + """"

def lambda_handler(event, context):
    try:
        config_bucket = os.environ.get("CONFIG_BUCKET")
        analyzer_name = os.environ.get("ANALYZER_NAME", ANALYZER_NAME)
        is_custom = os.environ.get("CUSTOM_ANALYZER", "false").lower() == "true"
        body = json.loads(event["body"]) if "body" in event else event
        session_id = body.get("session_id", "no_session")
        audit_mode = body.get("audit_mode", False)

        image_data = _get_image_data(body)
        config = _load_config_from_s3(config_bucket, analyzer_name, is_custom)
        analyzer = _initialize_analyzer(config, config_bucket, analyzer_name, is_custom)
        result = analyzer.analyze(image_data, body.get("aws_profile"), audit_mode)

        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket:
            try:
                s3_uri = save_result_to_s3(result=result, analyzer_name=analyzer_name,
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

def _load_config_from_s3(bucket: str, analyzer_name: str, is_custom: bool) -> dict:
    from foundation.s3_config_loader import load_manifest_from_s3
    manifest = load_manifest_from_s3(bucket, analyzer_name, custom=is_custom)
    return manifest.get("analyzer", manifest)

def _initialize_analyzer(config: dict, s3_bucket: str, analyzer_name: str, is_custom: bool):
    from foundation.analyzer_foundation import AnalyzerFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor

    analyzer = object.__new__(AnalyzerFoundation)
    analyzer.analyzer_type = analyzer_name
    analyzer.s3_bucket = s3_bucket
    analyzer.logger = logging.getLogger(f"foundation.{analyzer_name}")
    analyzer.config = config
    analyzer.global_settings = {
        "max_tokens": int(os.environ.get("MAX_TOKENS", "8000")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.1")),
        "max_image_size": int(os.environ.get("MAX_IMAGE_SIZE", "20971520")),
        "max_dimension": int(os.environ.get("MAX_DIMENSION", "2048")),
        "jpeg_quality": int(os.environ.get("JPEG_QUALITY", "85")),
        "cache_enabled": os.environ.get("CACHE_ENABLED", "True") == "True",
        "throttle_delay": float(os.environ.get("THROTTLE_DELAY", "1.0")),
        "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
    }
    analyzer.config_manager = ConfigurationManager()
    analyzer.prompt_loader = PromptLoader(config_source="s3", s3_bucket=s3_bucket, analyzer_name=analyzer_name, custom=is_custom)
    analyzer.image_processor = ImageProcessor()
    analyzer.bedrock_client = BedrockClient()
    analyzer.message_builder = MessageChainBuilder()
    analyzer.response_processor = ResponseProcessor()
    analyzer._configure_components()
    return analyzer
"""
        )

    def _add_gateway_targets(self) -> None:
        """Add custom analyzer Lambda functions as gateway targets using Custom Resource."""
        # Grant invoke permissions to gateway role for all custom analyzer functions
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

        for analyzer_name, lambda_function in self.functions.items():
            short_name = analyzer_name
            if short_name.endswith("_analyzer"):
                short_name = short_name[:-9]
            target_name = f"custom-{short_name.replace('_', '-')[:40]}"

            gateway_target = CustomResource(
                self,
                f"GatewayTarget-{analyzer_name}",
                service_token=provider.service_token,
                properties={
                    "GatewayId": self.gateway_id,
                    "TargetName": target_name,
                    "Description": f"Custom analyzer: {analyzer_name}",
                    "LambdaArn": lambda_function.function_arn,
                    "SchemaS3Uri": f"s3://{self.config_bucket_name}/custom-analyzers/schemas/{analyzer_name}.json",
                },
            )
            # Ensure S3 files are uploaded before creating gateway target
            if self.s3_deployment:
                gateway_target.node.add_dependency(self.s3_deployment)

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
            "CustomAnalyzerCount",
            value=str(len(self.functions)),
            description="Number of custom analyzers deployed",
        )
        if self.functions:
            CfnOutput(
                self,
                "CustomAnalyzerNames",
                value=",".join(list(self.functions.keys())),
                description="Names of deployed custom analyzers",
            )
