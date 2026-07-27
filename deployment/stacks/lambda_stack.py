"""Lambda Stack for BADGERS."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    aws_lambda as lambda_,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_s3 as s3,
    aws_ecr as ecr,
    Tags,
)
from constructs import Construct

if TYPE_CHECKING:
    from stacks.inference_profiles_stack import InferenceProfilesStack

# Container-based functions (too large for layers)
CONTAINER_FUNCTIONS = ["image_enhancer", "remediation_specialist"]


class LambdaSpecialistStack(Stack):
    """Stack for Lambda specialist functions."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_tags: dict[str, str],
        execution_role: iam.Role,
        config_bucket: s3.Bucket,
        output_bucket: s3.Bucket,
        jobs_table: dynamodb.ITable,
        ecr_repository: ecr.Repository,
        inference_profiles_stack: Optional[InferenceProfilesStack] = None,
        enabled_specialists: Optional[set[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_tags = deployment_tags
        self.execution_role = execution_role
        self.config_bucket = config_bucket
        self.output_bucket = output_bucket
        self.jobs_table = jobs_table
        self.ecr_repository = ecr_repository
        self.inference_profiles_stack = inference_profiles_stack
        self.enabled_specialists = enabled_specialists

        # Apply common tags to all resources
        self._apply_common_tags()

        # Create Lambda layer
        self.create_layer()

        # Create all specialist Lambda functions
        self.create_specialist_functions()

        # Create container-based Lambda functions
        self.create_container_functions()

        # Create outputs
        self.create_outputs()

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

    def create_layer(self):
        """Create Lambda layers from built zip files."""
        # Reference the pre-built foundation layer
        layer_path = Path("./lambdas/layer.zip")

        if not layer_path.exists():
            raise FileNotFoundError(
                f"Layer not found at {layer_path}. "
                "Run: cd lambdas && ./build_foundation_layer.sh"
            )

        self.foundation_layer = lambda_.LayerVersion(
            self,
            "SpecialistFoundationLayer",
            code=lambda_.Code.from_asset(str(layer_path)),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="Specialist foundation with Strands, Bedrock client, and utilities",
            layer_version_name="specialist-foundation",
        )

        # Apply resource-specific tags to layer
        self._apply_resource_tags(
            self.foundation_layer,
            "lambda-foundation-layer",
            "Specialist foundation layer with Strands and Bedrock client",
        )

        # Pillow layer (external)
        self.pillow_layer = lambda_.LayerVersion.from_layer_version_arn(
            self,
            "PillowLayer",
            f"arn:aws:lambda:{Stack.of(self).region}:770693421928:layer:Klayers-p312-pillow:2",
        )

        # Poppler layer for PDF conversion
        poppler_layer_path = Path("./lambdas/poppler-qpdf-layer.zip")

        if not poppler_layer_path.exists():
            raise FileNotFoundError(
                f"Poppler layer not found at {poppler_layer_path}. "
                "Run: cd lambdas && ./build_poppler_qdf_layer.sh"
            )

        self.poppler_layer = lambda_.LayerVersion(
            self,
            "PopplerLayer",
            code=lambda_.Code.from_asset(str(poppler_layer_path)),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="Poppler utilities for PDF processing",
            layer_version_name="poppler-utils",
        )

        # Apply resource-specific tags to poppler layer
        self._apply_resource_tags(
            self.poppler_layer,
            "lambda-poppler-layer",
            "Poppler utilities layer for PDF processing",
        )

        # Enhancement layer removed — enhancement runs in the container-based
        # agentic enhancer Lambda (image_enhancer) which bundles its own deps.
        # The build_enhancement_layer.sh script is retained on disk but not deployed.
        self.enhancement_layer = None

        # PDF processing layer for PDF manipulation/accessibility (pymupdf, pikepdf)
        pdf_processing_layer_path = Path("./lambdas/pdf-processing-layer.zip")

        if pdf_processing_layer_path.exists():
            self.pdf_processing_layer = lambda_.LayerVersion(
                self,
                "PdfProcessingLayer",
                code=lambda_.Code.from_asset(str(pdf_processing_layer_path)),
                compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
                description="PDF processing for accessibility tagging (pymupdf, pikepdf)",
                layer_version_name="pdf-processing",
            )

            # Apply resource-specific tags to pdf processing layer
            self._apply_resource_tags(
                self.pdf_processing_layer,
                "lambda-pdf-processing-layer",
                "PDF processing layer for accessibility tagging",
            )
        else:
            self.pdf_processing_layer = None
            logging.getLogger(__name__).warning(
                "PDF processing layer not found at %s. "
                "Run: cd lambdas && ./build_pdf_processing_layer.sh",
                pdf_processing_layer_path,
            )

    def create_specialist_functions(self):
        """Create specialist Lambda functions (filtered by enabled_specialists if set)."""
        # Get list of specialists from lambdas/code directory
        lambdas_dir = Path("./lambdas/code")

        if not lambdas_dir.exists():
            raise FileNotFoundError(f"Lambdas directory not found: {lambdas_dir}")

        specialist_dirs = sorted([d for d in lambdas_dir.iterdir() if d.is_dir()])

        self.functions = {}
        logger = logging.getLogger(__name__)

        for specialist_dir in specialist_dirs:
            specialist_name = specialist_dir.name
            # Skip container-based functions
            if specialist_name in CONTAINER_FUNCTIONS:
                continue
            # Skip disabled specialists
            if (
                self.enabled_specialists is not None
                and specialist_name not in self.enabled_specialists
            ):
                logger.info("Skipping disabled specialist: %s", specialist_name)
                continue
            function = self.create_specialist_function(specialist_name, specialist_dir)
            self.functions[specialist_name] = function

    def create_specialist_function(
        self, specialist_name: str, code_dir: Path
    ) -> lambda_.Function:
        """Create a single specialist Lambda function."""
        # Load schema for description
        schema_path = Path(f"./s3_files/schemas/{specialist_name}.json")
        description = self.get_tool_description(schema_path, specialist_name)

        # Environment variables
        environment = {
            "SPECIALIST_NAME": specialist_name,
            "BEDROCK_READ_TIMEOUT": "900",
            "CACHE_ENABLED": "True",
            "FAIL_AFTER_ERROR": "False",
            "CONFIG_BUCKET": self.config_bucket.bucket_name,
            "OUTPUT_BUCKET": self.output_bucket.bucket_name,
            "JOBS_TABLE_NAME": self.jobs_table.table_name,
            "JPEG_QUALITY": "85",
            "MAX_DIMENSIONS": "2048",
            "MAX_IMAGE_SIZE": "20971520",
            "MAX_TOKENS": "16000",
            "TEMPERATURE": "0.1",
            "THROTTLE_DELAY": "1.0",
        }

        # Add inference profile ARNs for cost tracking
        if self.inference_profiles_stack:
            environment.update(
                {
                    "CLAUDE_SONNET_PROFILE_ARN": self.inference_profiles_stack.claude_sonnet_profile_arn,
                    "CLAUDE_HAIKU_PROFILE_ARN": self.inference_profiles_stack.claude_haiku_profile_arn,
                    "NOVA_PREMIER_PROFILE_ARN": self.inference_profiles_stack.nova_premier_profile_arn,
                    "CLAUDE_OPUS_46_PROFILE_ARN": self.inference_profiles_stack.claude_opus_46_profile_arn,
                    "CLAUDE_OPUS_45_PROFILE_ARN": self.inference_profiles_stack.claude_opus_45_profile_arn,
                }
            )

        # Add poppler paths for pdf_to_images_converter
        if specialist_name == "pdf_to_images_converter":
            environment["PATH"] = "/opt/bin:/var/lang/bin:/usr/local/bin:/usr/bin:/bin"
            environment["LD_LIBRARY_PATH"] = "/opt/lib:/var/lang/lib:/lib64:/usr/lib64"
            environment["FONTCONFIG_PATH"] = "/opt/etc/fonts"
            environment["FONTCONFIG_CACHE"] = "/tmp/fontconfig-cache"

        # Determine layers for this function
        layers = [self.foundation_layer, self.pillow_layer]
        if specialist_name == "pdf_to_images_converter":
            layers.append(self.poppler_layer)

        # Attach PDF processing layer to functions that need PDF manipulation
        # NOTE: Container functions (remediation_specialist) are built via
        # _create_ecr_container_function and never reach this code path.
        # They bundle their own deps in the Docker image.
        pdf_processing_functions: list[str] = []
        if self.pdf_processing_layer and specialist_name in pdf_processing_functions:
            layers.append(self.pdf_processing_layer)

        # Enhancement layer removed — runs in container Lambda instead.
        # See lambdas/containers/image_enhancer/

        # Create function
        function = lambda_.Function(
            self,
            f"Function-{specialist_name}",
            function_name=f"badgers_{specialist_name}",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda_handler.lambda_handler",
            code=lambda_.Code.from_asset(str(code_dir)),
            role=self.execution_role,
            layers=layers,
            timeout=Duration.seconds(900),
            memory_size=2048,
            reserved_concurrent_executions=5,
            description=description,
            environment=environment,
        )

        # Apply resource-specific tags
        self._apply_resource_tags(
            function,
            f"lambda-{specialist_name}",
            description[:256] if len(description) > 256 else description,
        )

        return function

    def create_container_functions(self):
        """Create container-based Lambda functions from pre-built ECR images.

        Images must be pre-built and pushed to the shared ECR repository with
        function-specific tags (e.g., image_enhancer, remediation_specialist).

        Run: cd lambdas && ./build_container_lambdas.sh <ecr_repo_uri>
        """
        logger = logging.getLogger(__name__)

        for func_name in CONTAINER_FUNCTIONS:
            # Skip disabled specialists
            if (
                self.enabled_specialists is not None
                and func_name not in self.enabled_specialists
            ):
                logger.info("Skipping disabled container specialist: %s", func_name)
                continue
            # Check if image exists in ECR (tag = func_name)
            # CDK will fail at deploy time if image doesn't exist
            function = self._create_ecr_container_function(func_name)
            self.functions[func_name] = function
            logger.info("Created container function: %s", func_name)

    def _create_ecr_container_function(self, func_name: str) -> lambda_.Function:
        """Create a Lambda function from a pre-built ECR image.

        Container image functions do NOT support Lambda layers. All dependencies
        must be included in the container image itself.
        """
        schema_path = Path(f"./s3_files/schemas/{func_name}.json")
        description = self.get_tool_description(schema_path, func_name)

        environment = {
            "SPECIALIST_NAME": func_name,
            "BEDROCK_READ_TIMEOUT": "900",
            "CACHE_ENABLED": "True",
            "CONFIG_BUCKET": self.config_bucket.bucket_name,
            "OUTPUT_BUCKET": self.output_bucket.bucket_name,
            "JOBS_TABLE_NAME": self.jobs_table.table_name,
            "MAX_TOKENS": "16000",
            "TEMPERATURE": "0.1",
        }

        # Add inference profile ARNs for cost tracking
        if self.inference_profiles_stack:
            environment.update(
                {
                    "CLAUDE_SONNET_PROFILE_ARN": self.inference_profiles_stack.claude_sonnet_profile_arn,
                    "CLAUDE_HAIKU_PROFILE_ARN": self.inference_profiles_stack.claude_haiku_profile_arn,
                    "NOVA_PREMIER_PROFILE_ARN": self.inference_profiles_stack.nova_premier_profile_arn,
                    "CLAUDE_OPUS_46_PROFILE_ARN": self.inference_profiles_stack.claude_opus_46_profile_arn,
                    "CLAUDE_OPUS_45_PROFILE_ARN": self.inference_profiles_stack.claude_opus_45_profile_arn,
                }
            )

            # Image enhancer uses VISION_MODEL to select its model - point it at the application inference profile
            if func_name == "image_enhancer":
                environment["VISION_MODEL"] = (
                    self.inference_profiles_stack.claude_sonnet_46_profile_arn
                )

        # Container image functions do NOT support Lambda layers.
        # All dependencies (including foundation libs) must be baked into the container image.
        function = lambda_.Function(
            self,
            f"ContainerFunction-{func_name}",
            function_name=f"badgers_{func_name}",
            code=lambda_.Code.from_ecr_image(
                repository=self.ecr_repository,
                tag_or_digest=func_name,
            ),
            handler=lambda_.Handler.FROM_IMAGE,
            runtime=lambda_.Runtime.FROM_IMAGE,
            role=self.execution_role,
            timeout=Duration.seconds(900),
            memory_size=2048,
            reserved_concurrent_executions=5,
            description=description,
            environment=environment,
        )

        self._apply_resource_tags(
            function,
            f"lambda-container-{func_name}",
            description[:256] if len(description) > 256 else description,
        )

        return function

    def get_tool_description(self, schema_path: Path, specialist_name: str) -> str:
        """Get tool description from schema file."""
        if not schema_path.exists():
            return f"Specialist for {specialist_name.replace('_', ' ')}"

        try:
            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)

            if "inlinePayload" in schema and len(schema["inlinePayload"]) > 0:
                description = str(schema["inlinePayload"][0].get("description", ""))
                # Truncate to 256 chars (Lambda description limit)
                if len(description) > 256:
                    description = description[:253] + "..."
                return description
        except Exception as e:
            logging.getLogger(__name__).debug(
                "Could not load schema description for %s: %s", specialist_name, e
            )

        return f"Specialist for {specialist_name.replace('_', ' ')}"

    def create_outputs(self):
        """Create CloudFormation outputs."""
        # Essential outputs
        CfnOutput(
            self,
            "LayerArn",
            value=self.foundation_layer.layer_version_arn,
            description="Specialist foundation layer ARN",
            export_name=f"{Stack.of(self).stack_name}-LayerArn",
        )

        # Key orchestration function ARNs
        key_functions = [
            "pdf_to_images_converter",
            "classify_pdf_content",
        ]
        for func_name in key_functions:
            if func_name in self.functions:
                CfnOutput(
                    self,
                    f"{func_name.title().replace('_', '')}Arn",
                    value=self.functions[func_name].function_arn,
                    description=f"{func_name} function ARN",
                    export_name=f"{Stack.of(self).stack_name}-{func_name.title().replace('_', '')}-Arn",
                )

        # Nice-to-have outputs
        CfnOutput(
            self,
            "LayerVersion",
            value=str(self.foundation_layer.layer_version_arn.split(":")[-1]),
            description="Layer version number",
        )

        CfnOutput(
            self,
            "FunctionCount",
            value=str(len(self.functions)),
            description="Total number of specialist functions",
        )

        CfnOutput(
            self,
            "FunctionNamePrefix",
            value="badgers_",
            description="Function name prefix for filtering",
        )

        # All function ARNs as JSON
        function_arns = {
            name: func.function_arn for name, func in self.functions.items()
        }
        CfnOutput(
            self,
            "AllFunctionArns",
            value=json.dumps(function_arns, indent=2),
            description="All specialist function ARNs (JSON)",
        )
