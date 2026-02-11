"""S3 Stack for PDF Extractor POC."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    aws_s3 as s3,
    aws_kms as kms,
    aws_ssm as ssm,
    RemovalPolicy,
    Tags,
)
from constructs import Construct


class S3Stack(Stack):
    """Stack for S3 buckets."""

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

        # ID prefix for resource naming (lowercase for S3)
        id_prefix = "badgers"

        # Apply common tags to all resources
        self._apply_common_tags()

        # KMS key for S3 bucket encryption
        self.s3_kms_key = kms.Key(
            self,
            "S3EncryptionKey",
            alias=f"alias/{id_prefix}-s3-key-{deployment_id}",
            description="KMS key for S3 bucket encryption",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Config bucket for manifests and prompts
        self.config_bucket = s3.Bucket(
            self,
            "ConfigBucket",
            bucket_name=f"{id_prefix}-config-{deployment_id}",
            versioned=True,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.s3_kms_key,
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Source bucket for PDF uploads
        self.source_bucket = s3.Bucket(
            self,
            "SourceBucket",
            bucket_name=f"{id_prefix}-source-{deployment_id}",
            versioned=True,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.s3_kms_key,
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Output bucket for analysis results and temp files
        self.output_bucket = s3.Bucket(
            self,
            "OutputBucket",
            bucket_name=f"{id_prefix}-output-{deployment_id}",
            versioned=True,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.s3_kms_key,
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteTempAfter1Day",
                    prefix="temp/",
                    expiration=Duration.days(1),
                    noncurrent_version_expiration=Duration.days(1),
                )
            ],
        )

        # Apply resource-specific tags
        self._apply_resource_tags(
            self.s3_kms_key, "s3-encryption-key", "KMS key for S3 bucket encryption"
        )
        self._apply_resource_tags(
            self.config_bucket,
            "config-bucket",
            "S3 bucket for manifests and prompts and schemas",
        )
        self._apply_resource_tags(
            self.source_bucket, "source-bucket", "S3 bucket for PDF uploads"
        )
        self._apply_resource_tags(
            self.output_bucket,
            "output-bucket",
            "S3 bucket for analysis results and temp files",
        )

        # Outputs
        CfnOutput(
            self,
            "ConfigBucketName",
            value=self.config_bucket.bucket_name,
            description="Config bucket for manifests and prompts",
            export_name=f"{Stack.of(self).stack_name}-ConfigBucketName",
        )

        CfnOutput(
            self,
            "ConfigBucketArn",
            value=self.config_bucket.bucket_arn,
            description="Config bucket ARN",
            export_name=f"{Stack.of(self).stack_name}-ConfigBucketArn",
        )

        CfnOutput(
            self,
            "OutputBucketName",
            value=self.output_bucket.bucket_name,
            description="Output bucket for analysis results",
            export_name=f"{Stack.of(self).stack_name}-OutputBucketName",
        )

        CfnOutput(
            self,
            "OutputBucketArn",
            value=self.output_bucket.bucket_arn,
            description="Output bucket ARN",
            export_name=f"{Stack.of(self).stack_name}-OutputBucketArn",
        )

        CfnOutput(
            self,
            "SourceBucketName",
            value=self.source_bucket.bucket_name,
            description="Source bucket for PDF uploads",
            export_name=f"{Stack.of(self).stack_name}-SourceBucketName",
        )

        CfnOutput(
            self,
            "SourceBucketArn",
            value=self.source_bucket.bucket_arn,
            description="Source bucket ARN",
            export_name=f"{Stack.of(self).stack_name}-SourceBucketArn",
        )

        CfnOutput(
            self,
            "S3KmsKeyArn",
            value=self.s3_kms_key.key_arn,
            description="KMS key ARN for S3 bucket encryption",
            export_name=f"{Stack.of(self).stack_name}-S3KmsKeyArn",
        )

        # SSM Parameter for agent runtime to discover the config bucket
        self.config_bucket_param = ssm.StringParameter(
            self,
            "ConfigBucketNameParam",
            parameter_name="/badgers/config-bucket-name",
            string_value=self.config_bucket.bucket_name,
            description="Config bucket name for agent system prompt",
        )

        CfnOutput(
            self,
            "ConfigBucketParamName",
            value=self.config_bucket_param.parameter_name,
            description="SSM parameter path for config bucket name",
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
