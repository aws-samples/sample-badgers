"""ECR Stack for AgentCore Runtime container images."""

from aws_cdk import (
    Stack,
    CfnOutput,
    RemovalPolicy,
    Tags,
    aws_ecr as ecr,
)
from constructs import Construct


class AgentCoreECRStack(Stack):
    """Stack for ECR repository to store agent container images."""

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

        # ECR Repository for all container images (runtime + container lambdas)
        self.repository = ecr.Repository(
            self,
            "BadgersRepository",
            repository_name=f"badgers-{deployment_id}",
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep last 10 images",
                    max_image_count=10,
                )
            ],
        )

        # Apply tags
        self._apply_common_tags()
        self._apply_resource_tags(
            self.repository,
            "ecr-repository",
            "ECR repository for AgentCore Runtime agent container images",
        )

        # Outputs
        CfnOutput(
            self,
            "RepositoryUri",
            value=self.repository.repository_uri,
            description="ECR repository URI for agent container",
            export_name=f"{Stack.of(self).stack_name}-RepositoryUri",
        )

        CfnOutput(
            self,
            "RepositoryArn",
            value=self.repository.repository_arn,
            description="ECR repository ARN",
            export_name=f"{Stack.of(self).stack_name}-RepositoryArn",
        )

        CfnOutput(
            self,
            "RepositoryName",
            value=self.repository.repository_name,
            description="ECR repository name",
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
