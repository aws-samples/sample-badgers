"""AgentCore Memory Stack for PDF Extractor POC."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_bedrockagentcore as agentcore,
)
from constructs import Construct


class AgentCoreMemoryStack(Stack):
    """Stack for AgentCore Memory (short-term memory for session persistence)."""

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

        # Create Memory resource for STM
        self.memory = agentcore.CfnMemory(
            self,
            "badgers-memory",
            name=f"badgers_memory_{deployment_id}",
            description="Short-term memory for PDF analysis sessions",
            event_expiry_duration=90,  # days
        )

        # Apply resource-specific tags
        self._apply_resource_tags(
            self.memory,
            "agentcore-memory",
            "Short-term memory for PDF analysis session persistence",
        )

        # Outputs
        CfnOutput(
            self,
            "MemoryId",
            value=self.memory.attr_memory_id,
            description="AgentCore Memory ID",
            export_name=f"{self.stack_name}-MemoryId",
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
