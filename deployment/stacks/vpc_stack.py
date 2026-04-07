"""VPC Stack for BADGERS frontend infrastructure."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_ec2 as ec2,
)
from constructs import Construct


class VpcStack(Stack):
    """VPC with public and private subnets for ALB + Fargate."""

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
        self._apply_common_tags()

        self.vpc = ec2.Vpc(
            self,
            "FrontendVpc",
            vpc_name=f"badgers-frontend-{deployment_id}",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # VPC Flow Logs — avoids ec2.vpc.flow_logs_not_enabled
        self.vpc.add_flow_log(
            "FlowLog",
            traffic_type=ec2.FlowLogTrafficType.ALL,
        )

        self._apply_resource_tags(
            self.vpc, "frontend-vpc", "VPC for BADGERS frontend ALB and Fargate"
        )

        CfnOutput(
            self,
            "VpcId",
            value=self.vpc.vpc_id,
            description="VPC ID",
            export_name=f"{Stack.of(self).stack_name}-VpcId",
        )

    def _apply_common_tags(self) -> None:
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _apply_resource_tags(
        self, resource: Construct, name: str, description: str
    ) -> None:
        Tags.of(resource).add("resource_name", name)
        Tags.of(resource).add("resource_description", description)
