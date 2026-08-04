"""VPC Stack for BADGERS UI infrastructure.

Aligned with the media-contracts (MC) reference: public + private subnets for
the ECS Express Gateway UI service, a dedicated security group for interface
endpoints, and VPC endpoints so AWS API traffic stays off the NAT gateway.

Gateway endpoints (free, no hourly charge): S3, DynamoDB.
Interface endpoints: Bedrock Runtime, SSM, Secrets Manager.
"""

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_ec2 as ec2,
)
from constructs import Construct

try:  # cdk-nag is an optional synth-time aspect (enabled via CDK_NAG=1 in app.py)
    from cdk_nag import NagSuppressions

    _HAVE_CDK_NAG = True
except ImportError:  # pragma: no cover - cdk-nag present in the deploy venv
    _HAVE_CDK_NAG = False


class VpcStack(Stack):
    """VPC with public/private subnets and AWS service VPC endpoints."""

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
            vpc_name=f"badgers-ui-{deployment_id}",
            max_azs=2,
            nat_gateways=1,  # single NAT — cost optimised; increase for HA
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

        # ── Security groups ────────────────────────────────────────
        # Interface endpoint SG — accepts HTTPS from the UI task SG.
        self.endpoint_sg = ec2.SecurityGroup(
            self,
            "EndpointSG",
            vpc=self.vpc,
            security_group_name=f"badgers-endpoint-sg-{deployment_id}",
            description="Security group for BADGERS VPC interface endpoints",
            allow_all_outbound=False,
        )

        # UI task SG — the ECS Express Gateway service's ENIs.
        self.ui_task_sg = ec2.SecurityGroup(
            self,
            "UITaskSG",
            vpc=self.vpc,
            security_group_name=f"badgers-ui-task-sg-{deployment_id}",
            description="Security group for the BADGERS UI ECS task",
            allow_all_outbound=True,
        )

        self.endpoint_sg.add_ingress_rule(
            peer=ec2.Peer.security_group_id(self.ui_task_sg.security_group_id),
            connection=ec2.Port.tcp(443),
            description="HTTPS from the UI task to VPC interface endpoints",
        )

        # ── VPC Endpoints — keep AWS API traffic off NAT ───────────
        s3_endpoint = self.vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )
        Tags.of(s3_endpoint).add("Name", f"badgers-s3-endpoint-{deployment_id}")

        ddb_endpoint = self.vpc.add_gateway_endpoint(
            "DynamoDBEndpoint",
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
        )
        Tags.of(ddb_endpoint).add("Name", f"badgers-dynamodb-endpoint-{deployment_id}")

        self.bedrock_endpoint = self.vpc.add_interface_endpoint(
            "BedrockEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
            private_dns_enabled=True,
            security_groups=[self.endpoint_sg],
        )
        Tags.of(self.bedrock_endpoint).add(
            "Name", f"badgers-bedrock-endpoint-{deployment_id}"
        )

        ssm_endpoint = self.vpc.add_interface_endpoint(
            "SSMEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SSM,
            private_dns_enabled=True,
            security_groups=[self.endpoint_sg],
        )
        Tags.of(ssm_endpoint).add("Name", f"badgers-ssm-endpoint-{deployment_id}")

        sm_endpoint = self.vpc.add_interface_endpoint(
            "SecretsManagerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
            private_dns_enabled=True,
            security_groups=[self.endpoint_sg],
        )
        Tags.of(sm_endpoint).add(
            "Name", f"badgers-secretsmanager-endpoint-{deployment_id}"
        )

        # Convenience accessors for consuming stacks
        self.private_subnet_ids = [
            s.subnet_id
            for s in self.vpc.select_subnets(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ).subnets
        ]

        # ECS Express Mode picks the load balancer scheme from the subnets it is given:
        # public subnets get an internet-facing ALB (and public IPs on the tasks),
        # subnets with no internet gateway get an internal one. The UI needs the public
        # set to be reachable from a browser — see UI_PUBLIC_ACCESS in app.py.
        self.public_subnet_ids = [
            s.subnet_id
            for s in self.vpc.select_subnets(subnet_type=ec2.SubnetType.PUBLIC).subnets
        ]

        self._apply_resource_tags(
            self.vpc, "ui-vpc", "VPC for the BADGERS UI ECS service"
        )

        # ── Outputs ────────────────────────────────────────────────
        CfnOutput(
            self,
            "VpcId",
            value=self.vpc.vpc_id,
            description="VPC ID",
            export_name=f"{Stack.of(self).stack_name}-VpcId",
        )
        CfnOutput(
            self,
            "UITaskSecurityGroupId",
            value=self.ui_task_sg.security_group_id,
            description="Security group ID for the UI ECS task",
            export_name=f"{Stack.of(self).stack_name}-UITaskSecurityGroupId",
        )
        CfnOutput(
            self,
            "PrivateSubnetIds",
            value=",".join(self.private_subnet_ids),
            description="Comma-separated private subnet IDs",
            export_name=f"{Stack.of(self).stack_name}-PrivateSubnetIds",
        )
        CfnOutput(
            self,
            "PublicSubnetIds",
            value=",".join(self.public_subnet_ids),
            description="Comma-separated public subnet IDs",
            export_name=f"{Stack.of(self).stack_name}-PublicSubnetIds",
        )

        # ── CDK Nag suppressions ───────────────────────────────────
        if _HAVE_CDK_NAG:
            # The interface-endpoint security group's ingress CIDR is an intrinsic
            # (the VPC's CidrBlock attribute), which cdk-nag cannot resolve at
            # synth time, so EC23 raises a validation failure rather than a real
            # finding. Ingress is restricted to the VPC CIDR and to the UI task
            # security group on port 443 only.
            NagSuppressions.add_resource_suppressions(
                self.endpoint_sg,
                [
                    {
                        "id": "CdkNagValidationFailure",
                        "reason": (
                            "EC23 cannot validate an ingress rule whose CIDR is the "
                            "VPC CidrBlock intrinsic. Ingress is scoped to the VPC CIDR "
                            "and the UI task security group on TCP 443."
                        ),
                    },
                ],
            )

    def _apply_common_tags(self) -> None:
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _apply_resource_tags(
        self, resource: Construct, name: str, description: str
    ) -> None:
        Tags.of(resource).add("resource_name", name)
        Tags.of(resource).add("resource_description", description)
