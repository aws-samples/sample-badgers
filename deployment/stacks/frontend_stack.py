"""Frontend Stack — ALB + Fargate + ACM + Route53 for BADGERS unified UI."""

from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Tags,
    aws_certificatemanager as acm,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_elasticloadbalancingv2_actions as elbv2_actions,
    aws_route53 as route53,
    aws_route53_targets as route53_targets,
    aws_s3 as s3,
)
from constructs import Construct


class FrontendStack(Stack):
    """ALB + Fargate service for the unified UI behind Cognito auth.

    Accepts either VPN CIDR blocks or a prefix list ID for ALB ingress —
    pass one or both via alb_ingress_cidrs / alb_ingress_prefix_list_id.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        vpc: ec2.IVpc,
        user_pool: cognito.IUserPool,
        user_pool_domain: cognito.UserPoolDomain,
        ecr_repository: ecr.IRepository,
        hosted_zone_id: str,
        hosted_zone_name: str,
        domain_name: str,
        image_tag: str = "frontend",
        container_port: int = 7860,
        alb_ingress_cidrs: list[str] | None = None,
        alb_ingress_prefix_list_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags
        self._apply_common_tags()

        # ----- Route53 hosted zone lookup -----
        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(
            self,
            "HostedZone",
            hosted_zone_id=hosted_zone_id,
            zone_name=hosted_zone_name,
        )

        # ----- ACM certificate -----
        certificate = acm.Certificate(
            self,
            "FrontendCert",
            domain_name=domain_name,
            validation=acm.CertificateValidation.from_dns(hosted_zone),
        )

        # ----- Cognito app client for ALB auth (authorization code flow) -----
        alb_callback_url = f"https://{domain_name}/oauth2/idpresponse"

        self.alb_app_client = cognito.UserPoolClient(
            self,
            "ALBAppClient",
            user_pool=user_pool,
            user_pool_client_name=f"badgers-alb-client-{deployment_id}",
            generate_secret=True,
            auth_flows=cognito.AuthFlow(user_srp=True),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=[alb_callback_url],
                logout_urls=[f"https://{domain_name}/"],
            ),
            supported_identity_providers=[
                cognito.UserPoolClientIdentityProvider.COGNITO,
            ],
        )

        # ----- ECS cluster -----
        cluster = ecs.Cluster(
            self,
            "FrontendCluster",
            cluster_name=f"badgers-frontend-{deployment_id}",
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        # ----- ALB security group -----
        alb_sg = ec2.SecurityGroup(
            self,
            "AlbSg",
            vpc=vpc,
            description="ALB ingress — VPN CIDRs or prefix list",
            allow_all_outbound=True,
        )

        if alb_ingress_cidrs:
            for cidr in alb_ingress_cidrs:
                alb_sg.add_ingress_rule(
                    ec2.Peer.ipv4(cidr),
                    ec2.Port.tcp(443),
                    "HTTPS from VPN CIDR",
                )

        if alb_ingress_prefix_list_id:
            alb_sg.add_ingress_rule(
                ec2.Peer.prefix_list(alb_ingress_prefix_list_id),
                ec2.Port.tcp(443),
                "HTTPS from VPN prefix list",
            )

        if not alb_ingress_cidrs and not alb_ingress_prefix_list_id:
            # Fallback — caller must provide at least one
            raise ValueError(
                "Provide alb_ingress_cidrs and/or alb_ingress_prefix_list_id"
            )

        # ----- Fargate security group -----
        fargate_sg = ec2.SecurityGroup(
            self,
            "FargateSg",
            vpc=vpc,
            description="Fargate — ALB only",
            allow_all_outbound=True,
        )
        fargate_sg.add_ingress_rule(
            ec2.Peer.security_group_id(alb_sg.security_group_id),
            ec2.Port.tcp(container_port),
            "From ALB only",
        )

        # ----- Task definition -----
        task_def = ecs.FargateTaskDefinition(
            self,
            "FrontendTask",
            cpu=512,
            memory_limit_mib=1024,
        )

        container = task_def.add_container(
            "frontend",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repository, tag=image_tag),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="frontend"),
            port_mappings=[
                ecs.PortMapping(
                    container_port=container_port, protocol=ecs.Protocol.TCP
                )
            ],
            environment={
                "NODE_ENV": "production",
                "PORT": str(container_port),
                "AWS_REGION": Stack.of(self).region,
            },
            health_check=ecs.HealthCheck(
                command=[
                    "CMD-SHELL",
                    f"curl -f http://localhost:{container_port}/ || exit 1",
                ],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(15),
            ),
        )

        # ----- ALB access logs bucket -----
        alb_logs_bucket = s3.Bucket(
            self,
            "AlbAccessLogsBucket",
            bucket_name=f"badgers-fe-alb-logs-{deployment_id}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ----- ALB -----
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "FrontendAlb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            load_balancer_name=f"badgers-fe-{deployment_id}",
            drop_invalid_header_fields=True,
        )

        # ALB access logging — avoids elbv2.loadbalancer.access_logging_disabled
        alb.log_access_logs(alb_logs_bucket)

        # ----- Fargate service -----
        service = ecs.FargateService(
            self,
            "FrontendService",
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            security_groups=[fargate_sg],
            assign_public_ip=False,
            enable_execute_command=False,  # explicit — avoids ecs.service.execute_command_enabled
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
        )

        # ----- Target group -----
        target_group = elbv2.ApplicationTargetGroup(
            self,
            "FrontendTg",
            vpc=vpc,
            port=container_port,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[service],
            health_check=elbv2.HealthCheck(
                path="/",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
            ),
            target_group_name=f"badgers-fe-{deployment_id}",
        )

        # ----- HTTPS listener with Cognito auth + forward -----
        https_listener = alb.add_listener(
            "HttpsListener",
            port=443,
            certificates=[certificate],
            ssl_policy=elbv2.SslPolicy.FORWARD_SECRECY_TLS12_RES_GCM,
            default_action=elbv2_actions.AuthenticateCognitoAction(
                user_pool=user_pool,
                user_pool_client=self.alb_app_client,
                user_pool_domain=user_pool_domain,
                session_cookie_name="AWSELBAuthSessionCookie",
                session_timeout=Duration.hours(1),
                next=elbv2.ListenerAction.forward([target_group]),
            ),
        )

        # HTTP → HTTPS redirect
        alb.add_listener(
            "HttpRedirect",
            port=80,
            default_action=elbv2.ListenerAction.redirect(
                protocol="HTTPS",
                port="443",
                permanent=True,
            ),
        )

        # ----- Route53 alias -----
        route53.ARecord(
            self,
            "FrontendDns",
            zone=hosted_zone,
            record_name=domain_name,
            target=route53.RecordTarget.from_alias(
                route53_targets.LoadBalancerTarget(alb)
            ),
        )

        # ----- Tags -----
        self._apply_resource_tags(alb, "frontend-alb", "ALB for BADGERS unified UI")
        self._apply_resource_tags(
            cluster, "frontend-cluster", "ECS cluster for BADGERS frontend"
        )

        # ----- Outputs -----
        CfnOutput(
            self,
            "FrontendUrl",
            value=f"https://{domain_name}",
            description="Frontend URL",
        )
        CfnOutput(
            self,
            "AlbDnsName",
            value=alb.load_balancer_dns_name,
            description="ALB DNS name",
        )
        CfnOutput(
            self,
            "AlbAppClientId",
            value=self.alb_app_client.user_pool_client_id,
            description="Cognito app client ID for ALB auth",
            export_name=f"{Stack.of(self).stack_name}-AlbAppClientId",
        )

    def _apply_common_tags(self) -> None:
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _apply_resource_tags(
        self, resource: Construct, name: str, description: str
    ) -> None:
        Tags.of(resource).add("resource_name", name)
        Tags.of(resource).add("resource_description", description)
