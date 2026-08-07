"""ECS Stack for the BADGERS unified UI.

Ported from the media-contracts (MC) reference implementation. Replaces the
previous self-managed ALB + Fargate + ACM + Route53 stack with an ECS Express
Gateway Service, which provisions and manages its own load balancer and
endpoint (via the infrastructure role).

All container configuration is sourced from SSM Parameter Store via
``valueFrom`` references — no plaintext configuration in the service definition
or in deploy scripts.

SSM parameters read at task startup (prefix ``/badgers-{deployment_id}``):
  /agentcore-runtime-websocket-arn  (written here from the runtime stack value)
  /agentcore-gateway-id             (written here from the gateway stack value)
  /jobs-table-name                  (written by DynamoDBStack)
  /config-bucket-name               (written here)
  /output-bucket-name               (written here)
  /source-bucket-name               (written here)
  /node-env                         (written here, static)
  /ws-timeout-minutes               (written here, static)
  /cognito-user-pool-id             (written by CognitoStack)
  /cognito-domain                   (written by CognitoStack)
  /cognito-ui-client-id             (written by CognitoStack)

The task execution role gets ssm:GetParameters on the prefix so ECS can inject
values before the container starts.

The task role is scoped to exactly what the UI server calls:
  - bedrock-agentcore:InvokeAgentRuntime                   (HTTP invoke)
  - bedrock-agentcore:InvokeAgentRuntimeWithWebSocketStream (presigned wss:// chat)
  - bedrock-agentcore:ListGatewayTargets  (tool listing)
  - dynamodb Query/GetItem/DeleteItem on the jobs table (+ GSI)
  - s3 Get/Put/List on the config, source and output buckets
  - logs StartQuery/GetQueryResults (observability tab)
  - kms Decrypt/GenerateDataKey for the S3 CMK
  - ssm GetParameter(s) on the deployment prefix

After the Express service exists, an AwsCustomResource re-points the Cognito UI
client callback/logout URLs at the service endpoint (not known at synth time).
"""

from aws_cdk import (
    Stack,
    Tags,
    CfnOutput,
    CfnTag,
    RemovalPolicy,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_ssm as ssm,
    custom_resources as cr,
)
from constructs import Construct

try:  # cdk-nag is an optional synth-time aspect (enabled via CDK_NAG=1 in app.py)
    from cdk_nag import NagSuppressions

    _HAVE_CDK_NAG = True
except ImportError:  # pragma: no cover - cdk-nag present in the deploy venv
    _HAVE_CDK_NAG = False

CONTAINER_PORT = 7860


class ECSStack(Stack):
    """ECS Express Gateway Service and IAM roles for the BADGERS unified UI."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        config_bucket_arn: str,
        output_bucket_arn: str,
        source_bucket_arn: str,
        config_bucket_name: str,
        output_bucket_name: str,
        source_bucket_name: str,
        jobs_table_arn: str,
        kms_key_arn: str,
        private_subnet_ids: list[str],
        ui_task_sg_id: str,
        cognito_user_pool_id: str,
        cognito_ui_client_id: str,
        ecr_repository_uri: str,
        agentcore_runtime_websocket_arn: str,
        agentcore_gateway_id: str,
        stack_suffix: str,
        image_tag: str = "frontend",
        ws_timeout_minutes: str = "30",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags
        self._apply_common_tags()

        ssm_prefix = f"/badgers-{deployment_id}"
        ssm_wildcard = (
            f"arn:aws:ssm:{self.region}:{self.account}:parameter{ssm_prefix}/*"
        )

        def _ssm_arn(param_name: str) -> str:
            return f"arn:aws:ssm:{self.region}:{self.account}:parameter{param_name}"

        # ── SSM parameters written by this stack ───────────────────
        node_env_param = ssm.StringParameter(
            self,
            "NodeEnvParam",
            parameter_name=f"{ssm_prefix}/node-env",
            string_value="production",
            description="NODE_ENV for the UI container",
        )
        ws_timeout_param = ssm.StringParameter(
            self,
            "WsTimeoutParam",
            parameter_name=f"{ssm_prefix}/ws-timeout-minutes",
            string_value=ws_timeout_minutes,
            description="WebSocket request timeout in minutes",
        )
        runtime_arn_param = ssm.StringParameter(
            self,
            "RuntimeArnParam",
            parameter_name=f"{ssm_prefix}/agentcore-runtime-websocket-arn",
            string_value=agentcore_runtime_websocket_arn,
            description="AgentCore Runtime (WebSocket) ARN",
        )
        gateway_id_param = ssm.StringParameter(
            self,
            "GatewayIdParam",
            parameter_name=f"{ssm_prefix}/agentcore-gateway-id",
            string_value=agentcore_gateway_id,
            description="AgentCore Gateway ID",
        )
        config_bucket_param = ssm.StringParameter(
            self,
            "ConfigBucketParam",
            parameter_name=f"{ssm_prefix}/config-bucket-name",
            string_value=config_bucket_name,
            description="S3 config bucket name",
        )
        output_bucket_param = ssm.StringParameter(
            self,
            "OutputBucketParam",
            parameter_name=f"{ssm_prefix}/output-bucket-name",
            string_value=output_bucket_name,
            description="S3 output bucket name",
        )
        source_bucket_param = ssm.StringParameter(
            self,
            "SourceBucketParam",
            parameter_name=f"{ssm_prefix}/source-bucket-name",
            string_value=source_bucket_name,
            description="S3 source/upload bucket name",
        )

        # ── Task execution role (ECS pulls images + injects SSM) ───
        self.exec_role = iam.Role(
            self,
            "TaskExecRole",
            role_name=f"badgers-ecs-task-exec-{deployment_id}",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
        self.exec_role.add_to_policy(
            iam.PolicyStatement(
                sid="SSMGetParams",
                actions=["ssm:GetParameters", "ssm:GetParameter"],
                resources=[ssm_wildcard],
            )
        )
        # ECS needs to decrypt SecureString params / KMS-encrypted image layers
        self.exec_role.add_to_policy(
            iam.PolicyStatement(
                sid="KMSDecryptForInjection",
                actions=["kms:Decrypt"],
                resources=[kms_key_arn],
            )
        )

        # ── Task role (what the running container can do) ──────────
        self.task_role = iam.Role(
            self,
            "TaskRole",
            role_name=f"badgers-ecs-task-role-{deployment_id}",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Runtime permissions for the BADGERS UI ECS container",
        )

        # AgentCore — invoke the runtime (presigned WebSocket) and list gateway tools.
        #
        # The WebSocket path is a distinct action: InvokeAgentRuntime covers the HTTP
        # request/response API, while a presigned wss:// handshake to /runtimes/{arn}/ws
        # authorizes against InvokeAgentRuntimeWithWebSocketStream. Granting only the
        # former makes the handshake fail with 403 AccessDeniedException before the
        # upgrade, which never reaches the container and so never appears in its logs.
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeAgentCoreRuntime",
                actions=[
                    "bedrock-agentcore:InvokeAgentRuntime",
                    "bedrock-agentcore:InvokeAgentRuntimeWithWebSocketStream",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:runtime/*"
                ],
            )
        )
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="AgentCoreGatewayRead",
                actions=[
                    "bedrock-agentcore:ListGatewayTargets",
                    "bedrock-agentcore:GetGatewayTarget",
                    "bedrock-agentcore:GetGateway",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:gateway/*"
                ],
            )
        )

        # DynamoDB — job tracking reads and UI-initiated deletes
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="DynamoDBJobs",
                actions=[
                    "dynamodb:Query",
                    "dynamodb:GetItem",
                    "dynamodb:DeleteItem",
                ],
                resources=[jobs_table_arn, f"{jobs_table_arn}/index/*"],
            )
        )

        # S3 — config (read prompts/manifests), source (upload), output (results)
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ConfigReadWrite",
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"{config_bucket_arn}/*"],
            )
        )
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3SourceUpload",
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"{source_bucket_arn}/*"],
            )
        )
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3OutputRead",
                actions=["s3:GetObject"],
                resources=[f"{output_bucket_arn}/*"],
            )
        )
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3List",
                actions=["s3:ListBucket"],
                resources=[config_bucket_arn, source_bucket_arn, output_bucket_arn],
            )
        )

        # CloudWatch Logs Insights — the Observability tab
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="LogsInsights",
                actions=[
                    "logs:StartQuery",
                    "logs:GetQueryResults",
                    "logs:StopQuery",
                    "logs:DescribeLogGroups",
                ],
                resources=["*"],  # StartQuery requires log-group discovery
            )
        )

        # KMS — decrypt S3 objects and encrypt uploads
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="KMSDecryptAndEncrypt",
                actions=[
                    "kms:Decrypt",
                    "kms:DescribeKey",
                    "kms:GenerateDataKey",
                ],
                resources=[kms_key_arn],
            )
        )

        # SSM — server-side lookups at runtime
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                sid="SSMReadParams",
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[ssm_wildcard],
            )
        )

        # ── Infrastructure role (ECS manages the load balancer) ────
        self.infra_role = iam.Role(
            self,
            "InfraRole",
            role_name=f"badgers-ecs-infra-{deployment_id}",
            assumed_by=iam.ServicePrincipal("ecs.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"
                )
            ],
        )

        # ── CloudWatch log group ───────────────────────────────────
        log_group = logs.LogGroup(
            self,
            "UILogGroup",
            log_group_name=f"/ecs/badgers-ui-{deployment_id}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ── Express Gateway Service ────────────────────────────────
        ecr_image_uri = f"{ecr_repository_uri}:{image_tag}"

        self.express_service = ecs.CfnExpressGatewayService(
            self,
            "ExpressService",
            execution_role_arn=self.exec_role.role_arn,
            infrastructure_role_arn=self.infra_role.role_arn,
            task_role_arn=self.task_role.role_arn,
            service_name=f"badgers-ui-{deployment_id}",
            cluster="default",
            cpu="1024",
            memory="2048",
            health_check_path="/api/env",
            network_configuration=ecs.CfnExpressGatewayService.ExpressGatewayServiceNetworkConfigurationProperty(
                subnets=private_subnet_ids,
                security_groups=[ui_task_sg_id],
            ),
            primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
                image=ecr_image_uri,
                container_port=CONTAINER_PORT,
                environment=[
                    ecs.CfnExpressGatewayService.KeyValuePairProperty(
                        name="AWS_REGION",
                        value=self.region,
                    ),
                    ecs.CfnExpressGatewayService.KeyValuePairProperty(
                        name="PORT",
                        value=str(CONTAINER_PORT),
                    ),
                    # The admin Stacks tab builds CloudFormation stack names, which
                    # carry the per-deployment suffix. DEPLOYMENT_ID is the composite
                    # "{id}-{suffix}" used for resource names and the SSM prefix;
                    # STACK_SUFFIX is the bare suffix used in stack names.
                    ecs.CfnExpressGatewayService.KeyValuePairProperty(
                        name="DEPLOYMENT_ID",
                        value=deployment_id,
                    ),
                    ecs.CfnExpressGatewayService.KeyValuePairProperty(
                        name="STACK_SUFFIX",
                        value=stack_suffix,
                    ),
                ],
                secrets=[
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="NODE_ENV",
                        value_from=node_env_param.parameter_arn,
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="WS_TIMEOUT_MINUTES",
                        value_from=ws_timeout_param.parameter_arn,
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="AGENTCORE_RUNTIME_WEBSOCKET_ARN",
                        value_from=runtime_arn_param.parameter_arn,
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="AGENTCORE_GATEWAY_ID",
                        value_from=gateway_id_param.parameter_arn,
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="S3_CONFIG_BUCKET",
                        value_from=config_bucket_param.parameter_arn,
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="S3_OUTPUT_BUCKET",
                        value_from=output_bucket_param.parameter_arn,
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="S3_UPLOAD_BUCKET",
                        value_from=source_bucket_param.parameter_arn,
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="JOBS_TABLE_NAME",
                        value_from=_ssm_arn(f"{ssm_prefix}/jobs-table-name"),
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="COGNITO_USER_POOL_ID",
                        value_from=_ssm_arn(f"{ssm_prefix}/cognito-user-pool-id"),
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="COGNITO_DOMAIN",
                        value_from=_ssm_arn(f"{ssm_prefix}/cognito-domain"),
                    ),
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="COGNITO_UI_CLIENT_ID",
                        value_from=_ssm_arn(f"{ssm_prefix}/cognito-ui-client-id"),
                    ),
                ],
                aws_logs_configuration=ecs.CfnExpressGatewayService.ExpressGatewayServiceAwsLogsConfigurationProperty(
                    log_group=log_group.log_group_name,
                    log_stream_prefix="ui",
                ),
            ),
            tags=[CfnTag(key=k, value=v) for k, v in deployment_tags.items()],
        )

        # ── Cognito callback URL update (AwsCustomResource) ────────
        # The Express service endpoint is not known at synth time, so the UI
        # client's callback/logout URLs are re-pointed after the service exists.
        service_endpoint = self.express_service.attr_endpoint

        cognito_update_params = {
            "UserPoolId": cognito_user_pool_id,
            "ClientId": cognito_ui_client_id,
            "CallbackURLs": [f"https://{service_endpoint}/callback"],
            "LogoutURLs": [f"https://{service_endpoint}/"],
            # Re-specify existing client settings so they are preserved
            "AllowedOAuthFlows": ["code"],
            "AllowedOAuthScopes": ["openid", "email", "profile"],
            "AllowedOAuthFlowsUserPoolClient": True,
            "SupportedIdentityProviders": ["COGNITO"],
            "PreventUserExistenceErrors": "ENABLED",
            "EnableTokenRevocation": True,
            "AccessTokenValidity": 8,
            "IdTokenValidity": 8,
            "RefreshTokenValidity": 30,
            "TokenValidityUnits": {
                "AccessToken": "hours",
                "IdToken": "hours",
                "RefreshToken": "days",
            },
            "ExplicitAuthFlows": ["ALLOW_USER_SRP_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"],
        }

        cognito_callback_update = cr.AwsCustomResource(
            self,
            "CognitoCallbackUpdate",
            on_create=cr.AwsSdkCall(
                service="CognitoIdentityServiceProvider",
                action="updateUserPoolClient",
                parameters=cognito_update_params,
                physical_resource_id=cr.PhysicalResourceId.of(
                    f"cognito-callback-{deployment_id}"
                ),
            ),
            on_update=cr.AwsSdkCall(
                service="CognitoIdentityServiceProvider",
                action="updateUserPoolClient",
                parameters=cognito_update_params,
                physical_resource_id=cr.PhysicalResourceId.of(
                    f"cognito-callback-{deployment_id}"
                ),
            ),
            # No on_delete — the Cognito client is owned by CognitoStack
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=[
                            "cognito-idp:UpdateUserPoolClient",
                            "cognito-idp:DescribeUserPoolClient",
                        ],
                        resources=[
                            f"arn:aws:cognito-idp:{self.region}:{self.account}:userpool/{cognito_user_pool_id}",
                        ],
                    )
                ]
            ),
        )
        cognito_callback_update.node.add_dependency(self.express_service)

        # ── Outputs ────────────────────────────────────────────────
        CfnOutput(
            self,
            "ServiceArn",
            value=self.express_service.attr_service_arn,
            description="ECS Express Gateway service ARN",
            export_name=f"{self.stack_name}-ServiceArn",
        )
        CfnOutput(
            self,
            "ServiceEndpoint",
            value=self.express_service.attr_endpoint,
            description="Public endpoint for the BADGERS UI",
            export_name=f"{self.stack_name}-ServiceEndpoint",
        )
        CfnOutput(
            self,
            "ServiceUrl",
            value=f"https://{self.express_service.attr_endpoint}",
            description="BADGERS UI URL",
        )
        CfnOutput(
            self,
            "TaskRoleArn",
            value=self.task_role.role_arn,
            description="ECS task role ARN",
            export_name=f"{self.stack_name}-TaskRoleArn",
        )
        CfnOutput(
            self,
            "ExecRoleArn",
            value=self.exec_role.role_arn,
            description="ECS task execution role ARN",
            export_name=f"{self.stack_name}-ExecRoleArn",
        )

        # ── CDK Nag suppressions ───────────────────────────────────
        if _HAVE_CDK_NAG:
            NagSuppressions.add_resource_suppressions(
                self.task_role,
                [
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "logs:StartQuery requires log-group discovery across the "
                            "account for the Observability tab; CloudWatch Logs Insights "
                            "queries cannot be scoped to a single group at policy time."
                        ),
                        "appliesTo": ["Resource::*"],
                    },
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "S3 object paths use prefix-scoped wildcards — individual "
                            "object keys are dynamic job and specialist identifiers that "
                            "cannot be enumerated at deploy time."
                        ),
                    },
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "AgentCore runtime and gateway IDs are generated at deploy "
                            "time and are not available as CDK cross-stack references."
                        ),
                    },
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "DynamoDB GSI access requires the /index/* suffix — the table "
                            "ARN is already scoped to the specific jobs table."
                        ),
                    },
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "SSM parameters are prefix-scoped to the deployment; individual "
                            "parameter names are written by other stacks."
                        ),
                    },
                ],
                apply_to_children=True,
            )
            NagSuppressions.add_resource_suppressions(
                self.exec_role,
                [
                    {
                        "id": "AwsSolutions-IAM4",
                        "reason": (
                            "AmazonECSTaskExecutionRolePolicy is the AWS-managed policy "
                            "required for ECS task execution (image pull and log writes)."
                        ),
                        "appliesTo": [
                            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
                        ],
                    },
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "SSM parameters are prefix-scoped to the deployment — ECS reads "
                            "all parameters under this prefix at container startup."
                        ),
                    },
                ],
                apply_to_children=True,
            )
            NagSuppressions.add_resource_suppressions(
                self.infra_role,
                [
                    {
                        "id": "AwsSolutions-IAM4",
                        "reason": (
                            "AmazonECSInfrastructureRoleforExpressGatewayServices is the "
                            "AWS-managed policy required for ECS Express Gateway service "
                            "infrastructure management."
                        ),
                        "appliesTo": [
                            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"
                        ],
                    },
                ],
                apply_to_children=True,
            )
            NagSuppressions.add_stack_suppressions(
                self,
                [
                    {
                        "id": "AwsSolutions-IAM4",
                        "reason": (
                            "AwsCustomResource (CognitoCallbackUpdate) framework uses "
                            "AWSLambdaBasicExecutionRole — cannot be overridden."
                        ),
                        "appliesTo": [
                            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                        ],
                    },
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": (
                            "AwsCustomResource framework Lambda requires a wildcard for "
                            "log group creation."
                        ),
                    },
                    {
                        "id": "AwsSolutions-L1",
                        "reason": (
                            "AwsCustomResource framework manages its own Lambda runtime "
                            "version."
                        ),
                    },
                ],
            )

    def _apply_common_tags(self) -> None:
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)
