"""Cognito Stack for PDF Extractor POC."""

from aws_cdk import (
    Stack,
    CfnOutput,
    RemovalPolicy,
    SecretValue,
    Tags,
    aws_cognito as cognito,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class CognitoStack(Stack):
    """Stack for Cognito user pool and identity pool for AgentCore Gateway auth."""

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

        # User Pool for authentication
        self.user_pool = cognito.UserPool(
            self,
            "AgentCoreUserPool",
            user_pool_name=f"badgers-users-{deployment_id}",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(username=True, email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True)
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Resource Server for OAuth 2.0 scopes (required for client credentials)
        self.resource_server = self.user_pool.add_resource_server(
            "AgentCoreResourceServer",
            identifier="agentcore-gateway",
            scopes=[
                cognito.ResourceServerScope(
                    scope_name="invoke",
                    scope_description="Invoke AgentCore Gateway tools",
                )
            ],
        )

        # App Client for AgentCore Gateway with OAuth 2.0 Client Credentials
        # Note: Must be created after resource server to reference scopes
        self.user_pool_client = self.user_pool.add_client(
            "AgentCoreGatewayClient",
            user_pool_client_name=f"agentcore-gateway-client-{deployment_id}",
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True,
                admin_user_password=True,
            ),
            generate_secret=True,  # Required for client credentials flow
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(
                    client_credentials=True,  # Enable client credentials flow
                ),
                scopes=[cognito.OAuthScope.custom("agentcore-gateway/invoke")],
            ),
            prevent_user_existence_errors=True,
        )

        # Explicit dependency to ensure resource server is created first
        self.user_pool_client.node.add_dependency(self.resource_server)

        # Add domain for OAuth endpoints
        self.user_pool_domain = self.user_pool.add_domain(
            "AgentCoreDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=f"badgers-{deployment_id}"
            ),
        )

        # Store credentials in Secrets Manager for secure runtime access
        self.credentials_secret = secretsmanager.Secret(
            self,
            "CognitoCredentialsSecret",
            secret_name=f"badgers/cognito-config-{deployment_id}",
            description="Cognito client credentials for AgentCore Gateway",
            secret_object_value={
                "client_id": SecretValue.unsafe_plain_text(
                    self.user_pool_client.user_pool_client_id
                ),
                "client_secret": self.user_pool_client.user_pool_client_secret,
                "token_endpoint": SecretValue.unsafe_plain_text(
                    f"https://{self.user_pool_domain.domain_name}.auth.{Stack.of(self).region}.amazoncognito.com/oauth2/token"
                ),
            },
        )

        # Identity Pool for AWS credentials
        self.identity_pool = cognito.CfnIdentityPool(
            self,
            "AgentCoreIdentityPool",
            identity_pool_name=f"badgers_identity_{deployment_id}",
            allow_unauthenticated_identities=False,
            cognito_identity_providers=[
                cognito.CfnIdentityPool.CognitoIdentityProviderProperty(
                    client_id=self.user_pool_client.user_pool_client_id,
                    provider_name=self.user_pool.user_pool_provider_name,
                )
            ],
        )

        # Apply resource-specific tags
        self._apply_resource_tags(
            self.user_pool, "cognito-user-pool", "User pool for AgentCore Gateway auth"
        )
        self._apply_resource_tags(
            self.credentials_secret,
            "cognito-credentials-secret",
            "Secrets Manager secret for Cognito client credentials",
        )
        self._apply_resource_tags(
            self.identity_pool,
            "cognito-identity-pool",
            "Identity pool for AWS credentials",
        )

        # Outputs
        CfnOutput(
            self,
            "UserPoolId",
            value=self.user_pool.user_pool_id,
            description="Cognito User Pool ID for AgentCore Gateway",
            export_name=f"{Stack.of(self).stack_name}-UserPoolId",
        )

        CfnOutput(
            self,
            "UserPoolArn",
            value=self.user_pool.user_pool_arn,
            description="Cognito User Pool ARN",
            export_name=f"{Stack.of(self).stack_name}-UserPoolArn",
        )

        CfnOutput(
            self,
            "UserPoolClientId",
            value=self.user_pool_client.user_pool_client_id,
            description="Cognito User Pool Client ID for AgentCore Gateway",
            export_name=f"{Stack.of(self).stack_name}-UserPoolClientId",
        )

        CfnOutput(
            self,
            "IdentityPoolId",
            value=self.identity_pool.ref,
            description="Cognito Identity Pool ID",
            export_name=f"{Stack.of(self).stack_name}-IdentityPoolId",
        )

        CfnOutput(
            self,
            "UserPoolProviderUrl",
            value=self.user_pool.user_pool_provider_url,
            description="Cognito User Pool Provider URL (for OIDC)",
        )

        CfnOutput(
            self,
            "OAuthTokenEndpoint",
            value=f"https://{self.user_pool.user_pool_provider_name}.auth.{Stack.of(self).region}.amazoncognito.com/oauth2/token",
            description="OAuth 2.0 token endpoint for client credentials flow",
        )

        CfnOutput(
            self,
            "ResourceServerIdentifier",
            value="agentcore-gateway",
            description="Resource server identifier for OAuth scopes",
        )

        CfnOutput(
            self,
            "CredentialsSecretArn",
            value=self.credentials_secret.secret_arn,
            description="Secrets Manager ARN for Cognito credentials",
            export_name=f"{Stack.of(self).stack_name}-CredentialsSecretArn",
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
