"""Cognito Stack for BADGERS.

Aligned with the media-contracts (MC) reference implementation: one user pool
carrying two app clients, plus Managed Login v2 with explicit branding.

1. UI client (OIDC authorization-code + PKCE)
   - Used by the React frontend via react-oidc-context.
   - Public client (no secret), scopes ``openid email profile``.
   - Callback / logout URLs default to the local dev origin and are re-pointed
     at the deployed ECS service endpoint by the ECS stack after the Express
     Gateway service exists (its endpoint is not known at synth time).

2. Gateway client (OAuth 2.0 client-credentials, machine-to-machine)
   - Used by the AgentCore Runtime to mint tokens for Gateway auth.
   - Has a client secret, mirrored into Secrets Manager.
   - Scope: ``agentcore-gateway/invoke``.

Managed Login v2 requires an explicit branding style: an app client with no
branding returns "Login pages unavailable" at the hosted login URL. Custom
branding is loaded from ``deployment/assets/cognito-branding-definitions.json``
when present, otherwise Cognito-provided defaults are used.

Contracts preserved for other stacks:
  - ``credentials_secret`` — consumed by AgentCoreRuntimeWebSocketStack
  - export ``badgers-cognito-UserPoolId`` — consumed by AgentCoreGatewayStack
  - export ``badgers-cognito-UserPoolClientId`` — the M2M gateway client id,
    used as ``allowed_clients`` on the Gateway JWT authorizer
"""

import json
import os
from pathlib import Path

from aws_cdk import (
    Duration,
    Stack,
    CfnOutput,
    RemovalPolicy,
    SecretValue,
    Tags,
    aws_cognito as cognito,
    aws_secretsmanager as secretsmanager,
    aws_ssm as ssm,
)
from constructs import Construct

try:  # cdk-nag is an optional synth-time aspect (enabled via CDK_NAG=1 in app.py)
    from cdk_nag import NagSuppressions

    _HAVE_CDK_NAG = True
except ImportError:  # pragma: no cover - cdk-nag present in the deploy venv
    _HAVE_CDK_NAG = False


class CognitoStack(Stack):
    """Cognito User Pool with UI (OIDC/PKCE) and Gateway (M2M) app clients."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        ui_callback_url: str | None = None,
        ui_logout_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags
        self._apply_common_tags()

        # Callback / logout URLs — default to the local dev origin. The ECS stack
        # re-points these at the deployed service endpoint post-deploy.
        ui_callback_url = (
            ui_callback_url
            or os.environ.get("UI_CALLBACK_URL")
            or "http://localhost:7860/callback"
        )
        ui_logout_url = (
            ui_logout_url or os.environ.get("UI_LOGOUT_URL") or "http://localhost:7860/"
        )

        ssm_prefix = f"/badgers-{deployment_id}"

        # ── User Pool ──────────────────────────────────────────────
        self.user_pool = cognito.UserPool(
            self,
            "AgentCoreUserPool",
            user_pool_name=f"badgers-users-{deployment_id}",
            self_sign_up_enabled=False,  # admin-only provisioning — no public signup
            sign_in_aliases=cognito.SignInAliases(username=True, email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True)
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            mfa=cognito.Mfa.REQUIRED,
            mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            standard_threat_protection_mode=cognito.StandardThreatProtectionMode.FULL_FUNCTION,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ── Groups (drive UI role gating: admin sees deploy/admin tabs) ──
        self.admin_group = cognito.CfnUserPoolGroup(
            self,
            "AdminGroup",
            user_pool_id=self.user_pool.user_pool_id,
            group_name="admin",
            description="Full access — all UI tabs and admin API routes",
        )

        self.tester_group = cognito.CfnUserPoolGroup(
            self,
            "TesterGroup",
            user_pool_id=self.user_pool.user_pool_id,
            group_name="tester",
            description="Testing access — testing UI tabs and API routes only",
        )

        # ── Cognito domain (managed login v2) ──────────────────────
        self.domain_prefix = f"badgers-{deployment_id}"
        self.cfn_domain = cognito.CfnUserPoolDomain(
            self,
            "AgentCoreDomain",
            domain=self.domain_prefix,
            user_pool_id=self.user_pool.user_pool_id,
            managed_login_version=2,
        )

        domain_url = f"https://{self.domain_prefix}.auth.{Stack.of(self).region}.amazoncognito.com"
        authority = (
            f"https://cognito-idp.{Stack.of(self).region}.amazonaws.com"
            f"/{self.user_pool.user_pool_id}"
        )
        self.domain_url = domain_url
        self.authority = authority

        # ── Resource server for Gateway M2M scope ──────────────────
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

        # ── App Client 1: UI (OIDC, public, PKCE) ──────────────────
        self.ui_client = self.user_pool.add_client(
            "UIClient",
            user_pool_client_name=f"badgers-ui-{deployment_id}",
            generate_secret=False,  # public client — PKCE, no secret
            auth_flows=cognito.AuthFlow(user_srp=True),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=[ui_callback_url],
                logout_urls=[ui_logout_url],
            ),
            supported_identity_providers=[
                cognito.UserPoolClientIdentityProvider.COGNITO,
            ],
            prevent_user_existence_errors=True,
            access_token_validity=Duration.hours(8),
            id_token_validity=Duration.hours(8),
            refresh_token_validity=Duration.days(30),
            enable_token_revocation=True,
        )

        # ── App Client 2: AgentCore Gateway M2M (client credentials) ──
        # Export name UserPoolClientId must keep pointing at THIS client — the
        # Gateway JWT authorizer uses it as allowed_clients.
        self.user_pool_client = self.user_pool.add_client(
            "AgentCoreGatewayClient",
            user_pool_client_name=f"agentcore-gateway-client-{deployment_id}",
            generate_secret=True,  # required for client credentials flow
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True,
                admin_user_password=True,
            ),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(client_credentials=True),
                scopes=[cognito.OAuthScope.custom("agentcore-gateway/invoke")],
            ),
            prevent_user_existence_errors=True,
        )
        self.user_pool_client.node.add_dependency(self.resource_server)

        # ── Managed login branding (REQUIRED for managed login v2) ──
        # An app client with NO branding style returns "Login pages unavailable"
        # at the hosted login URL. Load exported branding when available; fall
        # back to Cognito-provided defaults otherwise.
        branding_file = (
            Path(__file__).parent.parent
            / "assets"
            / "cognito-branding-definitions.json"
        )
        if branding_file.exists():
            branding_data = json.loads(branding_file.read_text(encoding="utf-8"))
            branding_root = branding_data.get("ManagedLoginBranding", branding_data)
            branding_settings = branding_root.get("Settings", {})
            # Transform assets from PascalCase (API export format) to typed CDK
            # AssetTypeProperty instances. Deduplicate by (category, color_mode) —
            # Cognito rejects multiple assets per slot.
            raw_assets = branding_root.get("Assets", [])
            seen: set[tuple[str, str]] = set()
            branding_assets = [
                cognito.CfnManagedLoginBranding.AssetTypeProperty(
                    category=a.get("Category", ""),
                    color_mode=a.get("ColorMode", ""),
                    extension=a.get("Extension", ""),
                    bytes=a.get("Bytes", ""),
                )
                for a in raw_assets
                if (a.get("Category", ""), a.get("ColorMode", "")) not in seen
                and not seen.add((a.get("Category", ""), a.get("ColorMode", "")))
            ]

            ui_branding = cognito.CfnManagedLoginBranding(
                self,
                "UIManagedLoginBranding",
                user_pool_id=self.user_pool.user_pool_id,
                client_id=self.ui_client.user_pool_client_id,
                use_cognito_provided_values=False,
                settings=branding_settings,
                assets=branding_assets,
            )
        else:
            ui_branding = cognito.CfnManagedLoginBranding(
                self,
                "UIManagedLoginBranding",
                user_pool_id=self.user_pool.user_pool_id,
                client_id=self.ui_client.user_pool_client_id,
                use_cognito_provided_values=True,
            )
        ui_branding.node.add_dependency(self.cfn_domain)
        self.ui_branding = ui_branding

        # ── Secrets Manager — Gateway M2M credentials ──────────────
        # Shape preserved: AgentCoreRuntimeWebSocketStack reads this secret.
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
                    f"{domain_url}/oauth2/token"
                ),
            },
        )

        # ── Identity Pool for AWS credentials ──────────────────────
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

        # ── SSM parameters (cross-stack reads by the UI container) ──
        ssm.StringParameter(
            self,
            "UserPoolIdParam",
            parameter_name=f"{ssm_prefix}/cognito-user-pool-id",
            string_value=self.user_pool.user_pool_id,
            description="Cognito User Pool ID",
        )
        ssm.StringParameter(
            self,
            "UIClientIdParam",
            parameter_name=f"{ssm_prefix}/cognito-ui-client-id",
            string_value=self.ui_client.user_pool_client_id,
            description="Cognito UI app client ID (public, PKCE)",
        )
        ssm.StringParameter(
            self,
            "AuthorityParam",
            parameter_name=f"{ssm_prefix}/cognito-authority",
            string_value=authority,
            description="OIDC authority URL for react-oidc-context",
        )
        ssm.StringParameter(
            self,
            "DomainParam",
            parameter_name=f"{ssm_prefix}/cognito-domain",
            string_value=domain_url,
            description="Cognito hosted UI domain URL",
        )
        ssm.StringParameter(
            self,
            "GatewayClientIdParam",
            parameter_name=f"{ssm_prefix}/cognito-gateway-client-id",
            string_value=self.user_pool_client.user_pool_client_id,
            description="Cognito M2M app client ID (non-sensitive)",
        )

        # ── Resource tags ──────────────────────────────────────────
        self._apply_resource_tags(
            self.user_pool,
            "cognito-user-pool",
            "User pool for BADGERS UI and Gateway auth",
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

        # ── Outputs ────────────────────────────────────────────────
        CfnOutput(
            self,
            "UserPoolId",
            value=self.user_pool.user_pool_id,
            description="Cognito User Pool ID",
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
            description="Cognito M2M client ID for AgentCore Gateway",
            export_name=f"{Stack.of(self).stack_name}-UserPoolClientId",
        )
        CfnOutput(
            self,
            "UIClientId",
            value=self.ui_client.user_pool_client_id,
            description="Cognito UI app client ID (public, PKCE)",
            export_name=f"{Stack.of(self).stack_name}-UIClientId",
        )
        CfnOutput(
            self,
            "Authority",
            value=authority,
            description="OIDC authority for VITE_COGNITO_AUTHORITY",
            export_name=f"{Stack.of(self).stack_name}-Authority",
        )
        CfnOutput(
            self,
            "CognitoDomain",
            value=domain_url,
            description="Cognito domain for VITE_COGNITO_DOMAIN and logout URL",
            export_name=f"{Stack.of(self).stack_name}-CognitoDomain",
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
            value=f"{domain_url}/oauth2/token",
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

        # ── CDK Nag suppressions ───────────────────────────────────
        if _HAVE_CDK_NAG:
            NagSuppressions.add_resource_suppressions(
                self.user_pool,
                [
                    {
                        "id": "AwsSolutions-COG2",
                        "reason": (
                            "self_sign_up_enabled=False is intentional — this pool is "
                            "admin-provisioned only. MFA is REQUIRED via OTP."
                        ),
                    }
                ],
            )
            NagSuppressions.add_resource_suppressions(
                self.credentials_secret,
                [
                    {
                        "id": "AwsSolutions-SMG4",
                        "reason": (
                            "This secret mirrors a Cognito M2M client secret. Rotation is "
                            "performed by regenerating the Cognito app client secret and "
                            "updating this secret — Secrets Manager automatic rotation is "
                            "not applicable for Cognito-managed credentials."
                        ),
                    },
                ],
            )
            # CDK provisions an AwsCustomResource singleton Lambda for parts of the
            # user pool configuration; its role and runtime are framework-managed.
            NagSuppressions.add_stack_suppressions(
                self,
                [
                    {
                        "id": "AwsSolutions-IAM4",
                        "reason": (
                            "AwsCustomResource framework uses AWSLambdaBasicExecutionRole — "
                            "cannot be overridden."
                        ),
                        "appliesTo": [
                            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                        ],
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
        """Apply common deployment tags to all resources in this stack."""
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _apply_resource_tags(
        self, resource: Construct, name: str, description: str
    ) -> None:
        """Apply resource-specific name and description tags."""
        Tags.of(resource).add("resource_name", name)
        Tags.of(resource).add("resource_description", description)
