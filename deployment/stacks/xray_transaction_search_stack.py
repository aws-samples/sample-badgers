"""X-Ray Transaction Search Stack for BADGERS.

Enables CloudWatch Transaction Search at the account level (per-region),
which is a prerequisite for X-Ray tracing with AgentCore Gateway and Runtime.

This is a singleton resource per account/region — deploying when already
enabled will fail with AlreadyExists. If that happens, destroy this stack
or remove it from the deploy sequence.
"""

from aws_cdk import (
    Stack,
    Tags,
    aws_logs as logs,
    aws_xray as xray,
)
from constructs import Construct


class XRayTransactionSearchStack(Stack):
    """Account-level X-Ray Transaction Search enablement."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        indexing_percentage: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        for key, value in deployment_tags.items():
            Tags.of(self).add(key, value)

        account = Stack.of(self).account
        region = Stack.of(self).region
        partition = Stack.of(self).partition

        # Resource policy: allow X-Ray to write to CloudWatch Logs
        resource_policy = logs.CfnResourcePolicy(
            self,
            "XRayLogResourcePolicy",
            policy_name="TransactionSearchAccess",
            policy_document=(
                '{"Version":"2012-10-17","Statement":[{"Sid":"TransactionSearchXRayAccess",'
                '"Effect":"Allow","Principal":{"Service":"xray.amazonaws.com"},'
                '"Action":"logs:PutLogEvents","Resource":['
                f'"arn:{partition}:logs:{region}:{account}:log-group:aws/spans:*",'
                f'"arn:{partition}:logs:{region}:{account}:log-group:/aws/application-signals/data:*"'
                '],"Condition":{"ArnLike":{'
                f'"aws:SourceArn":"arn:{partition}:xray:{region}:{account}:*"'
                '},"StringEquals":{"aws:SourceAccount":"' + account + '"}}}]}'
            ),
        )

        # Enable Transaction Search (account-level singleton)
        transaction_search = xray.CfnTransactionSearchConfig(
            self,
            "XRayTransactionSearchConfig",
            indexing_percentage=indexing_percentage,
        )
        transaction_search.node.add_dependency(resource_policy)
