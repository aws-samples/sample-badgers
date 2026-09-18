"""CloudWatch vended log delivery for AgentCore resources, on one resource policy.

Delivery to CloudWatch Logs needs three L1s plus a grant:

    DeliverySource  (what emits: an AgentCore resource ARN + a log type)
      -> Delivery
    DeliveryDestination  (where it lands: a log group, or X-Ray)
    ResourcePolicy  (lets delivery.logs.amazonaws.com write to the log group)

The mixins in aws_cdk.mixins_preview assemble exactly those, but the policy they
create is a *stack singleton*: every to_log_group() call in a stack appends a
statement to one policy, and a second stack gets a second policy. Log delivery
therefore cost one policy per stack.

That matters because CloudWatch Logs resource policies share a hard,
NON-ADJUSTABLE quota of 10 per region across the whole account (Service Quotas
L-89892494), which several BADGERS deployments plus anything else in the account
compete for. deployment/scripts/common.sh has a preflight for it precisely
because exhausting it fails CloudFormation mid-deploy.

So this module owns the grant instead: one wildcard-scoped policy per deployment,
covering every log group the deployment delivers to, in any stack. One statement
rather than one per group also keeps the document small — the mixin-generated
policies were already 802 characters for two groups, and these documents have a
size cap.

X-Ray delivery needs no policy at all: an XRAY DeliveryDestination carries no
destination ARN and nothing writes to a log group, so deliver_traces_to_xray()
is free of the quota entirely.
"""

from aws_cdk import Stack, aws_logs as logs
from constructs import Construct

# All AgentCore log groups this project creates live under this prefix, in every
# stack. Wildcarding it is what lets one statement replace one-per-log-group.
AGENTCORE_LOG_GROUP_PREFIX = "/aws/bedrock-agentcore"


def _name(deployment_id: str, *parts: str) -> str:
    """Build a delivery source/destination name.

    Deliberately not the mixin's "cdk-<type>-source-<hash>" form. These names are
    account-unique, so a stack migrating from the mixin to these L1s would fail on
    a name collision if both spellings matched: CloudFormation creates the
    replacement before deleting the original.
    """
    return "-".join(["badgers", deployment_id, *parts])[:60]


def shared_delivery_policy(
    scope: Construct,
    construct_id: str,
    deployment_id: str,
) -> logs.CfnResourcePolicy:
    """Create the deployment's single log-delivery resource policy.

    Create this in exactly one stack per deployment. Any stack whose deliveries
    target a log group under AGENTCORE_LOG_GROUP_PREFIX is covered by it, but must
    be deployed *after* the owning stack — a Delivery whose destination log group
    has no grant yet fails to create. Every consumer already depends on the
    Gateway stack, which is why the policy lives there.

    Resource policies are account-and-region scoped and attach to nothing, so the
    name has to carry the deployment id or two deployments would overwrite each
    other's policy.
    """
    stack = Stack.of(scope)
    account = stack.account
    region = stack.region
    partition = stack.partition

    return logs.CfnResourcePolicy(
        scope,
        construct_id,
        policy_name=f"BadgersLogDelivery-{deployment_id}",
        policy_document=(
            '{"Version":"2012-10-17","Statement":[{'
            '"Sid":"BadgersAgentCoreLogDelivery",'
            '"Effect":"Allow",'
            '"Principal":{"Service":"delivery.logs.amazonaws.com"},'
            '"Action":["logs:CreateLogStream","logs:PutLogEvents"],'
            f'"Resource":"arn:{partition}:logs:{region}:{account}:log-group:'
            f'{AGENTCORE_LOG_GROUP_PREFIX}/*:*:log-stream:*",'
            '"Condition":{'
            f'"StringEquals":{{"aws:SourceAccount":"{account}"}},'
            f'"ArnLike":{{"aws:SourceArn":"arn:{partition}:logs:{region}:{account}:*"}}'
            "}}]}"
        ),
    )


def deliver_to_log_group(
    scope: Construct,
    construct_id: str,
    *,
    deployment_id: str,
    source_resource_arn: str,
    log_type: str,
    log_group: logs.ILogGroup,
) -> logs.CfnDelivery:
    """Deliver one log type from an AgentCore resource to a log group.

    The caller is responsible for the grant (shared_delivery_policy) and for
    ordering: this creates no policy, by design.
    """
    source = logs.CfnDeliverySource(
        scope,
        f"{construct_id}Source",
        name=_name(deployment_id, construct_id.lower(), "src"),
        log_type=log_type,
        resource_arn=source_resource_arn,
    )

    destination = logs.CfnDeliveryDestination(
        scope,
        f"{construct_id}Dest",
        name=_name(deployment_id, construct_id.lower(), "dst"),
        delivery_destination_type="CWL",
        destination_resource_arn=log_group.log_group_arn,
    )

    delivery = logs.CfnDelivery(
        scope,
        f"{construct_id}Delivery",
        delivery_source_name=source.name,
        delivery_destination_arn=destination.attr_arn,
    )
    # CfnDelivery references the source by name, not by Ref, so nothing tells
    # CloudFormation the source must exist first.
    delivery.node.add_dependency(source)
    delivery.node.add_dependency(destination)
    return delivery


def deliver_traces_to_xray(
    scope: Construct,
    construct_id: str,
    *,
    deployment_id: str,
    source_resource_arn: str,
) -> logs.CfnDelivery:
    """Deliver TRACES from an AgentCore resource to X-Ray.

    Needs no resource policy: the destination is X-Ray rather than a log group.
    Requires X-Ray Transaction Search to be enabled account-wide, which
    XRayTransactionSearchStack does.
    """
    source = logs.CfnDeliverySource(
        scope,
        f"{construct_id}Source",
        name=_name(deployment_id, construct_id.lower(), "src"),
        log_type="TRACES",
        resource_arn=source_resource_arn,
    )

    destination = logs.CfnDeliveryDestination(
        scope,
        f"{construct_id}Dest",
        name=_name(deployment_id, construct_id.lower(), "dst"),
        delivery_destination_type="XRAY",
    )

    delivery = logs.CfnDelivery(
        scope,
        f"{construct_id}Delivery",
        delivery_source_name=source.name,
        delivery_destination_arn=destination.attr_arn,
    )
    delivery.node.add_dependency(source)
    delivery.node.add_dependency(destination)
    return delivery
