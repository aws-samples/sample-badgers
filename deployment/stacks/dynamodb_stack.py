"""DynamoDB Stack for BADGERS job tracking.

Derived from the media-contracts (MC) reference implementation, extended from
MC's two-level (job, specialist) model to the three-level hierarchy BADGERS
needs:

    doc_id  ->  job_id  ->  subtask_id

  doc_id      one uploaded document. Minted as a UUID at upload time, so
              re-uploading the same file yields a new document.
  job_id      one analysis run over that document. Minted lazily by the server
              on the first specialist tool call of a turn, so a conversational
              turn that invokes no specialist creates no job record at all.
  subtask_id  one specialist invocation: one specialist against one page/image.

MC keys rows on (job_id, specialist), which assumes each specialist runs once
per job. That does not hold here: foundation/s3_result_saver.py writes results
to {session_id}/{specialist_name}/{specialist}_{image_identifier}_{timestamp},
i.e. BADGERS invokes the same specialist repeatedly within one run, once per
page or image. Keying on the specialist name alone would collapse every page
into a single row and retain only the last. Hence the subtask level.

Table: badgers-jobs-{deployment_id}
  PK  job_id      (String)
  SK  subtask_id  (String) — the reserved value 'orchestrator' for the
                             job-level row; otherwise
                             '{specialist}#{image_identifier}'

The subtask sort key is deterministic rather than a UUID, which yields both
uniqueness across the page fan-out and idempotency: a retry of the same
specialist against the same page upserts the same row and increments
retry_count instead of creating a duplicate.

Attributes (written by the orchestrator runtime and each specialist Lambda):
  doc_id           : owning document id
  specialist       : specialist name (also embedded in subtask_id)
  image_identifier : page/image this subtask analysed
  status           : PENDING | RUNNING | COMPLETE | FAILED
  result_s3_key    : S3 key where the specialist output was written
  started_at       : ISO-8601 timestamp
  completed_at     : ISO-8601 timestamp
  error            : error message (FAILED only)
  retry_count      : int
  session_id       : originating chat session
  reason           : why a new run was opened (job-level row only)
  ttl              : epoch seconds — records auto-expire after 30 days

GSI: status-index
  PK  status      — all subtasks in a given state (ops/monitoring, UI filters)
  SK  started_at

GSI: doc-index
  PK  doc_id      — every job and subtask belonging to one document
  SK  started_at

Query patterns: all subtasks of a job (Query on the table), all runs of a
document (doc-index), everything currently RUNNING or FAILED (status-index).
"""

from aws_cdk import (
    RemovalPolicy,
    Stack,
    Tags,
    CfnOutput,
    aws_dynamodb as dynamodb,
    aws_ssm as ssm,
)
from constructs import Construct


class DynamoDBStack(Stack):
    """DynamoDB table for specialist job state and checkpointing."""

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

        # ── Jobs table ─────────────────────────────────────────────
        self.jobs_table = dynamodb.Table(
            self,
            "SpecialistJobs",
            table_name=f"badgers-jobs-{deployment_id}",
            partition_key=dynamodb.Attribute(
                name="job_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="subtask_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            time_to_live_attribute="ttl",
        )

        # GSI for ops monitoring and UI status filters
        self.jobs_table.add_global_secondary_index(
            index_name="status-index",
            partition_key=dynamodb.Attribute(
                name="status",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="started_at",
                type=dynamodb.AttributeType.STRING,
            ),
            projection_type=dynamodb.ProjectionType.INCLUDE,
            non_key_attributes=[
                "job_id",
                "subtask_id",
                "doc_id",
                "specialist",
                "image_identifier",
                "result_s3_key",
                "error",
                "completed_at",
            ],
        )

        # GSI to list every job and subtask belonging to one document
        self.jobs_table.add_global_secondary_index(
            index_name="doc-index",
            partition_key=dynamodb.Attribute(
                name="doc_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="started_at",
                type=dynamodb.AttributeType.STRING,
            ),
            projection_type=dynamodb.ProjectionType.INCLUDE,
            non_key_attributes=[
                "job_id",
                "subtask_id",
                "specialist",
                "image_identifier",
                "status",
                "result_s3_key",
                "error",
                "completed_at",
            ],
        )

        # ── SSM parameter for runtime discovery ────────────────────
        ssm.StringParameter(
            self,
            "JobsTableParam",
            parameter_name=f"/badgers-{deployment_id}/jobs-table-name",
            string_value=self.jobs_table.table_name,
            description="DynamoDB jobs table name",
        )

        self._apply_resource_tags(
            self.jobs_table,
            "specialist-jobs-table",
            "DynamoDB table for BADGERS specialist job tracking",
        )

        # ── Outputs ────────────────────────────────────────────────
        CfnOutput(
            self,
            "JobsTableName",
            value=self.jobs_table.table_name,
            description="DynamoDB jobs table name",
            export_name=f"{Stack.of(self).stack_name}-JobsTableName",
        )
        CfnOutput(
            self,
            "JobsTableArn",
            value=self.jobs_table.table_arn,
            description="DynamoDB jobs table ARN",
            export_name=f"{Stack.of(self).stack_name}-JobsTableArn",
        )

    def _apply_common_tags(self) -> None:
        for key, value in self.deployment_tags.items():
            Tags.of(self).add(key, value)

    def _apply_resource_tags(
        self, resource: Construct, name: str, description: str
    ) -> None:
        Tags.of(resource).add("resource_name", name)
        Tags.of(resource).add("resource_description", description)
