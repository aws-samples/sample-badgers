"""Private Step Functions Standard runner for declared BADGERS workflows."""

from __future__ import annotations

import json

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    Tags,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
)
from constructs import Construct


class WorkflowRunnerStack(Stack):
    """Create a private, durable runner and its least-privilege roles."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        deployment_id: str,
        deployment_tags: dict[str, str],
        config_bucket: s3.IBucket,
        source_bucket: s3.IBucket,
        output_bucket: s3.IBucket,
        s3_kms_key: kms.IKey,
        jobs_table: dynamodb.ITable,
        foundation_layer: lambda_.ILayerVersion,
        specialist_functions: dict[str, lambda_.IFunction],
        config_prefix: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        if not config_prefix.startswith("releases/"):
            raise ValueError(
                "Workflow runner requires a SHA-pinned BADGERS_CONFIG_RELEASE_ID"
            )

        self.deployment_id = deployment_id
        self._apply_common_tags(deployment_tags)

        self.runner_role = iam.Role(
            self,
            "WorkflowRunnerRole",
            role_name=f"badgers-workflow-runner-{deployment_id}",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Private BADGERS workflow stage executor",
        )
        self.runner_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )
        # The source policy names the tenant layout the runner verifies. It
        # deliberately excludes arbitrary buckets and non-tenant source keys.
        self.runner_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadTenantSources",
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject"],
                resources=[source_bucket.arn_for_objects("tenants/*")],
            )
        )
        self.runner_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadPinnedReleaseConfiguration",
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject"],
                resources=[config_bucket.arn_for_objects("releases/*")],
            )
        )
        self.runner_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadAndWriteOperationArtifacts",
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[output_bucket.arn_for_objects("operations/*")],
            )
        )
        self.runner_role.add_to_policy(
            iam.PolicyStatement(
                sid="UseS3EncryptionKey",
                effect=iam.Effect.ALLOW,
                actions=["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"],
                resources=[s3_kms_key.key_arn],
                conditions={
                    "StringEquals": {
                        "kms:ViaService": f"s3.{self.region}.amazonaws.com"
                    }
                },
            )
        )
        self.runner_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadAndUpdateOperations",
                effect=iam.Effect.ALLOW,
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:Query",
                ],
                resources=[jobs_table.table_arn],
            )
        )
        if specialist_functions:
            self.runner_role.add_to_policy(
                iam.PolicyStatement(
                    sid="InvokePinnedSpecialists",
                    effect=iam.Effect.ALLOW,
                    actions=["lambda:InvokeFunction"],
                    resources=[
                        function.function_arn
                        for _, function in sorted(specialist_functions.items())
                    ],
                )
            )

        self.runner = lambda_.Function(
            self,
            "WorkflowStageHandler",
            function_name=f"badgers_workflow_runner_{deployment_id.replace('-', '_')}",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda_handler.lambda_handler",
            code=lambda_.Code.from_asset("./lambdas/code/workflow_runner"),
            role=self.runner_role,
            layers=[foundation_layer],
            timeout=Duration.minutes(15),
            memory_size=2048,
            reserved_concurrent_executions=2,
            environment={
                "DEPLOYMENT_ID": deployment_id,
                "CONFIG_BUCKET": config_bucket.bucket_name,
                "CONFIG_PREFIX": config_prefix,
                "SOURCE_BUCKET_NAME": source_bucket.bucket_name,
                "OUTPUT_BUCKET_NAME": output_bucket.bucket_name,
                "JOBS_TABLE_NAME": jobs_table.table_name,
                "MAX_SOURCE_BYTES": str(200 * 1024 * 1024),
                "MAX_WORKFLOW_PAGES": "500",
                "SPECIALIST_FUNCTIONS_JSON": json.dumps(
                    {
                        name: function.function_name
                        for name, function in sorted(specialist_functions.items())
                    },
                    sort_keys=True,
                ),
            },
        )

        workflow_role = iam.Role(
            self,
            "WorkflowStateMachineRole",
            role_name=f"badgers-workflow-state-machine-{deployment_id}",
            assumed_by=iam.ServicePrincipal(
                "states.amazonaws.com",
                conditions={
                    "ArnLike": {
                        "aws:SourceArn": (
                            f"arn:aws:states:{self.region}:{self.account}:stateMachine:"
                            f"badgers-workflow-{deployment_id}"
                        )
                    },
                    "StringEquals": {"aws:SourceAccount": self.account},
                },
            ),
            description="Runs only the private BADGERS workflow stage handler",
        )
        self.runner.grant_invoke(workflow_role)

        failure = self._task("FailOperation", "fail")
        cancelled = sfn.Succeed(self, "Cancelled")
        completed = sfn.Succeed(self, "Completed")
        finalize = self._task("Finalize", "finalize")
        finalize.add_catch(failure, result_path="$.failure")
        finalize.next(completed)

        correlate_page = self._task("CorrelatePage", "correlate_page")
        correlate_pages = self._page_map(
            "CorrelatePages",
            correlate_page,
            include_plan=False,
        )
        correlate_pages.add_catch(failure, result_path="$.failure")
        correlate_pages.next(
            self._cancel_choice("CorrelateCancellation", cancelled, finalize)
        )
        begin_correlation = self._task("BeginCorrelation", "begin_correlation")
        begin_correlation.add_catch(failure, result_path="$.failure")
        begin_correlation.next(correlate_pages)

        run_page_specialists = self._task(
            "RunPageSpecialists", "run_page_specialists"
        )
        run_specialists = self._page_map(
            "RunSpecialists",
            run_page_specialists,
            include_plan=True,
        )
        run_specialists.add_catch(failure, result_path="$.failure")
        run_specialists.next(
            self._cancel_choice("SpecialistCancellation", cancelled, begin_correlation)
        )

        plan = self._task("ResolvePlan", "plan")
        plan.add_catch(failure, result_path="$.failure")
        begin_specialist_work = self._task(
            "BeginSpecialistWork", "begin_specialist_work"
        )
        begin_specialist_work.add_catch(failure, result_path="$.failure")
        begin_specialist_work.next(run_specialists)
        plan.next(
            self._cancel_choice("PlanningCancellation", cancelled, begin_specialist_work)
        )

        classify = self._task("Classify", "classify")
        classify.add_catch(failure, result_path="$.failure")
        classify.next(
            self._cancel_choice("ClassificationCancellation", cancelled, plan)
        )

        convert = self._task("Convert", "convert")
        convert.add_catch(failure, result_path="$.failure")
        convert.next(self._cancel_choice("ConversionCancellation", cancelled, classify))

        validate = self._task("ValidateSource", "validate")
        validate.add_catch(failure, result_path="$.failure")
        validate.next(self._cancel_choice("ValidationCancellation", cancelled, convert))

        self.state_machine = sfn.StateMachine(
            self,
            "WorkflowStateMachine",
            state_machine_name=f"badgers-workflow-{deployment_id}",
            definition_body=sfn.DefinitionBody.from_chainable(validate),
            role=workflow_role,
            state_machine_type=sfn.StateMachineType.STANDARD,
            timeout=Duration.hours(6),
            tracing_enabled=True,
        )

        self._apply_resource_tags(
            self.runner, "workflow-stage-handler", "Private BADGERS workflow stage handler"
        )
        self._apply_resource_tags(
            self.state_machine,
            "workflow-state-machine",
            "Durable private BADGERS document workflow runner",
        )
        CfnOutput(
            self,
            "WorkflowStateMachineArn",
            value=self.state_machine.state_machine_arn,
            description="Private BADGERS workflow runner ARN",
            export_name=f"{Stack.of(self).stack_name}-WorkflowStateMachineArn",
        )
        CfnOutput(
            self,
            "WorkflowStageHandlerArn",
            value=self.runner.function_arn,
            description="Private BADGERS workflow stage handler ARN",
            export_name=f"{Stack.of(self).stack_name}-WorkflowStageHandlerArn",
        )

    def _task(self, construct_id: str, action: str) -> sfn_tasks.LambdaInvoke:
        task = sfn_tasks.LambdaInvoke(
            self,
            construct_id,
            lambda_function=self.runner,
            payload=sfn.TaskInput.from_object(
                {
                    "action": action,
                    "state.$": "$",
                    "execution_arn.$": "$$.Execution.Id",
                }
            ),
            payload_response_only=True,
            retry_on_service_exceptions=False,
            result_path="$",
            task_timeout=sfn.Timeout.duration(Duration.minutes(15)),
        )
        # Business errors are not retried: the handler turns them into a
        # failure manifest. Retry only service transport failures.
        task.add_retry(
            errors=[
                "Lambda.ServiceException",
                "Lambda.AWSLambdaException",
                "Lambda.SdkClientException",
            ],
            interval=Duration.seconds(2),
            max_attempts=3,
            backoff_rate=2.0,
        )
        return task

    def _cancel_choice(
        self, construct_id: str, cancelled: sfn.IChainable, next_state: sfn.IChainable
    ) -> sfn.Choice:
        return sfn.Choice(self, construct_id).when(
            sfn.Condition.boolean_equals("$.cancelled", True), cancelled
        ).otherwise(next_state)

    def _page_map(
        self, construct_id: str, processor: sfn.IChainable, *, include_plan: bool
    ) -> sfn.Map:
        selector: dict[str, object] = {
            "operation_id.$": "$.operation_id",
            "owner_sub.$": "$.owner_sub",
            "image.$": "$$.Map.Item.Value",
            "page_index.$": "$$.Map.Item.Index",
            "audit_mode.$": "$.audit_mode",
        }
        if include_plan:
            selector["plan.$"] = "$.plan"
        page_map = sfn.Map(
            self,
            construct_id,
            items_path="$.images",
            item_selector=selector,
            max_concurrency=5,
            result_path=sfn.JsonPath.DISCARD,
        )
        page_map.item_processor(processor)
        return page_map

    def _apply_common_tags(self, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            Tags.of(self).add(key, value)

    @staticmethod
    def _apply_resource_tags(resource: Construct, name: str, description: str) -> None:
        Tags.of(resource).add("resource_name", name)
        Tags.of(resource).add("resource_description", description)
