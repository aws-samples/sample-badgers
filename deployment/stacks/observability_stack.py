"""Profile-aware CloudWatch visibility for one BADGERS deployment."""

from __future__ import annotations

from collections.abc import Mapping

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    Tags,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cloudwatch_actions,
    aws_events as events,
    aws_events_targets as event_targets,
    aws_lambda as lambda_,
    aws_sns as sns,
    aws_stepfunctions as sfn,
)
from constructs import Construct


class ObservabilityStack(Stack):
    """Deploy the selected CloudWatch posture without requiring alert delivery."""

    MAX_METRICS_PER_ALARM = 10

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        deployment_id: str,
        deployment_tags: dict[str, str],
        mode: str,
        notification_kind: str,
        notification_target: str,
        specialist_functions: Mapping[str, lambda_.IFunction],
        workflow_state_machine: sfn.IStateMachine | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        if mode not in {"links", "baseline", "enhanced"}:
            raise ValueError("Observability mode must be links, baseline, or enhanced.")
        if notification_kind not in {"none", "sns", "eventbridge"}:
            raise ValueError("Unknown observability notification kind.")
        if (notification_kind == "none") != (not notification_target):
            raise ValueError("Notification kind and target must be configured together.")

        for key, value in deployment_tags.items():
            Tags.of(self).add(key, value)
        Tags.of(self).add("resource_name", f"badgers-observability-{deployment_id}")
        Tags.of(self).add(
            "resource_description",
            "CloudWatch dashboards and alarms for BADGERS deployment health",
        )

        CfnOutput(
            self,
            "ObservabilityMode",
            value=mode,
            description="Selected BADGERS observability posture",
        )
        if mode == "links":
            return

        functions = list(specialist_functions.values())
        if not functions:
            raise ValueError("Baseline observability requires at least one specialist function.")

        period = Duration.minutes(5)
        errors = cloudwatch.MathExpression(
            expression="SUM(METRICS())",
            using_metrics={
                f"error{index}": function.metric_errors(period=period, statistic="sum")
                for index, function in enumerate(functions)
            },
            period=period,
            label="Specialist Lambda errors",
        )
        throttles = cloudwatch.MathExpression(
            expression="SUM(METRICS())",
            using_metrics={
                f"throttle{index}": function.metric_throttles(period=period, statistic="sum")
                for index, function in enumerate(functions)
            },
            period=period,
            label="Specialist Lambda throttles",
        )
        alarms = []
        for group_index, start in enumerate(range(0, len(functions), self.MAX_METRICS_PER_ALARM), 1):
            group = functions[start : start + self.MAX_METRICS_PER_ALARM]
            suffix = "" if group_index == 1 else str(group_index)
            group_errors = cloudwatch.MathExpression(
                expression="SUM(METRICS())",
                using_metrics={f"error{start + index}": function.metric_errors(period=period, statistic="sum") for index, function in enumerate(group)},
                period=period,
                label="Specialist Lambda errors",
            )
            group_throttles = cloudwatch.MathExpression(
                expression="SUM(METRICS())",
                using_metrics={f"throttle{start + index}": function.metric_throttles(period=period, statistic="sum") for index, function in enumerate(group)},
                period=period,
                label="Specialist Lambda throttles",
            )
            alarms.extend([
            cloudwatch.Alarm(
                self,
                f"SpecialistErrorsAlarm{suffix}",
                alarm_name=f"badgers-{deployment_id}-specialist-errors{('-' + suffix) if suffix else ''}",
                alarm_description=(
                    "Suggested investigation: inspect specialist Lambda logs and the "
                    "correlated BADGERS job/workflow before defining escalation."
                ),
                metric=group_errors,
                threshold=1,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
            cloudwatch.Alarm(
                self,
                f"SpecialistThrottlesAlarm{suffix}",
                alarm_name=f"badgers-{deployment_id}-specialist-throttles{('-' + suffix) if suffix else ''}",
                alarm_description=(
                    "Suggested investigation: inspect Lambda concurrency, service quotas, "
                    "and traffic before defining escalation."
                ),
                metric=group_throttles,
                threshold=1,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
            ])

        dashboard_widgets: list[cloudwatch.IWidget] = [
            cloudwatch.TextWidget(
                markdown=(
                    "# BADGERS operational guidance\n"
                    "This POC creates CloudWatch-visible alarms without requiring an "
                    "alerting endpoint. Add your own ownership, escalation, and recovery "
                    "steps before treating these signals as a production runbook."
                ),
                width=24,
                height=4,
            ),
            cloudwatch.GraphWidget(
                title="Specialist Lambda health",
                left=[errors, throttles],
                width=12,
            ),
        ]

        if workflow_state_machine is not None:
            workflow_dimensions = {
                "StateMachineArn": workflow_state_machine.state_machine_arn,
            }
            failures = cloudwatch.Metric(
                namespace="AWS/States",
                metric_name="ExecutionsFailed",
                dimensions_map=workflow_dimensions,
                statistic="sum",
                period=period,
                label="Workflow failures",
            )
            timeouts = cloudwatch.Metric(
                namespace="AWS/States",
                metric_name="ExecutionsTimedOut",
                dimensions_map=workflow_dimensions,
                statistic="sum",
                period=period,
                label="Workflow timeouts",
            )
            alarms.extend(
                [
                    cloudwatch.Alarm(
                        self,
                        "WorkflowFailuresAlarm",
                        alarm_name=f"badgers-{deployment_id}-workflow-failures",
                        alarm_description=(
                            "Suggested investigation: inspect the Step Functions execution "
                            "history and workflow runner Lambda logs."
                        ),
                        metric=failures,
                        threshold=1,
                        evaluation_periods=1,
                        comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                        treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                    ),
                    cloudwatch.Alarm(
                        self,
                        "WorkflowTimeoutsAlarm",
                        alarm_name=f"badgers-{deployment_id}-workflow-timeouts",
                        alarm_description=(
                            "Suggested investigation: inspect execution duration, Lambda "
                            "timeouts, and document size before defining escalation."
                        ),
                        metric=timeouts,
                        threshold=1,
                        evaluation_periods=1,
                        comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                        treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                    ),
                ]
            )
            dashboard_widgets.append(
                cloudwatch.GraphWidget(
                    title="Workflow health",
                    left=[failures, timeouts],
                    width=12,
                )
            )
            if mode == "enhanced":
                execution_time = cloudwatch.Metric(
                    namespace="AWS/States",
                    metric_name="ExecutionTime",
                    dimensions_map=workflow_dimensions,
                    statistic="p95",
                    period=period,
                    label="Workflow execution time p95",
                )
                alarms.append(
                    cloudwatch.Alarm(
                        self,
                        "WorkflowLatencyAlarm",
                        alarm_name=f"badgers-{deployment_id}-workflow-latency",
                        alarm_description=(
                            "Suggested investigation: inspect Step Functions execution "
                            "history, document size, and specialist duration."
                        ),
                        metric=execution_time,
                        threshold=15 * 60 * 1000,
                        evaluation_periods=1,
                        comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                        treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                    )
                )
                dashboard_widgets.append(
                    cloudwatch.GraphWidget(
                        title="Enhanced processing latency",
                        left=[execution_time],
                        width=12,
                    )
                )

        dashboard_widgets.extend(
            cloudwatch.AlarmWidget(alarm=alarm, width=12) for alarm in alarms
        )
        self.dashboard = cloudwatch.Dashboard(
            self,
            "DeploymentDashboard",
            dashboard_name=f"badgers-{deployment_id}",
            widgets=[[widget] for widget in dashboard_widgets],
        )

        if notification_kind == "sns":
            topic = sns.Topic.from_topic_arn(self, "NotificationTopic", notification_target)
            for alarm in alarms:
                alarm.add_alarm_action(cloudwatch_actions.SnsAction(topic))
        elif notification_kind == "eventbridge":
            event_bus = events.EventBus.from_event_bus_arn(
                self, "NotificationEventBus", notification_target
            )
            events.Rule(
                self,
                "AlarmHandoffRule",
                description="Optionally forwards BADGERS CloudWatch alarm state changes.",
                event_pattern=events.EventPattern(
                    source=["aws.cloudwatch"],
                    detail_type=["CloudWatch Alarm State Change"],
                    detail={"alarmName": [alarm.alarm_name for alarm in alarms]},
                ),
                targets=[event_targets.EventBus(event_bus)],
            )

        CfnOutput(
            self,
            "DashboardName",
            value=self.dashboard.dashboard_name,
            description="BADGERS CloudWatch dashboard name",
        )
