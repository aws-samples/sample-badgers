"""Synth-time coverage for the deployable observability postures."""

from __future__ import annotations

from aws_cdk import App, Stack, assertions, aws_lambda as lambda_, aws_stepfunctions as sfn

from deployment.stacks.observability_stack import ObservabilityStack


def _stack(mode: str, *, notification_kind: str = "none") -> assertions.Template:
    app = App()
    workload = Stack(app, "Workload")
    function = lambda_.Function(
        workload,
        "Specialist",
        runtime=lambda_.Runtime.PYTHON_3_12,
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context): return event"),
    )
    state_machine = sfn.StateMachine(
        workload,
        "Workflow",
        definition_body=sfn.DefinitionBody.from_chainable(sfn.Pass(workload, "Done")),
    )
    notification_target = {
        "none": "",
        "sns": "arn:aws:sns:us-east-1:123456789012:badgers-alerts",
        "eventbridge": "arn:aws:events:us-east-1:123456789012:event-bus/badgers-alerts",
    }[notification_kind]
    observability = ObservabilityStack(
        app,
        "Observability",
        deployment_id="demo-a1b",
        deployment_tags={"environment": "poc", "deployment_id": "demo"},
        mode=mode,
        notification_kind=notification_kind,
        notification_target=notification_target,
        specialist_functions={"specialist": function},
        workflow_state_machine=state_machine,
    )
    return assertions.Template.from_stack(observability)


def test_links_mode_keeps_the_observability_stack_empty() -> None:
    template = _stack("links")

    template.resource_count_is("AWS::CloudWatch::Dashboard", 0)
    template.resource_count_is("AWS::CloudWatch::Alarm", 0)


def test_baseline_mode_deploys_dashboard_and_visible_alarms() -> None:
    template = _stack("baseline")

    template.resource_count_is("AWS::CloudWatch::Dashboard", 1)
    template.resource_count_is("AWS::CloudWatch::Alarm", 4)
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {"AlarmName": "badgers-demo-a1b-specialist-errors"},
    )


def test_enhanced_mode_adds_a_workflow_latency_alarm() -> None:
    template = _stack("enhanced")

    template.resource_count_is("AWS::CloudWatch::Alarm", 5)
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {"AlarmName": "badgers-demo-a1b-workflow-latency"},
    )


def test_sns_handoff_is_optional_and_attached_only_when_selected() -> None:
    template = _stack("baseline", notification_kind="sns")

    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "AlarmActions": [
                "arn:aws:sns:us-east-1:123456789012:badgers-alerts",
            ]
        },
    )


def test_eventbridge_handoff_is_created_only_when_selected() -> None:
    template = _stack("baseline", notification_kind="eventbridge")

    template.resource_count_is("AWS::Events::Rule", 1)
