"""Agent Observability page - View traces by Session ID from CloudWatch aws/spans."""

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import gradio as gr
from dotenv import load_dotenv

env_file = Path(__file__).parent.parent / "config" / ".env"
if env_file.exists():
    load_dotenv(env_file)

LOG_GROUP = "aws/spans"


def get_logs_client():
    """Get CloudWatch Logs client."""
    region = os.getenv("AWS_REGION", "us-west-2")
    return boto3.client("logs", region_name=region)


def run_query(
    logs_client,
    query: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 10000,
) -> list:
    """Run a CloudWatch Logs Insights query and wait for results."""
    try:
        resp = logs_client.start_query(
            logGroupName=LOG_GROUP,
            startTime=int(start_time.timestamp()),
            endTime=int(end_time.timestamp()),
            queryString=query,
            limit=limit,
        )
        query_id = resp["queryId"]

        for _ in range(60):
            time.sleep(1)
            result = logs_client.get_query_results(queryId=query_id)
            status = result["status"]
            if status in ("Complete", "Failed", "Cancelled", "Timeout"):
                break

        return result.get("results", [])
    except Exception as e:
        return [{"error": str(e)}]


def fetch_session_traces(session_id: str) -> str:
    """Fetch and format all traces for a session from aws/spans."""
    if not session_id or not session_id.strip():
        return "Please enter a Session ID (e.g., ws-session-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"

    session_id = session_id.strip()
    logs = get_logs_client()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)

    output = []
    output.append("=" * 70)
    output.append(f"SESSION: {session_id}")
    output.append("=" * 70)
    output.append("")

    # Step 1: Find all trace IDs for this session
    results = run_query(
        logs,
        f"fields traceId, name, @timestamp "
        f"| filter attributes.`aws.bedrock.agentcore.session_id` = '{session_id}' "
        f"   or attributes.`session.id` = '{session_id}' "
        f"   or attributes.`rpc.request.metadata.x-amzn-bedrock-agentcore-runtime-session-id` = '{session_id}' "
        f"| stats count() as cnt, min(@timestamp) as first_seen, max(@timestamp) as last_seen by traceId "
        f"| sort first_seen asc",
        start_time,
        end_time,
    )

    if not results or "error" in (results[0] if results else {}):
        # Try fulltext search as fallback
        results = run_query(
            logs,
            f"fields traceId, name, @message "
            f"| filter @message like '{session_id}' "
            f"| stats count() as cnt by traceId "
            f"| sort cnt desc",
            start_time,
            end_time,
        )

    trace_ids = []
    if results and "error" not in results[0]:
        for r in results:
            row = {f["field"]: f["value"] for f in r}
            tid = row.get("traceId", "")
            if tid:
                trace_ids.append(tid)
                output.append(f"Trace: {tid}  spans={row.get('cnt', '?')}")

    if not trace_ids:
        return f"No traces found for session {session_id}\n\nMake sure:\n1. The 'aws/spans' log group exists\n2. The session is from the last 24 hours"

    output.append(f"\nFound {len(trace_ids)} trace(s)")
    output.append("")

    # Step 2: Fetch all spans for all traces
    all_spans = []
    for tid in trace_ids:
        results = run_query(
            logs,
            f"fields @message | filter traceId = '{tid}' | sort @timestamp asc | limit 10000",
            start_time,
            end_time,
        )
        for r in results:
            for f in r:
                if f["field"] == "@message":
                    try:
                        doc = json.loads(f["value"])
                        doc["_traceId"] = tid
                        all_spans.append(doc)
                    except json.JSONDecodeError:
                        pass

    output.append(f"Total Spans: {len(all_spans)}")

    # Step 3: Extract events and stats
    all_events = []
    span_names_with_events: dict[str, int] = {}
    span_names_without_events: dict[str, int] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    models_used: set[str] = set()
    tools_used: list[str] = []

    for doc in all_spans:
        name = doc.get("name", "?")
        span_id = doc.get("spanId", "")
        trace_id = doc.get("_traceId", "")
        events = doc.get("events", [])
        attrs = doc.get("attributes", {})

        # Collect token usage
        if "gen_ai.usage.input_tokens" in attrs:
            total_input_tokens = max(
                total_input_tokens, int(attrs["gen_ai.usage.input_tokens"])
            )
        if "gen_ai.usage.output_tokens" in attrs:
            total_output_tokens = max(
                total_output_tokens, int(attrs["gen_ai.usage.output_tokens"])
            )
        if "gen_ai.request.model" in attrs:
            models_used.add(attrs["gen_ai.request.model"])

        if events:
            span_names_with_events[name] = span_names_with_events.get(name, 0) + len(
                events
            )
        else:
            span_names_without_events[name] = span_names_without_events.get(name, 0) + 1

        for evt in events:
            evt_name = evt.get("name", "unnamed")
            evt_time_ns = evt.get("timeUnixNano", "0")
            evt_attrs = evt.get("attributes", {})

            ts = ""
            try:
                ts = datetime.fromtimestamp(int(evt_time_ns) / 1e9).strftime(
                    "%H:%M:%S.%f"
                )[:-3]
            except (ValueError, OSError):
                pass

            # Extract tool names
            content = evt_attrs.get("content", evt_attrs.get("message", ""))
            if isinstance(content, str) and "toolUse" in content:
                tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                tools_used.extend(tool_matches)

            all_events.append(
                {
                    "trace_id": trace_id,
                    "span_name": name,
                    "span_id": span_id,
                    "event_name": evt_name,
                    "event_time": ts,
                    "event_time_ns": evt_time_ns,
                    "event_attrs": evt_attrs,
                }
            )

    all_events.sort(key=lambda e: int(e.get("event_time_ns", 0)))

    # Summary
    output.append(f"Total Events: {len(all_events)}")
    output.append("")

    output.append("── Token Usage ──")
    output.append(f"  Input:  {total_input_tokens:,}")
    output.append(f"  Output: {total_output_tokens:,}")
    output.append(f"  Total:  {total_input_tokens + total_output_tokens:,}")
    output.append("")

    if models_used:
        output.append("── Models Used ──")
        for model in sorted(models_used):
            output.append(f"  • {model}")
        output.append("")

    if tools_used:
        output.append("── Tools Called ──")
        for tool, cnt in Counter(tools_used).most_common():
            output.append(f"  {tool}: {cnt}x")
        output.append("")

    output.append("── Event Types ──")
    for name, cnt in Counter(e["event_name"] for e in all_events).most_common():
        output.append(f"  {cnt:>4}  {name}")
    output.append("")

    output.append("── Span Types (with events) ──")
    for name, cnt in sorted(span_names_with_events.items(), key=lambda x: -x[1])[:15]:
        output.append(f"  {cnt:>4}  {name}")
    output.append("")

    # Event timeline
    output.append("=" * 70)
    output.append(f"EVENT TIMELINE ({len(all_events)} events)")
    output.append("=" * 70)

    for i, evt in enumerate(all_events[:150]):
        evt_name = evt["event_name"]
        span_name = evt["span_name"]
        ts = evt["event_time"]
        attrs = evt["event_attrs"]

        # Format attributes concisely
        summary = ""
        content = attrs.get("content", attrs.get("message", attrs.get("body", "")))

        if isinstance(content, str):
            if "toolUse" in content:
                tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                summary = f"tools=[{', '.join(tool_matches)}]"
            elif "toolResult" in content:
                summary = "[tool result]"
            elif len(content) > 120:
                summary = content[:120] + "..."
            elif content:
                summary = content

        if not summary and attrs:
            summary = json.dumps(attrs, default=str)[:120]

        output.append(f"\n[{i+1:3d}] {ts}  {evt_name}")
        output.append(f"      span: {span_name}")
        if summary:
            output.append(f"      {summary}")

    if len(all_events) > 150:
        output.append(f"\n... and {len(all_events) - 150} more events")

    return "\n".join(output)


with gr.Blocks() as demo:
    gr.Markdown("# 📊 Agent Observability")
    gr.Markdown(
        "View all traces and spans for a session from CloudWatch `aws/spans` (last 24 hours)"
    )

    with gr.Row():
        session_input = gr.Textbox(
            label="Session ID",
            placeholder="ws-session-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            scale=4,
        )
        refresh_btn = gr.Button("🔍 Fetch Session", variant="primary", scale=1)

    trace_output = gr.Textbox(
        label="Trace Output",
        lines=45,
        max_lines=70,
        interactive=False,
    )

    refresh_btn.click(
        fn=fetch_session_traces,
        inputs=[session_input],
        outputs=[trace_output],
    )

    session_input.submit(
        fn=fetch_session_traces,
        inputs=[session_input],
        outputs=[trace_output],
    )
