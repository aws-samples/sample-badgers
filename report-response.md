# Response: authenticated cross-user access to BADGERS session and job data

## Assessment

**Status: Confirmed security issue.** This is not expected behavior and is not
an operational-only bug.

The UI correctly requires an authenticated Cognito identity for `/api/*` routes
except the load balancer health check. However, three route groups did not
authorize the requested resource against that identity:

- `GET /api/chat-sessions` enumerated every transcript in the shared task
  filesystem.
- `GET /api/chat-sessions/:sid` returned a transcript based only on its session
  ID.
- `GET /api/jobs/:jobId` and `GET /api/jobs?doc_id=` returned job/subtask data
  without verifying the `owner_sub` on the job-level `orchestrator` record.

Code review also found an adjacent use of the same untrusted session ID:
`POST /api/observability` could query CloudWatch traces for any supplied session
ID. Those trace results can include prompts, tool inputs, and tool outputs.

The issue is an authenticated authorization bypass / insecure direct object
reference. Any ordinary authenticated BADGERS user who knows, guesses, or first
enumerates a session ID could access another user's transcript on the same UI
task. A user who knows a job or document ID could access processing status and
per-page subtask details for another user's job. This can expose document-derived
content, confidential prompts, agent output, and processing metadata.

The report was reproduced against commit `adc285b`, which was the main-branch
head at the time of the reporter's testing. The current remediation is based on
the latest `main` (`609d7c5`).

## Customer impact and compatibility review

This is a **security-correctness change with intentional behavior changes**.
There are currently no confirmed customer installations or customer Production
deployments of this service. Therefore, no customer impact or exposure has been
confirmed. If an installation is identified, normal users continue to use the
same endpoints, request shapes, and response shapes for data they own.

| Area | Customer impact after deployment | Breaking change? |
| --- | --- | --- |
| Chat transcript list/read | Users can list and read only their own new session transcripts. Access to another user's transcript returns `404`. | Intentional security boundary; not a supported customer capability. |
| Job status by `job_id` or `doc_id` | Users can see only jobs whose job-level record has an `owner_sub` matching their Cognito subject. Foreign jobs return `404`; foreign document lookups return an empty job list. | Intentional security boundary; not a supported customer capability. |
| Observability | Users can inspect traces only for a session with a transcript in their caller-specific namespace. Manually pasting another user's session ID no longer works. | Intentional security boundary. |
| Existing flat transcript logs | Pre-remediation transcript files lack an owner field, so they cannot be safely migrated to a user-specific location. They will no longer be served by the chat-session endpoints after deployment. | **Yes, limited historical-data compatibility impact.** Existing chat history will not appear in the UI. New transcripts are unaffected. |

There is no API version or client request change. The one response-level
difference is that a missing or unauthorized transcript now returns an HTTP
`404` rather than a successful response containing `"Session not found"`.
Clients should already handle non-success responses for unavailable resources;
this is required to avoid silently treating a foreign resource as readable.

The implementation does not change document ingestion, BADGERS analysis,
specialist routing, report ownership, DynamoDB schema, or public endpoint
exposure. It only applies resource-level authorization to existing authenticated
UI routes and changes new transcript storage from a shared flat directory to an
opaque caller-specific directory.

Deployment, customer-installation, and exposure status are **not yet
established**. Before any Production deployment or impact conclusion, the
service owner should confirm whether BADGERS is deployed outside development
environments and, if so, identify the affected accounts/environments. For each
identified installation, assess available ECS task/log retention and CloudWatch
access records to determine whether the affected routes were used across
distinct authenticated identities. Legacy transcripts should not be attributed
or re-exposed without independently verifiable ownership evidence.

## Mitigation implemented

The pending code change on branch `fix` implements the following:

1. Stores new chat transcripts under an opaque directory derived from the
   authenticated caller's Cognito subject.
2. Lists and reads transcripts only from that caller-specific directory.
3. Authorizes a single-job read by requiring a matching `owner_sub` on the
   `orchestrator` row, returning the same `404` response for missing and foreign
   job IDs.
4. Finds jobs for a document through the existing `owner-index` first, then
   revalidates ownership from the authoritative job-level row.
5. Requires a caller-owned transcript before querying CloudWatch traces for a
   session.
6. Adds regression coverage with two distinct authenticated identities. The
   tests verify that one identity cannot enumerate/read the other identity's
   transcript, inspect its trace by session ID, or retrieve its job data.

Local validation completed:

```text
cd ui
npm test
npm run build
node --check server/routes/core.js
git diff --check
```

All checks passed. No production deployment has been performed.

## Required review and approval before Production

- [ ] **BADGERS service leadership:** Review the customer-impact assessment,
  especially the intentional loss of UI access to unattributable legacy chat
  transcripts, and approve deployment.
- [ ] **AppSec Affinity Engineer:** Review the authorization model, test
  coverage, and decision to fail closed for legacy flat logs; approve
  deployment.
- [ ] **Service owner / CloudOps:** Confirm the production rollout window and
  perform the post-deployment two-identity verification below.

## Mitigation action items

| Owner | Action | Proposed ECD |
| --- | --- | --- |
| BADGERS service owner | Establish the deployment inventory: confirm whether any customer or Production installations exist, and identify the affected accounts/environments if they do. | 2026-10-21 |
| BADGERS service owner | Review, merge, and deploy the tested `fix` branch after leadership and AppSec approval. | 2026-10-23 |
| BADGERS service owner + CloudOps | For each confirmed installation, assess retained ECS task files and relevant CloudWatch access evidence for potential cross-user use; retain findings in the security ticket. Do not attempt to migrate legacy flat transcripts without authoritative ownership data. | 2026-10-23 |
| AppSec Affinity Engineer | Review the remediation and approve the production security posture. | 2026-10-21 |
| BADGERS service owner | Post-deployment verification: use two separate test identities; verify each identity sees only its own session list/transcript, receives `404` for the other identity's session/job, receives no jobs for the other identity's document ID, and cannot query the other identity's traces. | 2026-10-23 |

The ECDs above are proposed dates and require confirmation by the assigned
owners and leadership.

## Requested ticket disposition

Please classify this as a confirmed security issue. The mitigation is
implemented and tested but is pending the required leadership and AppSec
Affinity Engineer review/approval before Production deployment.
