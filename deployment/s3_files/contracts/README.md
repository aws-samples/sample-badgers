# BADGERS integration contracts

These versioned JSON Schema documents define the shared non-UI integration
boundary. They are staged into every configuration release under `contracts/`.

| Contract | Purpose |
| --- | --- |
| `badgers-agent-request-v1.json` | Headless Agent API request |
| `badgers-operation-result-v1.json` | Terminal pointer manifest returned by Headless and future Gateway tools |
| `badgers-operation-record-v1.json` | Durable operation-index record |
| `badgers-operation-plan-v1.json` | Deterministic, private workflow plan |
| `badgers-workflow-definition-v1.json` | Approved declarative workflow definition |
| `badgers-routing-policy-v1.json` | Classifier-led routing policy |
| `badgers-gateway-mcp-v1.json` | Future Gateway workflow-tool catalog |

The result contract contains S3 references and bounded summary metadata only.
Specialist, classification, correlation, and report artifacts remain in their
native formats in the private output bucket.

`badgers-agent-request/v1` requires a private `s3://` document URI, a session
ID, and an instruction. It accepts only `auto` and `manual` routing modes. It
does not accept output prefixes, role ARNs, Lambda ARNs, or arbitrary
configuration URIs.

For `manual` requests, `requested_capabilities` may contain only approved
capability tags, never specialist or Lambda identifiers. A resolved plan must
include every workflow invariant. The authoritative catalog is staged from
`workflows/`, `routing_policies/`, and `workflow_metadata/` into each
agent-runtime or workflow-runner profile's immutable release.
