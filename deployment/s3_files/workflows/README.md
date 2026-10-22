[← BADGERS home](../../../README.md)

# BADGERS workflow definitions

Workflow files are declarative, release-pinned configuration. A workflow
chooses invariant steps, allowed capability tags, supported routing modes, and
one routing policy. It must not contain a Lambda ARN, role ARN, URL, script, or
customer-supplied executable code.

In the private 6.0 candidate, `standard-document-analysis` is the implemented
Headless workflow. It accepts PDF, PNG, and JPEG sources, supports `auto` and
`manual` routing, and returns the bounded operation-result pointer contract.
It is staged inside a deployment-private configuration release; it is not a
published Git source release. See the repository
[Release Guide](../../../RELEASE_GUIDE.md).

Each file name must equal its `workflow_id`, for example
`standard-document-analysis.json`. The staging process validates every
workflow against the selected product profile before it creates the immutable
release. A capability must be both allowed by the workflow and supplied by the
selected specialist pack.

`invariant_capabilities` maps execution invariants to capability tags. Manual
capability selection is permitted only where the workflow explicitly enables
it, and the selection must include all invariant capabilities. Auto plans use
classifier facts and the referenced routing policy to add approved work.
