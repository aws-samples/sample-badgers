# BADGERS release guide

This guide separates a source-code release from a deployment configuration
release. They solve different problems and must never be treated as
interchangeable.

## Current release status

| Channel | Purpose | Customer use |
| --- | --- | --- |
| Stable tags: `v5.0.0`, `v5.0.1`, `v5.0.2` | Immutable snapshots of the established platform | Pin production deployments to `v5.0.2` unless an earlier tag is explicitly required. |
| `main` | Stable source line | Maintainers prepare stable fixes here; customers should still prefer a tag. |
| `dev` | Shared integration line | Contains validated integration work that is not necessarily a customer release. |
| `release/6.0.0-beta` | Private 6.0 release candidate | Not published and not tagged. It is for maintainers only until an explicit beta tag is approved. |

The current 6.0 candidate includes product profiles, the Headless operation
runner, and native PDF/DOCX/PPTX work integrated from `dev`. It is not a
customer release. No `v6.0.0-beta.*` tag exists yet.

## Release terminology

### Source release

A source release is a Git tag such as `v5.0.2` or the future
`v6.0.0-beta.1`. A tag is immutable and identifies the exact source,
documentation, tests, and infrastructure definitions a customer used. Never
move, recreate, or reuse a published tag.

### Configuration release

A configuration release is the content-addressed SHA created by deployment
Step 3. It stages prompts, manifests, schemas, workflow definitions, routing
policies, and shared prompt core under `releases/<sha256>/` in that
deployment's private configuration bucket.

It is deliberately **not** a Git release and is not shared between customer
deployments. The SHA lets a deployed runtime, Gateway, and specialist set pin
the same immutable configuration even as the source repository continues to
change. Headless requires a configuration release because its durable runner
will not start without one.

### Deployment identity

`DEPLOYMENT_ID` and `STACK_SUFFIX` identify one customer deployment in one AWS
account and Region. They are neither a Git version nor a configuration-release
identifier. A deployment ID is profile-bound: create a new deployment rather
than converting Full into Headless, changing a Specialist pack set in place,
or reusing an ID for a different product shape.

## Product status in the 6.0 candidate

| Product | Available in this branch | Deliberately not included |
| --- | --- | --- |
| BADGERS Full | UI, runtime, internal AgentCore Gateway, all enabled specialists, native PDF/DOCX/PPTX intake, session memory, reports, and specialist administration | Nothing intentionally omitted from the Full product. |
| BADGERS Headless | Authenticated BADGERS-owned Agent API, curated PDF/image workflow packs, private Step Functions runner, DynamoDB operation state, private S3 artifacts, and opt-in native DOCX/PPTX templates | Browser UI, WebSocket transport, session memory, and external customer MCP tools. DOCX/PPTX remain absent from the default starter selection. |
| BADGERS Specialists | Direct IAM-authorized Lambda invocation of selected packs with shared prompts, schemas, and model profiles | Agent, Gateway, Cognito, UI, durable operation ledger, and an HTTP/MCP endpoint. |

The existing AgentCore Gateway remains the private tool plane used by BADGERS
runtime components. The separate **BADGERS Gateway** product for
customer-owned agents is not released in this candidate: its intended
workflow-level MCP tools are `badgers_submit_analysis`,
`badgers_get_analysis`, `badgers_cancel_analysis`, and
`badgers_get_workflow_catalog`. Raw individual-specialist MCP tools remain
out of scope.

## Headless contract boundary

The implemented `badgers-agent-request/v1` route accepts private S3 source
URIs. The default configuration supports PDF, PNG, and JPEG documents and
offers `auto` and `manual` routing. Auto routing uses the classifier as the
primary signal; manual routing accepts approved capability tags, never Lambda
or specialist names. A deployment that explicitly selects a native processor
and its matching reviewed template additionally supports DOCX or PPTX through
content-type routing.

The caller authenticates with the dedicated Headless Cognito client. The
runtime derives the operation owner from the validated token and rejects
caller-supplied identity, artifact prefix, role, Lambda, and configuration
URI fields. Results are bounded `badgers-operation-result/v1` pointer
manifests. The underlying source, classification, specialist, correlation,
and report artifacts remain private and tenant-scoped in S3.

Native DOCX/PPTX capability is integrated in Full and can be explicitly
enabled for Headless, but it is not silently advertised by the Headless starter
workflow. Enabling it requires the matching processor and reviewed template in
the deployment's desired state, then a normal CLI update that regenerates and
deploys the private configuration release.

## Maintainer workflow

1. Land feature work into `dev` only after its focused tests and profile-aware
   synthesis pass.
2. Merge the stable line into `dev` whenever a stable fix lands, so the
   integration line includes the current security baseline.
3. Checkpoint the beta work in small commits. Do not merge into a dirty beta
   worktree.
4. Merge `dev` through a temporary integration branch into
   `release/6.0.0-beta`. Resolve product-profile, workflow, deployment, and
   documentation interactions there, then run the combined suite.
5. Fast-forward the beta branch only after the integration branch passes its
   tests and Full, Headless, and Specialists synthesis.
6. Before any beta publication, update this guide, the root README,
   product guides, deployment reference, contracts, stack documentation, and
   changelog as one documentation set.
7. Publish `v6.0.0-beta.1` only after an explicit approval. Push the branch
   first only if external collaboration requires it; tag the exact approved
   commit afterward.
8. Keep `main` on the established release line until beta acceptance,
   migration guidance, deployment preflights, and security review are
   complete. Merge beta into `main` only for the approved 6.0 release.

## Beta release gate

Before creating a `v6.0.0-beta.*` tag, verify all of the following:

- The beta branch has a clean working tree and its intended commit is recorded.
- Full, Headless, and Specialists focused tests pass.
- CDK synthesis passes for all three profiles using real build artifacts in
  the release environment.
- Deployment preflight requirements, model access, profile prerequisites, and
  configuration-release staging are documented and exercised.
- Headless authentication, tenant ownership, operation status, cancellation,
  artifact privacy, and result-pointer behavior are verified.
- The changelog lists included capabilities, known limitations, migration
  impact, and deferred Gateway MCP work.
- The source tag, release notes, and customer support statement have explicit
  approval.

For deployment mechanics, see the
[Deployment Reference](deployment/DEPLOYMENT_README.md). For the source and
configuration contracts, see
[Integration Contracts](deployment/s3_files/contracts/README.md).
