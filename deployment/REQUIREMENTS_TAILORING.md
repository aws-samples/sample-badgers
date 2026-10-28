<sub>[← BADGERS home](../README.md) · [Deployment reference](DEPLOYMENT_README.md) · [Profiles and packs](profiles/README.md) · [Release guide](../RELEASE_GUIDE.md)</sub>

---

# Requirements tailoring guide

BADGERS is a reference application with reusable components, not a promise that
every source format, output contract, throughput target, or infrastructure
standard is already solved. Start with a concrete requirement, select the
supported deployment path, and add an extension only where there is a measured
gap.

This guide records the current boundaries so an evaluation can distinguish a
supported feature from a proposed integration.

---

## What is supported today

| Requirement | Use the existing product this way | Boundary to understand |
| --- | --- | --- |
| Page-oriented PDF or image analysis | Use the standard Headless workflow or Full agent path. The Headless runner enforces its declared sequence in Step Functions. | It is a visual page workflow: conversion, classification, selected specialists, and correlation each have explicit stages. |
| Native PDF inspection | Use the Full agent’s `native_pdf_processor` first. It reads the digital text layer and creates per-page native/ocr/hybrid/vision routing evidence. | The Headless standard workflow remains the visual page workflow. A complete native-first selective-enrichment state-machine workflow is a future workflow design, not an advertised template. |
| Native DOCX or PPTX intake | Select the matching container specialist and approved native template in the CLI or local planner. The deployment graph automatically includes ECR image build/push. | The native result is a private, versioned processor artifact. Requirements for visual follow-on must explicitly define the desired routing and result contract. |
| Macro-free XLSX workbook intake | Select the `workbook-analysis` pack in any product. In Headless, also enable `native-xlsx-analysis` for durable workflow execution. The processor reads bounded OOXML cells, formulas, tables, and object anchors without executing workbook content, conditionally adds pasted-raster-image Vision evidence, and returns canonical BADGERS XML. | Formulas, macros/scripts, connection refresh, and external links are never executed. Native cells and formulas remain authoritative. Native chart semantic extraction and consumer-specific output adapters remain separate extensions. |
| Direct specialist use | Deploy the Specialists profile and invoke selected Lambda functions directly. | The caller owns orchestration, request validation, retry policy, and result adaptation. |
| Full UI and conversational routing | Use the Full profile. | Agent-directed routing is prompt-governed; use Headless when a hard, declared workflow order is required. |
| Reusable packs | Compose packs and direct specialists using the terminal selector or local planner, then run the CLI reconciliation. | Desired state is local until the CLI publishes the immutable configuration release and updates the dependent stacks. |

---

## Requirements that need an explicit extension

| Requirement | Current status | Recommended next decision |
| --- | --- | --- |
| A consumer-specific `ExtractedDocument` object | BADGERS writes versioned operation manifests that point to private specialist artifacts; canonical page artifacts are XML. | Define the consumer’s exact object/schema, chunking rules, provenance needs, and error behavior. Build the adapter at that integration boundary rather than hard-code one consumer’s type into BADGERS. |
| Pulumi-managed infrastructure | This repository deploys with CDK; no Pulumi module is shipped. | Keep the BADGERS profile, pack, routing, and artifact contracts as the portable boundary. Build and own a Pulumi implementation only after deciding which AWS resources and lifecycle it must control. |
| Low-latency, high-throughput processing | The durable Headless workflow favors isolated artifacts and bounded concurrency. Its runner has reserved concurrency 2 and its page maps have maximum concurrency 5; specialists do not each reserve concurrency 5. | Benchmark the actual corpus, Region, models, page counts, and result-quality target. Then choose a latency budget, concurrency limits, batch strategy, and retry/error policy. |
| Native-first PDF with selective visual enrichment | Native PDF intake produces deterministic route evidence for every page. | Define the failure/quality policy for `textract_ocr`, `hybrid`, and `vision_ai` pages, then implement and benchmark the declared state-machine continuation. |

---

## Requirement discovery checklist

Before creating a new pack, processor, workflow, or infrastructure module,
capture these decisions:

1. **Inputs:** formats, maximum size/page count, encrypted or macro-enabled
   document policy, and whether native semantics must be retained.
2. **Outputs:** consumer schema, text ordering, coordinates, tables/charts,
   provenance, partial-success rules, and retention/access expectations.
3. **Processing policy:** which document conditions allow native extraction,
   OCR, visual models, human review, or failure.
4. **Performance:** target percentile latency, throughput, concurrency,
   acceptable cost, corpus composition, and benchmark acceptance criteria.
5. **Operations:** deployment/IaC owner, identity model, data residency,
   tenancy, observability, rollback, and support ownership.

Do not solve these by modifying an agent prompt alone. Put deterministic
requirements in a versioned workflow, routing policy, processor contract, or
consumer adapter, and preserve the decision in the immutable configuration
release.

---

## Evaluation notes

The October 2026 Trevi review correctly identified the prior visual PDF
workflow, third-party Pillow layer, and package metadata as risks. The
repository now has a repository-built foundation layer containing Pillow;
standard and custom specialists no longer import KLayers. Deployment refuses a
stale foundation archive, so the layer must be rebuilt after dependency or
foundation-code changes.

The review’s broad claims that BADGERS has no Office support, has only
region-inspection tests, or reserves concurrency 5 per specialist are no
longer accurate. DOCX/PPTX/XLSX native paths and their deployment dependencies
are available, test coverage includes workflow, native PDF, DOCX, PPTX, XLSX,
routing, and deployment-state behavior, and the durable workflow’s concurrency
is bounded as described above.

The remaining gaps in the preceding table are deliberate product boundaries,
not hidden fallbacks. They should be evaluated against real requirements and a
representative benchmark before they become shipped capabilities.
