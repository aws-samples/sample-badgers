# BADGERS Schema Mapping Reference

This document maps between the three vocabularies used in BADGERS and the external academic standards that consumers are most likely to target.

## Internal Vocabularies

BADGERS uses two complementary vocabularies at different stages of the pipeline:

| Vocabulary | Schema File | Stage | Purpose |
|---|---|---|---|
| **Elements** | [`badgers-elements.xsd`](badgers-elements.xsd) | Analysis (per-specialist) | Semantic content classification — "what is this?" |
| **Content Tree** (Spine) | [`badgers-content-tree.xsd`](badgers-content-tree.xsd) | Correlation (unified output) | PDF/UA structure tagging — "how should this be tagged?" |

The **elements vocabulary** is what individual specialists produce. The **content tree vocabulary** is what the correlation specialist produces after merging all specialist outputs into a single hierarchical document spine. The content tree uses PDF/UA (ISO 14289-1) tag names directly.

---

## Elements ↔ Content Tree Mapping

| Elements Specialist Type | Elements Sub-types | Content Tree (PDF/UA) Tag(s) | Notes |
|---|---|---|---|
| `document_title` | — | `H1` | Mapped to top-level heading |
| `section` | Chapter, Main section, Subsection, Sub-subsection | `Sect` | Sections become structural grouping containers |
| `heading` | Title (H1), Section heading (H2), Subsection heading (H3), Sub-subsection heading (H4), Lower-level heading (H5, H6), Running head | `H1`, `H2`, `H3`, `H4`, `H5`, `H6` | Direct mapping by level. Running head → `Artifact` |
| `paragraph` | Body, Introduction, Conclusion | `P` | All paragraph sub-types map to `P` |
| `citation` | In-text, Footnote, Endnote | `Reference` (inline marker) + `Note` (body) | In-text citations become `Reference` elements; footnote/endnote text becomes `Note` |
| `reference` | Book, Journal article, Web page, Conference paper | `BibEntry` | All reference sub-types map to `BibEntry` |
| `caption` | Figure caption, Table caption, Image caption | `Caption` | Direct mapping; `parent_ref` links to parent `Figure` or `Table` |
| `quote` | Block quote, Inline quote, Pull quote | `Quote` or `BlockQuote` (block) / `Span` (inline) | Block quotes get their own `Sect` with `tag="BlockQuote"` |
| `equation` | Inline equation, Display equation | `Formula` | `placement="Inline"` or `placement="Block"` |
| `list` | Bulleted, Numbered, Definition | `L` → `LI` → `Lbl` + `LBody` | Full substructure in content tree |
| `footnote` | — | `Note` | Paired with `Reference` via `ref` attribute |
| `table_of_contents` | — | `TOC` → `TOCI` | Table of contents with item sub-elements |
| `abstract` | — | `P` (within a `Sect`) | No dedicated PDF/UA tag; rendered as paragraph in a labeled section |
| `keyword` | — | *(enrichment only)* | Captured in `<enrichments><keyword_topic>` — no PDF/UA structure tag |
| *(no equivalent)* | — | `Table`, `THead`, `TBody`, `TR`, `TH`, `TD` | Table substructure only exists in the content tree |
| *(no equivalent)* | — | `Link` | Inline hyperlinks — only in content tree |
| *(no equivalent)* | — | `Code` | Code blocks — only in content tree |
| *(no equivalent)* | — | `Artifact` | Non-content elements (page numbers, headers, footers, watermarks) |
| *(no equivalent)* | — | `Span` | Inline formatting — only in content tree |

---

## Content Tree → External Standards Mapping

### JATS XML (Journal Article Tag Suite)

For scholarly publishing workflows. JATS is the standard for NLM/PubMed/PMC journal articles.

| Content Tree Tag | JATS Element | Notes |
|---|---|---|
| `H1` | `<article-title>` | Document title |
| `H2` | `<sec><title>` | Section heading (section wraps in `<sec>`) |
| `H3`–`H6` | `<sec><title>` (nested) | Nested section levels |
| `P` | `<p>` | Direct mapping |
| `Figure` | `<fig>` | Alt text → `<alt-text>` |
| `Caption` | `<caption><p>` | Inside parent `<fig>` or `<table-wrap>` |
| `Table` | `<table-wrap><table>` | THead/TBody/TR/TH/TD map to HTML table model |
| `L` | `<list>` | `list-type="bullet"` or `list-type="order"` |
| `LI` | `<list-item>` | Direct mapping |
| `Formula` | `<disp-formula>` / `<inline-formula>` | By `placement` attribute |
| `BibEntry` | `<ref><mixed-citation>` | Inside `<ref-list>` |
| `Reference` | `<xref ref-type="bibr">` | Inline citation marker |
| `Note` | `<fn>` | Inside `<fn-group>` |
| `Link` | `<ext-link>` | `ext-link-type="uri"` |
| `Code` | `<code>` | JATS v1.2+ |
| `Quote`/`BlockQuote` | `<disp-quote>` | Direct mapping |
| `Artifact` | *(omit)* | Not applicable in JATS |
| `TOC` | *(omit)* | Not applicable in JATS |

### ALTO XML (Analyzed Layout and Text Object)

For digital library layout-aware OCR workflows. ALTO preserves spatial coordinates.

| Content Tree Tag | ALTO Element | Notes |
|---|---|---|
| `P` | `<TextBlock>` | Each paragraph → a TextBlock |
| `H1`–`H6` | `<TextBlock STYLEREFS="heading">` | Headings are TextBlocks with style refs |
| `Figure` | `<Illustration>` | Alt text → `<Description>` (ALTO v4) |
| `Table` | `<ComposedBlock TYPE="table">` | Table → ComposedBlock containing TextBlocks |
| `L` | `<ComposedBlock TYPE="list">` | List → ComposedBlock |
| `Artifact` | *(omit or mark as* `<GraphicalElement>` *)* | Page decorations |
| Grid coordinates | `HPOS`, `VPOS`, `WIDTH`, `HEIGHT` | BADGERS grid → pixel coordinates via page dimensions |

> **Note:** ALTO is layout-centric, not semantic. The content tree's semantic tags (Formula, BibEntry, Note) don't have ALTO equivalents — they would be preserved as metadata annotations or lost in a pure ALTO export.

### Dublin Core / Schema.org

For general metadata interoperability. See also [`badgers-elements.jsonld`](badgers-elements.jsonld).

| Content Tree Tag | Dublin Core | Schema.org | Notes |
|---|---|---|---|
| `H1` | `dc:title` | `schema:name` | Document title |
| `P` | `dc:description` (if abstract) | `schema:text` | General text |
| `Figure` | — | `schema:image` | With `schema:caption` from Caption |
| `BibEntry` | `dc:references` | `schema:citation` | Bibliographic reference |
| `Note` | — | `schema:comment` | Footnotes/endnotes |
| `Formula` | — | `schema:mathExpression` | Mathematical content |
| `Table` | — | `schema:Table` | Tabular data |
| `TOC` | `dcterms:tableOfContents` | — | Table of contents |

### PREMIS (Preservation Metadata)

For digital preservation workflows. PREMIS tracks provenance events — pairs well with W3C PROV.

| BADGERS Concept | PREMIS Element | Notes |
|---|---|---|
| Analysis run | `<premis:event>` with `eventType="analysis"` | One event per specialist execution |
| Source document | `<premis:object>` (intellectual entity) | The input PDF |
| Content tree output | `<premis:object>` (representation) | The correlation XML |
| Specialist Lambda | `<premis:agent>` with `agentType="software"` | Agent that performed the event |
| Foundation model | `<premis:agent>` with `agentType="software"` | Model version as agent identifier |
| Session provenance | `<premis:event>` chain | Ordered sequence of analysis events |

---

## Quick Reference: Which Schema to Use

| If you need to... | Use this schema | Map to this external standard |
|---|---|---|
| Classify document elements semantically | Elements (`badgers-elements.xsd`) | Dublin Core / Schema.org via JSON-LD |
| Build an accessible PDF structure tree | Content Tree (`badgers-content-tree.xsd`) | PDF/UA (ISO 14289-1) — already native |
| Export to a scholarly publishing pipeline | Content Tree | JATS XML |
| Integrate with a digital library system | Content Tree + Elements | ALTO XML (layout) + Dublin Core (metadata) |
| Track provenance of analysis results | Content Tree metadata + PROV | PREMIS or W3C PROV-O/PROV-XML |
| Serve linked-data consumers | Elements JSON-LD context | Schema.org / Dublin Core |

---

## Versioning

This mapping tracks BADGERS v5.0 (September 2026). The content tree schema version is `2.0` (introduced with the hierarchical `content_tree` replacing the flat `content_spine`). The elements vocabulary has no independent version number — it tracks the BADGERS release.
