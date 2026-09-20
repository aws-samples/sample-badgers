# BADGERS Element Vocabulary

The BADGERS element vocabulary defines the document element types and sub-types that specialists extract during analysis. This vocabulary is the canonical type system shared across all BADGERS specialists that perform structural document analysis.

## Schema Files

| File | Format | Purpose |
|------|--------|---------|
| [`badgers-content-tree.xsd`](badgers-content-tree.xsd) | XML Schema (XSD) | **Primary interop schema** — formalizes the correlation specialist's content tree (PDF/UA tags) |
| [`badgers-elements.xsd`](badgers-elements.xsd) | XML Schema (XSD) | Formal schema for validating specialist XML output |
| [`badgers-elements.jsonld`](badgers-elements.jsonld) | JSON-LD Context | Linked-data context mapping BADGERS types to [Schema.org](https://schema.org/) and [Dublin Core](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) |
| [`MAPPING.md`](MAPPING.md) | Markdown | Rosetta stone — maps between Elements, Content Tree, JATS, ALTO, Dublin Core, and PREMIS |

> **Which schema should I use?** If you're integrating BADGERS output into an external system, start with [`MAPPING.md`](MAPPING.md) — it tells you which schema maps to which standard. The **content tree** (`badgers-content-tree.xsd`) is the primary interoperability surface; the **elements** vocabulary (`badgers-elements.xsd`) is the semantic analysis layer.

## Using the JSON-LD Context

Include the context in any BADGERS JSON output to make it interoperable with linked-data consumers:

```json
{
  "@context": "https://raw.githubusercontent.com/aws-samples/sample-badgers/main/schemas/badgers-elements.jsonld",
  "elements": [
    {
      "order": 1,
      "type": "heading",
      "sub_type": "Section heading (H2)",
      "content": "Introduction",
      "location": { "page": 1, "position": "top-center" }
    }
  ]
}
```

## Element Types

14 element types are defined, each with optional sub-types for finer classification.

### `document_title`

The main title of the entire document.

- **Schema.org mapping**: [`schema:name`](https://schema.org/name)

---

### `section`

A major division of the document, often containing multiple subsections or paragraphs.

- **Schema.org mapping**: [`schema:hasPart`](https://schema.org/hasPart)

| Sub-type |
|----------|
| Chapter |
| Main section |
| Subsection |
| Sub-subsection |

---

### `heading`

A title or subtitle within the document, used to organize content and indicate hierarchy.

- **Schema.org mapping**: [`schema:headline`](https://schema.org/headline)

| Sub-type |
|----------|
| Title (H1) |
| Section heading (H2) |
| Subsection heading (H3) |
| Sub-subsection heading (H4) |
| Lower-level heading (H5, H6) |
| Running head |

---

### `paragraph`

A self-contained unit of text, typically focusing on a single idea or topic.

- **Schema.org mapping**: [`schema:text`](https://schema.org/text)

| Sub-type |
|----------|
| Body paragraph |
| Introduction paragraph |
| Conclusion paragraph |

---

### `citation`

A reference to a source of information, typically found within the text or as a footnote.

- **Schema.org mapping**: [`schema:citation`](https://schema.org/citation)

| Sub-type |
|----------|
| In-text citation |
| Footnote citation |
| Endnote citation |

---

### `reference`

A detailed description of a source, usually found in a bibliography or reference list.

- **Dublin Core mapping**: [`dcterms:references`](http://purl.org/dc/terms/references)

| Sub-type |
|----------|
| Book reference |
| Journal article reference |
| Web page reference |
| Conference paper reference |

---

### `caption`

A brief explanation or description accompanying a figure, table, or image.

- **Schema.org mapping**: [`schema:caption`](https://schema.org/caption)

| Sub-type |
|----------|
| Figure caption |
| Table caption |
| Image caption |

---

### `quote`

A direct reproduction of text from another source.

- **Schema.org mapping**: [`schema:citation`](https://schema.org/citation)

| Sub-type |
|----------|
| Block quote |
| Inline quote |
| Pull quote |

---

### `equation`

A mathematical expression, often set apart from the main text.

- **Schema.org mapping**: [`schema:mathExpression`](https://schema.org/mathExpression)

| Sub-type |
|----------|
| Inline equation |
| Display equation |

---

### `list`

A series of items presented in a structured format.

- **Schema.org mapping**: [`schema:itemListElement`](https://schema.org/itemListElement)

| Sub-type |
|----------|
| Bulleted list |
| Numbered list |
| Definition list |

---

### `footnote`

Additional information or a citation placed at the bottom of a page.

- **Dublin Core mapping**: [`dcterms:description`](http://purl.org/dc/terms/description)

---

### `table_of_contents`

A list of the main sections or chapters of a document, typically including page numbers.

- **Dublin Core mapping**: [`dcterms:tableOfContents`](http://purl.org/dc/terms/tableOfContents)

---

### `abstract`

A brief summary of the document's main points, typically found at the beginning of academic papers.

- **Dublin Core mapping**: [`dcterms:abstract`](http://purl.org/dc/terms/abstract)

---

### `keyword`

A word or phrase that represents the main topics or themes of the document.

- **Schema.org mapping**: [`schema:keywords`](https://schema.org/keywords)

---

## Citation Styles

When a `citation` or `reference` element is detected, the specialist also identifies the citation style:

| Style | Standard |
|-------|----------|
| APA | American Psychological Association |
| MLA | Modern Language Association |
| Chicago | Chicago Manual of Style |
| Harvard | Harvard Referencing System |
| IEEE | Institute of Electrical and Electronics Engineers |
| Vancouver | Vancouver/ICMJE style |

---

## Versioning

This vocabulary is versioned with the BADGERS release. The current version tracks `BADGERS v5.0` (September 2026). Breaking changes to element types or sub-types will increment the major version.

## Source of Truth

The canonical definitions live in:
- **XSD**: `deployment/s3_files/prompts/elements_specialist/schema/schema_elements.xsd`
- **Dictionary**: `deployment/s3_files/prompts/elements_specialist/elements_dictionary.xml`

The files in this `schemas/` directory are published copies for external consumption.
