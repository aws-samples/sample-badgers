<mads_compliance_guide>
# MADS XML Schema Compliance Guide for LLMs

This guide ensures LLMs generate MADS XML that validates correctly against the MADS 2.1 schema on the first attempt.

## Schema Declaration

Always include the proper schema declaration at the beginning of MADS documents:

```xml
<madsCollection xmlns="http://www.loc.gov/mads/v2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xsi:schemaLocation="http://www.loc.gov/mads/v2 http://www.loc.gov/standards/mads/v2/mads-2-1.xsd">
    <!-- mads records go here -->
</madsCollection>
```

## Critical Validation Rules

### 1. Structure and Element Ordering

- **Root Structure**: All MADS XML must use either `<mads>` (single record) or `<madsCollection>` (multiple records) as the root element
- **Required Order in `<mads>`**:
  1. `<authority>` (required, at least one)
  2. `<related>` (optional)
  3. `<variant>` (optional)
  4. Other metadata elements (optional)
- **Required Elements**: Each `<mads>` record MUST have:
  - At least one `<authority>` element
  - One `<recordInfo>` element with creation metadata
- **Global Metadata**: Each metadata element (like `<recordInfo>`) must be inside a `<mads>` element, not directly under `<madsCollection>`

### 2. Element Content Constraints

| Element              | Valid Child Elements                                                                                      | Notes                           |
| -------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `<name>`             | `<namePart>`, `<description>`                                                                             | Not `<title>` or other elements |
| `<titleInfo>`        | `<title>`, `<subTitle>`, `<partNumber>`, `<partName>`, `<nonSort>`                                        | Not `<namePart>`                |
| `<topic>`            | Text content only                                                                                         | No child elements allowed       |
| `<organizationInfo>` | `<startDate>`, `<endDate>`, `<descriptor>`                                                                | Not `<fieldOfEndeavor>`         |
| `<personInfo>`       | `<birthDate>`, `<deathDate>`, `<birthPlace>`, `<deathPlace>`, `<gender>`, `<nationality>`, `<descriptor>` | Not `<fieldOfEndeavor>`         |

### 3. Attribute Value Restrictions

| Element     | Attribute | Valid Values                                                                  | Invalid Values                                    |
| ----------- | --------- | ----------------------------------------------------------------------------- | ------------------------------------------------- |
| `<related>` | `type`    | `earlier`, `later`, `parentOrg`, `broader`, `narrower`, `equivalent`, `other` | `references`, `host`, or any non-enumerated value |
| `<variant>` | `type`    | `acronym`, `abbreviation`, `translation`, `expansion`, `other`                | Any non-enumerated value                          |
| `<name>`    | `type`    | `personal`, `corporate`, `conference`, `family`                               | Any non-enumerated value                          |

### 4. Custom Relationship Types

For relationships not covered by enumerated values:
- Always use `type="other"`
- Add `otherType="custom-relationship"` to specify the custom relationship

```xml
<related type="other" otherType="cites">
    <!-- relationship content -->
</related>
```

## Common Error Patterns to Avoid

1. **Never place `<fieldOfEndeavor>` in `<personInfo>` or `<organizationInfo>`**
   - Incorrect: `<organizationInfo><fieldOfEndeavor>Value</fieldOfEndeavor></organizationInfo>`
   - Correct: `<organizationInfo><descriptor>Value</descriptor></organizationInfo>`
   - Correct: `<fieldOfActivity>Value</fieldOfActivity>` (as a separate element)

2. **Never use non-standard values for the `type` attribute**
   - Incorrect: `<related type="references">`
   - Correct: `<related type="other" otherType="references">`

3. **Never put `<namePart>` inside `<titleInfo>`**
   - Incorrect: `<titleInfo><namePart>Title</namePart></titleInfo>`
   - Correct: `<titleInfo><title>Title</title></titleInfo>`

4. **Never put child elements inside `<topic>`**
   - Incorrect: `<topic><namePart>Subject</namePart></topic>`
   - Correct: `<topic>Subject</topic>`

5. **Never place metadata elements outside of a `<mads>` record**
   - Incorrect: `<madsCollection><mads>...</mads><recordInfo>...</recordInfo></madsCollection>`
   - Correct: `<madsCollection><mads>...<recordInfo>...</recordInfo></mads></madsCollection>`

## Complete Example Template

```xml
<?xml version="1.0" encoding="UTF-8"?>
<madsCollection xmlns="http://www.loc.gov/mads/v2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xsi:schemaLocation="http://www.loc.gov/mads/v2 http://www.loc.gov/standards/mads/v2/mads-2-1.xsd">

    <!-- Person record -->
    <mads version="2.1">
        <authority>
            <name type="personal">
                <namePart type="family">Smith</namePart>
                <namePart type="given">John</namePart>
                <namePart type="date">1980-</namePart>
            </name>
        </authority>
        <variant type="other">
            <name type="personal">
                <namePart>Smith, J.</namePart>
            </name>
        </variant>
        <related type="other" otherType="collaboratesWith">
            <name type="personal">
                <namePart>Jones, Sarah</namePart>
            </name>
        </related>
        <personInfo>
            <birthDate>1980</birthDate>
            <nationality>American</nationality>
        </personInfo>
        <fieldOfActivity>Computer Science</fieldOfActivity>
        <recordInfo>
            <recordCreationDate encoding="w3cdtf">2024-05-13</recordCreationDate>
            <recordContentSource>Manual entry</recordContentSource>
            <languageOfCataloging>
                <languageTerm type="code" authority="iso639-2b">eng</languageTerm>
            </languageOfCataloging>
        </recordInfo>
    </mads>

    <!-- Organization record -->
    <mads version="2.1">
        <authority>
            <name type="corporate">
                <namePart>Example Corporation</namePart>
            </name>
        </authority>
        <variant type="acronym">
            <name type="corporate">
                <namePart>ExCorp</namePart>
            </name>
        </variant>
        <related type="broader">
            <name type="corporate">
                <namePart>Parent Industries</namePart>
            </name>
        </related>
        <organizationInfo>
            <startDate>2000</startDate>
            <descriptor>Technology company</descriptor>
        </organizationInfo>
        <fieldOfActivity>Software Development</fieldOfActivity>
        <recordInfo>
            <recordCreationDate encoding="w3cdtf">2024-05-13</recordCreationDate>
            <recordContentSource>Manual entry</recordContentSource>
            <languageOfCataloging>
                <languageTerm type="code" authority="iso639-2b">eng</languageTerm>
            </languageOfCataloging>
        </recordInfo>
    </mads>

    <!-- Topic record -->
    <mads version="2.1">
        <authority>
            <topic>Artificial Intelligence</topic>
        </authority>
        <variant type="acronym">
            <topic>AI</topic>
        </variant>
        <related type="broader">
            <topic>Computer Science</topic>
        </related>
        <recordInfo>
            <recordCreationDate encoding="w3cdtf">2024-05-13</recordCreationDate>
            <recordContentSource>Manual entry</recordContentSource>
            <languageOfCataloging>
                <languageTerm type="code" authority="iso639-2b">eng</languageTerm>
            </languageOfCataloging>
        </recordInfo>
    </mads>

    <!-- Work title record -->
    <mads version="2.1">
        <authority>
            <titleInfo>
                <title>Introduction to MADS</title>
                <subTitle>A Comprehensive Guide</subTitle>
            </titleInfo>
        </authority>
        <related type="other" otherType="hasAuthor">
            <name type="personal">
                <namePart>Zhang, Wei</namePart>
            </name>
        </related>
        <recordInfo>
            <recordCreationDate encoding="w3cdtf">2024-05-13</recordCreationDate>
            <recordContentSource>Manual entry</recordContentSource>
            <languageOfCataloging>
                <languageTerm type="code" authority="iso639-2b">eng</languageTerm>
            </languageOfCataloging>
        </recordInfo>
    </mads>
</madsCollection>
```
</mads_compliance_guide>