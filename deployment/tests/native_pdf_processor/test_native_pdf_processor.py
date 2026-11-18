import fitz



def _page_with_text(text: str) -> fitz.Page:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    # Hold the document through the page's parent reference for the test.
    return page


def test_linear_digital_text_stays_native_and_emits_valid_spine(native_pdf):
    page = _page_with_text(
        "Executive summary\nRevenue grew steadily in the first half of the year."
    )

    analysis = native_pdf._analyze_page(page, 1, tagged=False)
    decision = native_pdf._decide_route(analysis)
    xml = native_pdf._build_native_spine(
        analysis=analysis,
        document_path="s3://source/report.pdf",
        session_id="test",
        route=decision["route"],
        decision=decision,
    )

    if not (decision["route"] == "native_pdf"):
        raise AssertionError
    if not (decision["native_fidelity"] == "complete"):
        raise AssertionError
    if not (native_pdf._validate_native_spine(xml, 1)):
        raise AssertionError
    if not ("native_pdf_text" in xml):
        raise AssertionError


def test_native_spine_validation_rejects_entity_declaration(native_pdf):
    malicious_xml = """<!DOCTYPE unified_document [<!ENTITY payload "unsafe">]>
    <unified_document schema_version="2.0" page="1">
      <metadata><source_document>&payload;</source_document></metadata>
      <content_tree><sect id="sect_root" /></content_tree>
    </unified_document>"""

    if not (native_pdf._validate_native_spine(malicious_xml, 1) is False):
        raise AssertionError


def test_semantic_visual_reference_routes_to_hybrid(native_pdf):
    page = _page_with_text(
        "Figure 2 shows the proposed architecture and workflow for data processing."
    )

    decision = native_pdf._decide_route(native_pdf._analyze_page(page, 1, tagged=False))

    if not (decision["route"] == "hybrid"):
        raise AssertionError
    if not (decision["visual_semantic_signal"] is True):
        raise AssertionError


def test_no_native_text_routes_to_textract_before_vision(native_pdf):
    document = fitz.open()
    page = document.new_page()

    decision = native_pdf._decide_route(native_pdf._analyze_page(page, 1, tagged=False))

    if not (decision["route"] == "textract_ocr"):
        raise AssertionError
    if not (decision["native_fidelity"] == "incomplete"):
        raise AssertionError


def test_password_protected_pdf_is_detectable_before_page_analysis(tmp_path):
    protected = tmp_path / "protected.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Confidential")
    document.save(
        protected,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()

    opened = fitz.open(protected)
    try:
        if not (bool(opened.needs_pass) is True):
            raise AssertionError
    finally:
        opened.close()
