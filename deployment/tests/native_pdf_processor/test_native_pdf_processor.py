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

    assert decision["route"] == "native_pdf"
    assert decision["native_fidelity"] == "complete"
    assert native_pdf._validate_native_spine(xml, 1)
    assert "native_pdf_text" in xml


def test_semantic_visual_reference_routes_to_hybrid(native_pdf):
    page = _page_with_text(
        "Figure 2 shows the proposed architecture and workflow for data processing."
    )

    decision = native_pdf._decide_route(native_pdf._analyze_page(page, 1, tagged=False))

    assert decision["route"] == "hybrid"
    assert decision["visual_semantic_signal"] is True


def test_no_native_text_routes_to_textract_before_vision(native_pdf):
    document = fitz.open()
    page = document.new_page()

    decision = native_pdf._decide_route(native_pdf._analyze_page(page, 1, tagged=False))

    assert decision["route"] == "textract_ocr"
    assert decision["native_fidelity"] == "incomplete"


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
        assert bool(opened.needs_pass) is True
    finally:
        opened.close()
