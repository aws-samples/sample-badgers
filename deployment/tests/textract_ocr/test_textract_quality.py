def test_coherent_textract_lines_pass_quality_gate(textract_ocr):
    lines = [
        {"Text": "Revenue grew 12 percent year over year.", "Confidence": 98.0},
        {"Text": "Operating margin expanded in every region.", "Confidence": 97.0},
    ]

    result = textract_ocr._quality(lines)

    if not (result["quality_passed"] is True):
        raise AssertionError
    if not (result["quality_score"] >= 80):
        raise AssertionError


def test_short_or_low_confidence_ocr_escalates_to_vision(textract_ocr):
    result = textract_ocr._quality([{"Text": "x", "Confidence": 42.0}])

    if not (result["quality_passed"] is False):
        raise AssertionError
    if not ("too_little_recovered_text" in result["reasons"]):
        raise AssertionError
