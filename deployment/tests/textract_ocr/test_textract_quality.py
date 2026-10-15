def test_coherent_textract_lines_pass_quality_gate(textract_ocr):
    lines = [
        {"Text": "Revenue grew 12 percent year over year.", "Confidence": 98.0},
        {"Text": "Operating margin expanded in every region.", "Confidence": 97.0},
    ]

    result = textract_ocr._quality(lines)

    assert result["quality_passed"] is True
    assert result["quality_score"] >= 80


def test_short_or_low_confidence_ocr_escalates_to_vision(textract_ocr):
    result = textract_ocr._quality([{"Text": "x", "Confidence": 42.0}])

    assert result["quality_passed"] is False
    assert "too_little_recovered_text" in result["reasons"]
