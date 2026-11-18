"""Tests for the comparison + cap pure functions (KNOWLEDGE.md §4.2 items 6-7, §4.5 property 4)."""

import comparison as c


# --- candidate extraction ---


def test_extract_candidates_or_phrase():
    if not (c.extract_candidates("could be 18 or 16") == ["18", "16"]):
        raise AssertionError


def test_extract_candidates_vs_phrase():
    if not (c.extract_candidates("87 vs 67") == ["87", "67"]):
        raise AssertionError


def test_extract_candidates_quoted():
    if not (c.extract_candidates("Could be 'Smith' or 'Smyth'") == ["Smith", "Smyth"]):
        raise AssertionError


def test_extract_candidates_none_when_single():
    if not (c.extract_candidates("looks like 42") == []):
        raise AssertionError
    if not (c.extract_candidates("") == []):
        raise AssertionError


# --- match ---


def test_exact_match():
    cands = c.extract_candidates("could be 18 or 16")
    if not (c.match_reading("18", cands) == "18"):
        raise AssertionError


def test_no_match():
    cands = c.extract_candidates("could be 18 or 16")
    if not (c.match_reading("19", cands) is None):
        raise AssertionError


def test_match_case_insensitive():
    if not (c.match_reading("smith", ["Smith", "Smyth"]) == "Smith"):
        raise AssertionError


def test_match_no_candidates():
    if not (c.match_reading("18", []) is None):
        raise AssertionError


# --- confidence ---


def test_confidence_lowest_among_differing_positions():
    # "18" vs "16" differ only at index 1; that char is medium, so overall medium
    per_char = [
        {"char": "1", "confidence": "high"},
        {"char": "8", "confidence": "medium"},
    ]
    if not (c.confidence_from_per_character(per_char, ["18", "16"]) == "medium"):
        raise AssertionError


def test_confidence_ignores_agreeing_positions():
    # differing char (index 1) is high; the low char at index 0 agrees between candidates
    per_char = [{"char": "1", "confidence": "low"}, {"char": "8", "confidence": "high"}]
    if not (c.confidence_from_per_character(per_char, ["18", "16"]) == "high"):
        raise AssertionError


def test_confidence_no_candidates_uses_overall_min():
    per_char = [{"char": "4", "confidence": "high"}, {"char": "2", "confidence": "low"}]
    if not (c.confidence_from_per_character(per_char, []) == "low"):
        raise AssertionError


def test_confidence_empty_per_character_is_low():
    if not (c.confidence_from_per_character([], ["18", "16"]) == "low"):
        raise AssertionError


# --- cap (§4.2 item 7, §4.5 property 4) ---


def test_cap_small_and_highly_enlarged():
    # shorter side 99 (< 100), scale > 10 -> cap. Other side large to isolate the boundary.
    conf, capped = c.apply_cap("high", [99, 200], 10.5)
    if not (capped is True):
        raise AssertionError
    if not (conf == "medium"):
        raise AssertionError


def test_no_cap_when_side_is_100():
    # shorter side exactly 100 is NOT under 100 -> no cap.
    conf, capped = c.apply_cap("high", [100, 200], 10.5)
    if not (capped is False):
        raise AssertionError
    if not (conf == "high"):
        raise AssertionError


def test_no_cap_when_scale_is_exactly_10():
    # scale must EXCEED 10; exactly 10 does not cap.
    conf, capped = c.apply_cap("high", [99, 200], 10.0)
    if not (capped is False):
        raise AssertionError
    if not (conf == "high"):
        raise AssertionError


def test_cap_when_scale_just_over_10():
    conf, capped = c.apply_cap("high", [99, 200], 10.01)
    if not (capped is True):
        raise AssertionError
    if not (conf == "medium"):
        raise AssertionError


def test_cap_does_not_raise_low():
    conf, capped = c.apply_cap("low", [50, 50], 20.0)
    if not (capped is True):
        raise AssertionError
    if not (conf == "low"):
        raise AssertionError


def test_property_4_capped_implies_not_high():
    for start in ("high", "medium", "low"):
        conf, capped = c.apply_cap(start, [50, 50], 20.0)
        if not (capped is True):
            raise AssertionError
        if not (conf != "high"):
            raise AssertionError
