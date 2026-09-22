"""Tests for the blind reader prompt (KNOWLEDGE.md §4.5 property 5, §4.2 item 5).

The Bedrock call itself is not exercised here — it needs the foundation layer and AWS. These
tests cover the prompt construction and the defensive response parsing, which are pure.
"""

import blind_reader as br


# --- property 5: the reading prompt carries no concern, candidates, or specialist name ---


def test_prompt_excludes_concern_and_candidates():
    concern = "could be 18 or 16"
    candidates = ["18", "16"]
    specialist = "tables"
    for task in ("transcribe", "assign_rows"):
        prompt = br.build_prompt_text(task)
        assert concern not in prompt
        assert specialist not in prompt
        for candidate in candidates:
            assert candidate not in prompt


def test_prompt_excludes_word_candidates():
    concern = "Could be 'Smith' or 'Smyth'"
    prompt = br.build_prompt_text("transcribe")
    assert concern not in prompt
    assert "Smith" not in prompt
    assert "Smyth" not in prompt


def test_prompt_allows_unreadable_and_placeholder():
    prompt = br.build_prompt_text("transcribe")
    assert "UNREADABLE" in prompt
    assert "[?]" in prompt


def test_assign_rows_prompt_mentions_ticks():
    assert "tick" in br.build_prompt_text("assign_rows").lower()
    assert "tick" not in br.build_prompt_text("transcribe").lower()


def test_messages_carry_image_then_instruction():
    msgs = br.build_messages("QUJD", "transcribe")
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == "QUJD"
    assert content[1]["type"] == "text"


# --- defensive response parsing ---


def test_parse_reading_valid_json():
    text = '{"reading": "18", "per_character": [{"char": "1", "confidence": "high"}], "marks": [], "notes": "n"}'
    parsed = br._parse_reading(text)
    assert parsed["reading"] == "18"
    assert parsed["per_character"] == [{"char": "1", "confidence": "high"}]
    assert parsed["notes"] == "n"


def test_parse_reading_strips_code_fence():
    text = (
        '```json\n{"reading": "42", "per_character": [], "marks": [], "notes": ""}\n```'
    )
    assert br._parse_reading(text)["reading"] == "42"


def test_parse_reading_normalises_bad_confidence():
    text = '{"reading": "5", "per_character": [{"char": "5", "confidence": "totally-sure"}]}'
    assert br._parse_reading(text)["per_character"][0]["confidence"] == "low"


def test_parse_reading_non_json_falls_back_to_raw_text():
    parsed = br._parse_reading("UNREADABLE")
    assert parsed["reading"] == "UNREADABLE"
    assert parsed["per_character"] == []


# --- S3 prompts are editable; the in-code fallback must not drift from the shipped files ---


def test_in_code_defaults_match_shipped_s3_files():
    from pathlib import Path

    prompts_dir = (
        Path(__file__).resolve().parents[2]
        / "s3_files"
        / "prompts"
        / "region_inspector"
    )
    for slot, filename in br.PROMPT_FILES.items():
        shipped = (prompts_dir / filename).read_text().strip()
        assert shipped == br.default_prompts()[slot], f"fallback drift in {slot}"


def test_custom_prompts_flow_through_and_still_exclude_concern():
    # An operator-edited prompt dict is honored, and property 5 still holds as long as the
    # code never injects the concern (it doesn't — concern is not a parameter here).
    custom = {
        "system": "Read the crop. Answer UNREADABLE if illegible; use [?] per character.",
        "transcribe": "Transcribe only.",
        "assign_rows": "Report tick values for each mark.",
    }
    text = br.build_prompt_text("transcribe", custom)
    assert "Transcribe only." in text
    assert "could be 18 or 16" not in text
    assert "18" not in text and "16" not in text
