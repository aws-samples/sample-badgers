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
        if not (concern not in prompt):
            raise AssertionError
        if not (specialist not in prompt):
            raise AssertionError
        for candidate in candidates:
            if not (candidate not in prompt):
                raise AssertionError


def test_prompt_excludes_word_candidates():
    concern = "Could be 'Smith' or 'Smyth'"
    prompt = br.build_prompt_text("transcribe")
    if not (concern not in prompt):
        raise AssertionError
    if not ("Smith" not in prompt):
        raise AssertionError
    if not ("Smyth" not in prompt):
        raise AssertionError


def test_prompt_allows_unreadable_and_placeholder():
    prompt = br.build_prompt_text("transcribe")
    if not ("UNREADABLE" in prompt):
        raise AssertionError
    if not ("[?]" in prompt):
        raise AssertionError


def test_assign_rows_prompt_mentions_ticks():
    if not ("tick" in br.build_prompt_text("assign_rows").lower()):
        raise AssertionError
    if not ("tick" not in br.build_prompt_text("transcribe").lower()):
        raise AssertionError


def test_messages_carry_image_then_instruction():
    msgs = br.build_messages("QUJD", "transcribe")
    if not (len(msgs) == 1):
        raise AssertionError
    content = msgs[0]["content"]
    if not (content[0]["type"] == "image"):
        raise AssertionError
    if not (content[0]["source"]["media_type"] == "image/png"):
        raise AssertionError
    if not (content[0]["source"]["data"] == "QUJD"):
        raise AssertionError
    if not (content[1]["type"] == "text"):
        raise AssertionError


# --- defensive response parsing ---


def test_parse_reading_valid_json():
    text = '{"reading": "18", "per_character": [{"char": "1", "confidence": "high"}], "marks": [], "notes": "n"}'
    parsed = br._parse_reading(text)
    if not (parsed["reading"] == "18"):
        raise AssertionError
    if not (parsed["per_character"] == [{"char": "1", "confidence": "high"}]):
        raise AssertionError
    if not (parsed["notes"] == "n"):
        raise AssertionError


def test_parse_reading_strips_code_fence():
    text = (
        '```json\n{"reading": "42", "per_character": [], "marks": [], "notes": ""}\n```'
    )
    if not (br._parse_reading(text)["reading"] == "42"):
        raise AssertionError


def test_parse_reading_normalises_bad_confidence():
    text = '{"reading": "5", "per_character": [{"char": "5", "confidence": "totally-sure"}]}'
    if not (br._parse_reading(text)["per_character"][0]["confidence"] == "low"):
        raise AssertionError


def test_parse_reading_non_json_falls_back_to_raw_text():
    parsed = br._parse_reading("UNREADABLE")
    if not (parsed["reading"] == "UNREADABLE"):
        raise AssertionError
    if not (parsed["per_character"] == []):
        raise AssertionError


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
        if not (shipped == br.default_prompts()[slot]):
            raise AssertionError(f"fallback drift in {slot}")


def test_custom_prompts_flow_through_and_still_exclude_concern():
    # An operator-edited prompt dict is honored, and property 5 still holds as long as the
    # code never injects the concern (it doesn't — concern is not a parameter here).
    custom = {
        "system": "Read the crop. Answer UNREADABLE if illegible; use [?] per character.",
        "transcribe": "Transcribe only.",
        "assign_rows": "Report tick values for each mark.",
    }
    text = br.build_prompt_text("transcribe", custom)
    if not ("Transcribe only." in text):
        raise AssertionError
    if not ("could be 18 or 16" not in text):
        raise AssertionError
    if not ("18" not in text and "16" not in text):
        raise AssertionError
