"""Comparison and cap — pure functions for inspect_region_tool (KNOWLEDGE.md §4.2 items 6-7).

None of this touches Bedrock or S3. The blind read produces a ``reading`` and a
``per_character`` confidence list; these functions compare that reading to the specialist's
candidates (parsed from ``concern``) and decide a final confidence, then cap it when the crop
was enlarged past the point where new detail could exist.

Correctness property 4 (§4.5): ``capped`` implies ``confidence`` is not ``"high"``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

SMALL_REGION_PX = 100     # shorter source side below this is near the resolution limit
CAP_SCALE = 10            # enlargement beyond this, combined with a small source, caps confidence

# Confidence vocabulary, lowest to highest.
_ORDER = {"low": 0, "medium": 1, "high": 2}
_BY_RANK = {rank: label for label, rank in _ORDER.items()}


def _min_level(levels: List[str]) -> str:
    ranks = [_ORDER[v] for v in levels if v in _ORDER]
    if not ranks:
        return "low"
    return _BY_RANK[min(ranks)]


def _cap_to(confidence: str, ceiling: str) -> str:
    """Lower ``confidence`` to ``ceiling`` if it is higher; otherwise leave it."""
    c = _ORDER.get(confidence, 0)
    top = _ORDER.get(ceiling, 0)
    return _BY_RANK[min(c, top)]


def extract_candidates(concern: str) -> List[str]:
    """Pull the alternative readings out of a freeform concern string.

    Handles the shapes specialists actually produce: quoted alternatives
    ("'Smith' or 'Smyth'"), "could be 18 or 16", "18 vs 16", "18 / 16". Returns a list only
    when at least two distinct candidates are found; otherwise an empty list, which callers
    treat as "no candidates to compare against".
    """
    if not concern:
        return []

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", concern)
    if len(quoted) >= 2:
        candidates = [q.strip() for q in quoted if q.strip()]
        return candidates if len(candidates) >= 2 else []

    parts = re.split(r"\bor\b|\bvs\.?\b|/|,", concern, flags=re.IGNORECASE)
    candidates = []
    for part in parts:
        cleaned = re.sub(
            r"^(could be|might be|maybe|possibly|either|reads?|is|value)\s+",
            "",
            part.strip(),
            flags=re.IGNORECASE,
        ).strip()
        cleaned = cleaned.strip(".:;")
        if cleaned:
            candidates.append(cleaned)

    # Drop duplicates while preserving order.
    seen = set()
    unique = []
    for c in candidates:
        key = c.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique if len(unique) >= 2 else []


def match_reading(reading: str, candidates: List[str]) -> Optional[str]:
    """Return the candidate that equals ``reading`` (case-insensitively), or None."""
    if not reading:
        return None
    r = reading.strip()
    for candidate in candidates:
        c = candidate.strip()
        if c == r or c.casefold() == r.casefold():
            return candidate
    return None


def _differing_positions(candidates: List[str]) -> set:
    """Character indices at which the candidates do not all agree."""
    if len(candidates) < 2:
        return set()
    max_len = max(len(c) for c in candidates)
    differing = set()
    for i in range(max_len):
        chars = {c[i] if i < len(c) else None for c in candidates}
        if len(chars) > 1:
            differing.add(i)
    return differing


def confidence_from_per_character(
    per_character: List[Dict[str, str]], candidates: List[str]
) -> str:
    """Confidence = the lowest per-character confidence among the characters that differ
    between candidates; with no candidates, the lowest per-character confidence overall
    (KNOWLEDGE.md §4.2 item 6).

    Falls back to the overall minimum when the differing positions do not line up with any
    read character (e.g. the reading is shorter than the candidates)."""
    levels = [pc.get("confidence") for pc in per_character]
    levels = [lvl for lvl in levels if lvl in _ORDER]
    if not levels:
        return "low"

    differing = _differing_positions(candidates or [])
    if differing:
        selected = [
            per_character[i].get("confidence")
            for i in differing
            if i < len(per_character) and per_character[i].get("confidence") in _ORDER
        ]
        if selected:
            return _min_level(selected)

    return _min_level(levels)


def apply_cap(
    confidence: str, source_px_size: List[int], scale_factor: float
) -> Tuple[str, bool]:
    """Cap confidence to at most ``medium`` when the shorter source side is under 100 px AND
    the scale factor exceeds 10 (KNOWLEDGE.md §4.2 item 7 / §4.5 property 4)."""
    shorter = min(source_px_size)
    capped = shorter < SMALL_REGION_PX and scale_factor > CAP_SCALE
    if capped:
        confidence = _cap_to(confidence, "medium")
    return confidence, capped
