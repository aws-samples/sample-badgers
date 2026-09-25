"""region_inspector — crop/resize core for the BADGERS inspect_region_tool.

Ported from the prototype ``region_inspector-02.py`` (see
``.archive/badgers-region-inspector-knowledge-02/artifacts/region_inspector/``) with the
three deltas KNOWLEDGE.md §5 requires:

1. A bad (empty or inverted) region no longer raises on the first one. It gets a per-region
   ``error`` and the others continue.
2. ``reason`` is replaced by ``task`` / ``flagged_by`` / ``concern``. The grid overlay is
   derived from ``task == "assign_rows"`` rather than a separate ``grid`` flag.
3. Each result records the ``image_uri`` the crop came from.

Input is page image bytes plus regions in either normalized 0-1 page coordinates or page
pixels. Output is one crop per region, resized so the long side is the vision model's working
size, with metadata stating exactly how much real detail the crop contains.

A crop of an image adds no new pixels. The metadata records the enlargement so downstream
confidence is not inflated by it. The vision-model read, comparison/cap, and S3 I/O live in
the sibling modules and the Lambda handler, not here.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image, ImageDraw

TARGET_LONG_SIDE = 1568  # long side handed to the vision model
DEFAULT_PAD = 0.15  # context padding: 15% of region width/height on each side
SMALL_REGION_PX = (
    100  # below this (shorter side, source px) fine features are near the limit
)

TRANSCRIBE = "transcribe"
ASSIGN_ROWS = "assign_rows"
VALID_TASKS = (TRANSCRIBE, ASSIGN_ROWS)


@dataclass
class RegionRequest:
    region_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    task: str = TRANSCRIBE  # "transcribe" | "assign_rows"
    flagged_by: str = ""  # specialist that raised the flag (not shown to the reader)
    concern: str = ""  # the specialist's uncertainty (withheld from the blind read)

    @property
    def grid(self) -> bool:
        """The page-coordinate tick overlay is only for the row-assignment task."""
        return self.task == ASSIGN_ROWS


@dataclass
class RegionResult:
    region_id: str
    image_uri: str  # the image URI this crop came from
    task: str
    flagged_by: str
    concern: str
    page_px_size: Optional[List[int]] = None  # full page w, h
    source_px_box: Optional[List[int]] = (
        None  # padded region in page pixels: x1, y1, x2, y2
    )
    source_px_size: Optional[List[int]] = (
        None  # region w, h in page pixels (real information)
    )
    output_px_size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    detail: Optional[str] = None  # "interpolated" | "native" | "downsampled"
    notes: List[str] = field(default_factory=list)
    png_base64: str = ""
    error: Optional[str] = None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _padded_box(r: RegionRequest, pad: float, page_size):
    x1, y1, x2, y2 = map(float, (r.x1, r.y1, r.x2, r.y2))
    if any(value > 1.0 for value in (x1, y1, x2, y2)):
        width, height = page_size
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    x1, y1, x2, y2 = map(_clamp01, (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{r.region_id}: empty or inverted region")
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return (_clamp01(x1 - pw), _clamp01(y1 - ph), _clamp01(x2 + pw), _clamp01(y2 + ph))


def _draw_grid(img: Image.Image, box_norm, step: float = 0.01) -> Image.Image:
    """Left-edge ticks labelled in PAGE-normalized y, so marks in the crop map back to page rows."""
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    _, y1, _, y2 = box_norm
    y = (int(y1 / step) + 1) * step
    while y < y2:
        py = round((y - y1) / (y2 - y1) * img.height)
        d.line([(0, py), (14, py)], fill=(220, 0, 0), width=2)
        d.text((16, py - 6), f"{y:.2f}", fill=(220, 0, 0))
        y += step
    return img


def inspect_regions(
    image_bytes: bytes,
    regions: List[RegionRequest],
    image_uri: str = "",
    pad: float = DEFAULT_PAD,
    target_long_side: int = TARGET_LONG_SIDE,
) -> List[RegionResult]:
    """Crop, pad, and resize each region. Returns one RegionResult per request, in request
    order. An empty or inverted region yields a result carrying ``error`` while the rest are
    processed (delta 1)."""
    page = Image.open(io.BytesIO(image_bytes))
    page.load()
    W, H = page.size
    results: List[RegionResult] = []

    for r in regions:
        result = RegionResult(
            region_id=r.region_id,
            image_uri=image_uri,
            task=r.task,
            flagged_by=r.flagged_by,
            concern=r.concern,
            page_px_size=[W, H],
        )
        try:
            box = _padded_box(r, pad, (W, H))
        except ValueError as exc:
            result.error = str(exc)
            results.append(result)
            continue

        px = [
            round(box[0] * W),
            round(box[1] * H),
            round(box[2] * W),
            round(box[3] * H),
        ]
        sw, sh = px[2] - px[0], px[3] - px[1]
        crop = page.crop(px)

        s = target_long_side / max(sw, sh)
        out = crop.resize((max(1, round(sw * s)), max(1, round(sh * s))), Image.LANCZOS)
        detail = "interpolated" if s > 1.0 else ("downsampled" if s < 1.0 else "native")

        notes: List[str] = []
        if s > 1.0:
            notes.append(
                f"Enlarged {s:.1f}x by Lanczos interpolation; no detail beyond "
                f"the {sw}x{sh} source pixels."
            )
        if min(sw, sh) < SMALL_REGION_PX:
            notes.append(
                f"Region is {sw}x{sh} source px. Features such as open vs closed loops "
                "are near the resolution limit; treat readings that depend on them as uncertain."
            )
        if r.grid:
            out = _draw_grid(out, box)

        buf = io.BytesIO()
        out.convert("RGB").save(buf, format="PNG")

        result.source_px_box = px
        result.source_px_size = [sw, sh]
        result.output_px_size = list(out.size)
        result.scale_factor = round(s, 3)
        result.detail = detail
        result.notes = notes
        result.png_base64 = base64.b64encode(buf.getvalue()).decode()
        results.append(result)

    return results
