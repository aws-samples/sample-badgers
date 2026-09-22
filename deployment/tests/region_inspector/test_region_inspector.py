"""Tests for the region_inspector crop core (ported from the prototype and extended for the
three deltas in KNOWLEDGE.md §5 and the geometry correctness properties in §4.5)."""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

import region_inspector as ri

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "passenger_list_brig_sophia.png"
)


@pytest.fixture(scope="module")
def page_bytes():
    return FIXTURE.read_bytes()


def norm(x1, y1, x2, y2, w=1030, h=558):
    return x1 / w, y1 / h, x2 / w, y2 / h


def test_fixture_dimensions(page_bytes):
    assert Image.open(io.BytesIO(page_bytes)).size == (1030, 558)


def test_tonnage_region_metadata(page_bytes):
    r = ri.inspect_regions(
        page_bytes, [ri.RegionRequest("t", *norm(948, 105, 1000, 150))]
    )[0]
    assert r.source_px_box == [940, 98, 1008, 157]
    assert r.source_px_size == [68, 59]
    assert max(r.output_px_size) == ri.TARGET_LONG_SIDE
    assert r.detail == "interpolated"
    assert r.scale_factor > 20
    assert any("Enlarged" in n for n in r.notes)
    assert any("near the resolution limit" in n for n in r.notes)


def test_output_png_decodes_to_reported_size(page_bytes):
    r = ri.inspect_regions(
        page_bytes, [ri.RegionRequest("c", *norm(335, 396, 395, 428))]
    )[0]
    img = Image.open(io.BytesIO(base64.b64decode(r.png_base64)))
    assert list(img.size) == r.output_px_size
    assert img.format == "PNG"


def test_padding_is_15_percent_each_side(page_bytes):
    r = ri.inspect_regions(page_bytes, [ri.RegionRequest("p", 0.4, 0.4, 0.6, 0.6)])[0]
    # region 0.2 wide -> pad 0.03 each side -> 0.37..0.63
    assert r.source_px_box == [
        round(0.37 * 1030),
        round(0.37 * 558),
        round(0.63 * 1030),
        round(0.63 * 558),
    ]


def test_edge_region_is_clamped(page_bytes):
    r = ri.inspect_regions(page_bytes, [ri.RegionRequest("e", 0.95, 0.95, 1.2, 1.1)])[0]
    assert r.source_px_box[2] == 1030 and r.source_px_box[3] == 558
    # property 2: source box lies within the page
    assert r.source_px_box[0] >= 0 and r.source_px_box[1] >= 0


# --- Delta 1: a bad region yields a per-region error instead of raising ---


@pytest.mark.parametrize("box", [(0.5, 0.5, 0.4, 0.6), (0.5, 0.5, 0.5, 0.6)])
def test_inverted_or_empty_region_gets_error_not_raise(page_bytes, box):
    r = ri.inspect_regions(page_bytes, [ri.RegionRequest("bad", *box)])[0]
    assert r.error is not None
    assert r.png_base64 == ""
    assert r.source_px_box is None


def test_bad_region_does_not_stop_the_others(page_bytes):
    reqs = [
        ri.RegionRequest("good1", *norm(948, 105, 1000, 150)),
        ri.RegionRequest("bad", 0.5, 0.5, 0.4, 0.6),
        ri.RegionRequest("good2", *norm(335, 396, 395, 428)),
    ]
    res = ri.inspect_regions(page_bytes, reqs)
    # property 6: results returned in request order with their IDs
    assert [r.region_id for r in res] == ["good1", "bad", "good2"]
    assert res[0].error is None and res[0].png_base64
    assert res[1].error is not None
    assert res[2].error is None and res[2].png_base64


# --- Delta 2: task drives the grid overlay (no separate grid flag) ---


def test_large_region_has_no_small_region_note(page_bytes):
    r = ri.inspect_regions(
        page_bytes, [ri.RegionRequest("big", *norm(620, 440, 1000, 558))]
    )[0]
    assert r.source_px_size == [467, 136]
    assert not any("near the resolution limit" in n for n in r.notes)


def test_assign_rows_task_overlays_grid(page_bytes):
    box = norm(620, 440, 1000, 558)
    plain = ri.inspect_regions(
        page_bytes, [ri.RegionRequest("a", *box, task="transcribe")]
    )[0]
    grid = ri.inspect_regions(
        page_bytes, [ri.RegionRequest("b", *box, task="assign_rows")]
    )[0]
    assert plain.output_px_size == grid.output_px_size
    assert plain.png_base64 != grid.png_base64


# --- Delta 3: image_uri is recorded on every result ---


def test_image_uri_recorded_on_every_result(page_bytes):
    uri = "s3://bucket/session/images/page_001.png"
    reqs = [
        ri.RegionRequest("good", *norm(948, 105, 1000, 150)),
        ri.RegionRequest("bad", 0.5, 0.5, 0.4, 0.6),
    ]
    res = ri.inspect_regions(page_bytes, reqs, image_uri=uri)
    assert all(r.image_uri == uri for r in res)


def test_passthrough_fields_recorded(page_bytes):
    r = ri.inspect_regions(
        page_bytes,
        [
            ri.RegionRequest(
                "t",
                *norm(948, 105, 1000, 150),
                task="transcribe",
                flagged_by="tables",
                concern="87 vs 67",
            )
        ],
    )[0]
    assert r.flagged_by == "tables"
    assert r.concern == "87 vs 67"
    assert r.task == "transcribe"


# --- §4.5 geometry properties ---


def test_detail_interpolated_iff_scale_over_one(page_bytes):
    # Same region, three target sizes, to exercise all three detail classes deterministically
    # (the fixture is smaller than the default 1568 long side, so nothing downsamples there).
    box = norm(620, 440, 1000, 558)  # source long side 467 px
    interp = ri.inspect_regions(page_bytes, [ri.RegionRequest("i", *box)])[0]
    assert interp.scale_factor > 1.0 and interp.detail == "interpolated"
    native = ri.inspect_regions(
        page_bytes, [ri.RegionRequest("n", *box)], target_long_side=467
    )[0]
    assert native.scale_factor == 1.0 and native.detail == "native"
    down = ri.inspect_regions(
        page_bytes, [ri.RegionRequest("d", *box)], target_long_side=100
    )[0]
    assert down.scale_factor < 1.0 and down.detail == "downsampled"


def test_every_output_long_side_is_target(page_bytes):
    boxes = [
        norm(948, 105, 1000, 150),
        norm(335, 396, 395, 428),
        norm(620, 440, 1000, 558),
    ]
    res = ri.inspect_regions(
        page_bytes, [ri.RegionRequest(str(i), *b) for i, b in enumerate(boxes)]
    )
    for r in res:
        assert max(r.output_px_size) == ri.TARGET_LONG_SIDE
