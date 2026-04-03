"""PDF accessibility tagging utilities."""

from .tag_mapping import TAG_MAP, map_tag
from .bbox_resolver import resolve_text_bboxes, resolve_figure_bboxes
from .content_stream import wrap_content_stream_fixed
from .structure_tree import build_structure_tree
from .pdf_accessibility_models import TagRegion, VALID_TAGS

__all__ = [
    "TAG_MAP",
    "map_tag",
    "resolve_text_bboxes",
    "resolve_figure_bboxes",
    "wrap_content_stream_fixed",
    "build_structure_tree",
    "TagRegion",
    "VALID_TAGS",
]
