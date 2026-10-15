"""Extension-keyed document intake routing for BADGERS.

The router deliberately selects a processing contract from the source filename,
not from an optimistic content sniff. Each selected processor still validates
the bytes appropriate to its format before processing them.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".gif"}
_ROUTES = {
    ".docx": {
        "input_type": "docx",
        "processor_tool": "process_docx_tool",
        "processor_specialist": "docx_processor",
        "result_contract": "page_plan",
    },
    ".pdf": {
        "input_type": "pdf",
        "processor_tool": "process_native_pdf_tool",
        "processor_specialist": "native_pdf_processor",
        "result_contract": "page_plan",
    },
    ".pptx": {
        "input_type": "pptx",
        "processor_tool": "process_pptx_tool",
        "processor_specialist": "pptx_processor",
        "result_contract": "page_plan",
    },
}
for _extension in _IMAGE_EXTENSIONS:
    _ROUTES[_extension] = {
        "input_type": "image",
        "processor_tool": "pdf_to_images_converter",
        "processor_specialist": "pdf_to_images_converter",
        "result_contract": "page_images",
    }


def _extension(document_path: str) -> str:
    """Return the lower-case source extension without trusting URL query content."""
    source_path = document_path.split("?", 1)[0].split("#", 1)[0]
    return PurePosixPath(source_path).suffix.lower()


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Choose the first processing tool and state its output contract."""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        document_path = str(body.get("document_path") or "").strip()
        if not document_path:
            raise ValueError("Missing required parameter: document_path.")
        extension = _extension(document_path)
        if extension == ".doc":
            raise ValueError(
                "Legacy .doc files are not supported. Save the document as .docx or PDF."
            )
        if extension == ".docm":
            raise ValueError(
                "Macro-enabled .docm files are not supported. Remove macros and save as .docx."
            )
        if extension == ".xlsx":
            raise ValueError(
                f"{extension} is reserved for a future Office intake feature and is not supported yet."
            )
        route = _ROUTES.get(extension)
        if route is None:
            accepted = ", ".join(sorted(_ROUTES))
            raise ValueError(
                f"Unsupported file extension {extension or '(none)'}. Accepted extensions: {accepted}."
            )

        result = {
            **route,
            "document_path": document_path,
            "extension": extension,
            "session_id": body.get("session_id") or "",
            "job_id": body.get("job_id") or "",
            "doc_id": body.get("doc_id") or "",
        }
        return {
            "statusCode": 200,
            "body": json.dumps({"success": True, "result": json.dumps(result)}),
        }
    except ValueError as error:
        return {
            "statusCode": 400,
            "body": json.dumps({"success": False, "error": str(error)}),
        }
