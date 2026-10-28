#!/usr/bin/env python3
"""Run the XLSX Vision component only against the checked-in local sample.

The sample workbook and generated result live below ``.archive`` and are
ignored by Git. This script never calls Bedrock unless ``--invoke-bedrock`` is
given explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROCESSOR_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROCESSOR_DIRECTORY.parents[3]
DEFAULT_WORKBOOK = (
    REPOSITORY_ROOT
    / ".archive/features/xlxs-files/samples/badgers-workbook-inspection-sample.xlsx"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / ".archive/features/xlxs-files/samples/badgers-workbook-vision-output.json"
)
MANIFEST = REPOSITORY_ROOT / "deployment/s3_files/manifests/xlsx_processor.json"
PROMPTS = REPOSITORY_ROOT / "deployment/s3_files/prompts/xlsx_processor"

sys.path.insert(0, str(PROCESSOR_DIRECTORY))
sys.path.insert(0, str(REPOSITORY_ROOT / "deployment/badgers-foundation"))

from visual_analysis import (  # noqa: E402
    VisualAnalysisError,
    analyze_workbook_visuals,
    load_local_visual_configuration,
    prepare_visual_analysis,
)
from workbook_inspector import inspect_xlsx  # noqa: E402
from xlsx_content_tree import build_xlsx_content_tree_xml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--xml-output",
        type=Path,
        help="Canonical BADGERS XML output (defaults to --output with .xml suffix).",
    )
    parser.add_argument(
        "--invoke-bedrock",
        action="store_true",
        help="Actually send the sample's embedded PNG to Bedrock Vision.",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="Bedrock region when --invoke-bedrock is supplied (default: us-west-2).",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=5,
        help="Maximum embedded images to analyze (1–5; default: 5).",
    )
    args = parser.parse_args()

    if not args.workbook.is_file():
        parser.error(f"Sample workbook was not found: {args.workbook}")
    workbook_bytes = args.workbook.read_bytes()
    inspection = inspect_xlsx(workbook_bytes)
    prepared, skipped = prepare_visual_analysis(
        workbook_bytes, inspection, max_objects=args.max_objects
    )

    if not args.invoke_bedrock:
        output = {
            "schema_version": "badgers-xlsx-visual-analysis/v1",
            "status": "prepared_not_invoked",
            "sample_workbook": str(args.workbook),
            "prepared_objects": [
                {
                    "sheet_name": item["sheet_name"],
                    "object_id": item["object"]["object_id"],
                    "object_kind": item["object"]["kind"],
                    "content_type": item["content_type"],
                    "image_bytes": len(item["image_bytes"]),
                }
                for item in prepared
            ],
            "skipped_objects": skipped,
        }
    else:
        system_prompt, model_config = load_local_visual_configuration(MANIFEST, PROMPTS)
        output = analyze_workbook_visuals(
            workbook_bytes,
            inspection,
            system_prompt=system_prompt,
            model_config=model_config,
            max_objects=args.max_objects,
            aws_region=args.region,
        )
        output["sample_workbook"] = str(args.workbook)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inspection["visual_analysis"] = output
    xml_output = args.xml_output or args.output.with_suffix(".xml")
    xml_output.parent.mkdir(parents=True, exist_ok=True)
    xml_output.write_bytes(
        build_xlsx_content_tree_xml(
            inspection,
            source_document=str(args.workbook),
            session_id="local-xlsx-vision-sample",
        )
    )
    print(f"Wrote {output['status']} result to {args.output} and canonical XML to {xml_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VisualAnalysisError as error:
        print(f"XLSX Vision analysis failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
