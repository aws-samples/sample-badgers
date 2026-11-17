#!/usr/bin/env python3
"""Create a Lambda-safe ZIP archive when the host does not provide ``zip``."""

from __future__ import annotations

import argparse
import os
import stat
import zipfile
from pathlib import Path


def zip_directory(source: Path, output: Path) -> None:
    """Archive *source* with forward-slash paths and Unix executable modes."""
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
            info = zipfile.ZipInfo(file_path.relative_to(source).as_posix())
            info.create_system = 3
            mode = stat.S_IMODE(file_path.stat().st_mode)
            if file_path.read_bytes()[:4] == b"\x7fELF":
                mode |= 0o111
            info.external_attr = (mode & 0xFFFF) << 16
            with file_path.open("rb") as input_file:
                archive.writestr(info, input_file.read(), compress_type=zipfile.ZIP_DEFLATED)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    zip_directory(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
