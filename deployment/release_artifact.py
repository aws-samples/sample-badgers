"""Build a deterministic, hash-addressed BADGERS configuration release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .config_stager import stage_profile_config
    from .profile_resolver import resolve_profile
except ImportError:
    from config_stager import stage_profile_config
    from profile_resolver import resolve_profile


def build_release(profile_name: str, destination: Path, packs: tuple[str, ...] = ()) -> str:
    """Stage config, write a canonical manifest, and return its content hash."""
    profile = resolve_profile(profile_name, specialist_packs=packs or None)
    stage_profile_config(profile, destination)
    files = {
        path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format": 1,
        "profile": profile.to_dict(),
        "files": files,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    release_id = hashlib.sha256(encoded).hexdigest()
    manifest["release_id"] = release_id
    (destination / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return release_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hash-addressed configuration release.")
    parser.add_argument("profile")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--pack", action="append", default=[])
    args = parser.parse_args()
    release_id = build_release(args.profile, args.destination, tuple(args.pack))
    print(release_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
