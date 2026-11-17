#!/usr/bin/env python3
"""A small Docker credential helper that obtains ECR tokens only on demand."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys

_ECR_HOST = re.compile(r"^(?P<account>\d{12})\.dkr\.ecr\.(?P<region>[a-z]{2}-[a-z]+-\d)\.amazonaws\.com$")


def _host() -> str:
    return sys.stdin.read(8192).strip()


def main() -> int:
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command in {"store", "erase"}:
        return 0
    if command != "get":
        return 1
    host = _host()
    match = _ECR_HOST.fullmatch(host)
    if match is None:
        print("credentials not found in native keychain", file=sys.stderr)
        return 1
    aws = shutil.which("aws")
    if aws is None:
        print("aws CLI was not found", file=sys.stderr)
        return 1
    result = subprocess.run(
        [aws, "ecr", "get-login-password", "--region", match.group("region")],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print((result.stderr or "could not obtain ECR credentials")[:300], file=sys.stderr)
        return 1
    print(json.dumps({"ServerURL": host, "Username": "AWS", "Secret": result.stdout.strip()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
