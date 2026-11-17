"""Create the isolated Docker config used by BADGERS image-build commands."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import sys
from pathlib import Path


def _private_root(repo_root: Path) -> Path:
    root = (repo_root / ".deploy-state" / "docker-config").resolve()
    if root.name != "docker-config" or root.parent.name != ".deploy-state":
        raise ValueError("Refusing to create a Docker config outside .deploy-state/docker-config")
    return root


def configure(repo_root: Path, python: str | None = None) -> Path:
    """Rebuild a token-free config and install the local credential-helper shim."""
    root = _private_root(repo_root)
    if root.exists():
        shutil.rmtree(root)
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    os.chmod(bin_dir, 0o700)
    config = {"credsStore": "badgers-ecr"}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)
    helper = repo_root / "deployment" / "scripts" / "docker_credential_ecr.py"
    interpreter = python or sys.executable
    if os.name == "nt":
        shim = bin_dir / "docker-credential-badgers-ecr.cmd"
        shim.write_text(f'@"{interpreter}" "{helper}" %*\r\n', encoding="utf-8")
    else:
        shim = bin_dir / "docker-credential-badgers-ecr"
        shim.write_text(
            "#!/bin/sh\nexec " + shlex.quote(interpreter) + " " + shlex.quote(str(helper)) + ' "$@"\n',
            encoding="utf-8",
        )
    os.chmod(shim, 0o700)
    return root


if __name__ == "__main__":
    print(configure(Path(sys.argv[1]).resolve()))
