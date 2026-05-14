"""Small shared utilities for the GPU worker."""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def is_non_empty_file(path: Path) -> bool:
    """Return True when the path exists and has a non-zero size."""

    return path.is_file() and path.stat().st_size > 0


def safe_unlink(path: Path) -> None:
    """Delete a file if it exists."""

    try:
        path.unlink()
    except FileNotFoundError:
        return


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_shell_command(command: str) -> None:
    """Run a shell command and raise a blunt error if it fails."""

    LOGGER.info("Running command: %s", command)
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return

    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    detail = stderr or stdout or "no output"
    raise RuntimeError(f"Command failed ({completed.returncode}): {command}. Output: {detail[:500]}")
