"""Small shared utilities for the GPU worker."""

from __future__ import annotations

import hashlib
import logging
import stat
import subprocess
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def ensure_no_symlink_path(
    trusted_root: Path,
    candidate: Path,
    *,
    require_leaf: bool,
    require_regular_file: bool = False,
    action: str = "Path",
) -> Path:
    """Return a lexical in-root path after lstat-checking every child component.

    ``trusted_root`` itself is configuration-owned and may resolve through a
    platform mount symlink. Every component *below* that root is inspected with
    ``lstat`` so a job directory cannot redirect reads or writes elsewhere.
    Missing suffixes are allowed only when ``require_leaf`` is false, which is
    used while creating new worker-owned paths.
    """

    raw_root = Path(trusted_root).expanduser().absolute()
    root = raw_root.resolve(strict=False)
    raw_candidate = Path(candidate).expanduser()
    if not raw_candidate.is_absolute():
        raw_candidate = raw_root / raw_candidate
    try:
        relative = raw_candidate.relative_to(raw_root)
    except ValueError:
        try:
            relative = raw_candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"{action} escapes trusted root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{action} has an unsafe path component")

    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        is_leaf = index == len(relative.parts) - 1
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if require_leaf:
                raise RuntimeError(f"{action} path is missing") from None
            return current.joinpath(*relative.parts[index + 1 :])
        except OSError as exc:
            raise RuntimeError(f"{action} path could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{action} path contains an unsafe symlink")
        if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"{action} ancestor is not a directory")
        if is_leaf and require_regular_file and not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{action} is not a regular file")
    return current


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
