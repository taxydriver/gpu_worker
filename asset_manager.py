"""Asset resolution and download helpers."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Event, Lock, Thread

import portalocker
import requests
from pydantic import BaseModel

from gpu_worker.asset_registry import PROVISIONERS, get_asset_group
from gpu_worker.config import get_settings
from gpu_worker.utils import is_non_empty_file, safe_unlink, sha256_file


LOGGER = logging.getLogger(__name__)
_DOWNLOAD_STATUS_LOCK = Lock()
_DOWNLOAD_STATUS: dict[str, object] | None = None
# Suppress the download banner for the first few seconds of an entry: a warm
# model that hf_hub_download resumes/verifies briefly reports a non-zero
# downloaded_bytes against its total before it settles to "already present",
# flashing a phantom "downloading 22%" that self-corrects. Only surface a
# status once it has been live past this window.
_DOWNLOAD_STATUS_MIN_ELAPSED_SEC = 3.0
_HF_INCOMPLETE_GLOB = "*.incomplete"
_PROVISIONED_GROUPS: set[str] = set()
_PROVISIONED_GROUPS_LOCK = Lock()

# HuggingFace resolve URL pattern: .../resolve/<revision>/<path_in_repo>
_HF_URL_RE = re.compile(
    r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)"
)


def _parse_hf_url(url: str) -> tuple[str, str, str] | None:
    """Return (repo_id, revision, filename) if url is a HuggingFace resolve URL."""
    m = _HF_URL_RE.match(url)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _hf_hub_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


class EnsureAssetsResult(BaseModel):
    """Result from ensuring an asset group exists locally."""

    downloaded_assets: list[str]
    asset_check_sec: float
    download_sec: float


def active_download_status() -> dict[str, object] | None:
    """Return a snapshot of the active model download, if any."""

    with _DOWNLOAD_STATUS_LOCK:
        if not _DOWNLOAD_STATUS:
            return None
        elapsed = _DOWNLOAD_STATUS.get("elapsed_sec")
        if isinstance(elapsed, (int, float)) and elapsed < _DOWNLOAD_STATUS_MIN_ELAPSED_SEC:
            # Too young to trust: a warm model being resumed/verified flashes a
            # phantom percentage here. Hide it until it has been live a moment.
            return None
        return dict(_DOWNLOAD_STATUS)


def _format_bytes(num_bytes: int) -> str:
    """Return a compact human-readable byte count."""

    value = float(max(num_bytes, 0))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _set_download_status(
    *,
    asset_name: str,
    downloaded_bytes: int,
    total_bytes: int | None,
    started_at: float,
    source: str,
) -> None:
    elapsed_sec = max(time.monotonic() - started_at, 0.001)
    mb_per_sec = downloaded_bytes / elapsed_sec / (1024 * 1024)
    pct = (
        min(max(downloaded_bytes / total_bytes * 100.0, 0.0), 100.0)
        if total_bytes and total_bytes > 0
        else None
    )
    with _DOWNLOAD_STATUS_LOCK:
        global _DOWNLOAD_STATUS
        _DOWNLOAD_STATUS = {
            "asset": asset_name,
            "source": source,
            "downloaded_bytes": downloaded_bytes,
            "downloaded": _format_bytes(downloaded_bytes),
            "total_bytes": total_bytes,
            "total": _format_bytes(total_bytes) if total_bytes else None,
            "pct": pct,
            "mb_per_sec": mb_per_sec,
            "elapsed_sec": elapsed_sec,
            "updated_at": time.time(),
        }


def _clear_download_status(asset_name: str) -> None:
    with _DOWNLOAD_STATUS_LOCK:
        global _DOWNLOAD_STATUS
        if _DOWNLOAD_STATUS and _DOWNLOAD_STATUS.get("asset") == asset_name:
            _DOWNLOAD_STATUS = None


def _hf_incomplete_prefix(metadata_name: str) -> str:
    """Return the Hugging Face local-dir incomplete-file prefix for a metadata file."""

    digest = hashlib.sha1(metadata_name.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode()


def _verify_sha256(path: Path, expected_sha256: str) -> None:
    """Verify the SHA-256 checksum for a downloaded file."""

    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() == expected_sha256.lower():
        return

    safe_unlink(path)
    raise RuntimeError(
        f"Checksum mismatch for {path}. Expected {expected_sha256}, got {actual_sha256}"
    )


def _aria2c_available() -> bool:
    """Return True if aria2c is installed on this system."""
    import shutil
    return shutil.which("aria2c") is not None


def _download_hf_hub(
    url: str,
    destination: Path,
    *,
    asset_name: str,
    started_at: float,
) -> None:
    """Download a HuggingFace file using huggingface_hub.

    huggingface_hub handles XetHub (cas-bridge.xethub.hf.co) natively — the
    old requests/aria2c path times out because HuggingFace migrated large models
    to XetHub's chunked protocol. Setting HF_XET_HIGH_PERFORMANCE=1 enables the
    hf_xet Rust extension for parallel multi-part transfers.

    Also supports resume: if local_dir/.cache has a partial download from a prior
    interrupted attempt, hf_hub_download picks up from where it left off.
    """
    import os as _os
    _os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    from huggingface_hub import hf_hub_download
    from huggingface_hub._local_folder import get_local_download_paths

    parsed = _parse_hf_url(url)
    assert parsed is not None
    repo_id, revision, filename = parsed
    dry_run_info = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=destination.parent,
        dry_run=True,
    )
    total_bytes = int(getattr(dry_run_info, "file_size", 0) or 0) or None
    local_paths = get_local_download_paths(local_dir=destination.parent, filename=filename)
    incomplete_prefix = _hf_incomplete_prefix(local_paths.metadata_path.name)

    def _hf_local_downloaded_bytes() -> int:
        candidates = [local_paths.file_path, destination]
        candidates.extend(
            local_paths.metadata_path.parent.glob(f"{incomplete_prefix}.{_HF_INCOMPLETE_GLOB}")
        )
        sizes: list[int] = []
        for candidate in candidates:
            try:
                if candidate.is_file():
                    sizes.append(candidate.stat().st_size)
            except OSError:
                continue
        return max(sizes, default=0)

    LOGGER.info(
        "Downloading asset (hf_hub): %s repo=%s revision=%s file=%s",
        asset_name, repo_id, revision, filename,
    )

    from huggingface_hub.utils import tqdm as hf_tqdm
    stop_progress = Event()

    def _log_local_progress() -> None:
        last_logged_at = started_at
        last_logged_size = -1
        while not stop_progress.wait(2):
            downloaded_bytes = _hf_local_downloaded_bytes()
            if downloaded_bytes <= 0:
                continue
            _set_download_status(
                asset_name=asset_name,
                downloaded_bytes=downloaded_bytes,
                total_bytes=total_bytes,
                started_at=started_at,
                source="hf_hub",
            )
            now = time.monotonic()
            if downloaded_bytes == last_logged_size or now - last_logged_at < 5:
                continue
            mb_per_sec = downloaded_bytes / max(now - started_at, 0.001) / (1024 * 1024)
            if total_bytes:
                pct = downloaded_bytes / total_bytes * 100.0
                LOGGER.info(
                    "Downloading asset progress: %s %s/%s (%.1f%%) %.2f MB/s",
                    asset_name,
                    _format_bytes(downloaded_bytes),
                    _format_bytes(total_bytes),
                    pct,
                    mb_per_sec,
                )
            else:
                LOGGER.info(
                    "Downloading asset progress: %s %s downloaded %.2f MB/s",
                    asset_name,
                    _format_bytes(downloaded_bytes),
                    mb_per_sec,
                )
            last_logged_size = downloaded_bytes
            last_logged_at = now

    progress_thread = Thread(
        target=_log_local_progress,
        name=f"asset-download-progress-{asset_name}",
        daemon=True,
    )
    progress_thread.start()

    class DownloadProgressTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_log_at = started_at
            _set_download_status(
                asset_name=asset_name,
                downloaded_bytes=max(int(self.n or 0), _hf_local_downloaded_bytes()),
                total_bytes=int(self.total) if self.total else total_bytes,
                started_at=started_at,
                source="hf_hub",
            )

        def update(self, n=1):
            result = super().update(n)
            now = time.monotonic()
            downloaded_bytes = max(int(self.n or 0), _hf_local_downloaded_bytes())
            current_total_bytes = int(self.total) if self.total else total_bytes
            _set_download_status(
                asset_name=asset_name,
                downloaded_bytes=downloaded_bytes,
                total_bytes=current_total_bytes,
                started_at=started_at,
                source="hf_hub",
            )
            if now - self._last_log_at >= 5 and downloaded_bytes > 0:
                mb_per_sec = downloaded_bytes / max(now - started_at, 0.001) / (1024 * 1024)
                if current_total_bytes:
                    pct = downloaded_bytes / current_total_bytes * 100.0
                    LOGGER.info(
                        "Downloading asset progress: %s %s/%s (%.1f%%) %.2f MB/s",
                        asset_name,
                        _format_bytes(downloaded_bytes),
                        _format_bytes(current_total_bytes),
                        pct,
                        mb_per_sec,
                    )
                else:
                    LOGGER.info(
                        "Downloading asset progress: %s %s downloaded %.2f MB/s",
                        asset_name,
                        _format_bytes(downloaded_bytes),
                        mb_per_sec,
                    )
                self._last_log_at = now
            return result

    # hf_hub_download saves as local_dir/<filename>; rename to destination (.part path)
    try:
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=destination.parent,
                tqdm_class=DownloadProgressTqdm,
            )
        )
    finally:
        stop_progress.set()
        progress_thread.join(timeout=1)
        _clear_download_status(asset_name)

    if local_path != destination:
        os.replace(local_path, destination)

    elapsed_sec = max(time.monotonic() - started_at, 0.001)
    size = destination.stat().st_size
    mb_per_sec = size / elapsed_sec / (1024 * 1024)
    LOGGER.info(
        "Download complete (hf_hub): %s %s in %.1fs (%.2f MB/s)",
        asset_name, _format_bytes(size), elapsed_sec, mb_per_sec,
    )


def _download_to_path(url: str, destination: Path, *, asset_name: str) -> None:
    """Download a URL to a local destination.

    Priority:
    1. huggingface_hub for HuggingFace URLs — handles XetHub natively and supports
       resume. The old requests/aria2c paths time out on HF's XetHub CDN.
    2. aria2c (16-connection parallel) when installed.
    3. Single-connection requests stream as final fallback.
    """
    started_at = time.monotonic()

    try:
        if _parse_hf_url(url) and _hf_hub_available():
            _download_hf_hub(url, destination, asset_name=asset_name, started_at=started_at)
        elif _aria2c_available():
            _download_aria2c(url, destination, asset_name=asset_name, started_at=started_at)
        else:
            _download_requests(url, destination, asset_name=asset_name, started_at=started_at)
    finally:
        _clear_download_status(asset_name)


def _download_aria2c(
    url: str,
    destination: Path,
    *,
    asset_name: str,
    started_at: float,
) -> None:
    """Download via aria2c with 16 parallel connections."""
    import subprocess

    settings = get_settings()
    LOGGER.info("Downloading asset (aria2c): %s url=%s", asset_name, url)

    # DL:287MiB or DL:2.4GiB etc — aria2c summary speed line
    _DL_RE = re.compile(r"\bDL:(\d+(?:\.\d+)?)(G|M|K)?i?B\b", re.IGNORECASE)
    # downloaded/total(pct%) — e.g. 12.6GiB/15.3GiB(82%)
    _PROGRESS_RE = re.compile(r"(\d+(?:\.\d+)?)(G|M|K)?i?B/(\d+(?:\.\d+)?)(G|M|K)?i?B\((\d+)%\)")

    def _parse_mib(value: float, unit: str) -> float:
        u = unit.upper() if unit else "B"
        if u == "G":
            return value * 1024
        if u == "M":
            return value
        if u == "K":
            return value / 1024
        return value / (1024 * 1024)

    cmd = [
        "aria2c",
        "--split=16",
        "--max-connection-per-server=16",
        "--min-split-size=10M",
        "--max-tries=5",
        "--retry-wait=3",
        f"--timeout={min(int(settings.model_download_timeout_sec), 600)}",
        "--continue=true",
        "--file-allocation=none",
        "--summary-interval=5",
        "--show-console-readout=false",
        f"--dir={destination.parent}",
        f"--out={destination.name}",
        url,
    ]
    with tempfile.TemporaryFile(mode="w+t") as output:
        proc = subprocess.Popen(
            cmd,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_log_at = started_at
        read_pos = 0
        last_dl_mib: float | None = None
        last_downloaded_mib: float | None = None

        while True:
            now = time.monotonic()

            # Read new aria2c output lines and parse speed + progress
            output.seek(read_pos)
            new_text = output.read()
            read_pos = output.tell()
            for line in new_text.splitlines():
                m_dl = _DL_RE.search(line)
                if m_dl:
                    last_dl_mib = _parse_mib(float(m_dl.group(1)), m_dl.group(2) or "B")
                m_prog = _PROGRESS_RE.search(line)
                if m_prog:
                    last_downloaded_mib = _parse_mib(float(m_prog.group(1)), m_prog.group(2) or "B")

            if now - last_log_at >= 5:
                elapsed_sec = max(now - started_at, 0.001)
                if last_downloaded_mib is not None:
                    avg_mb_per_sec = last_downloaded_mib / elapsed_sec
                    downloaded_bytes = int(last_downloaded_mib * 1024 * 1024)
                else:
                    current_size = destination.stat().st_size if destination.exists() else 0
                    avg_mb_per_sec = current_size / elapsed_sec / (1024 * 1024)
                    downloaded_bytes = current_size
                _set_download_status(
                    asset_name=asset_name,
                    downloaded_bytes=downloaded_bytes,
                    total_bytes=None,
                    started_at=started_at,
                    source="aria2c",
                )
                size_str = _format_bytes(int((last_downloaded_mib or 0) * 1024 * 1024)) if last_downloaded_mib else "?"
                if last_dl_mib is not None:
                    LOGGER.info(
                        "Downloading asset progress (aria2c): %s %s downloaded avg=%.2f MB/s current=%.2f MB/s",
                        asset_name, size_str, avg_mb_per_sec, last_dl_mib,
                    )
                else:
                    LOGGER.info(
                        "Downloading asset progress (aria2c): %s %s downloaded avg=%.2f MB/s",
                        asset_name, size_str, avg_mb_per_sec,
                    )
                last_log_at = now

            returncode = proc.poll()
            if returncode is not None:
                break
            time.sleep(1)

        output.seek(0)
        captured_output = output.read()

    if proc.returncode != 0:
        safe_unlink(destination)
        raise RuntimeError(
            f"aria2c failed for {asset_name}: {captured_output[-500:]}"
        )

    if not is_non_empty_file(destination):
        raise RuntimeError(f"Downloaded file is empty: {destination}")

    elapsed_sec = max(time.monotonic() - started_at, 0.001)
    size = destination.stat().st_size
    mb_per_sec = size / elapsed_sec / (1024 * 1024)
    LOGGER.info(
        "Download complete (aria2c): %s %s in %.1fs (%.2f MB/s)",
        asset_name,
        _format_bytes(size),
        elapsed_sec,
        mb_per_sec,
    )
    _clear_download_status(asset_name)


def _download_requests(
    url: str,
    destination: Path,
    *,
    asset_name: str,
    started_at: float,
) -> None:
    """Single-connection streaming download via requests (fallback)."""
    settings = get_settings()
    deadline = started_at + settings.model_download_timeout_sec

    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length") or 0)
        downloaded_bytes = 0
        last_log_at = started_at
        LOGGER.info(
            "Downloading asset: %s total=%s url=%s",
            asset_name,
            _format_bytes(total_bytes) if total_bytes > 0 else "unknown",
            url,
        )
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=settings.download_chunk_size):
                if not chunk:
                    continue
                now = time.monotonic()
                if now > deadline:
                    raise TimeoutError(f"Download timed out for {url}")
                handle.write(chunk)
                downloaded_bytes += len(chunk)

                if now - last_log_at >= 5:
                    elapsed_sec = max(now - started_at, 0.001)
                    mb_per_sec = downloaded_bytes / elapsed_sec / (1024 * 1024)
                    _set_download_status(
                        asset_name=asset_name,
                        downloaded_bytes=downloaded_bytes,
                        total_bytes=total_bytes if total_bytes > 0 else None,
                        started_at=started_at,
                        source="requests",
                    )
                    if total_bytes > 0:
                        pct = (downloaded_bytes / total_bytes) * 100
                        LOGGER.info(
                            "Downloading asset progress: %s %s/%s (%.1f%%) %.2f MB/s",
                            asset_name,
                            _format_bytes(downloaded_bytes),
                            _format_bytes(total_bytes),
                            pct,
                            mb_per_sec,
                        )
                    else:
                        LOGGER.info(
                            "Downloading asset progress: %s %s downloaded %.2f MB/s",
                            asset_name,
                            _format_bytes(downloaded_bytes),
                            mb_per_sec,
                        )
                    last_log_at = now

    if not is_non_empty_file(destination):
        raise RuntimeError(f"Downloaded file is empty: {destination}")

    elapsed_sec = max(time.monotonic() - started_at, 0.001)
    mb_per_sec = downloaded_bytes / elapsed_sec / (1024 * 1024)
    LOGGER.info(
        "Download complete: %s %s in %.1fs (%.2f MB/s)",
        asset_name,
        _format_bytes(downloaded_bytes),
        elapsed_sec,
        mb_per_sec,
    )
    _clear_download_status(asset_name)


def _ensure_single_asset(asset: dict[str, str]) -> str | None:
    """Ensure one asset exists locally and return its path if downloaded."""

    asset_name = asset["name"]
    target_path = Path(asset["path"]).expanduser()
    url = asset["url"]
    checksum = asset.get("sha256")

    if is_non_empty_file(target_path):
        LOGGER.info("Asset already present: %s -> %s", asset_name, target_path)
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{target_path}.lock")
    temp_path = Path(f"{target_path}.part")

    LOGGER.info("Ensuring asset: %s -> %s", asset_name, target_path)
    try:
        with portalocker.Lock(
            str(lock_path),
            mode="a+",
            timeout=get_settings().model_download_timeout_sec,
        ):
            if is_non_empty_file(target_path):
                LOGGER.info("Asset became available while waiting for lock: %s", target_path)
                return None

            safe_unlink(temp_path)
            _download_to_path(url, temp_path, asset_name=asset_name)
            if checksum:
                _verify_sha256(temp_path, checksum)
            os.replace(temp_path, target_path)
            LOGGER.info("Downloaded asset: %s", target_path)
            return str(target_path)
    except requests.RequestException as exc:
        safe_unlink(temp_path)
        raise RuntimeError(f"Failed to download asset {asset_name} from {url}: {exc}") from exc
    except portalocker.exceptions.LockException as exc:
        raise RuntimeError(f"Timed out waiting for asset lock: {lock_path}") from exc
    except Exception:
        safe_unlink(temp_path)
        raise


def is_asset_group_warm(asset_group: str) -> bool:
    """Return True when every asset for a group has a non-empty file on disk.

    Used at startup to recover warm state without redownloading: if a group's
    files are present locally, ComfyUI can load them on demand and the worker
    should advertise the group as warm to the broker.

    Returns False for unknown asset groups or any missing/empty asset file.
    """
    try:
        assets = get_asset_group(asset_group)
    except ValueError:
        return False
    if not assets:
        return False
    for asset in assets:
        path = Path(asset["path"]).expanduser()
        if not is_non_empty_file(path):
            return False
    return True


def ensure_asset_group(asset_group: str) -> EnsureAssetsResult:
    """Ensure all assets for a named group exist locally."""

    assets = get_asset_group(asset_group)
    downloaded_assets: list[str] = []
    asset_check_sec = 0.0
    download_sec = 0.0

    for asset in assets:
        check_started = time.monotonic()
        already_present = is_non_empty_file(Path(asset["path"]).expanduser())
        asset_check_sec += time.monotonic() - check_started
        if already_present:
            LOGGER.info("Asset already present: %s", asset["path"])
            continue

        download_started = time.monotonic()
        downloaded = _ensure_single_asset(asset)
        download_sec += time.monotonic() - download_started
        if downloaded:
            downloaded_assets.append(downloaded)

    return EnsureAssetsResult(
        downloaded_assets=downloaded_assets,
        asset_check_sec=asset_check_sec,
        download_sec=download_sec,
    )


def ensure_asset_group_provisioned(asset_group: str) -> bool:
    """Run a group-specific provisioner once in this worker process.

    Provisioners install runtime pieces asset downloads cannot express (for
    example Comfy custom nodes). They run before the caller decides whether a
    Comfy restart is needed, so unloaded nodes are never advertised as ready.
    """

    relative_script = PROVISIONERS.get(asset_group)
    if not relative_script:
        return False
    with _PROVISIONED_GROUPS_LOCK:
        if asset_group in _PROVISIONED_GROUPS:
            return False
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / relative_script
        if not script.is_file():
            raise RuntimeError(f"Provisioner missing for {asset_group}: {script}")
        try:
            subprocess.run(
                ["bash", str(script)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()[:1000]
            raise RuntimeError(f"Provisioner failed for {asset_group}: {detail}") from exc
        _PROVISIONED_GROUPS.add(asset_group)
        return True
