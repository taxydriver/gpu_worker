"""requirements.txt is the human-maintained declaration of intent;
requirements.lock is what deploy_gpu.py actually installs into a release
venv (worker_release.py:905). They can silently diverge: Pillow was added to
requirements.txt but never carried into requirements.lock, so every release
venv built after that point still had no PIL, and wan_i2v renders burned a
full 233s WAN pass before dying on the import (G410). Nothing had asserted
that every package requirements.txt declares is actually present in the lock.
"""

from __future__ import annotations

import re
from pathlib import Path

GPU_WORKER_ROOT = Path(__file__).resolve().parents[1]


def _normalize(name: str) -> str:
    # PEP 503: "-", "_" and "." are equivalent separators in a package name.
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_package_names(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        if name:
            names.add(_normalize(name))
    return names


def test_every_declared_dependency_is_present_in_the_lock() -> None:
    requirements = _declared_package_names(
        (GPU_WORKER_ROOT / "requirements.txt").read_text()
    )
    lock = _declared_package_names((GPU_WORKER_ROOT / "requirements.lock").read_text())
    missing = requirements - lock
    assert not missing, (
        f"requirements.txt declares {sorted(missing)} but requirements.lock "
        "(what deploy_gpu.py actually installs) does not pin it — see G410"
    )
