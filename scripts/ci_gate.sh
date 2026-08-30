#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${repo_root}"
PYTHONPATH="$(dirname "${repo_root}")${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m pytest -q --tb=short
