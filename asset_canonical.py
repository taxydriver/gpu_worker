"""Canonical asset-group naming.

One canonical identifier per logical asset group. Aliases (legacy capability
names like 'wan_i2v', 'flux2_stills') resolve to the canonical form (the keys
of ASSET_REGISTRY, e.g. 'wan_i2v_v1', 'flux_stills_v1').

Keep the alias table in sync with Filmforge/backend/app/render_broker/asset_canonical.py.
"""

from __future__ import annotations

from gpu_worker.asset_registry import ASSET_REGISTRY, CAPABILITY_ASSET_GROUPS


def canonical_asset_group(name: str | None) -> str | None:
    """Return the canonical asset-group identifier for a name or alias.

    Returns None for None/empty/unknown input. Idempotent: canonical(canonical(x)) == canonical(x).
    """
    if not name:
        return None
    key = str(name).strip()
    if not key:
        return None
    if key in ASSET_REGISTRY:
        return key
    aliases = CAPABILITY_ASSET_GROUPS.get(key, [])
    for candidate in aliases:
        if candidate in ASSET_REGISTRY:
            return candidate
    return None


def canonicalize_groups(names: list[str] | None) -> list[str]:
    """Canonicalize a list of asset-group names, dropping unknowns and dedup."""
    if not names:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        canonical = canonical_asset_group(name)
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out
