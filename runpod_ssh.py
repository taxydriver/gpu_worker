#!/usr/bin/env python3
"""Print the SSH connection string for the active filmforge_comfy RunPod pod."""

from __future__ import annotations
import sys
from pathlib import Path

POD_NAME = "filmforge_comfy"


def main() -> int:
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "filmforge_backend" / "app" / ".env"

    api_key = None
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

    if not api_key:
        print(f"ERROR: RUNPOD_API_KEY not found in {env_path}", file=sys.stderr)
        return 1

    try:
        import runpod  # type: ignore[import]
    except ImportError:
        print("ERROR: runpod package not installed. Run: pip install runpod", file=sys.stderr)
        return 1

    runpod.api_key = api_key
    pods = runpod.get_pods()
    pod = next((p for p in pods if p.get("name") == POD_NAME and p.get("desiredStatus") == "RUNNING"), None)
    if not pod:
        # Fall back to any pod with that name regardless of status
        pod = next((p for p in pods if p.get("name") == POD_NAME), None)
    if not pod:
        print(f"ERROR: No pod named '{POD_NAME}' found", file=sys.stderr)
        return 1

    ports = (pod.get("runtime") or {}).get("ports") or []
    ssh = next((x for x in ports if x["privatePort"] == 22 and x["isIpPublic"]), None)
    if not ssh:
        print("ERROR: SSH port not available yet — pod may still be booting", file=sys.stderr)
        return 1

    print(f"root@{ssh['ip']} -p {ssh['publicPort']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
