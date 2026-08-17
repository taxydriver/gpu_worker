#!/usr/bin/env python3
"""Stage, cut over, or roll back a FilmForge worker security release.

This command never contacts a provider or git remote. ``stage`` is filesystem
only and ``prepare`` controls the staged tunnel. ``cutover`` necessarily
restarts the worker and sends an authenticated probe to the configured backend
endpoint; it succeeds only when that backend proves the real public route.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from gpu_worker.worker_release import (
        SecureProfileLayout,
        SecureWorkerContract,
        StagedSecureProfile,
        WorkerReleaseError,
        build_cutover_receipt_template,
        cutover_secure_profile,
        prepare_secure_profile,
        rollback_secure_profile,
        stage_secure_profile,
    )
except ModuleNotFoundError:  # direct execution from the gpu_worker checkout
    from worker_release import (  # type: ignore[no-redef]
        SecureProfileLayout,
        SecureWorkerContract,
        StagedSecureProfile,
        WorkerReleaseError,
        build_cutover_receipt_template,
        cutover_secure_profile,
        prepare_secure_profile,
        rollback_secure_profile,
        stage_secure_profile,
    )


def _layout(args: argparse.Namespace) -> SecureProfileLayout:
    return SecureProfileLayout(
        systemd_root=args.systemd_root,
        state_root=args.state_root,
    )


def _write_json_0600(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stage(args: argparse.Namespace) -> int:
    contract = SecureWorkerContract(
        release_id=args.release_id,
        worker_code_release_id=args.worker_code_release_id,
        worker_unit=args.worker_unit,
        tunnel_unit=args.tunnel_unit,
        worker_port=args.worker_port,
        worker_public_url=args.worker_public_url,
        tunnel_local_url=args.tunnel_local_url,
        worker_exec=args.worker_exec,
        worker_module_dir=args.worker_module_dir,
        worker_secret_source=args.worker_secret_source,
        tunnel_secret_source=args.tunnel_secret_source,
        backend_probe_secret_source=args.backend_probe_secret_source,
        tunnel_exec_source=args.tunnel_exec_source,
        tunnel_binary_source=args.tunnel_binary_source,
        edge_provider=args.edge_provider,
        profile_mode=args.profile_mode,
        worker_count=args.worker_count,
    )
    staged = stage_secure_profile(contract, _layout(args))
    print(
        json.dumps(
            {
                "ok": True,
                "operation": "stage",
                "release_id": staged.release_id,
                "stage_receipt": str(staged.stage_receipt),
                "profile_mode": args.profile_mode,
                "public_override_present": args.profile_mode == "migration",
                "worker_profile_active": False,
                "first_install_guard_present": args.profile_mode == "first-install",
                "next_operation": "prepare",
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare(args: argparse.Namespace) -> int:
    layout = _layout(args)
    prepare_secure_profile(
        release_id=args.release_id,
        layout=layout,
    )
    receipt_template: str | None = None
    if args.receipt_template:
        release_dir = layout.releases_root / args.release_id
        staged = StagedSecureProfile(
            release_id=args.release_id,
            release_dir=release_dir,
            worker_dropin=Path(),
            tunnel_dropin=Path(),
            stage_receipt=release_dir / "stage-receipt.json",
        )
        _write_json_0600(
            args.receipt_template,
            build_cutover_receipt_template(staged),
        )
        receipt_template = str(args.receipt_template)
    print(
        json.dumps(
            {
                "ok": True,
                "operation": "prepare",
                "release_id": args.release_id,
                "tunnel_loaded_and_active": True,
                "worker_profile_unchanged": True,
                "receipt_template": receipt_template,
            },
            sort_keys=True,
        )
    )
    return 0


def _cutover(args: argparse.Namespace) -> int:
    cutover_secure_profile(
        release_id=args.release_id,
        receipt_path=args.receipt,
        layout=_layout(args),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "operation": "cutover",
                "release_id": args.release_id,
                "public_override_present": False,
                "worker_profile_active": True,
                "loopback_only_verified": True,
                "backend_authenticated_route_verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _rollback(args: argparse.Namespace) -> int:
    layout = _layout(args)
    stage_receipt = layout.releases_root / args.release_id / "stage-receipt.json"
    try:
        profile_mode = str(json.loads(stage_receipt.read_text())["profile_mode"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise WorkerReleaseError("rollback profile mode could not be verified") from None
    rollback_secure_profile(
        release_id=args.release_id,
        layout=layout,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "operation": "rollback",
                "release_id": args.release_id,
                "profile_mode": profile_mode,
                "public_override_present": profile_mode == "migration",
                "worker_stopped": profile_mode == "first-install",
                "tunnel_stopped": True,
                "worker_profile_active": False,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systemd-root",
        type=Path,
        default=Path("/etc/systemd/system"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("/etc/filmforge/worker-security"),
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    stage = subparsers.add_parser(
        "stage",
        help="stage a complete secure profile without changing the live public override",
    )
    stage.add_argument("--release-id", required=True)
    stage.add_argument(
        "--worker-code-release-id",
        required=True,
        help="exact immutable worker code release pinned by ExecStart",
    )
    stage.add_argument("--worker-unit", required=True)
    stage.add_argument("--tunnel-unit", required=True)
    stage.add_argument(
        "--edge-provider", choices=("cloudflared", "caddy"), default="cloudflared",
        help="cloudflared named tunnel (default) or direct Caddy TLS edge",
    )
    stage.add_argument("--worker-port", required=True, type=int)
    # ADR-0009: workers behind the single edge; 1 (default) is the byte-
    # identical single-worker profile every existing caller stages.
    stage.add_argument("--worker-count", type=int, default=1)
    stage.add_argument("--worker-public-url", required=True)
    stage.add_argument("--tunnel-local-url", required=True)
    stage.add_argument("--worker-exec", required=True, type=Path)
    stage.add_argument("--worker-module-dir", required=True, type=Path)
    stage.add_argument("--worker-secret-source", required=True, type=Path)
    stage.add_argument("--tunnel-secret-source", required=True, type=Path)
    stage.add_argument("--backend-probe-secret-source", required=True, type=Path)
    stage.add_argument(
        "--tunnel-exec-source",
        required=True,
        type=Path,
        help="repo-managed stable tunnel launcher installed into the release",
    )
    stage.add_argument(
        "--tunnel-binary-source",
        required=True,
        type=Path,
        help="concrete cloudflared executable copied into the immutable profile",
    )
    stage.add_argument(
        "--profile-mode",
        choices=("migration", "first-install"),
        default="migration",
        help="migration preserves an existing 99 override; first-install requires no public profile",
    )
    stage.set_defaults(handler=_stage)

    prepare = subparsers.add_parser(
        "prepare",
        help="reload and restart only the staged tunnel while the worker stays safe",
    )
    prepare.add_argument("--release-id", required=True)
    prepare.add_argument(
        "--receipt-template",
        type=Path,
        help="write the false-by-default verifier template after tunnel prepare",
    )
    prepare.set_defaults(handler=_prepare)

    cutover = subparsers.add_parser(
        "cutover",
        help="remove the public override only with a complete fresh readiness receipt",
    )
    cutover.add_argument("--release-id", required=True)
    cutover.add_argument("--receipt", required=True, type=Path)
    cutover.set_defaults(handler=_cutover)

    rollback = subparsers.add_parser(
        "rollback",
        help="restore the preserved public override before restarting the worker",
    )
    rollback.add_argument("--release-id", required=True)
    rollback.set_defaults(handler=_rollback)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkerReleaseError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
