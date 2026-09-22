"""Identity department: a separate, isolated ComfyUI on its own card (2026-09-22).

Director ruling: the identity kit (InfiniteYou, PuLID, ArcFace, YOLO-seg, Fun-VACE replace)
gets FULL isolation from the production ComfyUI -- its own root, venv and custom_nodes --
because a shared card/runtime OOMed the production chain twice (Endayya, ledger M24).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from gpu_worker import asset_registry, deploy_gpu, identity_service

ROOT = Path(__file__).resolve().parents[1]


def _script(plan):
    return deploy_gpu.verda_rehydrate_script(
        public_ip="203.0.113.9",
        worker_port=9000,
        comfy_port=8188,
        worker_count=0,
        remote_root="/opt/filmforge_gpu_worker",
        worker_plan=plan,
    )


def _bash_fn(script: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", script, re.S | re.M)
    assert m, f"{name} not rendered"
    return m.group(0)


def _run(snippet: str) -> str:
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, check=True).stdout.strip()


# ── plan + department routing ─────────────────────────────────────────────────


def test_identity_is_a_plan_department() -> None:
    assert deploy_gpu.parse_worker_plan("generation,identity") == ["generation", "identity"]


def test_rendered_script_is_valid_bash() -> None:
    script = _script(["generation", "identity"])
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_identity_advertises_only_identity_and_generation_is_not_narrowed() -> None:
    fn = _bash_fn(_script(["generation", "identity"]), "caps_for_dept")
    assert _run(fn + 'caps_for_dept identity') == "identity_v1"
    # GPU0 keeps the broad generation default (never narrow a generation box at deploy).
    assert _run(fn + 'unset WORKER_CAPABILITIES; caps_for_dept generation') == (
        "flux2_stills,wan_i2v,ltx_i2v,character_loras"
    )


def test_identity_runs_its_own_comfy_root_and_generation_keeps_production() -> None:
    script = _script(["generation", "identity"])
    fns = _bash_fn(script, "comfy_root_for_dept") + _bash_fn(script, "runs_comfy")
    prelude = 'COMFY_ROOT=/workspace/ComfyUI; IDENTITY_COMFY_ROOT=/workspace/ComfyUI_identity\n'
    assert _run(prelude + fns + "comfy_root_for_dept generation") == "/workspace/ComfyUI"
    assert _run(prelude + fns + "comfy_root_for_dept identity") == "/workspace/ComfyUI_identity"
    for dept, expected in (("generation", "yes"), ("identity", "yes"), ("vision", "no"), ("audio", "no")):
        assert _run(prelude + fns + f"runs_comfy {dept} && echo yes || echo no") == expected


def test_helpers_are_defined_before_first_use() -> None:
    # runs_comfy used before its definition resolves to "command not found" (false) and
    # would DISABLE every ComfyUI unit, production included.
    script = _script(["generation", "identity"])
    for name in ("runs_comfy", "comfy_root_for_dept", "assert_gpu_ownership"):
        defined = script.index(f"{name}() {{")
        uses = [m.start() for m in re.finditer(rf"\b{name}\b", script) if m.start() != defined]
        assert uses and min(uses) > defined, name


def test_comfy_units_pin_each_card_and_use_the_department_root() -> None:
    script = _script(["generation", "identity"])
    assert "Environment=CUDA_VISIBLE_DEVICES=${idx}\nExecStart=$dept_comfy_root/.venv/bin/python main.py" in script
    assert "Environment=COMFY_OUTPUT_DIR=$dept_comfy_root/output" in script


def test_identity_root_is_prepared_before_units_and_provisioned_after() -> None:
    script = _script(["generation", "identity"])
    root_stage = script.index("IDENTITY_STAGE=root bash provision_identity.sh")
    first_unit = script.index('cat > "/etc/systemd/system/comfyui-gpu${idx}.service"')
    full = script.index("  bash provision_identity.sh")
    assert root_stage < first_unit < full


def test_identity_provisioner_fires_from_the_plan_not_only_the_capability() -> None:
    script = _script(["generation", "identity"])
    block = script[script.index('_identity_wanted=""'):script.index('if test -n "$_identity_wanted"; then')]
    gate = 'WORKER_PLAN=(generation identity); WORKER_CAPABILITIES=flux2_stills,wan_i2v\n' + block + 'echo "${_identity_wanted:-no}"'
    assert _run("set -euo pipefail\n" + gate) == "1"
    gate_no = 'WORKER_PLAN=(generation generation); WORKER_CAPABILITIES=flux2_stills\n' + block + 'echo "${_identity_wanted:-no}"'
    assert _run("set -euo pipefail\n" + gate_no) == "no"
    gate_empty = 'WORKER_PLAN=(); WORKER_CAPABILITIES=\n' + block + 'echo "${_identity_wanted:-no}"'
    assert _run("set -euo pipefail\n" + gate_empty) == "no"


def test_smoke_checks_isolation_and_gpu_ownership() -> None:
    script = _script(["generation", "identity"])
    assert "isolation broken" in script
    assert "assert_gpu_ownership" in script
    for cls in ("InfuseNetApply", "ApplyPulidFlux", "ExtractFacePoseImage", "WanVaceToVideo"):
        assert cls in script


# ── assets ────────────────────────────────────────────────────────────────────


def test_identity_asset_group_and_provisioner() -> None:
    group = asset_registry.ASSET_REGISTRY["identity_v1"]
    names = {a["name"] for a in group}
    assert {"identity_flux1_dev_fp8", "infu_sim_stage1_infusenet_bf16", "pulid_flux_v0_9_1",
            "buffalo_l_w600k_r50", "antelopev2_glintr100", "yolov8x_seg"} <= names
    assert all(a["url"].startswith("https://") and "black-forest-labs" not in a["url"] for a in group)
    assert asset_registry.CAPABILITY_ASSET_GROUPS["identity_v1"] == ["identity_v1", "wan_vace_v1"]
    assert asset_registry.PROVISIONERS["identity_v1"] == "gpu_worker/provision_identity.sh"


def test_provisioner_is_valid_bash_and_never_writes_production_nodes() -> None:
    text = (ROOT / "provision_identity.sh").read_text()
    subprocess.run(["bash", "-n", str(ROOT / "provision_identity.sh")], check=True)
    assert 'local d="$ID_ROOT/custom_nodes/$1"' in text
    assert '$COMFY/custom_nodes/ComfyUI' not in text  # nodes never land in production
    assert "1c979397c5c80f5ac83a2473a2f7d4503104110f" in text  # InfiniteYou pin
    assert "a80912fc3435c358607bf4b43a58dbcbebdb09ff" in text  # PuLID pin


def _patch_program() -> str:
    text = (ROOT / "provision_identity.sh").read_text()
    return text.split("<<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]


def test_pulid_patch_adds_kwargs_once(tmp_path: Path) -> None:
    src = tmp_path / "pulidflux.py"
    src.write_text(
        "def forward_orig(\n    self,\n    img,\n    guidance = None,\n    control=None,\n) -> Tensor:\n    return img\n"
    )
    prog = tmp_path / "patch.py"
    prog.write_text(_patch_program())
    subprocess.run(["python3", str(prog), str(src)], check=True, capture_output=True)
    once = src.read_text()
    assert "control=None, **kwargs) -> Tensor:" in once
    subprocess.run(["python3", str(prog), str(src)], check=True, capture_output=True)
    assert src.read_text() == once
    compile(once, str(src), "exec")


# ── worker service ────────────────────────────────────────────────────────────


def test_readiness_requires_declaration_and_install(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IDENTITY_COMFY_ROOT", str(tmp_path))
    assert identity_service.identity_readiness(["identity_v1"])["ready"] is False
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").write_text("")
    for node in ("ComfyUI_InfiniteYou", "ComfyUI-PuLID-Flux"):
        (tmp_path / "custom_nodes" / node).mkdir(parents=True)
    w = tmp_path / "models/insightface/models/buffalo_l"
    w.mkdir(parents=True)
    (w / "w600k_r50.onnx").write_text("")
    assert identity_service.identity_readiness(["identity_v1"])["ready"] is True
    undeclared = identity_service.identity_readiness(["flux2_stills", "wan_i2v"])
    assert undeclared["ready"] is False and "not_declared" in undeclared["reasons"]


def test_runner_inherits_the_worker_gpu_pin_and_refuses_unpinned(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert identity_service.runner_env()["CUDA_VISIBLE_DEVICES"] == "1"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError, match="not pinned"):
        identity_service.runner_env()


def test_rendered_worker_units_pin_every_worker_process() -> None:
    # The scorer runs inside the identity GPU's worker process: that unit must be pinned.
    script = _script(["generation", "identity"])
    unit = script[script.index('cat > "/etc/systemd/system/filmforge-worker-gpu${idx}.service" <<UNIT'):]
    assert "Environment=CUDA_VISIBLE_DEVICES=${idx}" in unit.split("UNIT\n", 2)[1]


def test_score_batches_a_whole_take_in_one_runner_call(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_runner(req, timeout_sec=900):
        calls.append(req)
        return {"scorer": "insightface/buffalo_l/w600k_r50", "frames": [], "frames_scored": 0}

    monkeypatch.setattr(identity_service, "run_runner", fake_runner)
    fetched = []
    identity_service.score(
        identity_service.IdentityScoreRequest(truth_url="https://x/t.png", video_url="https://x/v.mp4",
                                              person_id="grandfather"),
        lambda url, dest, kind: fetched.append((url, kind)),
    )
    assert len(calls) == 1 and calls[0]["mode"] == "score" and calls[0]["video"].endswith("take.mp4")
    assert fetched == [("https://x/t.png", "image"), ("https://x/v.mp4", "video")]


def _code_sync_block() -> str:
    text = (ROOT / "provision_identity.sh").read_text()
    return text.split("# --BEGIN-CODE-SYNC--\n", 1)[1].split("  # --END-CODE-SYNC--", 1)[0]


def test_root_sync_copies_every_code_file_and_excludes_only_root_dirs(tmp_path: Path) -> None:
    # 2026-09-22 first rent: an unanchored --exclude 'user*' dropped app/user_manager.py and
    # the identity ComfyUI crash-looped on import. Excludes must match ROOT entries only.
    prod, ident = tmp_path / "ComfyUI", tmp_path / "ComfyUI_identity"
    files = {
        "main.py": "m", "app/user_manager.py": "u", "app/model_manager.py": "mm",
        "comfy/ldm/models/autoencoder.py": "nested models dir", "app/temp/x.py": "nested temp",
        "comfy_extras/nodes_input/input.py": "i", "user_settings.py": "root user file",
    }
    for rel, body in files.items():
        (prod / rel).parent.mkdir(parents=True, exist_ok=True)
        (prod / rel).write_text(body)
    for root_dir in (".venv/bin", "models/checkpoints", "custom_nodes/prod_only_node", "user/default", "output", "temp"):
        (prod / root_dir).mkdir(parents=True, exist_ok=True)
        (prod / root_dir / "marker").write_text("prod")
    ident.mkdir()
    (ident / "custom_nodes/ComfyUI-PuLID-Flux").mkdir(parents=True)
    (ident / "custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py").write_text("identity node")
    subprocess.run(["bash", "-c", f'COMFY="{prod}"; ID_ROOT="{ident}"\n' + _code_sync_block()], check=True)
    for rel, body in files.items():
        if rel == "user_settings.py":
            continue  # a ROOT-level user* entry is excluded by design
        assert (ident / rel).read_text() == body, rel
    for root_dir in ("models", "custom_nodes/prod_only_node", "user", "output", "temp", ".venv"):
        assert not (ident / root_dir / "marker").exists(), root_dir
    # the identity root's own nodes survive --delete
    assert (ident / "custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py").read_text() == "identity node"


def test_provisioner_verifies_tracked_files_byte_for_byte() -> None:
    text = (ROOT / "provision_identity.sh").read_text()
    assert "git ls-files -z" in text and 'cmp -s "$COMFY/{}" "$ID_ROOT/{}"' in text


def test_root_sync_repairs_an_existing_broken_identity_root(tmp_path: Path) -> None:
    # The re-rent reuses the data volume, so the root stage must repair the broken copy
    # left by the first rent (missing app/user_manager.py), keep its venv and nodes, and
    # drop stale code that production no longer has.
    prod, ident = tmp_path / "ComfyUI", tmp_path / "ComfyUI_identity"
    (prod / "app").mkdir(parents=True)
    (prod / "app/user_manager.py").write_text("u")
    (prod / "main.py").write_text("m2")
    (ident / "app").mkdir(parents=True)
    (ident / "main.py").write_text("m1-stale")
    (ident / "app/removed_upstream.py").write_text("stale")
    (ident / ".venv/bin").mkdir(parents=True)
    (ident / ".venv/bin/python").write_text("identity venv")
    (ident / "custom_nodes/ComfyUI_InfiniteYou").mkdir(parents=True)
    subprocess.run(["bash", "-c", f'COMFY="{prod}"; ID_ROOT="{ident}"\n' + _code_sync_block()], check=True)
    assert (ident / "app/user_manager.py").read_text() == "u"
    assert (ident / "main.py").read_text() == "m2"
    assert not (ident / "app/removed_upstream.py").exists()
    assert (ident / ".venv/bin/python").read_text() == "identity venv"
    assert (ident / "custom_nodes/ComfyUI_InfiniteYou").is_dir()


def test_identity_deps_never_run_the_resolver() -> None:
    # Second rent (2026-09-22): resolving facexlib -> torch re-applied torch's declared CUDA
    # pins (cuda-toolkit 13.0.2 -> 12.8.1 + a full nvidia-*-cu12 set). Nothing may resolve.
    text = (ROOT / "provision_identity.sh").read_text()
    assert 'PIPN="$PIP --no-deps -c $FREEZE"' in text
    assert "pip list --format=freeze" in text  # constraints = the FULL freeze, not a hand list
    installs = [l for l in text.splitlines() if re.search(r"\$PIP[A-Z]* ", l) and not l.strip().startswith(("#", "PIPN="))]
    assert installs and all("$PIPN" in l for l in installs), installs
    assert "onnxruntime-gpu" not in text.split("IDENTITY_PKGS=", 1)[1].split('"', 2)[1]
    assert "-r " not in text  # node requirements.txt files are never fed to pip


def test_guard_compares_every_pre_existing_package(tmp_path: Path) -> None:
    text = (ROOT / "provision_identity.sh").read_text()
    guard = text[text.index("moved=$("):text.index('echo "[identity] added:')]
    before = tmp_path / "f.before"
    after = tmp_path / "f.after"
    before.write_text("cuda-toolkit==13.0.2\nml_dtypes==0.3.2\ntorch==2.11.0+cu128\n")
    prelude = f'FREEZE="{tmp_path}/f"; OWNED="^(ml_dtypes)=="\n'
    after.write_text("cuda-toolkit==13.0.2\nml_dtypes==0.6.0\nprotobuf==7.36.2\ntorch==2.11.0+cu128\n")
    assert subprocess.run(["bash", "-c", prelude + guard], capture_output=True).returncode == 0  # new + owned OK
    after.write_text("cuda-toolkit==12.8.1\nml_dtypes==0.6.0\ntorch==2.11.0+cu128\n")
    r = subprocess.run(["bash", "-c", prelude + guard], capture_output=True, text=True)
    assert r.returncode == 1 and "cuda-toolkit==13.0.2" in r.stderr


def test_root_stage_recopies_a_drifted_venv_from_a_reused_volume() -> None:
    text = (ROOT / "provision_identity.sh").read_text()
    block = text[text.index("  if [ -x \"$PY\" ]; then\n    local drift"):text.index("  if [ ! -x \"$PY\" ]; then")]
    assert '"$COMFY/.venv/bin/python" -m pip list --format=freeze' in block
    assert 'rm -rf "$ID_ROOT/.venv"' in block
    # the drift check runs BEFORE the copy-if-missing step, so a drifted copy is replaced
    assert text.index("local drift") < text.index('cp -a "$COMFY/.venv" "$ID_ROOT/.venv"')
