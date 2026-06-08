# GPU Worker Deployment

Canonical reference for deploying FilmForge GPU workers via Verda. Update this when anything changes.

## Quick start

```bash
cd gpu_worker
python3 deploy_ui.py          # opens http://localhost:7860
```

Pick **Verda → Create & Deploy**. The UI streams all output live.

---

## Validated configurations

| Instance type | GPU | VRAM | Location | Status |
|---|---|---|---|---|
| `2A100.44V` | 2× A100 | 2×40GB | FIN-01 | ✅ Validated (May 2026) |
| `1A100.22V` | 1× A100 | 22GB | FIN-01 | ✅ Spot available |
| `1A100.40S.22V` | 1× A100 SXM | 40GB | FIN-01 | ✅ On-demand |
| `1H100.80S.32V` | 1× H100 SXM | 80GB | FIN-02 | ✅ Spot available |
| `1B200.30V` | B200 (HGX) | 180GB | FIN-03 | ✅ Validated (Jun 2026) — spot, $2.14/hr |
| `1RTXPRO6000.30V.CC` | RTX PRO 6000 Blackwell | 96GB | FIN-03 | ❌ CUDA broken — see Known Issues |

**Always prefer FIN-03** — more GPU capacity and permanent volumes live there. FIN-01 is the fallback when FIN-03 has no availability.

---

## OS image

`ubuntu-24.04-cuda-12.8-open-docker`

- CUDA toolkit: 12.8 on disk, driver exposes CUDA 13.0
- GSP firmware included: Turing (tu10x), Ampere (ga10x), Ada (ad10x)
- GSP firmware **NOT included**: Blackwell (gb2xx) — causes CUDA error 802/101

**If you see "GPU Firmware: N/A" in the deploy log → wrong instance type. Stop and pick A100 or H100.**

---

## Volumes (FIN-03 permanent)

| Role | Volume ID | Size |
|---|---|---|
| OS | `90d1c888-1597-44ae-9de1-f1b3ed2e3b8a` | 100GB |
| Data / models | `d2c23270-6d0b-4ce4-8f0f-058b148a05d7` | 250GB |

These must be **detached** before a deploy. Rehydrate path (`fresh=false`) remounts them.

Fresh installs create new volumes — models need to re-download (~60–90 min for full WAN/FLUX suite).

> **FIN-01 fallback volumes** (if FIN-03 has no capacity):
> OS `34ec939d-a8c1-4ee2-9637-533e324dfe39`, Data `4ea18b04-564f-4218-ab79-e90d1ccc839b`

---

## Known issues

### RTX PRO 6000 `.CC` — Confidential Computing mode (2026-06-08, tested)
**Symptom:** `cuInit(0)` returns 802 (`CUDA_ERROR_SYSTEM_NOT_READY`). Persists regardless of wait time or PyTorch version.

**Root cause confirmed by live test:** The `.CC` suffix in `1RTXPRO6000.30V.CC` means **Confidential Computing PRODUCTION mode** — permanently enabled at the Verda hypervisor level.
```
CC status: ON
CC Environment: PRODUCTION
DRAM Encryption Mode: Enabled
```
Normal CUDA processes (PyTorch, ComfyUI) cannot run in CC mode without hardware attestation. There is no in-VM toggle.

**GPU firmware is fine** — `fw=580.126.09` (driver 580.x) loads correctly. The firmware was NOT the problem.

**Deploy script behavior:** Fast-fails immediately with a clear message when `nvidia-smi conf-compute -f` shows `CC status: ON`.

**Fix:** Use `1A100.22V`, `1H100.80S.32V`, or `1B200.30V`. If Verda offers a non-CC RTX PRO 6000 (without `.CC`), it would likely work fine.

---

### `vm delete` deletes volumes (CLI footgun)
**Never run `verda --agent vm delete <id> --yes`** — it soft-deletes ALL attached volumes including the model data volume.

If you already ran it, restore within 96h:
```python
from deploy_ui import _verda_volume_api_action
_verda_volume_api_action("<volume-id>", "restore")
```
The deploy UI teardown button (`/api/verda/teardown`) passes `volume_ids: []` correctly — use that instead.

---

### Deploy UI shows no output
**Fixed 2026-06-04:** SSE stream now replays accumulated logs for late-connecting browsers. Previously, if the browser connected after deploy started, it missed early log lines. Reconnect now always shows full output from the beginning.

---

## Rehydrate script checks (in order)

The `verda_rehydrate_script` in `deploy_gpu.py` runs on the remote VM and does:

1. `nvidia-smi` present?
2. GPU count ≥ 1?
3. **GPU firmware check** — exits immediately with clear error if any GPU has `GPU Firmware: N/A`
4. `/dev/vdb` mountable to `/mnt/data`?
5. ComfyUI at `/workspace/ComfyUI/main.py`?
6. ComfyUI venv Python executable?
7. CUDA 13 driver → reinstall torch for cu130 if needed
8. PyTorch CUDA validation (`torch.cuda.is_available()` must be True)
9. gpu_worker code found + venv present?
10. Install systemd units + start workers

If any step fails, the deploy log shows the exact failure with a fix hint. Output streams live to the UI.

---

## Cost guardrails

- **Verda UI**: blocks provision if per-GPU cost > $3/hr (confirmation dialog)
- **Always use spot** for dev/test runs — spot A100 in FIN-01 is ~$0.67/hr
- The GPU dropdown disables while availability is loading to prevent stale-selection accidents
