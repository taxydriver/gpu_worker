# GPU Worker Deployment

Canonical reference for deploying FilmForge GPU workers via Verda. Update this when anything changes.

## Quick start

```bash
cd gpu_worker
python3 deploy_ui.py          # opens http://localhost:7860
```

Pick **Verda → Create & Deploy**. The UI streams all output live.

---

## Atomic worker release and secure cutover

Worker code is uploaded as a content-addressed candidate under
`.../releases/sha256-<digest>/`. The candidate owns an isolated venv and never becomes `current`
until the receipt-gated worker reports that exact code release from `/health`. The installer never
copies into, mutates, or pulls a remote git checkout.

Production packaging accepts only a clean committed `gpu_worker` tree. Untracked or modified files
fail before a provider VM is created. Only git-tracked files are packaged, the commit and tracked
manifest digests are recorded, and the committed, fully pinned `requirements.lock` is installed.
The same prebuilt archive is retained through provider creation and SSH upload, so the bytes checked
before billing starts are the bytes installed on the worker.

This directly prevents the H100 failure where `/opt/filmforge_gpu_worker` was at `a526cb5`, had
local `asset_registry.py` changes plus an untracked `provision_recammaster.sh` equivalent to
`915de05`, and `git pull --ff-only ... || true` continued after the conflict. Worker deploy now
contains no worker pull. TLS/tunnel, backend bearer client, tokens, authoritative worker count,
URL/port cardinality, protected receipt paths, and deployment phase are preflighted before
provider/GPU mutation.

Credential values never enter unit files, manifests, process arguments, or the repo. Public
`--env KEY=VALUE` rejects credential-shaped keys; use a mode-`0600` `--env-file` or protected
backend env storage.

### Required state machine

1. **Stage code.** On a live migration use `WORKER_DEPLOY_PHASE=stage-code`; it uploads and
   validates the immutable code+venv, then exits before GPU, unit, process, or backend mutation.
2. **Stage profile.** The versioned profile contains the loopback bind, exact advertised HTTPS
   URL, exact code release, stable tunnel unit/launcher/binary/config/credential, worker and
   registration secrets, and independent backend-probe secret. Migration requires and preserves
   `99-public-url-override.conf`. Stage links only the tunnel unit/drop-in; it deliberately leaves
   worker `20-filmforge-secure-profile.conf` absent so tunnel preparation cannot stop or mutate the
   live worker. The known incident-era `10-secure-loopback.conf` and its exact env are copied into
   protected rollback storage but remain live until cutover; every other unmanaged worker/tunnel
   drop-in is refused. New idle hosts use `--profile-mode first-install`, never create a public
   override, and receive a managed `00-filmforge-staged-guard.conf` that prevents premature start.
3. **Prepare.** Reload and restart only the staged tunnel. The migration worker has no dependency
   on that tunnel yet and remains untouched behind `99`; a first-install worker must remain
   inactive. The launcher keeps the named tunnel process supervised; cutover authorization arms
   its continuous authenticated public-route watchdog.
4. **Provision a new idle host only.** After its first-install profile and `00` guard are prepared,
   rerun deploy with `WORKER_DEPLOY_PHASE=provision-only`. The remote gate revalidates every real
   staged artifact and mode, requires an uncut-over first-install receipt, refuses active services,
   processes, or occupied worker ports, and only then installs GPU/Comfy and inactive base units.
   It never starts the worker. Migration hosts skip this step.
5. **Verify readiness.** An independent verifier fills the false-by-default receipt only after
   proving the stable tunnel, backend bearer-sending client, backend registration, and worker
   secret fingerprint together.
6. **Cut over.** Migration removes `99`; first install starts and enables its worker and tunnel as
   one boot-persistent pair. Both verify loopback-only binding and then require a backend-origin
   probe exercising the real tunnel and bearer client.
   The proof includes the exact profile, code source, and dependency snapshot. Failure restores
   the verified `99` backup, removes the new worker dependency, restores the known incident profile,
   and restarts the old worker; first-install failure removes authorization and stops both units.
7. **Finalize code.** Rerun deploy with `WORKER_DEPLOY_PHASE=activate`. With a completed receipt it
   only verifies loopback `/health`, promotes `current`, and exits. It does not rewrite or restart
   services, so receipt-gated cutover remains the last service mutation.

Create protected files from `deploy/secure-profile/`, then stage exact candidate paths:

For a brand-new host, the executable order is: create/connect the idle VM, run `stage-code`, run
the `stage` and `prepare` commands below with `--profile-mode first-install`, then run
`provision-only`, obtain the independent receipt, cut over, and finally run `activate`. A stock VM
cannot pass `provision-only` before the real staged profile exists; that refusal is intentional.

```bash
sudo python3 manage_worker_release.py stage \
  --release-id <profile-release-id> \
  --worker-code-release-id sha256-<source-digest-prefix> \
  --worker-unit filmforge-worker-gpu0.service \
  --tunnel-unit filmforge-worker-tunnel-gpu0.service \
  --worker-port 9000 \
  --worker-public-url https://gpu0.example.com \
  --tunnel-local-url http://127.0.0.1:9000 \
  --worker-exec /opt/filmforge-worker-releases/releases/sha256-<source-digest-prefix>/.venv/bin/python \
  --worker-module-dir /opt/filmforge-worker-releases/releases/sha256-<source-digest-prefix> \
  --worker-secret-source /etc/filmforge/secrets/worker-gpu0.env \
  --tunnel-secret-source /etc/filmforge/secrets/tunnel-gpu0.env \
  --backend-probe-secret-source /etc/filmforge/secrets/backend-cutover-probe.env \
  --tunnel-exec-source deploy/bin/filmforge-worker-tunnel \
  --tunnel-binary-source /opt/instance-tools/bin/cloudflared \
  --profile-mode migration
```

Staging performs no daemon reload or restart. During migration it does **not** install worker
`20-filmforge-secure-profile.conf`; that link is created only inside the receipt-gated cutover.
For first install, the `00` guard exists while `20` remains absent. Prepare the exact tunnel and
create the verifier template:

```bash
sudo python3 manage_worker_release.py prepare \
  --release-id <profile-release-id> \
  --receipt-template /etc/filmforge/worker-security/cutover-receipt.json
```

After the independent verifier fills that fresh mode-`0600` receipt:

```bash
sudo python3 manage_worker_release.py cutover \
  --release-id <profile-release-id> \
  --receipt /etc/filmforge/worker-security/cutover-receipt.json
```

Cutover checks that systemd loaded the exact staged tunnel, verifies public-port closure, and
requires the post-restart authenticated backend probe. `BindsTo=` stops the worker if the tunnel
unit exits, while the launcher's authenticated edge-route watchdog forces that exit when a running
cloudflared process no longer carries traffic. Explicit recovery is:

```bash
sudo python3 manage_worker_release.py rollback --release-id <profile-release-id>
```

Rollback verifies and restores the saved `99` for migration, removes the managed worker profile,
restores the known incident-era files, stops/removes the staged tunnel, and reloads systemd. For a
first install it removes the authorization/profile, stops both units, and disables the units that
cutover enabled, so rollback remains fail-closed across reboot. Never remove
`99-public-url-override.conf` by hand.

The current state machine covers incident migration and first installation. A later secure-to-secure
profile/code replacement must not be improvised with `migration` or `first-install`; it requires a
dedicated old-profile CAS upgrade transaction before this runbook permits it.

### Vercel DNS + Caddy edge

The FilmForge infrastructure **Rent GPU** button now invokes this complete
first-install transaction with `--secure-one-click`. It performs the immutable
code preflight, Fly secret synchronization, Vercel A-record update, Caddy TLS
preparation, GPU provisioning, authenticated cutover, activation, and rollback
automatically. The UI must not manufacture the individual security variables;
they are derived and receipt-checked server-side.

One-click v1 deliberately accepts exactly one GPU worker per VM. Offers with
multiple GPUs are rejected before any provider call until a shared multi-port
TLS-edge contract is implemented. The manual state-machine commands below
remain the recovery and audit interface, not normal button-click instructions.

For Vercel-managed DNS, create an explicit `A` record for
`gpu-worker.anapana.ai` to the VM's stable public IPv4. Stage Caddy as the
versioned `caddy` edge provider; it is the only public listener (TCP 80/443 for
certificate issuance and renewal). Worker port 9000 remains loopback-only and
must never be published. Use the protected examples in `deploy/secure-profile/`:

```bash
sudo python3 manage_worker_release.py stage \
  --edge-provider caddy \
  --release-id <profile-release-id> \
  --worker-code-release-id sha256-<source-digest-prefix> \
  --worker-unit filmforge-worker-gpu0.service \
  --tunnel-unit filmforge-worker-edge-gpu0.service \
  --worker-port 9000 \
  --worker-public-url https://gpu-worker.anapana.ai \
  --tunnel-local-url http://127.0.0.1:9000 \
  --worker-exec /opt/filmforge-worker-releases/releases/sha256-<source-digest-prefix>/.venv/bin/python \
  --worker-module-dir /opt/filmforge-worker-releases/releases/sha256-<source-digest-prefix> \
  --worker-secret-source /etc/filmforge/secrets/worker-gpu0.env \
  --tunnel-secret-source /etc/filmforge/secrets/caddy-gpu0.env \
  --backend-probe-secret-source /etc/filmforge/secrets/backend-cutover-probe.env \
  --tunnel-exec-source deploy/bin/filmforge-worker-caddy \
  --tunnel-binary-source /opt/instance-tools/bin/caddy \
  --profile-mode first-install
```

Set `WORKER_EDGE_PROVIDER=caddy` in the deploy contract. Prepare still starts
only the edge and leaves the worker guarded. The independent receipt must set
`edge_tls_hostname_ready=true` after validating the certificate for the exact
hostname; after cutover the Caddy watchdog continuously makes authenticated
`/health` checks and exits after three failures, so `BindsTo=` fail-closes the
worker.

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

## InfiniteTalk capability redeploy (2026-08-08)

The generation worker can now be deployed/redeployed with canonical capability
`infinitetalk_v1`. The deploy sequence provisions the custom nodes before restarting generation
Comfy processes; `/health` advertises the capability only after exact weights, node classes,
wav2vec imports and Comfy reachability pass. The runtime/provisioner work is published at commit
`a526cb5`.

Two-person A2 workers must instead declare `infinitetalk_two_person_v1`. Secure cutover hashes the
approved Multi checkpoint during asset ensure, verifies the pinned wrapper and two-speaker patch,
and requires the explicit-mask/parallel-audio node schema before emitting the staged receipt.

Listener-stability workers declare `infinitetalk_two_person_v2`. Secure cutover reuses the same
pinned assets but additionally exercises the deterministic roomtone contract before v2 can be
advertised. A1 and A2-v1 capabilities never satisfy a v2 dispatch.

The backend's managed redeploy control accepts an explicit canonical capability list, permits only
one in-flight redeploy per instance, records the subprocess terminal status, redacts the remote
target from its receipt, and performs no provider VM/volume mutation beyond a read-only inventory
lookup. It is a worker fixup, not provisioning and not a render. There is currently no FilmForge
MCP action for this admin control; that missing surface is recorded as **G91**. Use the managed
infrastructure surface until G91 closes; do not substitute arbitrary shell access and call it a
FilmForge tool action.

Deployment/readiness does **not** mean FilmForge can render a talking shot. The missing owned
backend/MCP action is tracked as G90, and A1 proves still+audio—not V2V preservation.

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
