# Vast GPU Worker Deployment Runbook

This file records the exact deployment pattern used to get `gpu_worker` running on a remote Vast GPU instance with ComfyUI.

The goal is that next time the only missing ingredient is a fresh Vast instance plus working SSH access.

## What Was Deployed

We deployed:

- `gpu_worker` from this repo
- on a Vast GPU instance that already had ComfyUI installed
- with the backend talking to the worker over HTTPS through a Cloudflare quick tunnel

After the changes in this repo, the worker can handle:

- Flux2 stills
- JuggernautXL stills
- WAN I2V video
- Stable Audio 1

The backend now downloads worker outputs locally and no longer needs direct Comfy access for:

- Flux2 stills
- JuggernautXL stills
- WAN video
- Stable Audio 1 generation

## Remote Host Layout We Found

On the working Vast template, the important paths and ports were:

- ComfyUI repo: `/workspace/ComfyUI`
- Comfy API: `http://127.0.0.1:18188`
- Comfy output dir: `/workspace/ComfyUI/output`
- Comfy temp dir: `/workspace/ComfyUI/temp`
- Comfy input dir: `/workspace/ComfyUI/input`
- Comfy process manager: `supervisord`

Comfy lifecycle commands used:

- `supervisorctl stop comfyui`
- `supervisorctl start comfyui`

## Worker Install Location

We installed the worker here:

- code dir: `/workspace/filmforge_gpu_worker/gpu_worker`
- venv dir: `/workspace/filmforge_gpu_worker/.venv`

## One-Time Local Prerequisite

You need working SSH access to the Vast instance.

Typical command shape:

```bash
ssh -i ~/.ssh/vast_deploy -p <SSH_PORT> root@<SSH_HOST>
```

## Step 1: Copy Worker To Vast

From your local machine:

```bash
scp -i ~/.ssh/vast_deploy -P <SSH_PORT> -r /Users/vamsee/ML/projects/gpu_worker root@<SSH_HOST>:/workspace/filmforge_gpu_worker/
```

## Step 2: Install Python Environment On Vast

SSH into the box and run:

```bash
cd /workspace/filmforge_gpu_worker
python3 -m venv .venv
.venv/bin/pip install -r gpu_worker/requirements.txt
```

## Step 3: Start The Worker

Run this on the Vast box:

```bash
cd /workspace/filmforge_gpu_worker
export COMFY_BASE_URL=http://127.0.0.1:18188
export COMFY_OUTPUT_DIR=/workspace/ComfyUI/output
export COMFY_TEMP_DIR=/workspace/ComfyUI/temp
export COMFY_INPUT_DIR=/workspace/ComfyUI/input
export COMFY_STOP_CMD="supervisorctl stop comfyui"
export COMFY_START_CMD="supervisorctl start comfyui"

nohup .venv/bin/python -m uvicorn gpu_worker.app:app \
  --host 0.0.0.0 \
  --port 9000 \
  >/tmp/gpu_worker.log 2>&1 </dev/null &
```

Health check:

```bash
curl http://127.0.0.1:9000/health
```

Expected shape:

```json
{
  "worker_ok": true,
  "comfy_reachable": true,
  "comfy_base_url": "http://127.0.0.1:18188",
  "known_asset_groups": [
    "flux_stills_v1",
    "juggernaut_stills_v1",
    "stable_audio_v1",
    "wan_i2v_v1"
  ]
}
```

## Step 4: Expose The Worker

On this Vast instance, direct public access to `:9000` was not reliably usable, so we used a Cloudflare quick tunnel.

Command used on the Vast box:

```bash
/opt/instance-tools/bin/cloudflared tunnel --url http://127.0.0.1:9000 --no-autoupdate
```

That prints a temporary public URL like:

```text
https://<random-name>.trycloudflare.com
```

Use that as:

- A render broker worker registration in the backend

Important:

- this URL is temporary
- it changes when the tunnel restarts
- it is fine for testing, not for production

## Step 5: Backend Env

For broker-based multi-worker routing, each worker should receive:

```env
FILMFORGE_BACKEND_URL=https://<your-backend-url>
WORKER_REGISTRATION_TOKEN=<shared-token>
WORKER_MAX_CONCURRENT_JOBS=1
```

Use `WORKER_CAPABILITIES=flux2_stills` for a Flux stills worker and
`WORKER_CAPABILITIES=wan_i2v` for a WAN video worker. Keep
`GPU_WORKER_BASE_URL` only as a legacy local-dev single-worker fallback.

For a Vast instance with two GPUs, use the automated multi-GPU path:

```bash
./setup_gpu.sh --vast \
  --vast-gpu "RTX A4000" \
  --vast-worker-count 2 \
  --vast-max-price 0.40
```

This starts `comfyui-gpu0`/`filmforge-worker-gpu0` and
`comfyui-gpu1`/`filmforge-worker-gpu1`. The workers use ports `9000` and
`9001`, ComfyUI ports `18188` and `18189`, and separate
`WORKER_ID_FILE` values so they register as two broker workers.

## Step 6: What The Worker Downloads

The worker auto-downloads models on first use by asset group.

### `flux_stills_v1`

- `flux2-vae.safetensors`
- `flux2_dev_fp8mixed.safetensors`
- `mistral_3_small_flux2_bf16.safetensors`

### `juggernaut_stills_v1`

- `JuggernautXL_v9_RunDiffusionPhoto_v2.safetensors`
- `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
- `ip-adapter-plus-face_sdxl_vit-h.safetensors`

### `wan_i2v_v1`

- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `wan_2.1_vae.safetensors`
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`

### `stable_audio_v1`

- `stable-audio-open-1.0.safetensors`
- `t5-base.safetensors`

First run on a fresh box can take a long time because of these downloads.

## Step 7: Validation Sequence

Use this order.

### A. Worker health

```bash
curl https://<worker-url>/health
```

### B. Flux still smoke test

Test a direct worker `POST /run` for short local checks, or prefer the backend still generation path. For remote tunnel-backed workers, the backend now uses async `POST /jobs` plus `GET /jobs/{job_id}` polling to avoid long request timeouts.

### C. WAN video test

Run one backend still-to-video flow.

### D. Stable Audio test

Run one backend audio generation call.

### E. Full trailer job

Run a full trailer job after:

- stills work
- WAN clips work
- Stable Audio works

## Important Behavioral Notes

### Remote output flow

The worker returns:

- `outputs`
- `output_files`

The backend downloads worker files into:

- `/tmp/filmforge_gpu_worker_outputs` by default

### Worker-served files

The worker exposes:

- `GET /files/{root_name}/{relative_path}`

Allowed file roots come from:

- `COMFY_OUTPUT_DIR`
- `COMFY_TEMP_DIR`

### Remote input staging

The worker also stages backend-provided input files into Comfy input before submission.

This is used for:

- WAN source stills
- Juggernaut IP-Adapter portrait refs

The backend sends these as `comfy_input_files` in the `/run` request.

## Current Remote Start Command We Used

This is the exact working pattern used on the Vast box:

```bash
cd /workspace/filmforge_gpu_worker
export COMFY_BASE_URL=http://127.0.0.1:18188
export COMFY_OUTPUT_DIR=/workspace/ComfyUI/output
export COMFY_TEMP_DIR=/workspace/ComfyUI/temp
export COMFY_INPUT_DIR=/workspace/ComfyUI/input
export COMFY_STOP_CMD="supervisorctl stop comfyui"
export COMFY_START_CMD="supervisorctl start comfyui"
nohup .venv/bin/python -m uvicorn gpu_worker.app:app --host 0.0.0.0 --port 9000 >/tmp/gpu_worker.log 2>&1 </dev/null &
```

## Current Remote Debug Commands

Check worker:

```bash
curl http://127.0.0.1:9000/health
pgrep -af "uvicorn gpu_worker.app:app"
tail -n 100 /tmp/gpu_worker.log
```

Check Comfy:

```bash
curl http://127.0.0.1:18188/system_stats
supervisorctl status comfyui
```

Check model download progress:

```bash
ls -lh /workspace/ComfyUI/models/checkpoints/*.part
ls -lh /workspace/ComfyUI/models/diffusion_models/*.part
ls -lh /workspace/ComfyUI/models/text_encoders/*.part
ls -lh /workspace/ComfyUI/models/clip_vision/*.part
ls -lh /workspace/ComfyUI/models/ipadapter/*.part
```

## Known Operational Caveats

- The quick tunnel is temporary.
- The worker is currently started with `nohup`, not yet under `supervisor` or `systemd`.
- First-run model downloads can take a long time.
- If the worker process exits, the quick tunnel returns `502`.
- Long WAN renders should go through the worker's async `/jobs` API, not a single long-lived `/run` request.
- The backend must be restarted after env changes.

## Recommended Next-Time Checklist

When you ask to deploy on a new Vast instance, the checklist should be:

1. get working SSH
2. inspect actual Comfy path, port, and process manager
3. copy `gpu_worker`
4. create venv and install requirements
5. set `COMFY_BASE_URL`, `COMFY_OUTPUT_DIR`, `COMFY_TEMP_DIR`, `COMFY_INPUT_DIR`
6. set real `COMFY_START_CMD` and `COMFY_STOP_CMD`
7. start worker on `:9000`
8. create tunnel or expose public port
9. update backend `GPU_WORKER_*` env
10. restart backend
11. test health
12. test Flux still
13. test Juggernaut still
14. test WAN video
15. test Stable Audio
16. test one full trailer job

## Best Follow-Up Improvements

Still pending for production-grade deployment:

- run `gpu_worker` under `supervisor` or `systemd`
- add auth/signing on worker endpoints
- replace quick tunnel with a stable domain or exposed port
- optionally move final audio/video upload closer to worker-side storage
