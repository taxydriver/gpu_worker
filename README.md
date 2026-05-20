# GPU Worker

This folder contains a small GPU-side FastAPI service for FilmForge. It sits in front of a local ComfyUI instance and only owns four responsibilities:

1. Resolve and download required model assets for a requested asset group.
2. Restart local ComfyUI if any new assets were downloaded.
3. Submit the provided ComfyUI workflow payload and poll until completion.
4. Return structured outputs, timings, and blunt debug/error details.

It deliberately does not own cinematic logic, planning, prompt construction, or workflow mutation.

## Architecture

FilmForge backend -> GPU worker -> local ComfyUI

The intended flow is:

1. FilmForge sends either `POST /run` for synchronous local use or `POST /jobs` for remote async use with a `job_id`, `asset_group`, and raw `comfy_payload`.
2. The worker ensures all models for that asset group exist locally.
3. If any model was downloaded, the worker restarts local ComfyUI once.
4. If the backend includes `comfy_input_files`, the worker stages those files into ComfyUI's input directory and patches the target nodes before submission.
5. The worker submits the raw workflow payload to ComfyUI and polls `/history/{prompt_id}` until the prompt is complete or times out.
6. In sync mode, the worker returns output file paths, timing breakdowns, and debug data directly. In async mode, the worker stores that final `RunResponse` and exposes it through `GET /jobs/{job_id}`.
7. Generated files can be fetched back over the worker's HTTP API for remote-GPU deployments.

## Files

- `app.py`: FastAPI app with `/health`, `/run`, `/jobs`, and `/files/...`.
- `schemas.py`: Request and response schemas.
- `config.py`: Environment-based settings.
- `asset_registry.py`: Static asset groups and model URLs.
- `asset_manager.py`: Asset existence checks, locking, download, and checksum verification.
- `comfy_process.py`: Basic ComfyUI health and restart logic.
- `comfy_client.py`: Prompt submission, polling, and output extraction.
- `utils.py`: Small file and shell helpers.

## Current MVP Behavior

- Asset groups are static and local to `asset_registry.py`.
- Existing non-empty model files are treated as ready.
- Missing files are downloaded with streaming requests into `*.part` temp files.
- A per-file `*.lock` file is used to avoid duplicate downloads.
- Downloads are atomically renamed into place on success.
- If any asset was downloaded, ComfyUI is restarted once.
- Prompt completion is detected from `/history/{prompt_id}` when outputs exist or ComfyUI marks the prompt complete.
- Errors return `ok=false` with partial timing/debug data instead of depending on exception-shaped HTTP behavior.

## Environment Variables

The worker reads these env vars:

- `COMFY_BASE_URL` default: `http://127.0.0.1:8188`
- `COMFY_HEALTH_TIMEOUT_SEC` default: `60`
- `MODEL_DOWNLOAD_TIMEOUT_SEC` default: `1800`
- `WORKER_HOST` default: `0.0.0.0`
- `WORKER_PORT` default: `9000`
- `COMFY_START_CMD` optional, but required if the worker needs to restart ComfyUI
- `COMFY_STOP_CMD` optional
- `DOWNLOAD_CHUNK_SIZE` default: `8388608`
- `COMFY_OUTPUT_DIR` default: `/workspace/ComfyUI/output`
- `COMFY_TEMP_DIR` optional additional served root for temp outputs
- `COMFY_INPUT_DIR` default: `/workspace/ComfyUI/input`

## Important Setup Notes

- The default `asset_registry.py` entries now match the Flux2 and WAN 2.2 workflow files in this repo and point at public Hugging Face model URLs.
- The default model paths in `asset_registry.py` assume a Vast-style ComfyUI layout under `/workspace/ComfyUI/models/...`. Change them if your local layout is different.
- `COMFY_START_CMD` and `COMFY_STOP_CMD` are plain shell commands. This is intentionally simple MVP process control.
- `/files/{root_name}/{relative_path}` only serves files under configured roots such as `COMFY_OUTPUT_DIR`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r gpu_worker/requirements.txt
```

## Deploy To Vast

You can now rent, deploy, and optionally warm a fresh Vast worker in one command:

```bash
./setup_gpu.sh --vast \
  --vast-gpu "RTX 4090" \
  --vast-max-price 0.70 \
  --warm-asset-group flux_stills_v1 \
  --warm-asset-group wan_i2v_v1
```

This flow:

- searches Vast offers using the provided GPU, VRAM, and price filters
- rents the best matching instance with `vastai/comfyui:latest`
- attaches your SSH key and waits for SSH readiness
- deploys `gpu_worker` onto the rented box
- opens the public worker URL
- optionally pre-downloads model asset groups before returning

If you already have a Vast box and just want to deploy to it manually:

```bash
./setup_gpu.sh ssh -i ~/.ssh/vast_deploy -p 22981 root@61.206.39.5 -L 8080:localhost:8080
```

This shell wrapper passes the SSH command through to `deploy_gpu.py`. If you need the Python flags, separate them with `--`:

```bash
./setup_gpu.sh ssh -p 2880 root@104.189.178.117 -L 8080:localhost:8080 -- --skip-backend-restart
```

What it does:

- parses the SSH command and ignores local port-forwards like `-L`
- copies this `gpu_worker` folder to `/workspace/filmforge_gpu_worker`
- creates or refreshes the remote `.venv`
- starts `uvicorn gpu_worker.app:app` on remote `127.0.0.1:9000`
- opens a Cloudflare quick tunnel when `cloudflared` exists on the remote box
- registers itself with FilmForge through `FILMFORGE_BACKEND_URL` and the render broker heartbeat
- leaves the legacy backend `GPU_WORKER_BASE_URL` untouched unless `--update-backend-env` is used

Useful flags:

- `--update-backend-env` for legacy local-dev single-worker routing
- `--skip-backend-restart`

## Deploy To Verda

Verda supports two deploy modes:

- existing volumes: boot a detached OS volume and attach a detached model/data volume
- fresh install: create new OS/model volumes, install ComfyUI + gpu_worker, and download models

```bash
./setup_gpu.sh --verda
./setup_gpu.sh --verda-fresh
```

Default Verda volume targets:

- OS volume: `34ec939d-a8c1-4ee2-9637-533e324dfe39`
- data/model volume: `4ea18b04-564f-4218-ab79-e90d1ccc839b`
- location: `FIN-01`
- instance type: `2A100.44V`

Useful overrides:

```bash
./setup_gpu.sh --verda \
  --verda-location FIN-01 \
  --verda-instance-type 2A100.44V \
  --verda-os-volume-id <os-volume-id> \
  --verda-data-volume-id <data-volume-id> \
  --verda-worker-count 2
```

Fresh install overrides:

```bash
./setup_gpu.sh --verda-fresh \
  --verda-location FIN-01 \
  --verda-instance-type 2A100.44V \
  --verda-fresh-os-volume-size 100 \
  --verda-fresh-storage-size 250 \
  --warm-asset-group flux_stills_v1 \
  --warm-asset-group wan_i2v_v1 \
  --warm-asset-group stable_audio_v1
```

The deploy UI exposes the same flow under **Create & Deploy (Verda)**. It uses
the mode selector to choose existing volumes or fresh install, streams the
Verda provisioning log, then stores the resulting SSH target in the Verda Hosts
list for log viewing.

This flow:

- checks Verda auth, volume status, and GPU availability
- requires both OS and data volumes to be detached before provisioning
- creates an on-demand VM from the OS volume and attaches the data volume
- mounts the data volume at `/mnt/data`
- starts one ComfyUI service and one GPU worker service per detected GPU
- prints public worker URLs such as `http://<verda-ip>:9000`
- registers workers with the backend when `FILMFORGE_BACKEND_URL` and the
  worker registration token are available
- sends rolling per-asset-group performance stats in broker heartbeats so the
  backend can estimate worker ETA

Teardown is intentionally not automated in this first phase. Verda VM deletion
has a volume-deletion footgun; safe detach/delete flow should be implemented as
a separate command before using automated teardown.

For split workers, set different capabilities per deployment:

```bash
--env WORKER_NAME=flux-worker-1 \
--env WORKER_CAPABILITIES=flux2_stills \
--env FILMFORGE_BACKEND_URL=http://<backend>:8000 \
--env WORKER_REGISTRATION_TOKEN=<shared-token>
```

```bash
--env WORKER_NAME=wan-worker-1 \
--env WORKER_CAPABILITIES=wan_i2v \
--env FILMFORGE_BACKEND_URL=http://<backend>:8000 \
--env WORKER_REGISTRATION_TOKEN=<shared-token>
```
- `--start-backend`

## Run Locally

```bash
export COMFY_BASE_URL=http://127.0.0.1:8188
export COMFY_START_CMD="./start"
export COMFY_STOP_CMD="./stop"
uvicorn gpu_worker.app:app --host 0.0.0.0 --port 9000
```

If you want the host/port to follow the environment variables directly:

```bash
uvicorn gpu_worker.app:app --host "${WORKER_HOST:-0.0.0.0}" --port "${WORKER_PORT:-9000}"
```

## API

### `GET /health`

Returns:

```json
{
  "worker_ok": true,
  "comfy_reachable": true,
  "comfy_base_url": "http://127.0.0.1:8188",
  "known_asset_groups": ["flux_stills_v1", "juggernaut_stills_v1", "stable_audio_v1", "wan_i2v_v1"]
}
```

### `POST /assets/ensure`

Preloads one or more asset groups without running a workflow. This is used by
the automated Vast deploy path so a newly rented worker can download models
before it starts taking real jobs.

```bash
curl -X POST http://127.0.0.1:9000/assets/ensure \
  -H "Content-Type: application/json" \
  -d '{"asset_groups":["flux_stills_v1","wan_i2v_v1"]}'
```

### `POST /run`

Example:

```bash
curl -X POST http://127.0.0.1:9000/run \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job_123",
    "asset_group": "flux_stills_v1",
    "timeout_sec": 1800,
    "poll_interval_sec": 2.0,
    "comfy_payload": {
      "3": {
        "class_type": "KSampler",
        "inputs": {}
      }
    }
  }'
```

Success response shape:

```json
{
  "ok": true,
  "job_id": "job_123",
  "asset_group": "flux_stills_v1",
  "downloaded_assets": [],
  "restart_performed": false,
  "comfy_prompt_id": "abc-123",
  "outputs": ["example.png"],
  "output_files": [
    {
      "path": "/workspace/ComfyUI/output/example.png",
      "filename": "example.png",
      "download_url": "/files/output/example.png"
    }
  ],
  "timings": {
    "asset_check_sec": 0.01,
    "download_sec": 0.0,
    "restart_sec": 0.0,
    "comfy_run_sec": 12.4,
    "total_sec": 12.41
  },
  "debug": {
    "history_found": true,
    "comfy_base_url": "http://127.0.0.1:8188"
  },
  "error": null
}
```

### `POST /jobs`

Accepts the same request body as `POST /run`, but returns quickly with a worker-side async job id:

```json
{
  "job_id": "8f72d1d19f7f4af9a6fd0f1783c6d27e",
  "status": "queued"
}
```

### `GET /jobs/{job_id}`

Polls the async job created by `POST /jobs`:

```json
{
  "job_id": "8f72d1d19f7f4af9a6fd0f1783c6d27e",
  "status": "completed",
  "result": {
    "ok": true,
    "job_id": "job_123",
    "asset_group": "flux_stills_v1",
    "outputs": ["example.png"],
    "output_files": [
      {
        "path": "/workspace/ComfyUI/output/example.png",
        "filename": "example.png",
        "download_url": "/files/output/example.png"
      }
    ],
    "error": null
  },
  "error": null
}
```

Optional request field for remote WAN or other worker-staged inputs:

```json
"comfy_input_files": [
  {
    "node_id": "97",
    "filename": "full_backend_still_test_00001_.png",
    "input_name": "image",
    "source_path": "/workspace/ComfyUI/output/full_backend_still_test_00001_.png"
  }
]
```

## What This Worker Does Not Handle Yet

- Authentication or request signing.
- Remote job queues or concurrency limits across jobs.
- Model version negotiation beyond the static asset group mapping.
- Download retries, resumable downloads, or bandwidth throttling.
- Rich output typing beyond best-effort file path extraction.
- Production-grade ComfyUI lifecycle management through systemd, Docker, Kubernetes, or supervisor.
- Workflow validation or any FilmForge cinematic/planning logic.

## Remote Worker Output Bridge

The worker now exposes generated files over HTTP through:

- `GET /files/{root_name}/{relative_path}`

The backend prefers async `POST /jobs` plus `GET /jobs/{job_id}` for remote workers and tunnels, because long WAN renders can outlive a single proxied HTTP request. It falls back to sync `POST /run` for older or same-machine workers.

The backend uses the `output_files` metadata returned by the final worker response, downloads each file from the worker, stores it in a backend-local temp directory, and then continues with the existing upload/storage flow using that local path.

This is the MVP bridge that makes a worker on a separate GPU host usable without redesigning the rest of the render pipeline.

For worker-backed WAN jobs, the backend can also send `comfy_input_files` in the `/run` request. The worker stages those source stills into `COMFY_INPUT_DIR` and rewrites the matching `LoadImage` nodes before prompt submission, which removes the backend's need for direct access to the worker's ComfyUI `/upload/image` API.

The same staging path is also used for Juggernaut IP-Adapter stills. Portrait refs are staged into `COMFY_INPUT_DIR/characters` on the worker, so Juggernaut still generation can run remotely without backend access to the worker's ComfyUI upload API.

## Next Real-World TODOs

- Replace the default public model URLs if you want to use private mirrors or pre-staged internal artifacts.
- Replace the default model paths if your ComfyUI models directory differs.
- Set real `COMFY_START_CMD` and `COMFY_STOP_CMD` values for the target machine.
- Decide whether existing files should also be checksum-verified on startup or only on fresh download.
- Move from backend pull-downloads to direct worker uploads or object-storage handoff if you want to reduce cross-host file copying.
- Decide whether audio generation should continue to be mixed and uploaded by the backend or eventually move to a full worker-side music artifact pipeline.
