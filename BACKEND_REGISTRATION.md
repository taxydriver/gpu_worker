# GPU Worker Backend Registration

The GPU worker now automatically registers itself with the Filmforge backend when it starts. This enables dynamic discovery and failover without restarting the backend.

## How It Works

1. **GPU worker starts** → sends `POST /api/gpu-workers/register` to backend
2. **Backend stores worker info** in `gpu_workers` table
3. **GPU worker sends heartbeat** every 5 minutes to stay marked as active
4. **Backend dispatcher** queries active workers and routes jobs

## Configuration

Environment variables (set these where you deploy the GPU worker):

```bash
# Required:
FILMFORGE_BACKEND_URL=http://localhost:8000  # or https://filmforge-backend.fly.dev

# Optional (auto-detected if not set):
WORKER_NAME=local-gpu                        # unique identifier (defaults to hostname)
WORKER_PUBLIC_URL=http://localhost:9000      # public URL of this worker
REGISTER_WITH_BACKEND=true                   # disable with "false"
HEARTBEAT_INTERVAL_SEC=300                   # send heartbeat every N seconds
```

## Local Development

### 1. Start backend
```bash
cd Filmforge/backend
python -m uvicorn app.main:app --reload
```

### 2. Start GPU worker (registration automatic)
```bash
export FILMFORGE_BACKEND_URL=http://localhost:8000
export WORKER_NAME=local-gpu
export WORKER_PUBLIC_URL=http://localhost:9000

cd gpu_worker
python -m uvicorn app:app --host 0.0.0.0 --port 9000
```

### 3. Verify registration
```bash
curl http://localhost:8000/api/gpu-workers/active
```

You should see your worker registered and marked as active.

## Production (RunPod)

### Option 1: Environment variables in deploy script

In your `deploy_gpu.py` or startup script:

```bash
#!/bin/bash
export FILMFORGE_BACKEND_URL=https://filmforge-backend.fly.dev
export WORKER_NAME=runpod-$(date +%s)
export WORKER_PUBLIC_URL=https://${RUNPOD_POD_ID}.runpod.io
export HEARTBEAT_INTERVAL_SEC=300

python -m uvicorn gpu_worker.app:app --host 0.0.0.0 --port 9000
```

### Option 2: RunPod Pod Environment

Set in RunPod pod environment variables:

```
FILMFORGE_BACKEND_URL=https://filmforge-backend.fly.dev
WORKER_NAME=runpod-flux-workers
WORKER_PUBLIC_URL=https://{RUNPOD_POD_ENDPOINT}.runpod.io
REGISTER_WITH_BACKEND=true
HEARTBEAT_INTERVAL_SEC=300
```

The worker will auto-register when the pod starts.

### Option 3: Via RunPod SSH deployment

If using `deploy_gpu.py` to SSH into a RunPod instance:

```bash
python gpu_worker/deploy_gpu.py \
    --ssh-dest "root@ssh.runpod.io" \
    --runpod-pod-id xyz123 \
    --register-with-backend \
    --backend-url https://filmforge-backend.fly.dev
```

(You may need to enhance `deploy_gpu.py` to pass these env vars to the remote box.)

## Troubleshooting

### Worker not registering

**Check logs:**
```bash
# Local:
grep "\[backend\]" /tmp/filmforge_gpu_worker.log

# RunPod: SSH in and check
ssh -i ~/.ssh/runpod root@[pod-ip]
tail -f gpu_worker.log | grep "\[backend\]"
```

**Common issues:**
- `FILMFORGE_BACKEND_URL` not set → registration disabled
- Backend is down → "Registration failed: Connection refused"
- Network unreachable → "Registration error: HTTPConnectionError"

### Worker marked as inactive

The backend auto-detects stale workers:
- If heartbeat is >10 minutes old, worker is considered inactive
- Or you can manually deactivate:
  ```bash
  curl -X POST http://localhost:8000/api/gpu-workers/my-worker/set-inactive
  ```

### Multiple workers not load balancing

The backend currently returns the first active worker. For load balancing:

1. **Query all workers:**
   ```bash
   curl http://localhost:8000/api/gpu-workers/
   ```

2. **Implement selection in backend dispatcher** (future enhancement):
   - Round-robin: pick `workers[job_count % len(workers)]`
   - Least-loaded: query `jobs` table and pick worker with fewest active jobs
   - Random: `random.choice(workers)`

## Disabling Registration

If you want to keep using `GPU_WORKER_BASE_URL` env var instead:

```bash
export REGISTER_WITH_BACKEND=false
export GPU_WORKER_BASE_URL=http://localhost:9000
```

The backend will fall back to the env var if no DB worker is found.

## API Reference

### Register GPU Worker
```
POST /api/gpu-workers/register
{
  "name": "local-gpu",
  "base_url": "http://localhost:9000",
  "is_active": true
}
```

### Get Active Worker
```
GET /api/gpu-workers/active
```

### List All Workers
```
GET /api/gpu-workers/
```

### Send Heartbeat
```
POST /api/gpu-workers/{worker_name}/heartbeat
```

### Mark Inactive
```
POST /api/gpu-workers/{worker_name}/set-inactive
```

## Future Enhancements

- [ ] Health check endpoint to detect stale workers automatically
- [ ] Load balancing (round-robin, least-loaded, etc.)
- [ ] Worker metrics (jobs completed, avg time, errors)
- [ ] Automatic failover (retry job on different worker if one fails)
- [ ] GPU worker can call backend to deregister on graceful shutdown
