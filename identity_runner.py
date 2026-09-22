"""Identity-kit runner: ArcFace scoring and person masks, run INSIDE the identity venv.

Standalone on purpose -- it imports nothing from gpu_worker, because it runs under the
identity ComfyUI's Python (insightface / ultralytics live there, not in the worker venv).
The worker calls it as a subprocess, one request per process, so no model stays resident
between requests (lead ruling 2026-09-22: lazy loads, cap host RAM). Score ALL of a take's
frames in one request; a per-frame call would reload ArcFace every time.

    python identity_runner.py < request.json > reply.json

request: {"mode": "score", "models_root": ".../models/insightface", "anchor": "a.png",
          "images": [...] | "video": "t.mp4", "sample_fps": 2, "max_frames": 48,
          "threshold": 0.70, "min_face_px": 40}
         {"mode": "segment", "video": "in.mp4", "out": "mask.mp4", "yolo": ".../yolov8x-seg.pt",
          "roi": [x0,y0,x1,y1] | null, "n": 2, "dilate_px": 25}

Scorer = insightface buffalo_l (w600k_r50), deliberately NOT antelopev2 / glintr100, which
InfiniteYou and PuLID condition on (scoring with the same net flatters them). Thresholds are
per person and belong to the caller's identity record, not to this runner. Score against a
REAL photo (or a Director-accepted likeness), never a generated anchor: a drifted anchor
certifies the drift (2026-09-22). A pass is a floor check, never acceptance.
LICENCES: insightface models non-commercial; ultralytics AGPL-3.0.
"""
from __future__ import annotations

import json
import sys

SCORER = "insightface/buffalo_l/w600k_r50"
LICENCE = "non-commercial (insightface models); AGPL-3.0 (ultralytics)"


def _face_app(models_root: str):
    from insightface.app import FaceAnalysis

    import onnxruntime

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in onnxruntime.get_available_providers()]
    # ctx_id=0 is the FIRST VISIBLE device. The worker pins CUDA_VISIBLE_DEVICES to the
    # identity GPU before spawning this process, so 0 here is never the production card.
    app = FaceAnalysis(name="buffalo_l", root=models_root, providers=providers,
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    return app


def _frames(req: dict):
    import cv2

    if req.get("images"):
        for i, path in enumerate(req["images"]):
            img = cv2.imread(path)
            if img is not None:
                yield i, img
        return
    cap = cv2.VideoCapture(req["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(fps / float(req.get("sample_fps", 2))))
    k = taken = 0
    while taken < int(req.get("max_frames", 48)):
        ok, img = cap.read()
        if not ok:
            break
        if k % step == 0:
            yield k, img
            taken += 1
        k += 1


def _yaw_proxy(kps) -> float:
    """Signed head-turn proxy from the detector's 5 landmarks (eyes, nose, mouth corners):
    the nose's horizontal offset from the eye midpoint over the inter-eye distance.
    ~0 = frontal; |x| grows as the head turns (~0.5+ is a strong three-quarter). Not an
    angle -- compare it within a take (a jump vs the first frame = a turn to/away from camera)."""
    (lx, ly), (rx, ry), (nx, ny) = kps[0], kps[1], kps[2]
    eye_dist = max(abs(rx - lx), 1e-6)
    return round(float((nx - (lx + rx) / 2) / eye_dist), 3)


def _unit(v):
    import numpy as np

    return v / (np.linalg.norm(v) + 1e-9)


def score(req: dict) -> dict:
    import cv2
    import numpy as np

    app = _face_app(req["models_root"])
    threshold = float(req["threshold"])
    min_px = int(req.get("min_face_px", 40))
    anchor_img = cv2.imread(req["anchor"])
    anchor_faces = app.get(anchor_img) if anchor_img is not None else []
    if not anchor_faces:
        return {"scorer": SCORER, "licence": LICENCE, "error": "anchor_face_not_detected"}
    anchor = _unit(max(anchor_faces, key=lambda f: f.bbox[2] - f.bbox[0]).embedding)

    frames = []
    for index, img in _frames(req):
        faces = []
        for f in app.get(img):
            px = int(round(f.bbox[2] - f.bbox[0]))
            faces.append({"bbox": [round(float(x), 1) for x in f.bbox], "det_score": round(float(f.det_score), 3),
                          "face_px": px, "kps": [[round(float(a), 1), round(float(b), 1)] for a, b in f.kps],
                          "yaw_proxy": _yaw_proxy(f.kps),
                          "cosine": round(float(np.dot(anchor, _unit(f.embedding))), 4) if px >= min_px else None})
        scorable = [f["cosine"] for f in faces if f["cosine"] is not None]
        reason = None if scorable else ("too_small" if faces else "not_detected")
        best = max(scorable) if scorable else None
        frames.append({"frame": index, "faces": faces, "best_cosine": best, "threshold": threshold,
                       "passed": None if best is None else best >= threshold, "no_face_reason": reason})

    reply = {"scorer": SCORER, "licence": LICENCE, "threshold": threshold, "min_face_px": min_px}
    if req.get("images") and len(frames) == 1:
        reply.update(frames[0])
        return reply
    bests = [f["best_cosine"] for f in frames if f["best_cosine"] is not None]
    reply.update({"frames": frames, "frames_sampled": len(frames), "frames_scored": len(bests),
                  "min": min(bests) if bests else None,
                  "median": float(np.median(bests)) if bests else None,
                  "best_cosine": max(bests) if bests else None})
    return reply


def segment(req: dict) -> dict:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(req["yolo"])
    cap = cv2.VideoCapture(req["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w, h = int(cap.get(3)), int(cap.get(4))
    x0, y0, x1, y1 = req.get("roi") or (0, 0, w, h)
    n, dil = int(req.get("n", 2)), int(req.get("dilate_px", 25))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
    out = cv2.VideoWriter(req["out"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    counts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, classes=[0], conf=0.25, verbose=False, retina_masks=True, device=0
                          if _cuda() else "cpu")[0]
        mask = np.zeros((h, w), np.uint8)
        cands = []
        if r.masks is not None:
            for mk, box in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    cands.append((float(mk.sum()), mk))
        cands.sort(key=lambda t: -t[0])
        for _, mk in cands[:n]:
            mask |= (cv2.resize(mk, (w, h)) > 0.5).astype(np.uint8) * 255
        out.write(cv2.dilate(mask, kernel))
        counts.append(min(len(cands), n))
    out.release()
    return {"out": req["out"], "frames": len(counts), "frames_with_all": sum(c >= n for c in counts),
            "frames_with_none": counts.count(0), "licence": LICENCE}


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> int:
    req = json.load(sys.stdin)
    mode = req.get("mode")
    reply = score(req) if mode == "score" else segment(req) if mode == "segment" else {"error": f"unknown mode {mode!r}"}
    json.dump(reply, sys.stdout)
    return 0 if "error" not in reply else 2


if __name__ == "__main__":
    sys.exit(main())
