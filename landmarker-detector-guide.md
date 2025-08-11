# Detector & Landmarker Model Guide

*Last updated: Aug 6, 2025*

**Audience:** Claude Code (implementation). This supplements the main PRD without changing it.

## Goal

Select the best **face detector** and **landmarker** per scenario to improve preview/export quality and stability. Provide auto‑selection logic, configs, and tests.

---

## Models in repo (baseline)

* **Detectors:** `yoloface_8n.onnx` (very fast, small faces moderate)
* **Landmarkers:** `fan_68_5.onnx` (fast 68-pt), `2dfan4.onnx` (slower, more precise)
* **Related:** `bisenet_resnet_34.onnx` (segmentation/mask), `xseg_1.onnx` (alt mask), `arcface_w600k_r50.onnx` (embeddings), `fairface.onnx` (demographics classifier; optional)

> Optional additions (future‑friendly): **SCRFD‑2.5G / 10G**, **RetinaFace‑R50**, **YOLOv8n‑face**, **MediaPipe FaceMesh (468 pts)**. Wire them as pluggable providers; only `yoloface_8n` and FAN models are required for Phase 4.

---

## When to use which (quick matrix)

| Scenario                         | Detector                | Landmarker       | Notes                                        |        |                                          |
| -------------------------------- | ----------------------- | ---------------- | -------------------------------------------- | ------ | ---------------------------------------- |
| **Single, frontal, ≥256px face** | YOLOFace‑8n             | FAN‑68‑5         | Fast path; good default                      |        |                                          |
| **Small/distant faces (<160px)** | SCRFD‑10G\*             | 2DFAN4           | Better recall of tiny faces                  |        |                                          |
| \*\*Profile/tilted (             | yaw                     | >30°)\*\*        | RetinaFace‑R50\*                             | 2DFAN4 | Robust to pose; more landmarks help mask |
| **Crowded multi‑face scene**     | SCRFD‑2.5G\*            | FAN‑68‑5         | Many boxes; speed/recall balance             |        |                                          |
| **Webcam/real‑time**             | BlazeFace/YOLOFace‑8n\* | FAN‑68‑5         | Low latency                                  |        |                                          |
| **High‑fidelity masking**        | YOLOFace‑8n             | 2DFAN4 + BiSeNet | Landmarks guide mask; refine by segmentation |        |                                          |

\* Optional addition; if unavailable, fall back to YOLOFace‑8n + 2DFAN4 for the listed rows.

---

## Auto‑selection logic (Phase 4 scope)

**Inputs:** target frame size, detected face sizes, pose estimate (from landmarks), face count, job preset.

1. **Detector choice**

```text
If preset == "Fast" or realtime: detector = yoloface_8n
Else if min_face_px < 160 and SCRFD available: detector = scrfd_10g
Else if expected_profile and RetinaFace available: detector = retinaface_r50
Else: detector = yoloface_8n
```

2. **Landmarker choice**

```text
If high_fidelity_mask or pose>|30°|: landmarker = 2dfan4
Else: landmarker = fan_68_5
```

3. **Retry/Ensemble (lightweight)**

* If no faces found: retry with **lower threshold** and/or alternate detector (if present).
* If mask quality score < threshold (see below): re‑landmark with **2DFAN4** and recompute mask.

---

## Quality scoring & thresholds

**Face quality (per detection):**

* Size ≥ `min_face_px` (default 256)
* Blur: Variance of Laplacian ≥ `blur_var_min` (default 100.0)
* Pose: |yaw|,|pitch| ≤ `pose_max` (defaults 30°), else penalize
* Occlusion proxy: % of landmark points inside detector box ≥ 0.9

**Mask quality (for swap):**

* IoU(mask vs. convex‑hull(landmarks)) ≥ `mask_iou_min` (default 0.7)
* Edge smoothness: mean gradient ≤ `edge_grad_max` (prevents jagged edges)

If a score fails, switch to **2DFAN4** (for landmarks) or **alt detector** (if available), then recompute.

---

## Config keys (extend `facefusion.ini`)

```ini
[detector]
preferred = yoloface_8n         ; default
fallback = auto                 ; auto / scrfd_10g / retinaface_r50 / none
score_threshold = 0.3
min_face_px = 256
retry_lower_thresh = 0.15

[landmarker]
preferred = fan_68_5
hi_fidelity = 2dfan4            ; used when needed by logic
pose_max_deg = 30

[masking]
use_seg_refine = true           ; BiSeNet/XSeg refinement
mask_iou_min = 0.7
edge_grad_max = 25.0

[realtime]
enabled = false
max_latency_ms = 50
```

> Keep defaults aligned with current behavior when optional models are not present.

---

## Pipeline hooks (where to wire)

* `facefusion/face_detector.py`: add registry + factory, thresholds, retry path.
* `facefusion/face_landmarker.py`: expose both FAN variants; implement pose estimation.
* `facefusion/face_masker.py`: accept landmark set; optional seg‑refine step.
* `facefusion/uis/components/face_detector.py` & `face_landmarker.py`: add dropdowns + “Auto” badges.
* `facefusion/inference_manager.py`: provider capabilities (CPU/GPU), model warm‑up.

---

## Minimal pseudocode

```python
def choose_models(cfg, scene_info):
    det = cfg.detector.preferred
    if cfg.preset == 'fast' or cfg.realtime.enabled:
        det = 'yoloface_8n'
    elif scene_info.min_face_px < cfg.detector.min_face_px and have('scrfd_10g'):
        det = 'scrfd_10g'
    elif scene_info.expect_profile and have('retinaface_r50'):
        det = 'retinaface_r50'

    lmk = cfg.landmarker.preferred
    if cfg.masking.use_seg_refine or scene_info.pose_exceeds(cfg.landmarker.pose_max_deg):
        lmk = '2dfan4'
    return det, lmk
```

---

## UI/UX notes

* Detector & Landmarker sections show **Actual in use** (badge) when Auto overrides user pick.
* Tooltip: short pros/cons for each model (speed/accuracy/pose/tiny‑face).
* “Re‑analyze with high fidelity landmarks” button in Preview if mask score is low.

---

## Testing & Benchmarks

**Datasets (internal, non‑sensitive):**

* Small faces set (≤120px)
* Profile/large pose set (yaw 30–60°)
* Crowded scenes (≥5 faces)
* Standard frontal set (≥256px)

**Metrics:**

* Detection recall/precision @ IoU 0.5
* Landmark NME (normalized mean error) on annotated subset
* Mask IoU vs. landmark hull; swap boundary artifacts (heuristic score)
* Throughput (fps) and latency (p95)

**Acceptance:**

* Auto‑selection improves either recall or mask IoU ≥ **+5%** vs. fixed defaults on the relevant set, with ≤ **+20%** latency hit in non‑realtime mode.

---

## Adding a new model (checklist)

1. Drop `.onnx` under `models/detectors/` or `models/landmarkers/` + update registry.
2. Add loader in `inference_manager.py` with provider compatibility.
3. Map config string → model; update UI dropdown.
4. Provide default thresholds; add to tooltip.
5. Add to tests and benchmark harness.

---

## Future extensions

* **Model ensemble:** run two detectors; keep union with NMS.
* **FaceMesh (468‑pt) path:** enable dense landmarks for expression‑sensitive modules.
* **Domain robustness:** periodic eval across varied lighting/skin tones/age to ensure consistent performance.

---

## Deliverables

* Auto‑selection implemented with fallbacks.
* Config keys honored; Effective Config shows chosen models per job.
* UI reflects actual runtime choices.
* Bench scripts + report markdown in `benchmarks/`.
