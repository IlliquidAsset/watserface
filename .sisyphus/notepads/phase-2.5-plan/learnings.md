# Phase 2.5 Learnings

## Conventions and Patterns

*Accumulated knowledge from task execution will be appended here.*

---

## Point Tracking Research (Task 1 - Completed Jan 26, 2026)

### Key Findings

**Recommendation:** CoTracker3 (Meta AI, 2024) as primary solution, TAPIR as fallback.

**Performance Comparison (TAP-Vid DAVIS AJ Score):**
- CoTracker3: **73.2%** (SOTA)
- BootsTAPIR: 67.8%
- LocoTrack: 68.1%
- TAPIR: 62.9%
- PIPs: 42.0%

**Why CoTracker3:**
1. Best occlusion handling (critical for mayo/glasses)
2. Simplest architecture (fewer parameters than predecessors)
3. 10% better than TAPIR on real videos
4. Active Meta AI support
5. Tracks 10k+ points simultaneously
6. 60-frame temporal window (eliminates flicker)

**Integration Pattern:**
```python
from cotracker.predictor import CoTrackerPredictor

tracker = CoTrackerPredictor(checkpoint="cotracker3.pth", window_len=60)
tracks, visibility = tracker(video, queries=landmark_points)
# tracks: (T, N, 2) - trajectories
# visibility: (T, N) - occlusion masks
```

**Performance Targets:**
- Offline mode: 2-5 fps (60-frame window)
- Online mode: 15-20 fps (16-frame window)
- Memory: ~4GB VRAM (1080p)

**Not Recommended:**
- OmniMotion: Too slow (10-30 min/video, test-time optimization)
- PIPs: Superseded by TAPIR/CoTracker3

**Next Steps:**
1. Install CoTracker3 and test on mayo frame
2. Integrate with Depth-Anything V2 for temporal consistency
3. Create motion fields for ControlNet conditioning
4. Validate temporal coherence (no flicker)

**Full Report:** `docs/research/point-tracking-selection.md`

---

## ControlNet Pipeline Setup (Task 2 - Already Complete)

### Discovery
Task 2 was already implemented in a previous session:
- File: `watserface/inpainting/controlnet.py` (411 lines)
- Tests: `tests/test_controlnet_pipeline.py` (21 tests, all passing)
- Status: ✅ COMPLETE

### Test Results
```bash
pytest tests/test_controlnet_pipeline.py -v
========================= 21 passed, 1 warning in 6.59s =========================
```

### Implementation Details
- ControlNet pipeline class with depth and canny conditioning
- Auto device detection (MPS/CUDA/CPU)
- Default conditioning scale: 0.75 (configurable)
- Model IDs: `diffusers/controlnet-depth-sdxl-1.0-small`, `diffusers/controlnet-canny-sdxl-1.0`
- Output size: 512x512
- Factory function: `create_controlnet_pipeline()`

### Pattern Learned
**Check for existing implementations before delegating.** This is the second task found already complete (after foundation fixes). Previous sessions have done significant work.

---
