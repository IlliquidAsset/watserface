# Foundation Fix Plan: Video Swap Quality

## Context

### Blocker for Phase 2.5
Phase 2.5 (DKT/ControlNet transparency handling) is blocked by critical quality issues in the base video swap pipeline. User feedback: "Not a one of these video swaps is usable."

### Root Cause Analysis
Six critical issues identified through comprehensive testing:
1. Mouth interior detection broken (8.2px vs ~30px expected)
2. Face size jitter severe (22-30% variance frame-to-frame)
3. Eye quality degradation (SimSwap 512 worse than InSwapper 128)
4. Quality metrics inadequate (can't detect issues)
5. Temporal smoothing insufficient (only masks, not landmarks/bboxes)
6. XSeg cheekbone masking needs verification

### Why This Blocks Phase 2.5
- **DKT won't fix landmarker bugs**: Tracking wrong points accurately doesn't help
- **ControlNet won't fix jitter**: Generative inpainting assumes stable input
- **Must validate foundation**: Prove basic pipeline works before adding complexity

---

## Work Objectives

### Core Objective
Fix critical quality issues in video face swapping to achieve usable output before proceeding with Phase 2.5 transparency handling.

### Concrete Deliverables
- Mouth interior detection achieving ~30px opening measurement (vs current 8.2px)
- Face size variance reduced to <5% frame-to-frame (vs current 22-30%)
- Eye quality matching or exceeding InSwapper 128 baseline
- Quality metrics detecting perceptual issues (LPIPS, temporal consistency)

### Definition of Done
- [ ] All 4 tasks completed with passing verification
- [ ] User validation: "Video swaps are now usable"
- [ ] Metrics show: <5% jitter, >0.85 LPIPS, >0.9 temporal consistency
- [ ] Phase 2.5 unblocked and ready to proceed

### Must Have
- Backward compatibility with existing swap pipeline
- No breaking changes to API or CLI
- Quantitative metrics for each fix
- User validation before marking complete

### Must NOT Have (Guardrails)
- No new model dependencies without justification
- No performance regression (must maintain 30fps target)
- No modification of Phase 2.5 plan (keep it intact)
- No scope creep into Phase 2.5 features

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest in `tests/`)
- **User wants tests**: YES (TDD for objective metrics, manual QA for subjective quality)
- **Framework**: pytest

### TDD for Objective Metrics
Each task with measurable outcomes follows RED-GREEN-REFACTOR:
1. **RED**: Write failing test first
2. **GREEN**: Implement minimum code to pass
3. **REFACTOR**: Clean up while keeping green

### Manual QA for Subjective Quality
**CRITICAL**: User must validate visual quality for each fix.

**Evidence Required**:
- Before/after comparison videos
- Metric measurements (jitter %, LPIPS score, etc.)
- User confirmation: "This is now usable"

---

## Task Flow

```
#1 (Mouth Fix) ──┐
                 ├──> #4 (Quality Metrics) ──> User Validation ──> Unblock Phase 2.5
#2 (Jitter Fix) ─┤
                 │
#3 (Eye Fix) ────┘
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| A | #1, #2, #3 | Independent fixes, can develop in parallel |

| Task | Depends On | Reason |
|------|------------|--------|
| #4 | #1, #2, #3 | Metrics validate all fixes together |

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> Specify parallelizability for EVERY task.

- [ ] 1. Fix Mouth Interior Detection

  **What to do**:
  - Replace inner lip landmark-based mouth detection with outer lip polygon + inward expansion
  - Implement brightness/edge detection to find true mouth cavity opening
  - Target: Detect ~30px mouth opening (vs current 8.2px)
  - Update `watserface/face_masker.py` mouth interior mask generation
  - Test on frame 72 of zBambola.mp4 (corndog eating scene)

  **Must NOT do**:
  - Do not modify XSeg model or training pipeline
  - Do not change landmark detection model (2dfan4) - work around its limitations
  - Do not break existing mask types (box, region, etc.)

  **Parallelizable**: YES (with #2, #3)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `watserface/face_masker.py:147-154` - Current mouth interior mask logic (BROKEN)
  - `watserface/face_masker.py:100-120` - Existing mask blending patterns

  **API/Type References** (contracts to implement against):
  - 2dfan4 landmarks: Points 48-59 (outer lip), 60-67 (inner lip - BROKEN)
  - OpenCV contour/polygon operations for expansion

  **Test References** (testing patterns to follow):
  - `test_quality/mouth_size_analysis_detailed.png` - Shows current 8.2px bug
  - `test_quality/overlay_mouth_interior.png` - Visual evidence of small mask

  **Documentation References** (specs and requirements):
  - `test_quality/ANALYSIS_FINDINGS.md:1-18` - Mouth detection analysis
  - `.sisyphus/notepads/phase-2.5-plan/problems.md:16-20` - Root cause

  **External References** (libraries and frameworks):
  - OpenCV `cv2.convexHull()`, `cv2.dilate()` for polygon expansion
  - NumPy for geometric calculations

  **WHY Each Reference Matters**:
  - `face_masker.py` shows current broken logic - must replace, not patch
  - Test images provide ground truth for validation
  - Analysis docs explain why inner lip landmarks fail (detect edge, not cavity)

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_mouth_detection.py`
  - [ ] Test covers: Mouth opening measurement on frame 72 (target: 25-35px range)
  - [ ] `pytest tests/test_mouth_detection.py` → PASS

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using Python script:
    ```bash
    python test_swap_quality.py --video .assets/examples/zBambola.mp4 --frames 72 --output test_quality/mouth_fix_verification.png
    ```
    Expected: Mouth opening measurement 25-35px (vs 8.2px baseline)
  - [ ] Verify state:
    - Navigate to: `test_quality/mouth_fix_verification.png`
    - Action: Open image, check mouth mask overlay
    - Verify: Mask covers corndog inside mouth, not just lip line
    - Screenshot: Save to `.sisyphus/evidence/task-1-mouth-fix.png`

  **Evidence Required:**
  - [ ] Command output with measurement (copy-paste)
  - [ ] Before/after comparison image
  - [ ] User confirmation: "Mouth mask now covers food properly"

  **Commit**: YES
  - Message: `fix(masker): correct mouth interior detection using outer lip polygon`
  - Files: `watserface/face_masker.py`, `tests/test_mouth_detection.py`
  - Pre-commit: `pytest tests/test_mouth_detection.py`

- [ ] 2. Stabilize Face Size Jitter

  **What to do**:
  - Implement temporal stabilization for face bounding boxes and landmarks
  - Add exponential moving average (EMA) for bounding box coordinates
  - Add Savitzky-Golay filter for landmark smoothing (window=5, polyorder=2)
  - Target: Reduce face size variance to <5% frame-to-frame (vs current 22-30%)
  - Update `watserface/face_detector.py` and `watserface/face_helper.py`

  **Must NOT do**:
  - Do not change face detection model (YOLO)
  - Do not add optical flow (save for later if needed)
  - Do not break single-frame processing mode

  **Parallelizable**: YES (with #1, #3)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `watserface/face_masker.py:60-75` - Existing XSeg temporal smoothing (5-frame window)
  - `watserface/processors/modules/face_swapper.py` - Video frame processing loop

  **API/Type References** (contracts to implement against):
  - `scipy.signal.savgol_filter()` for landmark smoothing
  - NumPy for EMA calculation: `smoothed = alpha * current + (1-alpha) * previous`

  **Test References** (testing patterns to follow):
  - `test_quality/ANALYSIS_FINDINGS.md:20-37` - Jitter measurements and analysis

  **Documentation References** (specs and requirements):
  - `.sisyphus/notepads/phase-2.5-plan/problems.md:22-26` - Jitter blocker details

  **External References** (libraries and frameworks):
  - SciPy `savgol_filter` documentation
  - Temporal smoothing best practices for video processing

  **WHY Each Reference Matters**:
  - Existing XSeg temporal smoothing shows pattern to follow (5-frame window works)
  - Analysis docs provide exact measurements to validate against (<5% target)
  - SciPy provides proven smoothing algorithms

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_temporal_stabilization.py`
  - [ ] Test covers: Face size variance <5% on 30-frame sequence
  - [ ] `pytest tests/test_temporal_stabilization.py` → PASS

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using test script:
    ```bash
    python test_swap_quality.py --video .assets/examples/zBambola.mp4 --frames 30 --measure-jitter --output test_quality/jitter_fix_verification.mp4
    ```
    Expected: Face size variance <5% (vs 22-30% baseline)
  - [ ] Verify state:
    - Navigate to: `test_quality/jitter_fix_verification.mp4`
    - Action: Play video, observe face stability
    - Verify: No visible "breathing" or size flashing
    - Screenshot: Save key frames to `.sisyphus/evidence/task-2-jitter-fix.png`

  **Evidence Required:**
  - [ ] Command output with variance measurement (copy-paste)
  - [ ] Before/after comparison video
  - [ ] User confirmation: "Face size is now stable"

  **Commit**: YES
  - Message: `fix(detector): add temporal stabilization for bboxes and landmarks`
  - Files: `watserface/face_detector.py`, `watserface/face_helper.py`, `tests/test_temporal_stabilization.py`
  - Pre-commit: `pytest tests/test_temporal_stabilization.py`

- [ ] 3. Hybrid Eye Quality Fix

  **What to do**:
  - Implement hybrid swapper: InSwapper 128 for eye regions, SimSwap 512 for rest of face
  - Use landmarks 36-47 (eyes) to define eye region boundaries
  - Blend at boundaries using Gaussian blur (sigma=3)
  - Target: Eye quality matching InSwapper 128, face texture from SimSwap 512
  - Create new swapper option: `hybrid_inswapper_simswap`

  **Must NOT do**:
  - Do not modify existing swapper models (inswapper_128, simswap_512)
  - Do not add new model dependencies
  - Do not break existing swapper selection

  **Parallelizable**: YES (with #1, #2)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `watserface/processors/modules/face_swapper.py:100-150` - Existing swap logic
  - `watserface/face_masker.py:100-120` - Mask blending patterns

  **API/Type References** (contracts to implement against):
  - 2dfan4 landmarks: Points 36-41 (left eye), 42-47 (right eye)
  - OpenCV `cv2.GaussianBlur()` for boundary blending

  **Test References** (testing patterns to follow):
  - `test_quality/eye_quality_comparison.png` - Shows InSwapper vs SimSwap eyes
  - `test_quality/ANALYSIS_FINDINGS.md:40-54` - Eye quality analysis

  **Documentation References** (specs and requirements):
  - `.sisyphus/notepads/phase-2.5-plan/problems.md:28-32` - Eye quality blocker

  **External References** (libraries and frameworks):
  - OpenCV image blending techniques
  - Alpha compositing: `result = A * alpha + B * (1-alpha)`

  **WHY Each Reference Matters**:
  - Existing swapper code shows how to invoke models and blend results
  - Eye comparison image proves InSwapper has better perceptual quality
  - Mask blending patterns provide template for eye region blending

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_hybrid_swapper.py`
  - [ ] Test covers: Eye region uses InSwapper, face uses SimSwap
  - [ ] `pytest tests/test_hybrid_swapper.py` → PASS

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using test script:
    ```bash
    python test_swap_quality.py --video .assets/examples/zBambola.mp4 --frames 1 --swapper hybrid_inswapper_simswap --output test_quality/hybrid_eye_verification.png
    ```
    Expected: Eyes match InSwapper quality, face has SimSwap texture
  - [ ] Verify state:
    - Navigate to: `test_quality/hybrid_eye_verification.png`
    - Action: Open image, zoom to eyes at 3x
    - Verify: Eyes look natural (like InSwapper), no over-sharpening
    - Screenshot: Save to `.sisyphus/evidence/task-3-eye-fix.png`

  **Evidence Required:**
  - [ ] Command output (copy-paste)
  - [ ] Before/after eye comparison at 3x zoom
  - [ ] User confirmation: "Eyes now look natural"

  **Commit**: YES
  - Message: `feat(swapper): add hybrid InSwapper/SimSwap for better eye quality`
  - Files: `watserface/processors/modules/face_swapper.py`, `tests/test_hybrid_swapper.py`
  - Pre-commit: `pytest tests/test_hybrid_swapper.py`

- [ ] 4. Implement Quality Metrics

  **What to do**:
  - Add LPIPS (Learned Perceptual Image Patch Similarity) for perceptual quality
  - Add temporal consistency score using optical flow
  - Add facial feature alignment check (eye/mouth position accuracy)
  - Integrate into `test_swap_quality.py` for automated validation
  - Target: LPIPS >0.85, temporal consistency >0.9

  **Must NOT do**:
  - Do not add metrics that require ground truth (we don't have it)
  - Do not slow down processing (metrics are for testing only)
  - Do not break existing diff_mean/identity_score metrics

  **Parallelizable**: NO (depends on #1, #2, #3 for validation)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `test_swap_quality.py:50-100` - Existing metric calculation (diff_mean, identity_score)

  **API/Type References** (contracts to implement against):
  - `lpips` library: `lpips.LPIPS(net='alex')` for perceptual similarity
  - OpenCV optical flow: `cv2.calcOpticalFlowFarneback()` for temporal consistency

  **Test References** (testing patterns to follow):
  - `test_quality/ANALYSIS_FINDINGS.md:56-69` - Metrics inadequacy analysis

  **Documentation References** (specs and requirements):
  - `.sisyphus/notepads/phase-2.5-plan/problems.md:34-37` - Metrics blocker

  **External References** (libraries and frameworks):
  - LPIPS: https://github.com/richzhang/PerceptualSimilarity
  - OpenCV optical flow documentation

  **WHY Each Reference Matters**:
  - Existing test script shows where to integrate new metrics
  - LPIPS is proven perceptual metric (used in research)
  - Optical flow detects temporal inconsistencies

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_quality_metrics.py`
  - [ ] Test covers: LPIPS calculation, temporal consistency score
  - [ ] `pytest tests/test_quality_metrics.py` → PASS

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using enhanced test script:
    ```bash
    python test_swap_quality.py --video .assets/examples/zBambola.mp4 --frames 30 --all-metrics --output test_quality/metrics_validation.json
    ```
    Expected: LPIPS >0.85, temporal_consistency >0.9, jitter <5%
  - [ ] Verify state:
    - Navigate to: `test_quality/metrics_validation.json`
    - Action: Open JSON, check all metrics
    - Verify: All metrics meet targets
    - Screenshot: Save to `.sisyphus/evidence/task-4-metrics.png`

  **Evidence Required:**
  - [ ] Command output with all metrics (copy-paste)
  - [ ] JSON file with measurements
  - [ ] Confirmation: All 3 fixes (#1, #2, #3) validated by metrics

  **Commit**: YES
  - Message: `feat(testing): add LPIPS and temporal consistency metrics`
  - Files: `test_swap_quality.py`, `tests/test_quality_metrics.py`
  - Pre-commit: `pytest tests/test_quality_metrics.py`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `fix(masker): correct mouth interior detection using outer lip polygon` | watserface/face_masker.py, tests/test_mouth_detection.py | pytest tests/test_mouth_detection.py |
| 2 | `fix(detector): add temporal stabilization for bboxes and landmarks` | watserface/face_detector.py, watserface/face_helper.py, tests/test_temporal_stabilization.py | pytest tests/test_temporal_stabilization.py |
| 3 | `feat(swapper): add hybrid InSwapper/SimSwap for better eye quality` | watserface/processors/modules/face_swapper.py, tests/test_hybrid_swapper.py | pytest tests/test_hybrid_swapper.py |
| 4 | `feat(testing): add LPIPS and temporal consistency metrics` | test_swap_quality.py, tests/test_quality_metrics.py | pytest tests/test_quality_metrics.py |

---

## Success Criteria

### Verification Commands
```bash
# Run all foundation fix tests
pytest tests/test_mouth_detection.py tests/test_temporal_stabilization.py tests/test_hybrid_swapper.py tests/test_quality_metrics.py

# Generate comprehensive quality report
python test_swap_quality.py --video .assets/examples/zBambola.mp4 --frames 30 --all-metrics --output test_quality/foundation_fix_report.json

# Expected results:
# - Mouth opening: 25-35px (vs 8.2px baseline)
# - Face size variance: <5% (vs 22-30% baseline)
# - Eye quality: Matches InSwapper baseline
# - LPIPS: >0.85
# - Temporal consistency: >0.9
```

### Final Checklist
- [ ] All 4 tasks completed with passing tests
- [ ] User validation: "Video swaps are now usable"
- [ ] Metrics meet targets (documented in evidence/)
- [ ] No breaking changes to existing API
- [ ] Phase 2.5 unblocked and ready to proceed

### User Validation Required
**CRITICAL**: Before marking this plan complete, user MUST confirm:
- ✅ "Video swaps are now usable" (vs "not a one is usable")
- ✅ Mouth masks properly cover food/objects
- ✅ Face size is stable (no breathing/flashing)
- ✅ Eyes look natural (no over-sharpening)

---

## Next Steps After Completion

Once all 4 tasks pass and user validates:

1. **Update Phase 2.5 blocker status**:
   - Mark foundation issues as RESOLVED in `.sisyphus/notepads/phase-2.5-plan/problems.md`
   - Document which fixes unblocked which Phase 2.5 tasks

2. **Resume Phase 2.5 plan**:
   - Return to `.sisyphus/plans/phase-2.5-plan.md`
   - Start with Task 1: Research point tracking (TAPIR/CoTracker3)
   - Proceed through all 7 tasks with confidence

3. **Archive foundation fix evidence**:
   - Move `.sisyphus/evidence/` to `.sisyphus/evidence/foundation-fix/`
   - Create summary report linking to all evidence

---

## Notepad Structure

```
.sisyphus/notepads/foundation-fix-plan/
  learnings.md    # Patterns discovered, what worked
  decisions.md    # Technical choices made
  issues.md       # Problems encountered, solutions
  problems.md     # Unresolved blockers (if any)
```
