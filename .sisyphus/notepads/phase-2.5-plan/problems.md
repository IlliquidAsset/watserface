# Phase 2.5 Unresolved Blockers

## Active Problems

*Unresolved blockers that need attention will be appended here.*

---

## [2026-01-26] Critical Quality Issues Block Phase 2.5 Progress

### Context
User feedback on video swap outputs revealed multiple critical quality issues that make current outputs unusable. These must be resolved before proceeding with Phase 2.5 DKT/ControlNet work.

### Identified Blockers

1. **Mouth Interior Detection - BROKEN**
   - Inner lip landmarks (60-67) detect lip LINE not actual opening
   - Measured: 8.2px vs expected ~30px (37.5px outer lip)
   - Impact: Mouth interior mask too small, food preservation fails
   - Root cause: 2dfan4 landmarker limitation

2. **Face Size Jitter - SEVERE** 
   - 22-30% size variance frame-to-frame
   - Largest jump: 9,326px² (frame 22→23)
   - Impact: Visible "breathing" effect, unusable output
   - Root cause: No temporal stabilization on bboxes/landmarks

3. **Eye Quality Degradation**
   - SimSwap 512 has worse perceptual quality than InSwapper 128
   - Paradox: Higher technical metrics (2-3x sharpness) but worse appearance
   - Hypothesis: Over-sharpening creates artificial edges
   - Impact: Eyes look unnatural

4. **Quality Metrics Inadequate**
   - Current metrics (diff_mean, identity_score) miss critical issues
   - Missing: LPIPS, temporal stability, feature alignment
   - Impact: Can't detect quality problems automatically

5. **Temporal Smoothing Insufficient**
   - 5-frame window reduced flicker 74% but jitter remains
   - Only masks smoothed, not landmarks/bboxes
   - Impact: Still visible artifacts

### Decision Required

**Option A**: Fix critical quality issues first (mouth, jitter, eyes)
- Pros: Makes current pipeline usable, validates approach
- Cons: Delays Phase 2.5 DKT/ControlNet work
- Estimated: 3-5 tasks

**Option B**: Proceed with Phase 2.5 plan as-is
- Pros: Follows original plan, may solve issues via DKT
- Cons: Building on broken foundation, may waste effort
- Risk: DKT won't fix landmarker/jitter issues

**Recommendation**: Option A - Fix foundation first. DKT/ControlNet won't help if basic face detection and temporal stability are broken.

### Files Created
- `test_quality/ANALYSIS_FINDINGS.md` - Comprehensive analysis
- `test_quality/overlay_*.png` - Transparent mask visualizations
- `test_quality/mouth_size_analysis_detailed.png` - Landmark analysis
- `test_quality/eye_quality_comparison.png` - Model comparison


## [2026-01-26] DECISION: Creating Foundation Fix Plan

### Action Taken
Per Boulder continuation rules, I'm creating a separate foundation fix plan to unblock Phase 2.5. The original Phase 2.5 plan remains intact but is on hold until foundation issues are resolved.

### Rationale
1. **DKT won't fix landmarker bugs**: CoTracker3 tracks points accurately, but if 2dfan4 provides wrong initial points (inner lip at edge instead of cavity), tracking wrong points perfectly doesn't solve the problem.

2. **ControlNet won't fix temporal jitter**: Generative inpainting assumes stable input geometry. 22-30% face size variance will create flickering artifacts in the inpainted regions.

3. **User validation required**: "Not a one of these video swaps is usable" indicates foundation is broken. Must validate basic pipeline before adding complexity.

### New Plan
Creating `.sisyphus/plans/foundation-fix-plan.md` with 4 critical tasks:
1. Fix mouth interior detection (8.2px → ~30px)
2. Stabilize face size jitter (22-30% variance → <5%)
3. Hybrid eye quality (InSwapper eyes + SimSwap face)
4. Implement quality metrics (LPIPS, temporal consistency)

### Success Criteria
Once foundation fixes pass user validation, we can:
- Resume Phase 2.5 with confidence
- Or pivot based on learnings

### Status
- Phase 2.5 plan: ON HOLD (0/7 tasks)
- Foundation fix plan: CREATING NOW

## [2026-01-26] Phase 2.5 Blocked by Foundation Issues - UPDATED

### Original Blocker (Documented Earlier)
Phase 2.5 (DKT/ControlNet transparency) blocked by critical quality issues in base video swap pipeline.

### Investigation Results (This Session)
Comprehensive investigation revealed:
1. **Implementations exist** for all foundation fixes
2. **Tests pass** for all implementations
3. **Integration missing** - code exists but not wired into pipeline
4. **75% complete** - just needs ~180 lines of integration code

### Specific Blockers for Phase 2.5

**Task 1 (Research Point Tracking)**: CAN PROCEED
- Not blocked by foundation issues
- Can research TAPIR/CoTracker3 independently

**Tasks 2-7 (ControlNet/DKT Implementation)**: BLOCKED
- Require stable video pipeline
- Current pipeline has:
  - 22-30% face size jitter (makes DKT tracking unreliable)
  - Broken mouth detection (affects transparency compositing)
  - No quality metrics (can't validate improvements)

### Recommendation Update

**Option A: Fix Foundation First** (STILL RECOMMENDED)
- Complete integration work (~2-4 hours)
- Validate fixes work
- Then proceed with Phase 2.5

**Option B: Proceed with Phase 2.5 Anyway**
- Start with Task 1 (research only)
- Accept that Tasks 2-7 will have quality issues
- May need to redo work after foundation fixes

**Option C: Parallel Track**
- One agent: Foundation integration
- Another agent: Phase 2.5 research (Task 1)
- Merge when foundation complete

### Current Status
- Foundation fix: Investigation complete, integration pending
- Phase 2.5: 0/7 tasks complete, blocked by foundation
- Recommendation: Fix foundation first

### Decision Point
Awaiting user decision on which path to take.
