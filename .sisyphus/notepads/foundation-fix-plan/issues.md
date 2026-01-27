# Foundation Fix Issues & Solutions

## Problems Encountered

*Issues discovered during implementation and their solutions will be documented here.*

---

## [2026-01-26] Task 1: Mouth Interior Mask Not Integrated

### Problem
The `create_mouth_interior_mask()` function exists in `watserface/face_masker.py` (lines 290-312) and has passing tests, but it's **NOT integrated into the mask system**.

### Root Cause
- The function was created but never added to the `FaceMaskArea` or `FaceMaskRegion` types
- Current "mouth" area in `choices.py` line 27 uses ALL mouth landmarks (48-67)
- No mask type calls `create_mouth_interior_mask()`

### Evidence
```bash
$ grep -r "create_mouth_interior_mask" watserface/ --include="*.py" | grep -v "def create"
# Returns: NOTHING (only the definition, no calls)
```

### Impact
The fix for the 8.2px bug exists but isn't being used. Face swapping still uses the broken "mouth" area mask which includes both outer AND inner lip landmarks.

### Solution Required
Need to either:
1. **Option A**: Add new mask area type "mouth-interior" that calls `create_mouth_interior_mask()`
2. **Option B**: Modify existing "mouth" area to use `create_mouth_interior_mask()` instead of landmark-based convex hull

Recommendation: Option A (less breaking, more explicit)

### Files to Modify
1. `watserface/types.py` line 113: Add 'mouth-interior' to FaceMaskArea literal
2. `watserface/choices.py` line 23-32: Add 'mouth-interior' to face_mask_area_set (but it won't use landmark indices, needs special handling)
3. `watserface/face_masker.py` line 229: Modify `create_area_mask()` to check for 'mouth-interior' and call `create_mouth_interior_mask()`

### Status
Task 1 is NOT complete - function exists but not integrated. Marking as BLOCKED pending decision on integration approach.

## [2026-01-26] Task 2: Temporal Stabilization Not Integrated

### Problem
The `TemporalBBoxStabilizer` and `TemporalLandmarkStabilizer` classes exist in `watserface/face_helper.py` (lines 14-73) but are **NOT being used** in the face detection/swapping pipeline.

### Root Cause
- Classes were created but never instantiated or called
- Face detector and landmarker don't use these stabilizers
- No integration into video processing loop

### Evidence
```bash
$ grep -r "TemporalBBoxStabilizer\|TemporalLandmarkStabilizer" watserface/ --include="*.py" | grep -v "^.*class "
# Returns: NOTHING (only class definitions, no usage)
```

### Impact
The fix for 22-30% face size jitter exists but isn't being used. Video swapping still has severe jitter.

### Solution Required
Need to integrate stabilizers into the video processing pipeline:
1. Instantiate stabilizers in face detector/swapper
2. Call `stabilizer.update(bbox)` and `stabilizer.update(landmarks)` for each frame
3. Use stabilized values instead of raw detections
4. Reset stabilizers when processing new video

### Files to Modify
1. `watserface/face_detector.py` - Add bbox stabilizer
2. `watserface/face_landmarker.py` - Add landmark stabilizer  
3. `watserface/processors/modules/face_swapper.py` - Integrate stabilizers into video loop

### Status
Task 2 is NOT complete - classes exist but not integrated. Same pattern as Task 1.
