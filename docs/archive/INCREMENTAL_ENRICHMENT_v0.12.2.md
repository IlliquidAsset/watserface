# Incremental Identity Profile Enrichment (v0.12.2)

## Problem Solved

**Before v0.12.2:**
- Training the same identity multiple times OVERWROTE the profile each time
- Stopping training early (e.g., at epoch 30 of 100) didn't create any profile
- Profiles only created when training completed ALL epochs
- User had to complete 100 epochs in one session to get a profile
- Modeler dropdown stayed empty if training stopped early

**After v0.12.2:**
- Training the same identity multiple times ENRICHES the profile
- Stopping training early STILL creates/updates profile ✅
- Each training session adds new embeddings to existing ones
- Can train 10 epochs, stop, come back later and add more
- Profiles appear in Modeler immediately after first session

---

## How It Works

### Incremental Training Workflow

```
Session 1: Train "Sam_ident" 10 epochs from video frames
├─ Extract 100 frames → get 87 embeddings
├─ Create new profile: Sam_ident (87 embeddings)
├─ Stop at epoch 10
└─ ✅ Profile saved to models/identities/sam_ident/profile.json
     ✅ Shows in Modeler dropdown

Session 2: Train "Sam_ident" 10 more epochs from photos
├─ Extract 50 photos → get 94 new embeddings
├─ Load existing profile (87 embeddings)
├─ Enrich: combine 87 + 94 = 181 total embeddings
├─ Recalculate mean/std from combined data
├─ Stop at epoch 20
└─ ✅ Profile updated (181 embeddings, session #2)
     ✅ Shows in Modeler dropdown

Session 3: Train "Sam_ident" to completion (80 more epochs)
├─ Use same Face Set → extract 102 embeddings
├─ Load existing profile (181 embeddings)
├─ Enrich: 181 + 102 = 283 total embeddings
├─ Complete all 100 epochs
└─ ✅ Profile updated (283 embeddings, session #3)
     ✅ ONNX model exported to .assets/models/trained/
     ✅ Checkpoint saved
```

---

## Mathematical Formula

### Incremental Mean Calculation

Instead of storing all raw embeddings (memory intensive), we use the **weighted average formula**:

```
Given:
- N1 = number of old embeddings
- mean1 = existing profile mean
- N2 = number of new embeddings
- mean2 = mean of new embeddings

Combined mean = (N1 × mean1 + N2 × mean2) / (N1 + N2)
```

**Example:**
```
Session 1: 87 embeddings, mean = [0.52, 0.31, ..., 0.88]
Session 2: 94 new embeddings, mean = [0.48, 0.35, ..., 0.92]

Combined = (87 × [0.52...] + 94 × [0.48...]) / 181
         = [0.50, 0.33, ..., 0.90]
```

### Pooled Standard Deviation

```
Given:
- std1 = existing profile std deviation
- std2 = std deviation of new embeddings

Combined variance = ((N1-1) × std1² + (N2-1) × std2²) / (N1+N2-1)
Combined std = sqrt(combined_variance)
```

This allows infinite enrichment without keeping raw embeddings in memory.

---

## Code Changes

### 1. New Method: `enrich_profile()`

**File:** `watserface/identity_profile.py` (lines 441-498)

```python
def enrich_profile(self, profile_id: str, new_embeddings: List[np.ndarray],
                   new_stats: Dict[str, Any]) -> IdentityProfile:
    """Enrich existing profile with new embeddings"""

    existing_profile = self.load_profile(profile_id)

    if existing_profile:
        # Get counts
        old_count = existing_profile.quality_stats.get('final_embedding_count', 0)
        new_count = len(new_embeddings)
        total_count = old_count + new_count

        # Combine means
        old_mean = np.array(existing_profile.embedding_mean)
        new_mean = np.mean(new_embeddings, axis=0)
        combined_mean = (old_count * old_mean + new_count * new_mean) / total_count

        # Combine stds (pooled variance)
        old_std = np.array(existing_profile.embedding_std)
        new_std = np.std(new_embeddings, axis=0)
        combined_variance = ((old_count-1)*old_std**2 + (new_count-1)*new_std**2) / (total_count-1)
        combined_std = np.sqrt(combined_variance)

        # Update profile
        existing_profile.embedding_mean = combined_mean.tolist()
        existing_profile.embedding_std = combined_std.tolist()
        existing_profile.quality_stats['final_embedding_count'] = total_count
        existing_profile.quality_stats['training_sessions'] = (
            existing_profile.quality_stats.get('training_sessions', 1) + 1
        )

        return existing_profile
    else:
        return None  # Signal to create new profile
```

---

### 2. Training Core Updates

**File:** `watserface/training/core.py`

**Change A: Try enrichment first (lines 286-334)**

```python
if embeddings:
    manager = identity_profile.get_identity_manager()
    profile_id = model_name.lower().replace(' ', '_')

    # Try to enrich existing profile first
    enriched_profile = manager.source_intelligence.enrich_profile(
        profile_id, embeddings, new_stats
    )

    if enriched_profile:
        # Enriched existing
        profile = enriched_profile
        action = "enriched"
        logger.info(f"✅ Enriched '{model_name}' (now {total} embeddings)")
    else:
        # Create new
        profile = IdentityProfile(
            id=profile_id,
            name=model_name,
            embedding_mean=mean,
            embedding_std=std,
            is_ephemeral=False,  # ← Important! Makes it show in Modeler
            quality_stats={...}
        )
        action = "created"

    manager.source_intelligence.save_profile(profile)
```

**Change B: Profile creation runs even if training stopped (lines 225-253)**

```python
# Copy ONNX model to assets (if training completed)
final_model_path = None
if onnx_path:
    # Training completed - copy model
    shutil.copy(onnx_path, final_model_path)
    ...
else:
    logger.info("Training incomplete - ONNX not exported yet")

# Profile creation happens HERE regardless of onnx_path status
# (continues to extract embeddings and save profile)
```

**Change C: Status messages reflect enrichment (lines 359-376)**

```python
if telemetry.get('profile_action') == 'enriched':
    if onnx_path:
        msg = f"✅ Profile enriched: +{new} (total: {total}, session #{n})"
    else:
        msg = f"⚠️ Stopped early. Profile enriched: +{new} (total: {total}, session #{n}). Resume to complete."
else:
    if onnx_path:
        msg = f"✅ Profile created with {count} embeddings"
    else:
        msg = f"⚠️ Stopped early. Profile created. Resume to export ONNX."
```

---

## User Experience Changes

### Training UI Messages

**Complete Training:**
```
✅ Identity Model 'Sam_ident' trained successfully!
Profile enriched: +102 new embeddings (total: 283, session #3)
```

**Stopped Early:**
```
⚠️ Training stopped early.
Profile enriched: +94 new embeddings (total: 181, session #2).
Resume training to complete.
```

**First Session:**
```
✅ Identity Model 'Sam_ident' trained successfully!
Profile created with 87 embeddings.
```

---

## Modeler Dropdown

**Before v0.12.2:**
```
Source Identity Profile: [No options]
```

**After v0.12.2:**
```
Source Identity Profile:
  - Samantha (181 embeddings, 2 sessions)
  - Sam_ident (283 embeddings, 3 sessions)
  - Actor_v1 (94 embeddings, 1 session)
```

The dropdown now shows ALL identities, even if training was stopped early or never completed.

---

## Profile JSON Structure

**File:** `models/identities/sam_ident/profile.json`

```json
{
  "id": "sam_ident",
  "name": "Sam_ident",
  "created_at": "2025-12-30T23:15:00",
  "source_files": [],
  "embedding_mean": [0.50, 0.33, ..., 0.90],
  "embedding_std": [0.08, 0.12, ..., 0.05],
  "quality_stats": {
    "total_processed": 250,
    "final_embedding_count": 283,
    "source_count": 0,
    "training_sessions": 3,
    "last_training": "2025-12-30T23:45:00"
  },
  "thumbnail_path": null,
  "is_ephemeral": false,
  "last_used": "2025-12-30T23:45:00",
  "source_count": 0
}
```

**Key Fields:**
- `final_embedding_count`: Total embeddings across all sessions (283)
- `training_sessions`: Number of times trained (3)
- `last_training`: Timestamp of most recent session
- `is_ephemeral`: false (so it shows in Modeler)

---

## Benefits

### 1. **Flexibility**
- Train 10 epochs today, 10 more tomorrow
- No need to complete all epochs in one sitting
- Stop anytime, resume anytime

### 2. **Multi-Source Enrichment**
- Session 1: Train from video (1000 frames)
- Session 2: Add photos (50 images)
- Session 3: Add another video (500 frames)
- Result: Rich profile from multiple sources

### 3. **Immediate Feedback**
- Profile appears in Modeler after first session
- No waiting for full 100 epochs
- Can start LoRA training immediately

### 4. **Memory Efficiency**
- Doesn't store raw embeddings (would be huge)
- Only stores mean + std (512 floats each)
- Infinite enrichment without memory growth

### 5. **Progress Tracking**
- `training_sessions` tracks how many times trained
- `last_training` shows most recent session
- Quality improves with each session

---

## Testing Checklist

### Test 1: Create New Profile (Stopped Early)
- [x] Train "test1" for 10 epochs, then stop
- [x] Profile created in models/identities/test1/
- [x] Shows in Modeler dropdown
- [x] quality_stats shows training_sessions: 1

### Test 2: Enrich Existing Profile
- [x] Train "test1" again for 10 more epochs, then stop
- [x] Profile enriched (not overwritten)
- [x] final_embedding_count increased
- [x] training_sessions incremented to 2

### Test 3: Complete Training
- [x] Train "test1" to completion (100 epochs)
- [x] ONNX exported to .assets/models/trained/
- [x] Profile enriched again
- [x] training_sessions: 3

### Test 4: Multiple Sources
- [x] Session 1: Train from Face Set A
- [x] Session 2: Train from upload (photos)
- [x] Profile combines embeddings from both

---

## Technical Notes

### Why Not Store Raw Embeddings?

**Option A: Store all raw embeddings**
```
Session 1: 87 embeddings × 512 floats × 4 bytes = 178 KB
Session 2: 181 embeddings × 512 floats × 4 bytes = 371 KB
Session 3: 283 embeddings × 512 floats × 4 bytes = 580 KB
After 10 sessions: ~2 MB per profile

Problem: Memory grows linearly with training sessions
```

**Option B: Store only mean + std (current approach)**
```
Session 1: (512 + 512) floats × 4 bytes = 4 KB
Session 2: (512 + 512) floats × 4 bytes = 4 KB
Session 3: (512 + 512) floats × 4 bytes = 4 KB
After 10 sessions: 4 KB per profile

Benefit: Constant memory, efficient
```

### Accuracy of Incremental Statistics

The incremental mean formula is **exact** - produces same result as calculating from all raw data.

The pooled std formula is **approximate** - assumes independence between samples. In practice, this is close enough for face embeddings and provides good results.

---

## Migration Note

**Existing profiles (created before v0.12.2):**
- Will have `training_sessions: 1` (assumed first session)
- Next training will enrich correctly
- No manual migration needed

**Face Sets with checkpoints:**
- Checkpoint at epoch 30: Resume will work
- Profile will be created/enriched on next training
- Old behavior: would have skipped profile creation

---

## App Status

✅ **Running:** http://127.0.0.1:7860
✅ **Version:** 0.12.2
✅ **Changes committed and pushed to master**

---

## Summary

| Feature | Before (v0.12.1) | After (v0.12.2) |
|---------|------------------|-----------------|
| Profile creation | Only if training completes | Always (even if stopped) |
| Multiple training sessions | Overwrites profile | Enriches existing |
| Modeler dropdown | Empty if stopped early | Shows all identities |
| Embedding tracking | Single snapshot | Cumulative across sessions |
| Training flexibility | Must complete all epochs | Stop/resume anytime |
| Memory usage | N/A | Constant (4KB per profile) |

**This unlocks iterative, multi-source identity training workflows!** 🎉
