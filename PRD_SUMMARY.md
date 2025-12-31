# Training UI Redesign PRD - Quick Summary

**Full PRD**: `PRD_TRAINING_UI_REDESIGN.md`
**Target Version**: 0.12.0
**Executor**: Gemini Code / Jules (Gemini 3)
**Time Estimate**: 3-4 hours

---

## What's Being Built

### 1. **Two-Column Training Status**
```
┌─────────────────────────────────────────────────┐
│  📊 Identity Training Status                   │
├──────────────────┬──────────────────────────────┤
│  LEFT COLUMN     │  RIGHT COLUMN                │
│                  │                              │
│  Overall Progress│  📈 Loss Chart              │
│  [====42%====]   │  (Real-time graph)          │
│                  │                              │
│  Epoch Progress  │  📊 Metrics Panel           │
│  [====86%====]   │  • Device: mps              │
│                  │  • ETA: 12m 34s             │
│  Batch Progress  │  • Loss: 0.0234             │
│  [====21%====]   │  • Epoch: 42/100            │
└──────────────────┴──────────────────────────────┘
```

### 2. **Dark Glassmorphism**
- Dark gradient background (Void Black `#0D0D0D` with purple hints)
- Glass containers with blur effect
- Transparent overlays with `backdrop-filter`
- Glowing magenta/blurple accents and animations

### 3. **Custom Branded Buttons**
- **Start**: Blurple→Magenta gradient (`#4D4DFF` → `#FF00FF`) with glow
- **Stop**: Red glass effect with transparency
- Hover animations (lift + glow)
- Electric Lime (`#CCFF00`) for progress values

### 4. **M4 Mac Optimization**
- 12 threads (up from 4)
- 32GB memory limit (up from default)
- CoreML + Neural Engine utilization
- Target: 2-3x faster training

---

## Files to Modify

1. ✏️ `watserface/uis/assets/overrides.css` - Glassmorphism CSS
2. ✏️ `watserface/uis/layouts/training.py` - Two-column UI
3. ✏️ `watserface/training/train_instantid.py` - Remove progress() conflicts
4. ✏️ `watserface/training/core.py` - Update wrappers
5. ✏️ `watserface/uis/components/execution_thread_count.py` - M4 defaults
6. ✏️ `watserface/uis/components/memory.py` - M4 memory
7. ✏️ `watserface/uis/components/execution.py` - CoreML default
8. ✏️ `watserface/metadata.py` - Version → 0.12.0

---

## Key Technical Details

### Progress Bar HTML Template
```html
<div class="glass-progress-container">
    <div class="progress-header">
        <span class="progress-label">OVERALL</span>
        <span class="progress-value">42%</span>
    </div>
    <div class="progress-bar-track">
        <div class="progress-bar-fill" style="width: 42%">
            <div class="progress-bar-shimmer"></div>
        </div>
    </div>
</div>
```

### Telemetry Data Structure
```python
{
    'overall_progress': 0.42,
    'epoch_progress': 0.86,
    'batch_progress': 0.215,
    'loss': 0.0234,
    'loss_history': [(0, 0.045), (1, 0.038), ..., (42, 0.0234)],
    'eta': '12m 34s',
    'device': 'mps',
    'status': 'Training'
}
```

### M4 Optimization Code
```python
# Thread count auto-detection
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    return 12  # M4 Max: use 12 of 16 cores

# Memory optimization
if total_memory >= 48:  # M4 Max
    return 32  # Use 32GB, leave 16GB for system
```

---

## Success Metrics

- ✅ No progress bar/telemetry conflicts
- ✅ Smooth animations (60fps)
- ✅ Glassmorphism visible and performant
- ✅ Training speed: 10-15 min/epoch (down from 28 min)
- ✅ "Internet speed test" aesthetic achieved
- ✅ Version bumped to 0.12.0

---

## What You'll See

**Before**:
```
Identity Training Status
Epoch 2/100 - Batch 201/250 (80%) | Loss: 0.0618
Status: Training
Epoch Progress: 80%
...
[Progress bar overlaps this text ⚠️]
```

**After**:
```
┌─ 📊 Identity Training Status ───────────────────┐
│                                                  │
│  OVERALL                           42%  ▶───────│
│  [████████████░░░░░░░░░░░░░░]        │  Loss   │
│                                      │  Chart  │
│  CURRENT EPOCH                 86%  ▶  📈📊📉  │
│  [████████████████████░░░░░░]        │         │
│                                      │  Device │
│  BATCH                        21.5% ▶  mps     │
│  [█████░░░░░░░░░░░░░░░░░░░░░]        │  ETA    │
│                                      │  12m34s │
└──────────────────────────────────────┴─────────┘
[Glass effect, dark theme, animated shimmer ✨]
```

---

## Handoff Instructions for AI Executor

1. Read full PRD: `PRD_TRAINING_UI_REDESIGN.md`
2. Start with Phase 1 (CSS) - verify glassmorphism works
3. Work through phases sequentially
4. Test after each phase
5. Final test: Start Sam_ident training, verify all graphs update
6. Commit with message: "Redesign training UI with glassmorphism + M4 optimization (v0.12.0)"

---

**Questions?** Check the full PRD - it includes:
- Detailed CSS code
- HTML templates
- Python code snippets
- Debugging tips
- Rollback plan
- Testing checklist

**Ready to execute!** 🚀
