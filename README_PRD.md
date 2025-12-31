# Training UI Redesign - PRD Package

**Version**: 0.12.0
**Status**: Ready for Execution
**Executor**: Gemini Code / Jules (Gemini 3)

---

## 📦 Package Contents

This folder contains a complete PRD (Product Requirements Document) for redesigning the WatserFace training UI with dark glassmorphism, brand-aligned styling, and M4 Mac optimization.

### Documents:

1. **`PRD_TRAINING_UI_REDESIGN.md`** (MAIN)
   - Complete technical specification
   - Full CSS code
   - Implementation steps
   - Testing requirements
   - ~700 lines, comprehensive

2. **`PRD_SUMMARY.md`**
   - Quick reference (TL;DR)
   - Key points and file changes
   - Useful for quick lookups

3. **`VISUAL_MOCKUP.md`**
   - ASCII art mockups
   - Before/after comparison
   - Color reference
   - Animation timeline

4. **`brand guidelines.md`**
   - Official WatserFace brand colors
   - Typography specs
   - Logo assets

5. **`README_PRD.md`** (this file)
   - Package index
   - Quick start guide

---

## 🚀 Quick Start for AI Executor

### Step 1: Read the PRD
```bash
# Start here - full technical spec
open "PRD_TRAINING_UI_REDESIGN.md"
```

### Step 2: Review Visual Design
```bash
# See what you're building
open "VISUAL_MOCKUP.md"
```

### Step 3: Check Brand Guidelines
```bash
# Ensure color accuracy
open "brand guidelines.md"
```

### Step 4: Execute
Follow the implementation steps in `PRD_TRAINING_UI_REDESIGN.md`:
- Phase 1: CSS & Glassmorphism (30 min)
- Phase 2: Button Redesign (15 min)
- Phase 3: Training Status UI (90 min)
- Phase 4: Training Core Updates (30 min)
- Phase 5: M4 Optimization (30 min)
- Phase 6: Version Bump & Testing (15 min)

**Total**: ~3-4 hours

---

## 🎯 What's Being Built

### Problem:
- Progress bar overlaps telemetry text
- Ugly default Gradio styling
- No glassmorphism theme
- M4 Mac underutilized (training too slow)

### Solution:
- **Two-column layout**: Progress bars (left) + Loss chart & metrics (right)
- **Dark glassmorphism**: Frosted glass effect with WatserFace brand colors
- **Custom buttons**: Blurple→Magenta gradient (Start), Glass red (Stop)
- **M4 optimization**: 12 threads, 32GB memory, CoreML + Neural Engine

### Result:
- Professional "internet speed test" aesthetic
- 2-3x faster training (10-15 min/epoch vs 28 min)
- No progress conflicts or flickering
- Brand-aligned design

---

## 🎨 Brand Colors (Reference)

```
Glitch Magenta:  #FF00FF  (Primary - Progress bars, buttons)
Deep Blurple:    #4D4DFF  (Secondary - Gradients, accents)
Electric Lime:   #CCFF00  (CTAs - Percentage values, highlights)
Void Black:      #0D0D0D  (Background)
Ghost White:     #F2F2F2  (Text)
```

---

## 📁 Files to Modify

**8 files total** (all modifications, no new files):

1. ✏️ `watserface/uis/assets/overrides.css`
   - Add ~300 lines of glassmorphism CSS

2. ✏️ `watserface/uis/layouts/training.py`
   - Replace Textbox with two-column layout
   - Update button styling
   - Wire up new components

3. ✏️ `watserface/training/train_instantid.py`
   - Remove conflicting `progress()` calls
   - Add `loss_history` tracking
   - Update telemetry structure

4. ✏️ `watserface/training/core.py`
   - Update wrapper to generate HTML
   - Parse new telemetry format

5. ✏️ `watserface/uis/components/execution_thread_count.py`
   - M4 auto-detection (12 threads)

6. ✏️ `watserface/uis/components/memory.py`
   - M4 memory optimization (32GB)

7. ✏️ `watserface/uis/components/execution.py`
   - CoreML default for M4

8. ✏️ `watserface/metadata.py`
   - Bump version: 0.11.0 → 0.12.0

---

## ✅ Acceptance Criteria

Execution is **complete** when:

- [ ] Training Status displays as two columns
- [ ] Three animated progress bars visible (shimmer effect)
- [ ] Real-time loss chart updating smoothly
- [ ] Metrics panel showing device, ETA, loss
- [ ] Dark glassmorphism visible (blur, transparency, brand colors)
- [ ] Buttons styled with Blurple→Magenta gradient
- [ ] M4 Mac using 12 threads + 32GB memory
- [ ] Training speed: 10-15 min/epoch (measured)
- [ ] Version displays as 0.12.0
- [ ] No progress bar conflicts or flickering

---

## 🧪 Testing Checklist

After implementation:

### Visual:
1. Open Training tab
2. Verify glassmorphism background gradient
3. Verify frosted glass effect on containers
4. Check button hover effects (gradient shift + lift)
5. Start training:
   - Left: 3 progress bars animate smoothly
   - Right: Loss chart updates in real-time
   - No flickering or text overlap

### Performance:
1. Open Activity Monitor
2. Start training
3. Verify CPU usage: 12 cores active
4. Verify memory: ~32GB allocated
5. Measure epoch time (target: 10-15 min)
6. Check temperature (should stay <80°C)

### Functional:
1. Start/stop training works
2. Resume training works
3. Error scenarios display correctly
4. All telemetry fields populate

---

## 🔧 Debugging Tips

**If glassmorphism doesn't show**:
- Check `overrides.css` loaded (browser Network tab)
- Verify browser supports `backdrop-filter` (Safari 14+, Chrome 76+)
- Hard refresh (Cmd+Shift+R)

**If progress bars flicker**:
- Check update throttle (0.5s in `wrapped_start_identity_training`)
- Verify `progress()` calls removed from `train_instantid.py`

**If training crashes**:
- Reduce thread count to 8
- Reduce memory limit to 24GB
- Check MPS fallback enabled

**If loss chart doesn't update**:
- Print telemetry dict structure
- Verify `loss_history` format: `[(step, loss), ...]`
- Check LinePlot expects DataFrame or dict

---

## 📊 Expected Performance Gains

| Metric | Before (v0.11.0) | After (v0.12.0) | Improvement |
|--------|------------------|-----------------|-------------|
| Threads | 4 | 12 | 3x |
| Memory | Default (~16GB) | 32GB | 2x |
| Epoch Time | 28+ min | 10-15 min | 2-3x faster |
| CPU Cores Used | 25% | 75% | 3x utilization |
| Neural Engine | Unused | Active (CoreML) | ✅ |

---

## 🎬 Before/After Preview

### Before:
```
Plain text status, default red buttons, no glassmorphism,
progress bar conflicts with text, slow training
```

### After:
```
Dark glassmorphism, brand colors (Magenta/Blurple/Lime),
animated progress bars, real-time charts, smooth animations,
2-3x faster training, professional speed test aesthetic
```

---

## 📞 Support

### If stuck:
1. Re-read relevant section of `PRD_TRAINING_UI_REDESIGN.md`
2. Check `VISUAL_MOCKUP.md` for design reference
3. Verify colors match `brand guidelines.md`

### Rollback plan:
See "Rollback Plan" section in main PRD if M4 optimizations cause instability.

---

## 🏁 Final Deliverable

When complete, you should be able to:

1. Open WatserFace at http://127.0.0.1:7860
2. Navigate to Training tab
3. See dark glassmorphism theme
4. Start "Sam_ident" training from "Samantha_Migrated" Face Set
5. Watch:
   - Left column: 3 animated progress bars (shimmer sweeping)
   - Right column: Loss chart animating + metrics updating
   - Container glowing with magenta halo
   - Buttons with Blurple→Magenta gradient
6. Observe 2-3x faster epoch time
7. Version footer shows "0.12.0"

**Result**: Professional, brand-aligned, performant training UI 🎉

---

## 📝 Commit Message

When complete, commit with:

```bash
git add .
git commit -m "Redesign training UI with glassmorphism + M4 optimization (v0.12.0)

- Add dark glassmorphism theme with WatserFace brand colors
- Implement two-column training status (progress + telemetry)
- Add animated progress bars with shimmer/glow effects
- Add real-time loss chart with streaming data
- Redesign buttons with Blurple→Magenta gradient
- Optimize M4 Max: 12 threads, 32GB memory, CoreML + Neural Engine
- Fix progress bar conflicts (no more text overlap)
- Bump version: 0.11.0 → 0.12.0

Training speed improved 2-3x (10-15 min/epoch vs 28 min).
UI now matches internet speed test aesthetic.

🤖 Generated with Claude Code (https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

**Ready for execution!** 🚀

Hand this package to Gemini Code or Jules and they'll have everything needed to build a stunning, performant training UI in 3-4 hours.

**Good luck!** ✨
