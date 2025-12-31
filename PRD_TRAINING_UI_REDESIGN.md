# PRD: Training UI Redesign & M4 Mac Optimization

**Product**: WatserFace
**Current Version**: 0.11.0
**Target Version**: 0.12.0
**Priority**: HIGH
**Estimated Execution Time**: 3-4 hours
**Executor**: Gemini Code / Jules (Gemini 3)

---

## Problem Statement

### Current Issues

1. **Progress Bar Conflicts** (CRITICAL)
   - Gradio `progress()` updates overwrite telemetry text in Identity Training Status
   - Text-based progress competes with visual progress bar
   - Results in flickering, unreadable UI during training

2. **Poor Visual Design** (HIGH)
   - Training status is plain text dump
   - No visual graphs or charts
   - Doesn't feel "visually striking" or professional
   - No "internet speed test" vibe as intended

3. **Button Design Misalignment** (MEDIUM)
   - Large red buttons ("Start Identity Training", "Stop") don't match brand
   - Use default Gradio styling instead of custom brand guidelines
   - Feel clunky and unrefined

4. **Missing Glassmorphism** (HIGH)
   - Dark glassmorphism theme existed previously but disappeared
   - Current UI has flat, boring backgrounds
   - No depth, transparency, or modern aesthetic

5. **Underutilized M4 Mac** (CRITICAL for performance)
   - Current defaults: CoreML only, 4 threads, default memory
   - M4 Max capable of much more (16 performance cores, Neural Engine, 48GB unified memory)
   - Training taking 28+ minutes for 1 epoch - too slow

---

## Success Criteria

### Visual Design
- [ ] Training status displays as **two-column layout** (Progress left, Telemetry right)
- [ ] **Progress graphs**: Visual bars/charts for epoch progress and batch progress
- [ ] **Telemetry graphs**: Real-time loss graph over time
- [ ] **"Internet speed test" aesthetic**: Clean, animated, visually striking
- [ ] **Dark glassmorphism** restored throughout UI
- [ ] **Brand-aligned buttons**: Custom styled, not default Gradio red

### Performance
- [ ] M4 Mac defaults optimized for maximum safe utilization
- [ ] Training speed improved by 2-3x (target: ~10-15 min per epoch)
- [ ] No thermal throttling or crashes

### Code Quality
- [ ] No progress bar / telemetry conflicts
- [ ] Smooth, non-flickering UI updates
- [ ] Version bumped to 0.12.0

---

## Technical Specifications

### 1. Training Status UI Redesign

**File**: `watserface/uis/layouts/training.py`

**Current Design** (lines 177, 188):
```python
IDENTITY_STATUS = gradio.Textbox(label="Identity Training Status", value="Idle", interactive=False)
OCCLUSION_STATUS = gradio.Textbox(label="Occlusion Training Status", value="Idle", interactive=False)
```

**New Design** (replace Textbox with custom HTML + components):

```python
# Identity Training Status - Two Column Layout
with gradio.Group():
    gradio.Markdown("### 📊 Identity Training Status")
    with gradio.Row():
        # LEFT COLUMN: Progress Visualization
        with gradio.Column(scale=1):
            IDENTITY_OVERALL_PROGRESS = gradio.HTML(
                value='<div class="glass-progress-container">...</div>',
                label="Overall Progress"
            )
            IDENTITY_EPOCH_PROGRESS = gradio.HTML(
                value='<div class="glass-progress-container">...</div>',
                label="Epoch Progress"
            )
            IDENTITY_BATCH_PROGRESS = gradio.HTML(
                value='<div class="glass-progress-container">...</div>',
                label="Batch Progress"
            )

        # RIGHT COLUMN: Telemetry & Metrics
        with gradio.Column(scale=1):
            IDENTITY_LOSS_CHART = gradio.LinePlot(
                x="step",
                y="loss",
                title="Training Loss",
                height=200,
                show_label=False
            )
            IDENTITY_METRICS = gradio.HTML(
                value='<div class="glass-metrics-panel">...</div>',
                label="Metrics"
            )
```

**Key Changes**:
1. Replace single `Textbox` with `Group` containing two columns
2. LEFT: Three custom HTML progress bars (overall, epoch, batch)
3. RIGHT: LinePlot for loss over time + HTML metrics panel
4. Use glassmorphism CSS classes (defined below)

**Output Structure** (update `wrapped_start_identity_training()`):
- Instead of yielding formatted text string
- Yield dictionary with separate fields:
  ```python
  yield {
      'overall_progress': 0.42,  # 0.0 to 1.0
      'epoch_progress': 0.86,    # 0.0 to 1.0
      'batch_progress': 0.215,   # 0.0 to 1.0
      'epoch': 42,
      'total_epochs': 100,
      'batch': 215,
      'total_batches': 250,
      'loss': 0.0234,
      'loss_history': [(0, 0.045), (1, 0.038), ..., (42, 0.0234)],
      'eta': '12m 34s',
      'device': 'mps',
      'status': 'Training'
  }
  ```

---

### 2. Progress Bar HTML Components

**File**: `watserface/uis/assets/training_progress.html` (NEW)

Create reusable HTML template for animated progress bars:

```html
<div class="glass-progress-container">
    <div class="progress-header">
        <span class="progress-label">{label}</span>
        <span class="progress-value">{percentage}%</span>
    </div>
    <div class="progress-bar-track">
        <div class="progress-bar-fill" style="width: {percentage}%">
            <div class="progress-bar-shimmer"></div>
        </div>
    </div>
    <div class="progress-details">
        <span>{detail_text}</span>
    </div>
</div>
```

**Usage**:
- Overall Progress: `{label}="Overall", {percentage}=42, {detail_text}="Epoch 42/100"`
- Epoch Progress: `{label}="Current Epoch", {percentage}=86, {detail_text}="Batch 215/250"`
- Batch Progress: `{label}="Batch", {percentage}=21.5, {detail_text}="Processing..."`

---

### 3. Glassmorphism CSS

**File**: `watserface/uis/assets/overrides.css` (MODIFY)

Add dark glassmorphism theme:

```css
/* ===== DARK GLASSMORPHISM THEME ===== */

/* Global dark background gradient (brand colors) */
body, .gradio-container {
    background: linear-gradient(135deg, #0D0D0D 0%, #1a0d2e 50%, #0d1a2e 100%);
    background-attachment: fixed;
}

/* Glass morphism effect for all containers */
.block, .form, .panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
}

/* Glassmorphism for progress containers */
.glass-progress-container {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    transition: all 0.3s ease;
}

.glass-progress-container:hover {
    background: rgba(255, 255, 255, 0.12);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
}

/* Progress header */
.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.progress-label {
    font-size: 14px;
    font-weight: 600;
    color: #F2F2F2;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.progress-value {
    font-size: 24px;
    font-weight: 700;
    color: #CCFF00;
    text-shadow: 0 0 10px rgba(204, 255, 0, 0.5);
}

/* Progress bar track */
.progress-bar-track {
    width: 100%;
    height: 12px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    margin-bottom: 8px;
}

/* Progress bar fill with gradient (brand colors) */
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #4D4DFF 0%, #FF00FF 100%);
    border-radius: 8px;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

/* Shimmer animation inside progress bar */
.progress-bar-shimmer {
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(255, 255, 255, 0.3) 50%,
        transparent 100%
    );
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 200%; }
}

/* Progress details */
.progress-details {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
    text-align: center;
}

/* Metrics panel glassmorphism */
.glass-metrics-panel {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 20px;
    margin-top: 16px;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-label {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.7);
    font-weight: 500;
}

.metric-value {
    font-size: 15px;
    color: #CCFF00;
    font-weight: 700;
}

/* ===== CUSTOM BUTTON STYLES (BRAND-ALIGNED) ===== */

/* Primary buttons (Start Training) - BRAND COLORS */
button.primary, .primary-btn {
    background: linear-gradient(135deg, #4D4DFF 0%, #FF00FF 100%) !important;
    border: none !important;
    color: #F2F2F2 !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px 0 rgba(255, 0, 255, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

button.primary:hover, .primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(255, 0, 255, 0.6) !important;
    background: linear-gradient(135deg, #5D5DFF 0%, #FF10FF 100%) !important;
}

/* Stop buttons (danger variant) */
button.stop, .stop-btn {
    background: rgba(255, 77, 77, 0.2) !important;
    border: 1px solid rgba(255, 77, 77, 0.4) !important;
    color: #ff4d4d !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    border-radius: 8px !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

button.stop:hover, .stop-btn:hover {
    background: rgba(255, 77, 77, 0.3) !important;
    border-color: rgba(255, 77, 77, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* ===== LOSS CHART STYLING ===== */

.loss-chart-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px;
    height: 220px;
}

/* ===== INTERNET SPEED TEST AESTHETIC ===== */

/* Animated glow effect for active training */
.training-active {
    animation: glow 2s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
    }
    50% {
        box-shadow: 0 0 40px rgba(255, 0, 255, 0.6);
    }
}

/* Pulse animation for progress values */
.pulse-value {
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.7;
    }
}
```

---

### 4. Training Core Updates

**File**: `watserface/training/train_instantid.py`

**Current Issue**: Calling `progress()` every batch conflicts with text yields

**Fix**:
1. Remove `progress()` calls entirely (lines 159-165)
2. Only yield telemetry data (no progress bar updates)
3. Let UI handle visualization

**Modified yield**:
```python
# Build loss history for chart
loss_history.append((completed_steps, current_loss))

# Yield batch-level progress
batch_telemetry = {
    'overall_progress': ((epoch * total_batches) + (batch_idx + 1)) / (epochs * total_batches),
    'epoch_progress': (batch_idx + 1) / total_batches,
    'batch_progress': (batch_idx + 1) / total_batches,
    'epoch': epoch + 1,
    'total_epochs': epochs,
    'batch': batch_idx + 1,
    'total_batches': total_batches,
    'loss': current_loss,
    'loss_history': loss_history[-100:],  # Last 100 points for chart
    'device': str(device),
    'status': 'Training'
}
yield f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{total_batches}", batch_telemetry
```

**File**: `watserface/training/core.py`

Update `wrapped_start_identity_training()` to:
1. Parse telemetry dict
2. Generate HTML for progress bars
3. Update LinePlot data
4. Update metrics panel HTML

---

### 5. M4 Mac Optimization

**File**: `watserface/uis/components/execution_thread_count.py`

**Current Default**: 4 threads

**New Default for macOS ARM**:
```python
import platform

def get_default_execution_thread_count() -> int:
    """Get optimal thread count for current system"""
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # M4 Max: 16 performance cores
        # Use 12 threads (leave 4 for system)
        return 12
    else:
        return 4
```

**File**: `watserface/uis/components/memory.py`

**Current Default**: Conservative memory limit

**New Default for M4 Max**:
```python
def get_default_memory_limit() -> int:
    """Get optimal memory limit for current system"""
    import psutil
    total_memory = psutil.virtual_memory().total / (1024**3)  # GB

    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # M4 Max with 48GB RAM
        # Use 32GB for training (leave 16GB for system)
        if total_memory >= 48:
            return 32
        elif total_memory >= 32:
            return 20
        else:
            return int(total_memory * 0.6)
    else:
        return int(total_memory * 0.5)
```

**File**: `watserface/uis/components/execution.py`

**Current Default**: CoreML only

**New Default for M4 Mac**:
```python
def get_default_execution_providers() -> List[str]:
    """Get optimal execution providers for current system"""
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # M4 Mac: Use CoreML + CPU fallback
        # CoreML utilizes Neural Engine
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    else:
        return ['CPUExecutionProvider']
```

**Additional PyTorch Optimizations** (if using PyTorch backend):

**File**: `watserface/training/train_instantid.py` (add after device selection)

```python
# M4 Mac optimizations
if device.type == 'mps':
    torch.backends.mps.enable_fallback(True)  # Enable CPU fallback for unsupported ops
    torch.set_num_threads(12)  # Use 12 threads

    # Enable MPS optimizations
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Reduce memory fragmentation

    logger.info("M4 Mac optimizations enabled: 12 threads, MPS fallback", __name__)
```

---

### 6. Button Redesign

**File**: `watserface/uis/layouts/training.py`

**Current**:
```python
START_IDENTITY_BUTTON = gradio.Button("Start Identity Training", variant="primary")
STOP_IDENTITY_BUTTON = gradio.Button("Stop", variant="stop")
```

**New**:
```python
START_IDENTITY_BUTTON = gradio.Button(
    "▶ Start Training",
    variant="primary",
    elem_classes=["primary-btn"]
)
STOP_IDENTITY_BUTTON = gradio.Button(
    "⏹ Stop",
    variant="stop",
    elem_classes=["stop-btn"]
)
```

Apply same pattern to occlusion buttons.

---

### 7. Version Bump

**File**: `watserface/metadata.py`

Update version:
```python
METADATA =\
{
    'name': 'WatserFace',
    'version': '0.12.0',  # Changed from 0.11.0
    ...
}
```

---

## Implementation Steps

### Phase 1: CSS & Glassmorphism (30 min)
1. Update `watserface/uis/assets/overrides.css` with glassmorphism styles
2. Test that dark theme appears across all tabs
3. Verify glass effect on containers

### Phase 2: Button Redesign (15 min)
4. Update button definitions in `training.py`
5. Add `elem_classes` for custom styling
6. Test hover effects and animations

### Phase 3: Training Status UI (90 min)
7. Replace `Textbox` with two-column layout in `training.py`
8. Create progress bar HTML generator function
9. Create metrics panel HTML generator function
10. Add `gradio.LinePlot` for loss chart
11. Update `wrapped_start_identity_training()` to:
    - Parse telemetry dict
    - Generate progress bar HTML
    - Update loss chart data
    - Generate metrics HTML
12. Wire up all outputs to new components

### Phase 4: Training Core Updates (30 min)
13. Remove `progress()` calls from `train_instantid.py`
14. Add `loss_history` list tracking
15. Update telemetry dict structure
16. Test that data flows correctly to UI

### Phase 5: M4 Optimization (30 min)
17. Update `execution_thread_count.py` with auto-detection
18. Update `memory.py` with auto-detection
19. Update `execution.py` with CoreML default
20. Add MPS optimizations to `train_instantid.py`
21. Test training speed improvement

### Phase 6: Version Bump & Testing (15 min)
22. Update `metadata.py` to 0.12.0
23. Restart app
24. Run full training test
25. Verify:
    - No progress bar conflicts
    - Graphs animate smoothly
    - Loss chart updates in real-time
    - Glassmorphism visible
    - Buttons styled correctly
    - Training faster (monitor Activity Monitor)

---

## Reference Screenshots

**Desired "Internet Speed Test" Aesthetic**:
- Think: Speedtest.net or Fast.com
- Large animated progress circles/bars
- Real-time graphs with smooth animations
- Dark background with glowing accents
- Glassmorphism cards with blur effects
- Clean typography with clear metrics
- Pulsing/shimmer effects during activity

**Color Palette** (from brand guidelines):
- **Primary**: `#FF00FF` (Glitch Magenta)
- **Secondary**: `#4D4DFF` (Deep Blurple)
- **Accent/CTA**: `#CCFF00` (Electric Lime)
- **Background**: `#0D0D0D` (Void Black) with subtle gradient
- **Glass**: `rgba(255, 255, 255, 0.08)` with blur
- **Danger**: `#ff4d4d` (red for stop button)
- **Text**: `#F2F2F2` (Ghost White)

---

## Critical Files Checklist

### Files to Create:
- [ ] None (all modifications to existing files)

### Files to Modify:
1. [ ] `watserface/uis/assets/overrides.css` - Add glassmorphism CSS
2. [ ] `watserface/uis/layouts/training.py` - Redesign status UI, update buttons
3. [ ] `watserface/training/train_instantid.py` - Remove progress() calls, update telemetry
4. [ ] `watserface/training/core.py` - Update wrapper to handle new UI components
5. [ ] `watserface/uis/components/execution_thread_count.py` - M4 auto-detection
6. [ ] `watserface/uis/components/memory.py` - M4 memory optimization
7. [ ] `watserface/uis/components/execution.py` - CoreML default for M4
8. [ ] `watserface/metadata.py` - Version bump to 0.12.0

---

## Testing Requirements

### Visual Testing:
1. Open Training tab in browser
2. Verify dark glassmorphism background gradient
3. Verify glass effect on all containers (blur, transparency)
4. Verify button styling (gradients, hover effects)
5. Start training and observe:
   - Left column: 3 animated progress bars (overall, epoch, batch)
   - Right column: Real-time loss chart + metrics panel
   - No flickering or conflicts
   - Smooth animations (shimmer, glow effects)

### Performance Testing:
1. Check Activity Monitor during training:
   - CPU usage across cores
   - Memory usage (should use ~32GB on M4 Max)
   - No thermal throttling
2. Measure epoch time (target: 10-15 min, down from 28 min)
3. Verify no crashes or OOM errors

### Functional Testing:
1. Start training → verify all graphs update
2. Stop training → verify clean shutdown
3. Resume training → verify graphs continue correctly
4. Test error scenarios → verify error display still works

---

## Known Constraints

### Technical Limitations:
- Gradio LinePlot may have update rate limits (test with streaming data)
- MPS backend doesn't support all PyTorch ops (enable CPU fallback)
- CSS glassmorphism requires modern browser (Safari 14+, Chrome 76+)

### Performance Constraints:
- M4 Max thermal limits: ~100W sustained
- Memory bandwidth: 400 GB/s (don't exceed to avoid bottlenecks)
- Neural Engine: 38 TOPS (CoreML will utilize automatically)

---

## Rollback Plan

If training becomes unstable after M4 optimizations:

1. Revert thread count to 4:
   ```python
   return 4
   ```

2. Revert memory limit to conservative:
   ```python
   return int(total_memory * 0.5)
   ```

3. Revert execution providers to CPU only:
   ```python
   return ['CPUExecutionProvider']
   ```

4. Remove MPS optimizations from train_instantid.py

---

## Expected Outcomes

### User Experience:
- **Before**: "Progress bar overlaps telemetry, ugly red buttons, training takes forever"
- **After**: "Looks like a professional speed test, beautiful glassmorphism, training is 2x faster"

### Performance:
- **Before**: 28+ min per epoch, 4 threads, underutilized M4
- **After**: 10-15 min per epoch, 12 threads, optimized for M4 Neural Engine

### Visual Impact:
- **Before**: Flat, text-heavy, default Gradio styling
- **After**: Dark glassmorphism, animated graphs, internet speed test aesthetic

---

## Acceptance Criteria

This PRD is **complete** when:

1. [ ] Training Status displays as two columns (progress left, telemetry right)
2. [ ] Three animated progress bars visible (overall, epoch, batch)
3. [ ] Real-time loss chart updating smoothly
4. [ ] Metrics panel showing device, ETA, loss, etc.
5. [ ] Dark glassmorphism applied throughout UI
6. [ ] Buttons styled with gradients and hover effects
7. [ ] M4 Mac using 12 threads + 32GB memory by default
8. [ ] Training speed improved to 10-15 min per epoch
9. [ ] Version shows 0.12.0
10. [ ] No progress bar/telemetry conflicts or flickering

---

## Additional Notes for Executor

### Gradio Component Tips:
- Use `gradio.HTML()` for custom progress bars (more control than built-in)
- Use `gradio.LinePlot()` for loss chart (supports streaming updates)
- Use `elem_classes=["class-name"]` to apply CSS classes
- Update HTML components with `.update(value=new_html)`
- Update LinePlot with `.update(value=new_dataframe)`

### CSS Tips:
- Test glassmorphism in dark mode (Safari/Chrome dev tools)
- Use `backdrop-filter` for blur effect (requires `-webkit-` prefix)
- Animations should be smooth (use `cubic-bezier` easing)
- Test on actual device (glassmorphism performance varies)

### M4 Optimization Tips:
- Monitor with Activity Monitor → GPU History (verify Neural Engine usage)
- Use `torch.backends.mps.is_available()` to detect MPS support
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to reduce fragmentation
- Test with small batch first, then increase if stable

### Debugging:
- If progress bars don't appear: Check browser console for CSS errors
- If loss chart doesn't update: Print telemetry dict structure, verify data format
- If training crashes: Reduce thread count and memory limit
- If glassmorphism doesn't show: Verify `overrides.css` is loaded (check Network tab)

---

**End of PRD**

**Estimated Total Time**: 3-4 hours
**Priority**: HIGH
**Target Version**: 0.12.0
**Dependencies**: None (all changes self-contained)

Good luck, Gemini/Jules! 🚀
