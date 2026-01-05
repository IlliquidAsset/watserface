# WatserFace v1.0.0 Roadmap

## Vision: Unified 2.5D Face Synthesis Pipeline

WatserFace v1.0.0 establishes a production-ready, linear workflow for high-fidelity face synthesis with automatic occlusion handling through depth-aware generative inpainting.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STUDIO WORKFLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   IDENTITY   │───▶│    XSEG      │───▶│    MAP +     │───▶│   OUTPUT   │ │
│  │   BUILDER    │    │   TRAINER    │    │   EXECUTE    │    │   REVIEW   │ │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └─────┬──────┘ │
│         │                   │                   │                  │        │
│         ▼                   ▼                   ▼                  ▼        │
│    [Add Media]         [Auto-mask]         [Preview]          [Export]      │
│    [Train More]        [Annotate]          [Quality Check]    [Iterate]     │
│                        [Depth Est.]        [Feedback Loop]                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌──────────────┐               ┌──────────────┐
            │   2.5D       │               │  Diffusion   │
            │   Pipeline   │               │  Inpainter   │
            ├──────────────┤               ├──────────────┤
            │ Normal Maps  │──────────────▶│ ControlNet   │
            │ Depth Est.   │               │ Boundary     │
            │ Z-Capture    │               │ Synthesis    │
            └──────────────┘               └──────────────┘
```

---

## Phase 1: Unified Studio Layout (v1.0.0-alpha)

### Goal
Replace fragmented tabs (swap, training, modeler) with single linear workflow.

### Components

#### 1.1 Identity Builder
- **Add Media**: Drop images/videos containing source identity
- **Face Set Management**: Auto-extract faces, temporal smoothing
- **Train Identity**: InstantID LoRA training with progress feedback
- **Iterative Refinement**: Add more samples, train additional epochs

#### 1.2 Occlusion Trainer (XSeg)
- **Target Analysis**: Auto-detect occlusions in target video
- **Auto-Mask Generation**: Convex hull from landmarks + edge detection
- **Manual Annotation**: Gradio ImageEditor for manual mask painting
- **Depth Estimation**: MiDaS/ZoeDepth for translucent/transparent occlusions
- **Train XSeg**: U-Net training with live loss visualization

#### 1.3 Map + Execute (Combined)
- **Source-Target Mapping**: Assign identities to faces in target
- **Quality Preview**: Generate sample frames before full processing
- **Feedback Loop**: If quality insufficient, return to training with more epochs
- **Full Execution**: Process entire video/image set

#### 1.4 Output Review
- **Side-by-side Comparison**: Original vs synthesized
- **Quality Metrics**: SSIM, identity similarity score
- **Export Options**: Resolution, format, codec settings

---

## Phase 2: 2.5D Pipeline (v1.0.0-beta)

### Goal
Enable realistic relighting and depth-aware processing.

### Components

#### 2.1 Depth Capture
- **MediaPipe Z-Coordinate**: Already captured in `face_landmarker.py`
- **Normal Map Generation**: `create_normal_map()` in `face_helper.py`
- **Monocular Depth Estimation**: Integrate MiDaS for scene depth

#### 2.2 Translucent/Transparent Occlusion Handling
- **Problem**: Glass, water, smoke partially occlude faces
- **Solution**: Video-based depth estimation using temporal consistency
- **Method**: 
  1. Extract depth maps per frame
  2. Identify translucent regions (depth variance + color analysis)
  3. Estimate occlusion alpha from depth discontinuities

#### 2.3 Normal Map Integration
- **Output**: RGB image (R=X, G=Y, B=Z normals, 0-255)
- **Use Cases**:
  - Relighting in post-processing
  - ControlNet conditioning for inpainting
  - Physics-based rendering hints

---

## Phase 3: Generative Inpainting (v1.0.0-rc)

### Goal
Bridge the gap between XSeg masks and identity fragments using diffusion.

### The "Corndog Problem"
When a subject interacts with objects (eating, drinking, hands on face), the mesh deforms in ways not represented in training data. Standard face swap fails at these boundaries.

### Solution: Layered Composition

```
Layer Stack (bottom to top):
┌─────────────────────────────────────┐
│  Target Background                  │  ← Preserved
├─────────────────────────────────────┤
│  Swapped Face (Identity Fragment)   │  ← From face_swapper
├─────────────────────────────────────┤
│  Boundary Inpainting               │  ← DIFFUSION GENERATED
├─────────────────────────────────────┤
│  Occlusion Object                  │  ← XSeg masked, preserved
└─────────────────────────────────────┘
```

### Implementation

#### 3.1 Boundary Detection
- Dilate XSeg mask by N pixels to create "boundary zone"
- Extract normal map for boundary region
- Extract depth for boundary region

#### 3.2 ControlNet Conditioning
- **Normal Map ControlNet**: Guides 3D shape of synthesized boundary
- **Depth ControlNet**: Ensures correct depth ordering
- **Reference Image**: Swapped face provides color/texture reference

#### 3.3 Diffusion Inpainting
- **Model**: Stable Diffusion Inpainting or SDXL Inpainting
- **LoRA**: Optional fine-tuning on face boundary data
- **Input**: Boundary mask + normal map + depth + reference
- **Output**: Seamless boundary pixels

#### 3.4 Temporal Consistency
- **Optical Flow**: Warp previous frame's inpainting as prior
- **Video Diffusion**: AnimateDiff or similar for multi-frame coherence

---

## Phase 4: Production Hardening (v1.0.0)

### Performance
- [ ] ONNX optimization for all models
- [ ] TensorRT/CoreML acceleration paths
- [ ] Batch processing with memory management

### Quality
- [ ] Automated quality scoring (reject bad frames)
- [ ] A/B comparison tools
- [ ] Export presets (social media, broadcast, etc.)

### UX
- [ ] Single-page Studio layout
- [ ] Keyboard shortcuts
- [ ] Preset management (save/load configurations)
- [ ] Project files (save work in progress)

---

## Technical Specifications

### Dependencies (New for v1.0.0)

```
# Depth Estimation
transformers>=4.35.0  # MiDaS, ZoeDepth
timm>=0.9.0           # Vision backbones

# Diffusion Inpainting
diffusers>=0.24.0     # Stable Diffusion pipelines
controlnet-aux>=0.0.7 # ControlNet preprocessors
accelerate>=0.24.0    # Training acceleration

# Video Consistency
pytorch-optical-flow  # Temporal coherence
```

### Model Downloads

| Model | Size | Purpose |
|-------|------|---------|
| MiDaS v3.1 Large | 1.3GB | Monocular depth estimation |
| SD Inpainting v2.1 | 5.2GB | Boundary synthesis |
| ControlNet Normal | 1.4GB | Normal map conditioning |
| ControlNet Depth | 1.4GB | Depth conditioning |

### File Structure

```
watserface/
├── uis/
│   └── layouts/
│       └── studio.py          # NEW: Unified workflow
├── studio/
│   ├── orchestrator.py        # NEW: Pipeline orchestration
│   ├── identity_builder.py    # NEW: Identity management
│   ├── occlusion_trainer.py   # NEW: XSeg + depth training
│   └── quality_checker.py     # NEW: Quality feedback loop
├── depth/
│   ├── midas.py               # NEW: MiDaS integration
│   ├── zoe.py                 # NEW: ZoeDepth integration
│   └── temporal.py            # NEW: Video depth smoothing
├── inpainting/
│   ├── diffusion.py           # NEW: SD inpainting wrapper
│   ├── controlnet.py          # NEW: ControlNet integration
│   └── boundary.py            # NEW: Boundary detection
└── processors/modules/
    └── occlusion_inpainter.py # UPDATED: Diffusion integration
```

---

## Migration Path

### From v0.10.x

1. **Config**: `ui_layouts = studio` replaces `ui_layouts = swap training modeler`
2. **Models**: Existing trained models compatible
3. **Workflows**: Saved jobs continue to work
4. **API**: CLI unchanged, new `--studio` flag for unified mode

### Breaking Changes

- Tab-based layouts deprecated (still available via `ui_layouts = swap training modeler`)
- Blob layout merged into Studio as "Auto" mode

---

## Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| v1.0.0-alpha | Week 1 | Studio layout + orchestrator |
| v1.0.0-beta | Week 2 | 2.5D pipeline integration |
| v1.0.0-rc | Week 3-4 | Diffusion inpainting |
| v1.0.0 | Week 5 | Production hardening |

---

## Success Metrics

1. **Quality**: Corndog test passes (dynamic occlusion handling)
2. **Performance**: <30s per 1080p frame on consumer GPU
3. **UX**: Zero tab switches for complete workflow
4. **Reliability**: <1% frame rejection rate on standard test set
