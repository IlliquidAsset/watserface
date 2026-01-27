# WatserFace System Architecture

**Version:** 0.13.0-dev
**Last Updated:** 2026-01-25
**Status:** Active Development

---

## 1. Executive Summary

WatserFace is an advanced face synthesis platform that extends traditional face swapping with **physics-aware transparency handling** and **inverted compositing** techniques. The system is built on a foundation of pretrained ONNX models with LoRA fine-tuning for identity-specific adaptation.

### Core Innovation

**Traditional Face Swapping:** Detect → Swap → Mask → Blend
**WatserFace Approach:** Mask → Swap → DKT → Composite (conditional)

This inversion enables handling of impossible scenarios (transparent occlusions like glass, liquids, smoke) that traditional pipelines cannot solve.

---

## 2. Current State (v0.13.0-dev)

### What Works Today

✅ **Face Swapping** - Production-ready using pretrained models
✅ **Identity Training** - InstantID profile creation from source media
✅ **LoRA Fine-Tuning** - Source→Target specialized adapters
✅ **XSeg Occlusion Detection** - Real-time inference of opaque masks
✅ **Studio Workflow** - Unified UI for training and swapping

### Technology Stack

| Component | Implementation | Status |
|-----------|---------------|--------|
| **Face Detection** | RetinaFace, SCRFD, YOLO | ✅ Production |
| **Landmarks** | MediaPipe 478, 2DFAN4 68 | ✅ Production |
| **Face Swapping** | InSwapper, HyperSwap, SimSwap | ✅ Production |
| **Face Enhancement** | GFPGAN, CodeFormer | ✅ Production |
| **Identity Training** | InstantID (custom) | ✅ Production |
| **LoRA Training** | Custom PyTorch adapter | ⚠️ Architecture broken (see below) |
| **XSeg Masking** | U-Net (pretrained) | ✅ Production |
| **DKT Transparency** | Not implemented | 🔴 Planned (Phase 2.5) |
| **ControlNet Refinement** | Not implemented | 🔴 Planned (Phase 2.5/3) |

### Known Issues

**Critical:**
- ❌ **LoRA Training Broken**: `IdentityGenerator` model ignores source embedding (architecture flaw at watserface/training/train_instantid.py:77-83)
- ❌ **Gemini Producing Garbage**: Progressive quality degradation during LoRA training iterations

**Blockers for Production:**
- ⚠️ **No Baseline Validation**: Haven't proven non-occluded swaps match FaceSwap.dev quality
- ⚠️ **MediaPipe Integration Bug**: 478 landmarks produce blurry/low-res output (coordinate mapping issue)

---

## 3. Development Roadmap

### Milestone 0: Baseline Quality Validation 🎯 **NEXT**

**Goal:** Prove WatserFace produces pixel-perfect swaps on clean, non-occluded faces that match or exceed existing tools.

**Why This Matters:**
- Phase 2.5/3 innovations solve occlusion problems, not base swap quality
- Must validate foundation before adding complexity
- Competitive benchmark vs FaceSwap.dev, Rope, etc.

**Acceptance Criteria:**
- ✅ Identity Similarity ≥ 0.85 (cosine similarity)
- ✅ Visual Quality SSIM ≥ 0.90 on backgrounds
- ✅ No visible edge artifacts (human evaluation)
- ✅ Color consistency ΔE < 5.0
- ✅ Competitive with FaceSwap.dev (within 5% on all metrics)

**Test Plan:**
```python
# Clean swap test (no occlusions)
SOURCE = "clean_frontal_face_A.jpg"
TARGET = "clean_frontal_face_B.jpg"

# Run both systems
watserface_output = run_watserface(source, target)
faceswap_output = run_faceswap(source, target)

# Compare metrics
metrics = {
    'identity_similarity': cosine_sim(embed(result), embed(source)),
    'ssim': structural_similarity(result, target),
    'lpips': learned_perceptual_similarity(result, reference),
    'face_sharpness': laplacian_variance(crop_face(result)),
    'edge_artifacts': measure_discontinuity_at_mask_boundary(result, mask)
}
```

**Implementation:** See `docs/architecture/MILESTONE_0_BASELINE.md`

---

### Phase 2.5: DKT Transparency Handling (The "Mayonnaise Layer")

**Goal:** Solve transparent occlusion problem using Diffusion Knows Transparency (DKT) depth/normal estimation.

**The Mayonnaise Test:**
- **Problem:** Person eating corn dog with mayonnaise dripping
- **Traditional Approach:** XSeg masks mayo as "occlusion" → leaves hole in face
- **Phase 2.5 Approach:** DKT recognizes mayo as semi-transparent volume → estimates depth/refraction → inpaints through it with physically accurate light bending

**Architecture:**

```
Layer 1 (Opaque): XSeg Mask
    ↓
    Defines strict swap boundary (solid objects: hands, food)
    ↓
Layer 2 (Transmission): DKT Depth + Normal Maps
    ↓
    Estimates transparent volumes (glass, liquid, smoke)
    ↓
Layer 3 (Inpainting): Stable Diffusion ControlNet
    ↓
    Re-lights with depth/normal constraints
    ↓
FINAL COMPOSITE
```

**Technology:**
- DKT: Video diffusion prior for transparent volume estimation
- ControlNet: Depth and normal map conditioning
- Temporal Coherence: Multi-frame consistency for video

**Status:** Not implemented
**Implementation:** See `docs/architecture/PHASE_2.5_DKT_POC.md`

---

### Phase 3: Inverted Compositing Pipeline

**Goal:** Conditional execution - use inverted compositing only when needed, fall back to traditional for clean swaps.

**Decision Tree:**

```python
def process_frame(source_identity, target_frame):
    # Layer 0: Detection & Analysis
    occlusion_type = classify_occlusion(target_frame)

    # Layer 1: Primary Swap (ALWAYS RUN)
    if has_custom_lora(source_identity):
        swapped, confidence_mask = lora_swap_with_xseg(...)
    else:
        swapped = pretrained_swap(...)

    # Layer 2: Transparency (CONDITIONAL)
    if occlusion_type == 'transparent':
        transmission = dkt_composite(swapped, target_frame)
    else:
        transmission = None

    # Layer 3: Diffusion Refinement (CONDITIONAL)
    if needs_inpainting(confidence_mask):
        final = controlnet_inpaint(transmission or swapped, ...)
    else:
        final = swapped

    # Layer 4: Traditional Blend (FALLBACK)
    return paste_back(target_frame, final, confidence_mask, affine_matrix)
```

**Key Innovations:**
1. **LoRA + XSeg Dual-Head Output** - Single model outputs both swapped face and confidence mask
2. **DreamBooth Synthetic Data** - Generate unlimited training pairs for data-scarce scenarios
3. **Conditional Layer Execution** - Skip expensive DKT/ControlNet when not needed

**Status:** Planned
**Implementation:** See `docs/architecture/PHASE_3_INVERTED.md`

---

## 4. File Structure

```
watserface/
├── watserface/
│   ├── core.py                    # Main processing orchestrator
│   ├── face_analyser.py           # Detection & landmarks
│   ├── face_helper.py             # Warping, paste_back, poisson_paste_back
│   ├── face_masker.py             # XSeg inference
│   ├── identity_profile.py        # Multi-source identity averaging
│   │
│   ├── processors/
│   │   └── modules/
│   │       ├── face_swapper.py    # Main swap logic (568-621)
│   │       ├── face_enhancer.py   # GFPGAN, CodeFormer
│   │       ├── deep_swapper.py    # Advanced swap variants
│   │       └── ...
│   │
│   ├── training/
│   │   ├── core.py                # Training orchestrator
│   │   ├── trainers/
│   │   │   ├── identity.py        # InstantID training
│   │   │   ├── xseg.py            # XSeg U-Net training
│   │   │   └── train_lora.py      # ⚠️ BROKEN - needs architecture fix
│   │   ├── models/
│   │   │   └── lora_adapter.py    # LoRA layer implementation
│   │   └── datasets/
│   │       ├── instantid_dataset.py
│   │       ├── lora_dataset.py
│   │       └── xseg_dataset.py
│   │
│   └── uis/
│       ├── layouts/
│       │   ├── streamlined.py     # Main UI (default)
│       │   ├── studio.py          # Unified workflow
│       │   ├── training.py        # Identity training
│       │   └── modeler.py         # LoRA fine-tuning
│       └── components/            # 70+ reusable UI components
│
├── .assets/models/
│   ├── pretrained/                # Downloaded ONNX models
│   │   ├── inswapper_128.onnx
│   │   ├── hyperswap_1a_256.onnx
│   │   ├── simswap_unofficial_512.onnx
│   │   ├── xseg_1.onnx
│   │   ├── gfpgan_1.4.onnx
│   │   └── ...
│   └── trained/                   # User-trained models
│       ├── [identity_name].onnx
│       ├── [identity_name].pth    # Checkpoint
│       └── [name]_lora.onnx       # LoRA adapters
│
└── models/identities/             # Identity profiles (JSON)
    └── [identity_name].json       # Mean embeddings, metadata
```

---

## 5. Key Design Decisions

### 5.1 Why ONNX Instead of PyTorch Inference?

**Rationale:**
- ✅ **Performance:** ONNX Runtime 2-3x faster than PyTorch on CPU/MPS
- ✅ **Deployment:** Single file, no Python dependencies
- ✅ **Ecosystem:** Pretrained models widely available (InSwapper, SimSwap, etc.)

**Trade-off:** Training in PyTorch → Export to ONNX (extra conversion step)

---

### 5.2 Why Inverted Compositing Instead of End-to-End Model?

**Alternative:** Train a single large model that handles everything

**Our Approach:** Modular pipeline with conditional execution

**Why:**
- ✅ **Flexibility:** Use different models for different occlusion types
- ✅ **Efficiency:** Skip expensive DKT when not needed
- ✅ **Debuggability:** Each layer independently testable
- ✅ **Upgradability:** Replace components without retraining everything

**Trade-off:** More complex orchestration logic

---

### 5.3 Why LoRA Instead of Full Model Fine-Tuning?

**LoRA Advantages:**
- ✅ **Speed:** 2-10 hours vs 10-50 hours
- ✅ **Storage:** 1-5 MB vs 100-500 MB per model
- ✅ **Reusability:** One identity → unlimited target-specific LoRAs
- ✅ **Composability:** Can combine multiple LoRA adapters

**Trade-off:** Slightly lower quality (95-99% vs 100%)

**Status:** Architecture currently broken, needs fix before use

---

### 5.4 Paste Methods: Linear vs Poisson

**Current Implementation (watserface/face_helper.py:236-269):**

```python
# Linear blend (default)
def paste_back(frame, crop, mask, matrix):
    paste_vision_frame = paste * (1 - mask) + inverse * mask
    return frame

# Poisson blend (optional)
def poisson_paste_back(frame, crop, mask, matrix):
    try:
        return cv2.seamlessClone(inverse, frame, mask, center, cv2.MIXED_CLONE)
    except cv2.error:
        return paste_back(frame, crop, mask, matrix)  # Fallback
```

**FaceSwap.dev Approach:** Multi-band blending with Laplacian pyramids

**Action for Milestone 0:** Compare all three methods, select best for baseline

---

## 6. Performance Characteristics

### Hardware Targets

**Development:** M4 MacBook Air (16GB, fanless)
- 4 Performance + 6 Efficiency cores
- Apple Neural Engine (16-core)
- Thermal constraints (no fan)

**Production:** NVIDIA GPUs (CUDA)
- RTX 3070+ recommended (8GB+ VRAM)
- FP16 mixed precision training

### Current Benchmarks (M4 MacBook Air)

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| Face Detection | ~30 FPS | RetinaFace on MPS |
| Face Swap (128px) | ~15 FPS | InSwapper |
| Face Swap (512px) | ~2-3 FPS | SimSwap |
| Face Enhancement | ~5 FPS | GFPGAN |
| Identity Training | 2-3 min/epoch | 1000 frames, optimized |
| LoRA Training | 2-4 min/epoch | Rank 16 |

---

## 7. Dependencies

### Core Runtime
```
python >= 3.11
torch >= 2.0.0
onnxruntime >= 1.15.0
opencv-python >= 4.8.0
mediapipe >= 0.10.0
insightface >= 0.7.3
gradio >= 4.0.0
```

### Training Extensions
```
diffusers >= 0.25.0          # Phase 2.5/3 (ControlNet)
transformers >= 4.36.0       # Model architectures
accelerate >= 0.25.0         # Training optimization
peft >= 0.7.0                # LoRA implementation
segmentation-models-pytorch  # XSeg training
```

### Phase 2.5/3 (Planned)
```
depth-anything               # Monocular depth estimation
controlnet-aux              # ControlNet preprocessors
```

---

## 8. Quality Assurance

### Test Strategy

**Unit Tests:** Core functions (face_helper, face_masker)
**Integration Tests:** Full pipeline on reference images
**Visual Regression:** Perceptual hash comparison
**Benchmark Suite:** Performance tracking across versions

### Validation Datasets

**Clean Swaps (Milestone 0):**
- 20 frontal face pairs
- Various skin tones, genders, ages
- Controlled lighting
- No occlusions

**Occluded Swaps (Phase 2.5):**
- Corn dog + mayonnaise (transparent liquid)
- Glasses (transparent solid)
- Steam/smoke (semi-transparent gas)
- Hands covering face (opaque)

---

## 9. Migration Path

### From v0.10.0 → v0.13.0-dev

**Breaking Changes:**
- ⚠️ LoRA training workflow changed (waiting for architecture fix)
- ✅ Pretrained model swapping unchanged (backward compatible)
- ✅ Identity profiles compatible

**Upgrade Path:**
1. Backup `.assets/models/trained/`
2. Update dependencies: `pip install -r requirements.txt`
3. LoRA models: Wait for architecture fix or retrain

---

## 10. References

### Research Papers
- **InstantID**: Wang et al., 2024 - Zero-shot identity preservation
- **DKT (Diffusion Knows Transparency)**: Xu et al., 2025 - Transparent volume estimation
- **LoRA**: Hu et al., 2021 - Low-Rank Adaptation of Large Language Models
- **SimSwap**: Chen et al., 2020 - Face swapping with GAN

### Upstream Projects
- **FaceFusion**: Original codebase (v3.3.4 fork point)
- **InsightFace**: Face detection and recognition
- **MediaPipe**: 478-point facial landmarks

### Internal Documentation
- [Milestone 0: Baseline Validation](docs/architecture/MILESTONE_0_BASELINE.md)
- [Phase 2.5: DKT PoC](docs/architecture/PHASE_2.5_DKT_POC.md)
- [Phase 3: Inverted Compositing](docs/architecture/PHASE_3_INVERTED.md)
- [Training Guide](docs/guides/TRAINING_GUIDE.md)

---

## 11. Contributing

See `docs/development/CONTRIBUTING.md` for:
- Code style guidelines
- Testing requirements
- Pull request process
- Development environment setup

---

**Last Updated:** 2026-01-25
**Maintained By:** IlliquidAsset
**Based On:** FaceFusion by Henry Ruhs
