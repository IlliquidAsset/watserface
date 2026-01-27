# Phase 2.5: DKT Transparency Handling (The "Mayonnaise Layer")

**Goal:** Solve transparent occlusion problem using physics-aware depth and normal estimation.

**Status:** Not Started (blocked by Milestone 0)

**Priority:** 🔴 Critical Innovation - Core Differentiator

---

## 1. The Problem: Transparent Occlusions

### 1.1 The "Mayonnaise Test"

**Scenario:** Person eating corn dog with mayonnaise dripping over face

**Traditional Approach (WatserFace current + FaceSwap.dev):**
```
1. XSeg detects mayonnaise as opaque occlusion
2. Mask excludes mayonnaise region
3. Face swap applied to non-occluded areas
4. Result: HOLE where mayonnaise should be
```

**Visual:**
```
Original:        Traditional Swap:     Desired Result:
┌─────────┐     ┌─────────┐          ┌─────────┐
│  👤     │     │  👤     │          │  👤     │
│    🌭   │ →   │    🌭   │    vs    │    🌭   │
│  💧     │     │  ⬛ ← HOLE         │  💧 ← Realistic
└─────────┘     └─────────┘          └─────────┘
```

**The Fundamental Issue:** Mayo is **semi-transparent** - it both:
1. Transmits light (you can see through it)
2. Refracts light (bends rays according to physics)

XSeg treats everything as **binary**: face (1) or not-face (0). No concept of **transmission**.

---

### 1.2 Other Transparent Occlusion Scenarios

**Glasses:**
- Lens material refracts light
- Specular highlights on surface
- Thickness affects refraction angle

**Steam/Smoke:**
- Volumetric scattering
- Density varies spatially
- Color affected by background

**Water/Liquid:**
- Surface tension creates distortion
- Depth affects transparency
- Reflections on surface

**Current Solutions:** All fail for the same reason - no physics model

---

## 2. The Solution: DKT (Diffusion Knows Transparency)

### 2.1 What is DKT?

**Paper:** "Diffusion Knows Transparency: Unveiling Depth and Normals from Translucent Objects" (Xu et al., 2025)

**Key Insight:** Video diffusion models (trained on billions of videos) have learned **implicit physics** of how transparent materials behave.

**Core Capability:**
- Input: Video frames (temporal context)
- Output 1: **Depth map** - Z-distance of transparent volume
- Output 2: **Normal map** - Surface orientation for refraction

**Why Video (not single frame)?**
- Temporal consistency reveals transparent boundaries
- Motion parallax disambiguates depth
- Multi-frame averaging reduces noise

---

### 2.2 How DKT Differs from Traditional Depth Estimation

**MiDaS/ZoeDepth (Monocular Depth):**
```
Input: Single RGB image
Output: Depth map (opaque surfaces only)
Problem: Transparent objects appear as "holes" at background depth
```

**DKT (Physics-Aware Depth):**
```
Input: Video sequence (5-10 frames)
Output: Depth + Normal + Transmission mask
Advantage: Models transparent volume as continuous medium
```

**Example:**
```
Scene: Glass of water on table

MiDaS Output:
┌─────────┐
│ table   │ ← Depth = 2.0m
│  ⬛⬛    │ ← Glass interior = NaN (failure)
│ table   │ ← Depth = 2.0m
└─────────┘

DKT Output:
┌─────────┐
│ table   │ ← Depth = 2.0m
│  🔵🔵   │ ← Glass volume: Depth = 1.8m-2.0m (gradient)
│ table   │ ← Depth = 2.0m
└─────────┘
         + Normal map (refraction direction)
         + Transmission map (0.8 alpha)
```

---

## 3. Architecture: 3-Layer Compositing

### 3.1 Layer Breakdown

```
INPUT: Source Identity + Target Frame (with transparent occlusion)
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Opaque (XSeg)                                      │
├─────────────────────────────────────────────────────────────┤
│ Purpose: Define strict swap boundary                        │
│ Method:  XSeg U-Net inference                               │
│ Output:  Binary mask (1 = solid occlusion, 0 = face)        │
│ Example: Hands, food, hair                                  │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Transmission (DKT)                                 │
├─────────────────────────────────────────────────────────────┤
│ Purpose: Estimate transparent volumes                       │
│ Method:  DKT video diffusion model                          │
│ Input:   5-10 frame temporal window                         │
│ Output:  depth_map (Z), normal_map (XYZ), alpha_map (0-1)   │
│ Example: Mayonnaise, glass, smoke                           │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Inpainting (ControlNet)                            │
├─────────────────────────────────────────────────────────────┤
│ Purpose: Re-light swapped face with transparent refraction  │
│ Method:  Stable Diffusion ControlNet Inpaint                │
│ Controls:                                                    │
│   - Depth Map: 3D geometry constraints                      │
│   - Normal Map: Surface orientation for lighting            │
│   - Alpha Map: Transmission blending weight                 │
│ Prompt:  "high quality photo, realistic lighting"           │
│ Strength: 0.2 denoise (low - preserve Layer 1 quality)      │
└─────────────────────────────────────────────────────────────┘
                      ↓
                FINAL COMPOSITE
```

---

### 3.2 Mathematical Formulation

**Step 1: Traditional Swap (Layer 1)**
```
S_face = GAN_swap(source_identity, target_frame)
M_opaque = XSeg(target_frame)  # Binary mask
```

**Step 2: DKT Volume Estimation (Layer 2)**
```
# Input: Temporal window
frames = [target_frame_{t-2}, ..., target_frame_{t+2}]

# DKT inference
D_depth, N_normal, A_alpha = DKT(frames)

# Extract transparent region only
M_transparent = (A_alpha > 0.1) & (M_opaque == 0)
```

**Step 3: Physics-Based Composite (Layer 3)**
```
# Prepare ControlNet inputs
control_depth = D_depth * M_transparent
control_normal = N_normal * M_transparent

# Generate re-lit composite
F_composite = ControlNet_Inpaint(
    image = S_face,
    mask = M_transparent,
    control_depth = control_depth,
    control_normal = control_normal,
    prompt = "photo of person, natural lighting, photorealistic",
    negative_prompt = "cartoon, 3d render, painting, blur",
    denoise_strength = 0.2,  # Low - preserve face quality
    controlnet_conditioning_scale = 0.8  # Medium - guide not override
)

# Final alpha blend
alpha_expanded = A_alpha[:,:,None]  # Broadcast to RGB
result = (
    S_face * (1 - alpha_expanded) +  # Opaque face regions
    F_composite * alpha_expanded      # Transparent volumes
)
```

---

## 4. Implementation Plan

### 4.1 Milestone 1: Single Frame Test (Week 1)

**Goal:** Prove DKT + ControlNet works on a single corn dog frame

**Tasks:**
1. **Find/Implement DKT**
   - Option A: Use official DKT checkpoint (if available)
   - Option B: Use Depth-Anything V2 + custom transmission estimation
   - Option C: Fine-tune Stable Diffusion depth model on transparent objects

2. **Setup ControlNet Pipeline**
   ```python
   from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
   import torch

   # Load ControlNet for depth + normal
   controlnet_depth = ControlNetModel.from_pretrained(
       "lllyasviel/control_v11f1p_sd15_depth"
   )
   controlnet_normal = ControlNetModel.from_pretrained(
       "lllyasviel/control_v11p_sd15_normalbae"
   )

   # Multi-ControlNet pipeline
   pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
       "runwayml/stable-diffusion-inpainting",
       controlnet=[controlnet_depth, controlnet_normal]
   ).to("cuda")  # or "mps" for M4
   ```

3. **Test on Corn Dog Frame**
   ```python
   # Load test case
   source = load_image("test_data/source_face.jpg")
   target = load_image("test_data/corndog_with_mayo.jpg")

   # Layer 1: Traditional swap
   swapped = run_watserface_swap(source, target, mask_type="region")

   # Layer 2: DKT estimation (mock for now)
   depth_map = estimate_depth(target)  # Placeholder
   normal_map = estimate_normals(depth_map)
   alpha_map = detect_transparency(target, depth_map)

   # Layer 3: ControlNet inpaint
   result = pipe(
       prompt="photo of person eating, natural lighting",
       image=swapped,
       mask_image=alpha_map,
       control_image=[depth_map, normal_map],
       num_inference_steps=20,
       strength=0.2
   ).images[0]

   # Compare
   save_comparison_grid([target, swapped, result], "mayo_test.jpg")
   ```

4. **Success Criteria**
   - ✅ Mayonnaise region looks realistic (not hole or artifact)
   - ✅ Face identity preserved from Layer 1 swap
   - ✅ Lighting consistent with original frame
   - ✅ No obvious AI artifacts

---

### 4.2 Milestone 2: Depth Estimation Integration (Week 2)

**Goal:** Replace placeholder depth with real DKT or equivalent

**Options Analysis:**

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **DKT Official** | Best transparency handling | May not be released yet | Check availability |
| **Depth-Anything V2** | Open source, robust | No transparency-specific training | Good fallback |
| **Marigold** | Diffusion-based, high quality | Slow inference | For production |
| **ZoeDepth** | Fast, good for opaque | Fails on transparent | Not suitable |

**Implementation:**
```python
# watserface/depth/dkt_estimator.py

import torch
from transformers import pipeline

class DKTEstimator:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Large"):
        self.depth_model = pipeline(
            "depth-estimation",
            model=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def estimate(self, frames, temporal_window=5):
        """
        Estimate depth + normals from video frames

        Args:
            frames: List of RGB images (temporal context)
            temporal_window: Number of frames to use

        Returns:
            depth_map: (H, W) float32, metric depth
            normal_map: (H, W, 3) float32, surface normals
            alpha_map: (H, W) float32, transmission coefficient
        """
        # Temporal consistency via multi-frame median
        depth_maps = []
        for frame in frames:
            depth = self.depth_model(frame)["depth"]
            depth_maps.append(depth)

        # Median filter for stability
        depth_stack = np.stack(depth_maps, axis=0)
        depth_map = np.median(depth_stack, axis=0)

        # Compute normals from depth gradient
        normal_map = self.depth_to_normals(depth_map)

        # Detect transparency from depth variance
        depth_variance = np.var(depth_stack, axis=0)
        alpha_map = self.variance_to_transmission(depth_variance)

        return depth_map, normal_map, alpha_map

    def depth_to_normals(self, depth):
        """Convert depth map to normal map"""
        # Compute gradients
        zx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        zy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)

        # Normal vector: (-dz/dx, -dz/dy, 1)
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))

        # Normalize
        magnitude = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (magnitude + 1e-8)

        # Map to [0, 1] for visualization
        normal = (normal + 1) / 2

        return normal.astype(np.float32)

    def variance_to_transmission(self, variance, threshold=0.05):
        """
        High variance across frames → likely transparent

        Args:
            variance: (H, W) depth variance
            threshold: Variance cutoff

        Returns:
            alpha: (H, W) transmission coefficient (0=opaque, 1=fully transparent)
        """
        # Normalize variance
        variance_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)

        # Sigmoidal mapping
        alpha = 1 / (1 + np.exp(-10 * (variance_norm - threshold)))

        return alpha.astype(np.float32)
```

---

### 4.3 Milestone 3: ControlNet Optimization (Week 3)

**Goal:** Fine-tune ControlNet parameters for face swapping use case

**Hyperparameter Search:**
```python
# Grid search over key parameters
configs = {
    'denoise_strength': [0.1, 0.2, 0.3, 0.5],
    'controlnet_scale': [0.5, 0.8, 1.0, 1.2],
    'num_steps': [15, 20, 30, 50],
    'guidance_scale': [5.0, 7.5, 10.0]
}

best_config = None
best_score = -inf

for config in itertools.product(*configs.values()):
    result = run_phase_25(test_frame, config)
    score = evaluate_quality(result, ground_truth)

    if score > best_score:
        best_score = score
        best_config = config

print(f"Best config: {best_config}")
print(f"Score: {best_score:.3f}")
```

**Evaluation Metrics:**
- Identity preservation (cosine similarity)
- Transparency realism (human eval)
- Temporal consistency (optical flow)
- Inference speed (FPS)

---

### 4.4 Milestone 4: Video Integration (Week 4)

**Goal:** Process full video with temporal coherence

**Challenges:**
1. **Frame-to-frame consistency** - Avoid flicker
2. **Temporal window management** - Efficient frame buffering
3. **Performance** - Real-time or near-real-time

**Implementation:**
```python
# watserface/processors/modules/transparency_handler.py

class TransparencyHandler:
    def __init__(self, temporal_window=5):
        self.dkt = DKTEstimator()
        self.controlnet = load_controlnet_pipeline()
        self.frame_buffer = deque(maxlen=temporal_window)

    def process_video(self, video_path, source_identity):
        """Process video with DKT transparency handling"""

        cap = cv2.VideoCapture(video_path)
        writer = cv2.VideoWriter(...)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Add to temporal buffer
            self.frame_buffer.append(frame)

            # Need full buffer before processing
            if len(self.frame_buffer) < self.temporal_window:
                continue

            # Layer 1: Traditional swap
            swapped = swap_face(source_identity, frame)

            # Layer 2: DKT estimation
            depth, normal, alpha = self.dkt.estimate(list(self.frame_buffer))

            # Layer 3: ControlNet inpaint (only if transparent region detected)
            if alpha.max() > 0.1:
                result = self.controlnet(
                    swapped, depth, normal, alpha
                )
            else:
                result = swapped  # Skip expensive ControlNet

            writer.write(result)
            frame_idx += 1

        cap.release()
        writer.release()
```

---

## 5. Test Cases

### 5.1 Transparent Liquid (Mayo Test)

**Input:** Corn dog eating scene with mayonnaise
**Expected:** Realistic mayo drip over swapped face
**Metric:** Human eval - "looks real" score ≥ 4/5

### 5.2 Transparent Solid (Glasses)

**Input:** Person wearing prescription glasses
**Expected:** Lens refraction preserved, face visible through glass
**Metric:** Face detection confidence ≥ 0.8 through glasses

### 5.3 Volumetric (Steam/Smoke)

**Input:** Face partially obscured by steam
**Expected:** Gradual fade through volume, not hard cutoff
**Metric:** Depth gradient smooth (no discontinuities)

### 5.4 Negative Test (Opaque)

**Input:** Hand covering face (opaque occlusion)
**Expected:** Traditional XSeg mask works, DKT skipped
**Metric:** Performance identical to non-DKT pipeline

---

## 6. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| DKT Depth Estimation | < 500ms per frame | Batched inference |
| ControlNet Inpaint | < 2s per frame | M4 MPS or CUDA |
| Full Pipeline (with DKT) | < 3s per frame | Including all layers |
| Traditional Pipeline (skip DKT) | < 500ms per frame | Opaque occlusions |

**Optimization Strategies:**
- Cache depth maps for static scenes
- Skip DKT if alpha_map.max() < threshold
- Batch process frames when possible
- Use FP16 inference on GPU

---

## 7. Dependencies

### New Packages
```python
# requirements-phase25.txt
diffusers>=0.25.0
transformers>=4.36.0
controlnet-aux>=0.0.7
depth-anything  # Or alternative depth model
accelerate>=0.25.0  # For optimized inference
```

### Model Downloads (Automatic)
- `lllyasviel/control_v11f1p_sd15_depth` (1.4 GB)
- `lllyasviel/control_v11p_sd15_normalbae` (1.4 GB)
- `runwayml/stable-diffusion-inpainting` (5.2 GB)
- `depth-anything/Depth-Anything-V2-Large` (1.3 GB)

**Total:** ~9 GB additional storage

---

## 8. Fallback Strategy

If DKT proves too slow or unavailable:

**Plan B: Hybrid Approach**
1. Use Depth-Anything V2 for depth
2. Manual transmission estimation via color/texture analysis
3. Simpler alpha blending instead of full ControlNet

**Plan C: Enhanced XSeg**
1. Train XSeg to output soft masks (0-1 alpha) instead of binary
2. Multi-class segmentation: opaque, semi-transparent, transparent
3. Use alpha-aware blending without ControlNet

---

## 9. Success Criteria

Phase 2.5 is **COMPLETE** when:

✅ Mayonnaise test passes (human eval ≥ 4/5)
✅ Glasses test passes (face detection ≥ 0.8)
✅ Steam test shows smooth gradients
✅ Performance acceptable (< 3s per frame)
✅ Integration with existing pipeline seamless
✅ Fallback to traditional pipeline when appropriate

---

## 10. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| DKT model unavailable | High | Use Depth-Anything + custom alpha |
| ControlNet too slow | Medium | Lower resolution, FP16, batch processing |
| Quality worse than traditional | Critical | Make DKT optional, allow user toggle |
| Temporal flicker in video | Medium | Optical flow smoothing, frame blending |

---

## 11. References

- **DKT Paper:** Xu et al., "Diffusion Knows Transparency" (2025)
- **Depth-Anything V2:** https://github.com/DepthAnything/Depth-Anything-V2
- **ControlNet:** https://github.com/lllyasviel/ControlNet
- **Stable Diffusion Inpainting:** https://huggingface.co/runwayml/stable-diffusion-inpainting

---

**Status:** Not Started (blocked by Milestone 0)
**Owner:** TBD
**Dependencies:** Milestone 0 complete, ControlNet setup
**Last Updated:** 2026-01-25
