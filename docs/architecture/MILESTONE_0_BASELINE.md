# Milestone 0: Baseline Quality Validation

**Goal:** Prove WatserFace produces pixel-perfect swaps on clean, non-occluded faces that match or exceed existing tools.

**Priority:** 🎯 **CRITICAL - MUST COMPLETE BEFORE PHASE 2.5/3**

**Status:** Not Started

---

## 1. Why This Milestone Exists

### The Problem

Phase 2.5/3 innovations (DKT transparency, inverted compositing, ControlNet refinement) **solve occlusion problems, not base swap quality**.

Before adding complexity, we must validate that:
- ✅ WatserFace's foundation is solid
- ✅ Clean swaps match or beat FaceSwap.dev
- ✅ No regressions from architecture changes

### What We're Testing

**In Scope:**
- Non-occluded face swaps (clean source + clean target)
- Pretrained model quality (InSwapper, HyperSwap, SimSwap)
- Paste/blend methods (linear, Poisson, multi-band)
- Color correction and edge quality

**Out of Scope:**
- Occluded faces (Phase 2.5/3 handles this)
- LoRA training (currently broken, separate fix)
- XSeg masks (tested separately)

---

## 2. Acceptance Criteria

### Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Identity Similarity** | ≥ 0.85 | Cosine similarity of InsightFace embeddings |
| **Background Preservation** | SSIM ≥ 0.90 | Structural similarity on non-face regions |
| **Perceptual Quality** | LPIPS ≤ 0.15 | Learned perceptual distance |
| **Color Consistency** | ΔE < 5.0 | CIE2000 color difference |
| **Sharpness** | ≥ baseline | Laplacian variance |
| **Edge Artifacts** | None visible | Human evaluation |

### Qualitative Requirements

✅ **No visible seams** at mask boundary
✅ **Color matches** target lighting/skin tone
✅ **Sharpness maintained** (no blur from blending)
✅ **Natural appearance** (not "obviously AI")
✅ **Competitive with FaceSwap.dev** on same inputs

---

## 3. Test Plan

### 3.1 Test Dataset

**Clean Face Pairs (20 pairs):**

```
test_data/
├── clean_pairs/
│   ├── pair_001/
│   │   ├── source.jpg          # Clean frontal face A
│   │   ├── target.jpg          # Clean frontal face B
│   │   └── ground_truth.jpg    # (Optional) Manual reference
│   ├── pair_002/
│   ├── ...
│   └── pair_020/
```

**Selection Criteria:**
- ✅ Frontal faces (±15° rotation max)
- ✅ Good lighting (no harsh shadows)
- ✅ Clear features (eyes, nose, mouth visible)
- ✅ No occlusions (hands, objects, hair)
- ✅ Various demographics (skin tone, gender, age)
- ✅ Similar pose/expression (easier baseline)

### 3.2 Test Matrix

Run all combinations:

**Models:** `inswapper_128`, `hyperswap_1a_256`, `simswap_unofficial_512`
**Blend Methods:** `linear`, `poisson`, `multiband` (to implement)
**Enhancements:** `none`, `gfpgan_1.4` (80% blend)
**Face Masks:** `box`, `region`, `bisenet`

**Total Configurations:** 3 models × 3 blends × 2 enhancements × 3 masks = 54 variants per pair

---

### 3.3 Test Script

```python
# test_baseline_quality.py

import numpy as np
import cv2
from insightface.app import FaceAnalysis
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips

class BaselineValidator:
    def __init__(self):
        self.face_app = FaceAnalysis()
        self.lpips_model = lpips.LPIPS(net='alex')

    def validate_swap(self, source_path, target_path, output_path):
        """Run full validation suite on a swap"""

        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
        result = cv2.imread(output_path)

        metrics = {
            # Identity preservation
            'identity_similarity': self.measure_identity_similarity(result, source),

            # Background preservation
            'ssim_background': self.measure_background_preservation(result, target),
            'psnr_background': peak_signal_noise_ratio(target, result),

            # Perceptual quality
            'lpips': self.measure_perceptual_distance(result, target),

            # Color consistency
            'color_delta_e': self.measure_color_consistency(result, target),

            # Sharpness
            'face_sharpness': self.measure_sharpness(result),

            # Edge quality
            'edge_artifacts': self.detect_edge_artifacts(result, mask),
        }

        return metrics

    def measure_identity_similarity(self, swapped_image, source_image):
        """Cosine similarity of face embeddings"""
        source_faces = self.face_app.get(source_image)
        result_faces = self.face_app.get(swapped_image)

        if not source_faces or not result_faces:
            return 0.0

        source_emb = source_faces[0].normed_embedding
        result_emb = result_faces[0].normed_embedding

        return np.dot(source_emb, result_emb)

    def measure_background_preservation(self, result, target):
        """SSIM on non-face regions"""
        mask = self.get_face_mask(target)
        background_only = np.where(mask == 0)

        return structural_similarity(
            result[background_only],
            target[background_only],
            channel_axis=-1
        )

    def measure_perceptual_distance(self, result, target):
        """LPIPS perceptual loss"""
        result_tensor = self.to_tensor(result)
        target_tensor = self.to_tensor(target)

        return self.lpips_model(result_tensor, target_tensor).item()

    def measure_color_consistency(self, result, target):
        """CIE2000 color difference"""
        result_face = self.crop_face(result)
        target_face = self.crop_face(target)

        # Convert to LAB color space
        result_lab = cv2.cvtColor(result_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)

        # Compute mean delta E
        delta_e = self.cie2000_delta_e(result_lab, target_lab)
        return np.mean(delta_e)

    def measure_sharpness(self, image):
        """Laplacian variance"""
        face_crop = self.crop_face(image)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def detect_edge_artifacts(self, result, mask):
        """Detect discontinuities at mask boundary"""
        # Dilate mask to get boundary region
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        boundary = dilated - mask

        # Compute gradient magnitude at boundary
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Mean gradient at boundary
        boundary_gradient = grad_mag[boundary > 0].mean()

        # Score: lower is better
        return boundary_gradient

# Run validation
validator = BaselineValidator()

for pair in test_pairs:
    for config in configurations:
        output = run_watserface(pair.source, pair.target, config)
        metrics = validator.validate_swap(pair.source, pair.target, output)

        print(f"{pair.name} | {config.name}:")
        print(f"  Identity: {metrics['identity_similarity']:.3f}")
        print(f"  SSIM: {metrics['ssim_background']:.3f}")
        print(f"  LPIPS: {metrics['lpips']:.3f}")
        print(f"  ΔE: {metrics['color_delta_e']:.2f}")
        print(f"  Sharpness: {metrics['face_sharpness']:.1f}")
        print(f"  Edge: {metrics['edge_artifacts']:.2f}")

        # Check acceptance criteria
        passed = (
            metrics['identity_similarity'] >= 0.85 and
            metrics['ssim_background'] >= 0.90 and
            metrics['lpips'] <= 0.15 and
            metrics['color_delta_e'] < 5.0
        )

        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}\n")
```

---

### 3.4 Comparison with FaceSwap.dev

Run identical test on FaceSwap.dev:

```bash
# Setup FaceSwap.dev
cd /Users/kendrick/Documents/dev/faceswap.dev
source faceswap/venv/bin/activate

# Extract faces from source
python faceswap.py extract \
  -i test_data/clean_pairs/pair_001/source.jpg \
  -o test_data/faceswap_extract/source

# Extract faces from target
python faceswap.py extract \
  -i test_data/clean_pairs/pair_001/target.jpg \
  -o test_data/faceswap_extract/target

# Train model (or use pretrained)
python faceswap.py train \
  -A test_data/faceswap_extract/source \
  -B test_data/faceswap_extract/target \
  -m test_data/faceswap_models/pair_001

# Convert (swap)
python faceswap.py convert \
  -i test_data/clean_pairs/pair_001/target.jpg \
  -o test_data/faceswap_output/pair_001.jpg \
  -m test_data/faceswap_models/pair_001 \
  --mask-type predicted \
  --color-adjustment avg-color
```

**Compare Results:**
- Side-by-side visual inspection
- Run same metrics on both outputs
- Identify WatserFace weaknesses

---

## 4. Expected Issues & Fixes

### 4.1 Interpolation Artifacts

**Symptom:** Pixelation or blocky edges

**Diagnosis:**
```python
# watserface/face_helper.py:139
crop_vision_frame = cv2.warpAffine(..., flags=cv2.INTER_AREA)
```

**Problem:** `INTER_AREA` good for downsampling, bad for upsampling

**Fix:**
```python
if bounding_box[2] - bounding_box[0] > crop_size[0]:
    interpolation_method = cv2.INTER_AREA  # Downsampling
else:
    interpolation_method = cv2.INTER_LANCZOS4  # Upsampling
```

---

### 4.2 Color Mismatch

**Symptom:** Swapped face has different skin tone than target

**Diagnosis:** No color correction in paste_back

**Fix:** Add histogram matching
```python
def match_target_color(swapped_face, target_face, mask):
    """Match color statistics in LAB space"""
    swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)

    for i in range(3):
        swapped_lab[:,:,i] = (
            (swapped_lab[:,:,i] - swapped_lab[:,:,i].mean()) *
            (target_lab[:,:,i].std() / swapped_lab[:,:,i].std()) +
            target_lab[:,:,i].mean()
        )

    return cv2.cvtColor(swapped_lab, cv2.COLOR_LAB2BGR)
```

**Apply in paste_back:**
```python
# watserface/face_helper.py:236
def paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix):
    # Color correction before pasting
    crop_vision_frame = match_target_color(
        crop_vision_frame,
        temp_vision_frame,  # Target for matching
        crop_mask
    )

    # ... rest of paste logic
```

---

### 4.3 Edge Artifacts (Halo Effect)

**Symptom:** Visible seam at mask boundary

**Diagnosis:** Linear blending too abrupt

**Fix:** Implement multi-band blending
```python
def laplacian_pyramid_blend(img1, img2, mask, levels=6):
    """Multi-resolution blending for seamless composites"""

    # Build Gaussian pyramids
    GP_img1 = [img1]
    GP_img2 = [img2]
    GP_mask = [mask]

    for i in range(levels):
        GP_img1.append(cv2.pyrDown(GP_img1[i]))
        GP_img2.append(cv2.pyrDown(GP_img2[i]))
        GP_mask.append(cv2.pyrDown(GP_mask[i]))

    # Build Laplacian pyramids
    LP_img1 = [GP_img1[levels-1]]
    LP_img2 = [GP_img2[levels-1]]

    for i in range(levels-1, 0, -1):
        size = (GP_img1[i-1].shape[1], GP_img1[i-1].shape[0])
        L1 = GP_img1[i-1] - cv2.pyrUp(GP_img1[i], dstsize=size)
        L2 = GP_img2[i-1] - cv2.pyrUp(GP_img2[i], dstsize=size)
        LP_img1.append(L1)
        LP_img2.append(L2)

    # Blend each level
    LS = []
    for l1, l2, mask in zip(LP_img1, LP_img2, GP_mask):
        mask_3ch = np.stack([mask]*3, axis=-1)
        ls = l1 * mask_3ch + l2 * (1 - mask_3ch)
        LS.append(ls)

    # Reconstruct
    result = LS[0]
    for i in range(1, levels):
        size = (LS[i].shape[1], LS[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + LS[i]

    return result
```

---

### 4.4 Sharpness Loss

**Symptom:** Blurry result after blending

**Fix:** Unsharp mask after paste
```python
def apply_unsharp_mask(image, sigma=1.0, strength=0.5):
    """Sharpen image using unsharp mask"""
    blurred = cv2.GaussianBlur(image, (0,0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

# Apply after paste_back
result = paste_back(...)
result = apply_unsharp_mask(result, sigma=1.0, strength=0.3)
```

---

## 5. Implementation Checklist

### Phase 1: Test Infrastructure
- [ ] Create test dataset (20 clean face pairs)
- [ ] Implement `BaselineValidator` class
- [ ] Setup FaceSwap.dev comparison environment
- [ ] Create automated test runner

### Phase 2: Current State Baseline
- [ ] Run WatserFace on test dataset (all configs)
- [ ] Run FaceSwap.dev on same dataset
- [ ] Generate comparison report
- [ ] Identify quality gaps

### Phase 3: Implement Fixes
- [ ] Fix interpolation (INTER_LANCZOS4 for upsampling)
- [ ] Add color correction (LAB histogram matching)
- [ ] Implement multi-band blending
- [ ] Add unsharp mask option
- [ ] Test each fix in isolation

### Phase 4: Validation
- [ ] Re-run full test suite
- [ ] Verify acceptance criteria met
- [ ] Visual inspection (human evaluation)
- [ ] Document findings

### Phase 5: Production Integration
- [ ] Make best methods default
- [ ] Add configuration options for user choice
- [ ] Update UI tooltips/help text
- [ ] Performance benchmarks

---

## 6. Success Metrics Dashboard

```
╔══════════════════════════════════════════════════════════════╗
║              MILESTONE 0 VALIDATION RESULTS                  ║
╠══════════════════════════════════════════════════════════════╣
║  Test Cases: 20 pairs × 54 configs = 1080 swaps             ║
║                                                              ║
║  Identity Similarity:        0.87 ± 0.03  ✅ (target: ≥0.85)║
║  Background SSIM:            0.93 ± 0.02  ✅ (target: ≥0.90)║
║  Perceptual Quality (LPIPS): 0.12 ± 0.04  ✅ (target: ≤0.15)║
║  Color Consistency (ΔE):     4.2 ± 1.1   ✅ (target: <5.0) ║
║  Edge Artifacts:             None visible ✅                 ║
║                                                              ║
║  FaceSwap.dev Comparison:                                    ║
║    Identity:   WatserFace 0.87 vs FaceSwap 0.86  ✅ Better  ║
║    SSIM:       WatserFace 0.93 vs FaceSwap 0.91  ✅ Better  ║
║    LPIPS:      WatserFace 0.12 vs FaceSwap 0.14  ✅ Better  ║
║                                                              ║
║  Status: ✅ MILESTONE 0 COMPLETE                            ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 7. Risk Mitigation

### Risk: Current quality is worse than FaceSwap.dev

**Mitigation:**
- Start testing early to identify gaps
- Implement proven techniques from FaceSwap.dev
- Allow time for iterative fixes

### Risk: Optimization breaks quality

**Mitigation:**
- Regression test suite
- Version control checkpoints
- Separate optimization from functionality

### Risk: Test dataset not representative

**Mitigation:**
- Diverse demographics in dataset
- Community validation
- Real-world examples

---

## 8. Timeline Estimate

**Phase 1 (Test Infrastructure):** 2-3 days
**Phase 2 (Baseline Measurement):** 1 day
**Phase 3 (Implement Fixes):** 3-5 days
**Phase 4 (Validation):** 1-2 days
**Phase 5 (Production Integration):** 1 day

**Total:** 8-12 days

---

## 9. Exit Criteria

Milestone 0 is **COMPLETE** when:

✅ All acceptance criteria met (see Section 2)
✅ Test suite passing on 95%+ of cases
✅ Visual inspection confirms no artifacts
✅ Competitive with or better than FaceSwap.dev
✅ Documentation updated
✅ Default configuration optimized

**Only then proceed to Phase 2.5/3 development.**

---

## 10. References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
- [FaceSwap.dev Converter](https://github.com/deepfakes/faceswap/blob/master/lib/convert.py)
- [Multi-band Blending Paper](https://persci.mit.edu/pub_pdfs/pyramid83.pdf) - Burt & Adelson, 1983

---

**Status:** Not Started
**Owner:** TBD
**Last Updated:** 2026-01-25
