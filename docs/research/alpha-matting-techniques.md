# Alpha Matting Techniques for Semi-Transparent Occlusion Handling

## Overview

This document covers alpha matting approaches for the WatserFace "Mayonnaise Strategy" - compositing face swaps under semi-transparent occlusions.

**Core Formula:** `Final = (Dirty_Swap * (1 - Alpha)) + (Original_Target * Alpha)`

---

## 1. FaceMat (Recommended for Production)

### Repository & Paper

**Repository:** https://github.com/hyebin-c/FaceMat  
**Paper:** arXiv:2508.03055 (ACM Multimedia 2025)

**Citation:**
```bibtex
@article{cho2025facemat,
  title={Uncertainty-Guided Face Matting for Occlusion-Aware Face Transformation},
  author={Hyebin Cho and Jaehyup Lee},
  journal={arXiv preprint arXiv:2508.03055},
  year={2025}
}
```

### How It Works

FaceMat uses a **two-stage training pipeline** with uncertainty-guided knowledge distillation:

**Stage 1: Boundary-Aware Learning**
- Teacher model predicts alpha matte AND per-pixel uncertainty
- Uses Negative Log-Likelihood (NLL) loss
- Focuses on ambiguous boundary regions

**Stage 2: Uncertainty-Guided Distillation**
- Student model trained trimap-free
- Teacher's uncertainty map guides spatial weighting
- Forces focus on uncertain/ambiguous regions

### Key Features

| Feature | Description |
|---------|-------------|
| **Trimap-Free** | No auxiliary inputs at inference |
| **Uncertainty-Aware** | Per-pixel uncertainty guides learning |
| **Video Matting** | Built on RVM baseline |
| **Occlusion-Specific** | Designed for hands, hair, accessories, smoke |

### Integration

```python
from facemat import FaceMatPredictor

predictor = FaceMatPredictor(checkpoint="facemat_weights.pth")
alpha, uncertainty = predictor.predict(target_image)

# Composite
final = dirty_swap * (1 - alpha) + original_target * alpha
```

---

## 2. MiDaS Depth-to-Alpha (Current Approach)

### Basic Threshold

```python
import numpy as np
import cv2

def depth_to_alpha_threshold(depth_map, threshold=0.74):
    """
    Convert depth map to alpha matte using threshold.
    
    Args:
        depth_map: Normalized depth map [0, 1], where 0=near, 1=far
        threshold: Depth threshold (objects closer than this = foreground)
    
    Returns:
        alpha: Alpha matte [0, 1]
    """
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    alpha = np.where(depth_normalized < threshold, 1.0, 0.0)
    return alpha
```

### Soft Threshold with Gradient

```python
def depth_to_alpha_soft(depth_map, threshold=0.74, smoothness=0.05):
    """
    Convert depth map to alpha with soft transition.
    
    Args:
        depth_map: Normalized depth map [0, 1]
        threshold: Center of transition
        smoothness: Width of transition region
    
    Returns:
        alpha: Soft alpha matte [0, 1]
    """
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    alpha = 1.0 - np.clip((depth_normalized - threshold + smoothness/2) / smoothness, 0, 1)
    return alpha
```

### Pros/Cons

**Pros:**
- Already implemented and tested
- Fast inference (MiDaS is optimized)
- No additional training required

**Cons:**
- Struggles with semi-transparent occlusions (hair, smoke)
- Threshold tuning required per scene
- No semantic understanding

---

## 3. Segment Anything Model (SAM) for Matting

### Recent Developments (2026)

**SAMA: Segment And Matte Anything** (arXiv:2601.12147)
- Unified model for segmentation AND matting
- Extends SAM with alpha matte prediction

**MAM2: Matting Anything 2** (ICLR 2026)
- Video matting for diverse objects
- Promptable with points, boxes, or masks
- Handles transparent/intricate objects

### Integration Pattern

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)

alpha = masks[0].astype(float)
```

---

## 4. Semantic Segmentation to Alpha

### Binary Mask to Alpha

```python
def segmentation_to_alpha(seg_mask, blur_kernel=5):
    """
    Convert binary segmentation mask to alpha with edge smoothing.
    """
    if seg_mask.max() > 1:
        seg_mask = seg_mask / 255.0
    
    alpha = cv2.GaussianBlur(seg_mask.astype(np.float32), 
                             (blur_kernel, blur_kernel), 0)
    return alpha
```

### Multi-class Segmentation

```python
def multiclass_seg_to_alpha(seg_map, foreground_classes=[1, 2, 3]):
    """
    Convert multi-class segmentation to alpha matte.
    """
    alpha = np.isin(seg_map, foreground_classes).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    return alpha
```

---

## 5. Complete Compositing Pipeline

### Standard Mayonnaise Strategy

```python
def composite_with_alpha(dirty_swap, original_target, alpha):
    """
    Composite face swap under occlusion using alpha matte.
    
    Formula: Final = (Dirty_Swap * (1 - Alpha)) + (Original_Target * Alpha)
    
    Args:
        dirty_swap: Face-swapped image (swapped "through" occlusion)
        original_target: Original target image (with occlusion)
        alpha: Alpha matte [0, 1], where 1 = occlusion (foreground)
    
    Returns:
        final: Composited result
    """
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, None]
    
    final = dirty_swap * (1 - alpha) + original_target * alpha
    return final.astype(np.uint8)
```

### Uncertainty-Weighted Compositing

```python
def composite_with_uncertainty(dirty_swap, original_target, alpha, uncertainty):
    """
    Composite with uncertainty-aware blending.
    """
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, None]
    if len(uncertainty.shape) == 2:
        uncertainty = uncertainty[:, :, None]
    
    # Reduce alpha in uncertain regions
    alpha_adjusted = alpha * (1 - uncertainty * 0.5)
    
    final = dirty_swap * (1 - alpha_adjusted) + original_target * alpha_adjusted
    return final.astype(np.uint8)
```

---

## 6. Hybrid Approach (Recommended)

Combine depth and semantic information for robust alpha:

```python
def hybrid_alpha_extraction(image, depth_map, face_seg_mask):
    """
    Combine depth and semantic information for robust alpha.
    """
    # 1. Depth-based alpha (for 3D occlusions)
    depth_alpha = depth_to_alpha_soft(depth_map, threshold=0.74, smoothness=0.05)
    
    # 2. Semantic-based alpha (for hair, accessories)
    occlusion_mask = np.isin(face_seg_mask, [2, 3]).astype(np.float32)
    semantic_alpha = cv2.GaussianBlur(occlusion_mask, (5, 5), 0)
    
    # 3. Combine: take maximum (union of occlusions)
    alpha = np.maximum(depth_alpha, semantic_alpha)
    
    # 4. Refine edges
    alpha = cv2.bilateralFilter(alpha.astype(np.float32), 9, 75, 75)
    
    return alpha
```

---

## 7. Comparison Summary

| Method | Quality | Speed | Complexity | Best For |
|--------|---------|-------|------------|----------|
| **MiDaS Depth** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Simple occlusions, real-time |
| **FaceMat** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Hair, semi-transparent, production |
| **SAM/SAMA** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Interactive, diverse objects |
| **Hybrid** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Balanced quality/speed |

---

## 8. Recommendations for WatserFace

1. **Immediate:** Continue with MiDaS depth-to-alpha (threshold 0.74)
2. **Short-term:** Implement hybrid approach (depth + face segmentation)
3. **Medium-term:** Integrate FaceMat for production-quality results
4. **Long-term:** Monitor SAMA/MAM2 for stable releases

---

*Research conducted: January 2026*
*Document version: 1.0*
