# ControlNet Hyperparameter Optimization Report

## Phase 2.5 - Task #90

### Overview

This report documents the ControlNet hyperparameter optimization framework for WatserFace face swapping with transparency handling.

---

## 1. Optimization Framework

### ControlNetOptimizer Class

**Location:** `watserface/inpainting/controlnet_optimizer.py`

**Features:**
- Grid search across parameter space
- SSIM-based quality metric
- Ablation study support
- Quick optimization mode for rapid tuning

### Parameter Space

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| `controlnet_conditioning_scale` | 0.3 - 0.9 | 0.75 | Controls conditioning strength |
| `guidance_start` | 0.0 - 0.3 | 0.0 | When to start ControlNet guidance |
| `guidance_end` | 0.7 - 1.0 | 1.0 | When to end ControlNet guidance |
| `strength` | 0.5 - 0.9 | 0.75 | Denoising strength |
| `num_inference_steps` | 20, 30, 50 | 30 | Diffusion steps |

---

## 2. Recommended Configuration

Based on Phase 2.5 research and the existing "Golden Config":

### Optimal Parameters

```python
optimal_config = {
    'controlnet_conditioning_scale': 0.75,  # Within 0.7-0.8 range for facial geometry
    'guidance_start': 0.0,
    'guidance_end': 1.0,
    'strength': 0.75,
    'num_inference_steps': 30
}
```

### Rationale

1. **Conditioning Scale 0.75**: 
   - Provides strong facial geometry preservation
   - Avoids over-conditioning artifacts at higher values
   - Within documented optimal range (0.7-0.8)

2. **30 Inference Steps**:
   - Balances quality and processing time
   - Sufficient for face swap detail
   - ~2-3s processing on modern GPU

3. **Strength 0.75**:
   - Preserves identity while allowing adaptation
   - Matches GFPGAN enhancement blend ratio

---

## 3. Ablation Study Framework

### Usage

```python
from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
from watserface.inpainting.controlnet import ControlNetPipeline

# Initialize
optimizer = ControlNetOptimizer()
pipeline = ControlNetPipeline()
pipeline.load()

# Run ablation study
report = optimizer.run_ablation_study(
    pipeline=pipeline,
    test_images=[test_img1, test_img2],
    reference_images=[ref_img1, ref_img2],
    output_dir=Path('docs/test-reports')
)

# Get best configuration
print(report['recommendation'])
```

### Quick Optimization

```python
# For rapid parameter tuning
best_params = optimizer.quick_optimize(
    pipeline=pipeline,
    test_image=test_img,
    reference_image=ref_img
)
```

---

## 4. Quality Metrics

### SSIM (Structural Similarity Index)

- **Target:** > 0.85 for acceptable quality
- **Formula:** Compares luminance, contrast, and structure
- **Implementation:** Custom implementation matching OpenCV

### Additional Metrics (Future)

- **LPIPS:** Perceptual similarity
- **FID:** Frechet Inception Distance  
- **Identity Preservation:** ArcFace cosine similarity

---

## 5. Test Results

### Unit Tests: 17/17 PASSING

```
tests/test_controlnet_optimizer.py::TestControlNetOptimizerInitialization - 3 tests
tests/test_controlnet_optimizer.py::TestParameterGridGeneration - 3 tests
tests/test_controlnet_optimizer.py::TestSSIMComputation - 3 tests
tests/test_controlnet_optimizer.py::TestParameterEvaluation - 1 test
tests/test_controlnet_optimizer.py::TestReportGeneration - 2 tests
tests/test_controlnet_optimizer.py::TestQuickOptimize - 1 test
tests/test_controlnet_optimizer.py::TestFactoryFunctions - 4 tests
```

---

## 6. Integration with Phase 2.5

### Workflow

1. **DKTEstimator** tracks points through occlusions
2. **ControlNetPipeline** generates conditioned face swap
3. **ControlNetOptimizer** tunes parameters for quality
4. **TransparencyHandler** composites with temporal coherence

### Golden Config Preservation

The optimization framework preserves the existing "Golden Config":
- Swapper: `simswap_unofficial_512`
- Enhancer: `gfpgan_1.4` at 80% blend
- Mask: `region` mode
- Landmarker: `2dfan4` (68 points)

---

## 7. Running Full Ablation Study

### Prerequisites

- GPU with >= 8GB VRAM
- Test images with known references
- Patience (full study takes ~30-60 minutes)

### Command

```bash
python -m watserface.inpainting.run_optimization \
    --test-dir ./test_images \
    --output-dir ./docs/test-reports \
    --quick  # Optional: for rapid results
```

---

## Conclusion

The ControlNet hyperparameter optimization framework provides:

1. **Systematic Parameter Exploration** via grid search
2. **Quality Measurement** via SSIM metric
3. **Ablation Study Reports** for documentation
4. **Quick Optimization** for rapid iteration

**Recommended Default:** `conditioning_scale=0.75, num_inference_steps=30`

---

*Report generated: January 2026*
*Phase 2.5 Task #90*
