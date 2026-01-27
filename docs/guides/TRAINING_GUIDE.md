# WatserFace Training Guide

**Complete guide to training custom face models with WatserFace**

**Version:** 0.13.0-dev
**Last Updated:** 2026-01-25

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Training Workflow](#2-training-workflow)
3. [Identity Training](#3-identity-training)
4. [LoRA Fine-Tuning](#4-lora-fine-tuning)
5. [XSeg Occlusion Training](#5-xseg-occlusion-training)
6. [Performance Optimization](#6-performance-optimization)
7. [Troubleshooting](#7-troubleshooting)
8. [Advanced Topics](#8-advanced-topics)

---

## 1. Quick Start

### 1.1 Installation

```bash
# Install base dependencies
pip install -r requirements.txt

# Install training extensions
pip install -r requirements-training.txt
```

**Training Dependencies:**
- PyTorch 2.0+
- Transformers 4.30+
- Diffusers 0.20+
- Accelerate 0.25+

---

### 1.2 5-Minute First Training

```bash
# Launch UI
python watserface.py run

# In browser:
1. Go to "Training" tab
2. Upload 10-20 images of your face (source)
3. Set model name: "MyIdentity"
4. Set epochs: 30 (quick test)
5. Click "Start Identity Training"
6. Wait ~1-2 hours
7. Use trained model in "Swap" tab
```

---

## 2. Training Workflow

### 2.1 The Complete Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   TRAINING WORKFLOW                          │
└──────────────────────────────────────────────────────────────┘

PHASE 1: Identity Training (ONE TIME)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Source media (images/video of Person A)
        ↓
    Extract faces (max 1000, sampled)
        ↓
    Smooth landmarks (temporal filtering)
        ↓
    Train InstantID model (30-200 epochs)
        ↓
Output: Identity Profile (.json)
        - Mean embeddings (512-dim)
        - Saved to models/identities/

Time: 1-5 hours (depends on epochs)
✅ REUSE FOREVER - Train once, use everywhere

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 2: LoRA Training (PER TARGET) - OPTIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Identity Profile + Target Video
        ↓
    Load frozen base model
        ↓
    Add LoRA layers (rank 4-128)
        ↓
    Train adapter (50-200 epochs)
        ↓
Output: LoRA Model (.onnx)
        - Size: ~1% of full model
        - Saved to .assets/models/trained/

Time: 2-10 hours (depends on epochs)
✅ Create specialized models per target scene

Note: LoRA training currently has architecture issues
See ARCHITECTURE.md for details

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 3: Face Swapping (PRODUCTION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Identity Profile (or pretrained model) + Target media
        ↓
    Select model in Swap tab
        ↓
    Process frames
        ↓
Output: High-quality face swap

Time: Real-time to 2s per frame (depends on model)
```

---

### 2.2 Which Training Type Do You Need?

| Use Case | Training Type | Time | Quality |
|----------|--------------|------|---------|
| **Quick test** | Pretrained models only | 0 | Good (85%) |
| **General purpose** | Identity training | 1-5 hrs | Very good (90%) |
| **Specific target** | Identity + LoRA | 3-15 hrs | Excellent (95%)* |
| **Production quality** | All + XSeg | 5-20 hrs | Maximum (98%)* |

*LoRA currently broken - see ARCHITECTURE.md

---

## 3. Identity Training

### 3.1 Overview

**Goal:** Create a reusable identity profile from source media

**Output:** JSON file containing averaged face embeddings

**Use:** Load in Swap tab to apply this identity to any target

---

### 3.2 Step-by-Step

#### Step 1: Prepare Source Media

**Requirements:**
- 10-100+ images or video of same person
- Clear, well-lit faces
- Various angles and expressions (recommended)
- Resolution: 512x512+ (will auto-resize)

**Best Practices:**
```
Good source set:
✅ 50 images, various angles
✅ Different expressions (smiling, neutral, talking)
✅ Consistent lighting
✅ Minimal occlusions

Poor source set:
❌ 5 images, all frontal
❌ Heavily shadowed/backlit
❌ Sunglasses, masks, hands in front
```

#### Step 2: Launch Training Tab

```bash
python watserface.py run --ui-layouts training
```

Or set in `watserface.ini`:
```ini
ui_layouts = swap training
```

#### Step 3: Configure Training

**UI Options:**
- **Dataset Path:** Upload images or select folder
- **Model Name:** e.g., "Samantha" (alphanumeric, no spaces)
- **Epochs:** 30-200
  - 30: Quick test (~1 hour)
  - 100: Production (~5 hours)
  - 200: Maximum quality (~10 hours)
- **Batch Size:** 4 (default) or 8 (if enough VRAM)
- **Learning Rate:** 0.001 (default - usually don't change)

#### Step 4: Start Training

Click **"Start Identity Training"**

**What Happens:**
1. Frame extraction from video (or use images directly)
2. Face detection (MediaPipe 478 landmarks)
3. Temporal smoothing (Savitzky-Golay filter)
4. Model training (PyTorch on MPS/CUDA/CPU)
5. Checkpoint saving every 10 epochs
6. Final export to `models/identities/`

**Monitor Progress:**
- Epoch counter
- Loss curve (should decrease)
- ETA remaining
- Sample output (every N epochs)

#### Step 5: Resume if Interrupted

Training can be paused/resumed:

```bash
# Press Ctrl+C to pause

# Resume: Launch UI again, same model name
# Will detect checkpoint and continue from last epoch
```

---

### 3.3 Output Files

```
models/identities/
└── Samantha.json
    {
      "id": "samantha",
      "name": "Samantha",
      "embedding_mean": [512 floats],  # Average face embedding
      "embedding_std": [512 floats],   # Variance
      "num_frames": 847,                # Frames used
      "quality_stats": {
        "avg_face_score": 0.94,
        "landmark_variance": 12.3
      }
    }

.assets/models/trained/
├── Samantha.onnx          # ONNX model (for inference)
├── Samantha.pth           # PyTorch checkpoint (for resume)
└── Samantha.hash          # Model verification
```

---

### 3.4 Using Your Identity

**In Swap Tab:**
1. Go to Swap tab
2. Face Swapper Model dropdown → Select "Samantha"
3. Upload target media
4. Process as usual

---

## 4. LoRA Fine-Tuning

**Status:** ⚠️ Architecture currently broken

See [ARCHITECTURE.md](../ARCHITECTURE.md) for technical details on the bug.

**When Fixed:**

### 4.1 What is LoRA?

**LoRA (Low-Rank Adaptation):** Efficient fine-tuning method

**Benefits:**
- Train 100x faster than full model
- 1% file size (1-5 MB vs 100-500 MB)
- Create unlimited variations from one identity

**Use Case:** Source→Target pairs with specific lighting/style

---

### 4.2 Workflow (When Implemented)

```
1. Train Identity Profile (ONE TIME)
   ↓
2. For each target scene:
   - Go to Modeler tab
   - Select identity: "Samantha"
   - Upload target video
   - Train LoRA: "samantha_to_corndog"
   ↓
3. Use LoRA in Swap tab
   - Select "samantha_to_corndog_lora"
   - Superior quality for that specific target
```

---

### 4.3 Configuration (Future)

**LoRA Rank:**
- 4-8: Fast, 90% quality
- 16: Balanced (recommended)
- 32-64: High quality, slower
- 128: Maximum, very slow

**Training Epochs:**
- 50: Quick test
- 100: Good quality
- 200: Production

---

## 5. XSeg Occlusion Training

### 5.1 What is XSeg?

**XSeg:** U-Net model that segments face from occlusions

**Occlusions:** Hands, food, hair, objects covering face

**Output:** Binary mask (1 = face, 0 = occlusion)

---

### 5.2 When to Train XSeg

**Use Pretrained XSeg When:**
- General occlusions (hands, common objects)
- Standard use cases

**Train Custom XSeg When:**
- Unusual occlusions (specific props, costumes)
- High-precision masks needed
- Pretrained XSeg failing on your data

---

### 5.3 XSeg Training Workflow

#### Step 1: Prepare Target Video

Upload video with occlusions you want to handle

#### Step 2: Auto-Generate Masks

**Convex Hull Method:**
- Automatically creates masks from landmarks
- Works for simple cases
- Fast

**Manual Annotation (Better):**
- Use Gradio ImageEditor
- Paint over occlusions
- Frame-by-frame or keyframe interpolation

#### Step 3: Train XSeg Model

**Configuration:**
- Model Name: "MyXSeg"
- Epochs: 50-100
- Architecture: U-Net ResNet34 backbone

**Training Time:** 2-5 hours

#### Step 4: Use in Swap

```python
# In Swap tab
face_occluder_model = "MyXSeg"
```

---

## 6. Performance Optimization

### 6.1 Speed Improvements

**Frame Sampling (10x Speedup):**
```python
# Before: Use all 2500 frames (28 min/epoch)
# After: Sample 1000 frames uniformly (2-3 min/epoch)

max_frames = 1000  # Default in optimized version
```

**Device Selection:**
```python
# Automatic detection
# M4 MacBook Air: Uses MPS (Apple Silicon)
# NVIDIA GPU: Uses CUDA
# Fallback: CPU (slow)

# Force specific device (if needed):
execution_device_id = 0  # GPU 0
execution_providers = mps  # or cuda, cpu
```

**Batch Size:**
```python
# Larger batch = faster, but more memory
batch_size = 4   # Default (safe)
batch_size = 8   # If 16GB+ VRAM
batch_size = 16  # If 24GB+ VRAM
```

---

### 6.2 Memory Optimization

**M4 MacBook Air (16GB RAM):**
```python
# Recommended settings
batch_size = 4
max_frames = 1000
threads = 4  # Performance cores only

# Memory usage: ~10-12 GB (leaves 4-6 GB for system)
```

**NVIDIA GPU:**
```python
# Example: RTX 4080 (16GB VRAM)
batch_size = 8
max_frames = 2000
mixed_precision = True  # FP16 training
```

**Low Memory Systems (<8GB RAM):**
```python
batch_size = 1
max_frames = 500
gradient_accumulation_steps = 4  # Simulate batch_size=4
```

---

### 6.3 Checkpoint Strategy

**Auto-Save Every N Epochs:**
```python
save_interval = 10  # Save every 10 epochs

# Files saved:
# - model.pth: Full state (model + optimizer)
# - model_epoch_N.pth: Periodic backups
```

**Resume Training:**
```bash
# Automatically resumes if:
# 1. Same model name
# 2. Checkpoint exists (.pth file)
# 3. Frames already extracted (cached)

# Manual resume:
python watserface.py run
# UI detects checkpoint, shows "Resume from epoch 47?"
```

---

### 6.4 Expected Performance

**M4 MacBook Air:**
| Task | Time per Epoch | Total (100 epochs) |
|------|----------------|-------------------|
| Identity Training | 2-3 min | 3-5 hours |
| LoRA Training | 2-4 min | 3-7 hours |
| XSeg Training | 1-2 min | 2-3 hours |

**RTX 4080:**
| Task | Time per Epoch | Total (100 epochs) |
|------|----------------|-------------------|
| Identity Training | 30-60 sec | 50-100 min |
| LoRA Training | 1-2 min | 100-200 min |
| XSeg Training | 20-40 sec | 30-70 min |

---

## 7. Troubleshooting

### 7.1 Common Issues

#### "No faces found in dataset"

**Cause:** Frame extraction failed or video has no visible faces

**Fix:**
1. Check source video has clear faces
2. Try different source video/images
3. Check `.jobs/training_dataset_*/frames/` has extracted images
4. Reduce face_detector_score threshold

---

#### Training loss is NaN

**Cause:** Learning rate too high or gradient explosion

**Fix:**
```python
# Reduce learning rate
learning_rate = 0.0001  # From 0.001

# Reduce batch size
batch_size = 2  # From 4

# Delete checkpoint and restart
rm .jobs/training_dataset_*/model.pth
```

---

#### Training very slow (>10 min/epoch)

**Cause:** Using CPU instead of GPU/MPS

**Fix:**
```python
# Check device detection
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Force MPS on M4
execution_providers = mps

# Or force CUDA
execution_providers = cuda
```

---

#### Trained model not appearing in dropdown

**Cause:** Model not exported to correct location

**Fix:**
1. Check `.assets/models/trained/` contains `[name].onnx`
2. Click refresh button (🔄) in Swap tab
3. Restart UI
4. Check model has corresponding `.hash` file

---

#### Out of memory during training

**Cause:** Batch size too large or too many frames

**Fix:**
```python
# Reduce batch size
batch_size = 2  # Or 1

# Reduce frame count
max_frames = 500

# Enable gradient checkpointing (saves memory)
gradient_checkpointing = True
```

---

### 7.2 Quality Issues

#### Identity not preserved after training

**Symptoms:** Swapped face doesn't look like source

**Diagnosis:**
```python
# Check identity profile quality
cat models/identities/[name].json
# Look for:
# - avg_face_score < 0.8 (low quality frames)
# - num_frames < 100 (not enough data)
```

**Fix:**
1. Add more source images (aim for 50-100+)
2. Improve source quality (better lighting, clearer faces)
3. Train more epochs (100-200)
4. Check embedding similarity in profile

---

#### Blurry or low-res output

**Cause:** Model resolution mismatch

**Fix:**
```python
# Use higher-res swapper
face_swapper_model = simswap_unofficial_512  # Not inswapper_128

# Enable pixel boost
face_swapper_pixel_boost = 512x512  # Or 1024x1024

# Add enhancement
face_enhancer_model = gfpgan_1.4
face_enhancer_blend = 80  # 80% enhancement
```

---

#### Artifacts at face boundary

**Cause:** Mask quality or blending method

**Fix:**
```python
# Try Poisson blending
blend_method = poisson

# Or adjust mask feathering
face_mask_blur = 0.3  # Increase for softer edge

# Use better mask type
face_mask_types = region  # Or bisenet
```

---

## 8. Advanced Topics

### 8.1 Custom Dataset Preparation

**Automated Extraction:**
```bash
# Extract frames from video
python watserface.py run \
  --extract-source path/to/video.mp4 \
  --output-path datasets/my_source/

# Result: Extracted frames in datasets/my_source/frames/
```

**Manual Curation:**
```bash
# Review and delete bad frames
cd datasets/my_source/frames/
# Delete blurry, occluded, or off-angle frames
```

---

### 8.2 Training Resumption Strategies

**Incremental Quality:**
```
Round 1: Train 30 epochs (quick test)
         ↓ Check quality
Round 2: Resume to 50 epochs (if not good enough)
         ↓ Check quality
Round 3: Resume to 100 epochs (if still improving)
         ↓ Final check
```

**Multi-Target LoRAs from One Identity:**
```
1. Train "Samantha" identity ONCE (100 epochs)
   ↓
2. Train LoRA: samantha_to_scene1 (100 epochs)
3. Train LoRA: samantha_to_scene2 (100 epochs)
4. Train LoRA: samantha_to_scene3 (100 epochs)
   ↓
Result: 3 specialized models from 1 identity
```

---

### 8.3 Hyperparameter Tuning

**Grid Search:**
```python
# Test multiple configurations
configs = {
    'learning_rate': [0.0001, 0.0005, 0.001],
    'batch_size': [2, 4, 8],
    'lora_rank': [8, 16, 32]
}

best_config = None
best_quality = 0

for lr, bs, rank in itertools.product(*configs.values()):
    model = train_identity(learning_rate=lr, batch_size=bs)
    quality = evaluate_model(model, test_set)

    if quality > best_quality:
        best_quality = quality
        best_config = (lr, bs, rank)
```

---

### 8.4 Model Validation

**Identity Similarity Check:**
```python
from insightface.app import FaceAnalysis

app = FaceAnalysis()

# Extract embedding from trained model output
result_faces = app.get(swapped_image)
result_emb = result_faces[0].normed_embedding

# Compare to source
source_faces = app.get(source_image)
source_emb = source_faces[0].normed_embedding

similarity = np.dot(result_emb, source_emb)
print(f"Identity similarity: {similarity:.3f}")
# Target: ≥ 0.85 for good quality
```

---

## 9. Best Practices Summary

### ✅ Do This

- Start with 30-50 epoch test before committing to 200
- Use high-quality source images (clear, well-lit)
- Enable checkpointing (auto-saves progress)
- Monitor training loss (should decrease steadily)
- Validate model quality before production use
- Keep source datasets organized

### ❌ Don't Do This

- Train on low-res or heavily compressed sources
- Skip validation step (always test first)
- Delete checkpoints mid-training (you'll lose progress)
- Ignore out-of-memory errors (reduce batch size)
- Train beyond convergence (wastes time)
- Mix multiple identities in one training run

---

## 10. Getting Help

**Documentation:**
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [Milestone 0](../architecture/MILESTONE_0_BASELINE.md) - Quality validation
- [Phase 2.5](../architecture/PHASE_2.5_DKT_POC.md) - Transparency handling

**Community:**
- GitHub Issues: Report bugs
- GitHub Discussions: Ask questions
- Training logs: Share for troubleshooting

**Debug Logs:**
```bash
# Enable verbose logging
python watserface.py run --log-level debug

# Logs saved to: .logs/training_[timestamp].log
```

---

## 11. FAQ

**Q: How much data do I need?**
A: Minimum 10 images, recommended 50-100+ for production quality

**Q: Can I train on CPU?**
A: Yes, but 10-20x slower. GPU/MPS strongly recommended.

**Q: How long does training take?**
A: Identity: 1-5 hours. LoRA: 3-10 hours (when fixed). XSeg: 2-5 hours.

**Q: Can I use the model commercially?**
A: Check original model licenses. InstantID: Research only. InSwapper: Permissive.

**Q: Do I need to retrain for each target?**
A: No! Identity profiles work on any target. LoRA is optional for extra quality.

**Q: What's the difference between Identity and LoRA training?**
A: Identity = general-purpose face model. LoRA = specialized fine-tune for specific target.

**Q: Can I combine multiple identities?**
A: Not in a single model. Train separate identities, then use in different projects.

---

**Last Updated:** 2026-01-25
**Version:** 0.13.0-dev
**Status:** Active Development
