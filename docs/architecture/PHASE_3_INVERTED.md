# Phase 3: Inverted Compositing Pipeline

**Goal:** Intelligent conditional execution - use inverted compositing for transparent occlusions, traditional approach for clean swaps.

**Status:** Not Started (blocked by Phase 2.5)

**Priority:** 🟡 Optimization - Performance & Quality Balance

---

## 1. The Hybrid Strategy

### 1.1 Why Not Always Use DKT?

**Problem:** Phase 2.5 DKT pipeline is expensive:
- Depth estimation: ~500ms
- ControlNet inpaint: ~2s
- **Total:** ~3s per frame vs 500ms traditional

**Observation:** 80% of frames have NO transparent occlusions

**Solution:** Conditional execution based on occlusion classification

---

### 1.2 Decision Tree

```python
def process_frame_intelligent(source_identity, target_frame):
    """
    Adaptive pipeline selection based on occlusion type
    """

    # Step 0: Analyze frame
    occlusion_type, confidence = classify_occlusion(target_frame)

    # Decision tree
    if occlusion_type == 'none':
        # Fast path: No occlusions detected
        return traditional_swap(source_identity, target_frame)

    elif occlusion_type == 'opaque':
        # Medium path: XSeg handles solid objects well
        return traditional_swap_with_xseg(source_identity, target_frame)

    elif occlusion_type == 'transparent':
        # Slow path: Requires DKT + ControlNet
        return dkt_transparency_swap(source_identity, target_frame)

    else:
        # Fallback: Unknown occlusion type
        return traditional_swap_with_xseg(source_identity, target_frame)
```

**Performance Impact:**
```
Video with 1000 frames:
  - 800 frames: clean (500ms each) = 400s
  - 150 frames: opaque (600ms each) = 90s
  - 50 frames: transparent (3s each) = 150s
  Total: 640s (10.6 minutes)

vs Always DKT:
  - 1000 frames × 3s = 3000s (50 minutes)

Speedup: 4.7x faster
```

---

## 2. Occlusion Classification

### 2.1 Multi-Class Classifier

**Architecture:** Lightweight CNN trained on labeled examples

```python
# watserface/processors/modules/occlusion_classifier.py

class OcclusionClassifier:
    """
    Classify occlusion type for intelligent pipeline routing
    """

    CLASSES = {
        0: 'none',        # Clean swap, no occlusions
        1: 'opaque',      # Solid objects (hands, food, hair)
        2: 'transparent'  # Glass, liquid, smoke
    }

    def __init__(self, model_path='.assets/models/occlusion_classifier.onnx'):
        self.session = onnxruntime.InferenceSession(model_path)

    def classify(self, frame, face_bbox):
        """
        Args:
            frame: (H, W, 3) RGB image
            face_bbox: (x1, y1, x2, y2) face location

        Returns:
            class_name: str ('none', 'opaque', 'transparent')
            confidence: float (0-1)
        """
        # Crop to face region + context
        x1, y1, x2, y2 = face_bbox
        margin = int(0.3 * (x2 - x1))  # 30% margin
        crop = frame[
            max(0, y1-margin):min(frame.shape[0], y2+margin),
            max(0, x1-margin):min(frame.shape[1], x2+margin)
        ]

        # Resize to classifier input (224x224)
        crop_resized = cv2.resize(crop, (224, 224))

        # Normalize
        crop_norm = (crop_resized / 255.0 - 0.5) / 0.5

        # Inference
        logits = self.session.run(
            None,
            {'input': crop_norm[None].transpose(0, 3, 1, 2).astype(np.float32)}
        )[0]

        # Softmax
        probs = np.exp(logits) / np.exp(logits).sum()
        class_id = probs.argmax()
        confidence = probs[class_id]

        return self.CLASSES[class_id], confidence
```

---

### 2.2 Training the Classifier

**Dataset Collection:**
```
occlusion_dataset/
├── none/
│   ├── clean_swap_001.jpg
│   ├── clean_swap_002.jpg
│   └── ... (1000 images)
├── opaque/
│   ├── hand_covering_face_001.jpg
│   ├── food_in_front_001.jpg
│   └── ... (1000 images)
└── transparent/
    ├── glasses_001.jpg
    ├── mayo_drip_001.jpg
    ├── steam_001.jpg
    └── ... (1000 images)
```

**Training Script:**
```python
# train_occlusion_classifier.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, ImageFolder

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

dataset = ImageFolder('occlusion_dataset/', transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Export to ONNX
torch.onnx.export(model, dummy_input, 'occlusion_classifier.onnx')
```

---

## 3. LoRA + XSeg Dual-Head Output

### 3.1 The Architecture Problem

**Current LoRA (Broken):**
```python
# watserface/training/train_instantid.py:77-83
def forward(self, x, id_emb):
    x = self.enc(x)
    x = self.res_blocks(x)  # id_emb NEVER USED!
    x = self.dec(x)
    return x  # Single output: swapped face
```

**Proposed LoRA+XSeg:**
```python
class IdentityGeneratorDualHead(nn.Module):
    """
    Two-head output: swapped face + confidence mask
    """

    def __init__(self, base_channels=64):
        super().__init__()

        # Shared encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # Identity injection (AdaIN)
        self.adain = AdaptiveInstanceNorm(base_channels*2, emb_dim=512)

        # Shared residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_channels*2) for _ in range(6)
        ])

        # HEAD 1: Swapped Face Decoder
        self.face_decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 3, 7, padding=3),
            nn.Sigmoid()
        )

        # HEAD 2: Confidence Mask Decoder (XSeg-style)
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, target_frame, source_embedding):
        """
        Args:
            target_frame: (B, 3, H, W) target image
            source_embedding: (B, 512) identity vector

        Returns:
            swapped_face: (B, 3, H, W) face swap result
            confidence_mask: (B, 1, H, W) self-predicted quality mask
        """
        # Encode
        x = self.enc(target_frame)

        # Inject source identity via AdaIN
        x = self.adain(x, source_embedding)

        # Shared processing
        x = self.res_blocks(x)

        # Dual heads
        swapped_face = self.face_decoder(x)
        confidence_mask = self.mask_decoder(x)

        return swapped_face, confidence_mask
```

---

### 3.2 Training Objective

**Multi-Task Loss:**
```python
def dual_head_loss(pred_face, pred_mask, target_face, target_mask, source_emb):
    """
    Combined loss for both heads
    """

    # Loss 1: Face reconstruction
    L_recon = F.l1_loss(pred_face, target_face)

    # Loss 2: Perceptual loss (VGG features)
    L_perceptual = perceptual_loss(pred_face, target_face)

    # Loss 3: Identity preservation
    pred_emb = extract_embedding(pred_face)
    L_identity = 1 - F.cosine_similarity(pred_emb, source_emb)

    # Loss 4: Mask accuracy (XSeg supervision)
    L_mask = F.binary_cross_entropy(pred_mask, target_mask)

    # Loss 5: Mask-weighted face quality
    # High mask confidence → strong face reconstruction
    L_masked_recon = (pred_mask * F.l1_loss(pred_face, target_face, reduction='none')).mean()

    # Combined
    total_loss = (
        1.0 * L_recon +
        0.5 * L_perceptual +
        2.0 * L_identity +
        0.3 * L_mask +
        0.5 * L_masked_recon
    )

    return total_loss
```

**Benefits:**
1. ✅ Single inference pass (faster)
2. ✅ Mask and swap guaranteed aligned
3. ✅ Model learns to predict its own confidence
4. ✅ Self-aware of occlusions during training

---

## 4. DreamBooth Synthetic Data Generation

### 4.1 The Data Scarcity Problem

**Challenge:** Training LoRA requires paired data:
- 1000+ frames of source identity
- 1000+ frames of target scene
- **Problem:** User may only have 10-20 source images

**Solution:** Generate synthetic training pairs using Stable Diffusion DreamBooth

---

### 4.2 DreamBooth Pipeline

**Step 1: Fine-tune SD on Source Identity**
```python
# watserface/training/dreambooth_finetune.py

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from accelerate import Accelerator

class DreamBoothTrainer:
    def __init__(self, source_images, identity_name="sks"):
        self.source_images = source_images
        self.identity_token = f"<{identity_name}>"

    def train(self, num_epochs=500, learning_rate=5e-6):
        """
        Fine-tune SD to learn source identity

        Output: SD model that can generate "photo of <sks> person ..."
        """

        # Load base SD model
        model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )

        # Prepare dataset
        dataset = self.create_dreambooth_dataset(
            images=self.source_images,
            instance_prompt=f"photo of {self.identity_token} person",
            class_prompt="photo of person"  # Regularization
        )

        # Training loop
        optimizer = torch.optim.AdamW(model.unet.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for batch in dataset:
                # Forward
                loss = model(
                    batch['pixel_values'],
                    batch['input_ids']
                ).loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save
        model.save_pretrained(f"models/dreambooth/{self.identity_token}")

        return model
```

**Step 2: Generate Synthetic Variations**
```python
def generate_synthetic_training_set(dreambooth_model, num_samples=1000):
    """
    Generate diverse synthetic images of source identity

    Variations:
      - Expressions: smiling, neutral, surprised, mouth open
      - Poses: frontal, left turn, right turn, looking up/down
      - Lighting: natural, studio, dramatic, backlit
      - Background: plain, outdoor, indoor
    """

    prompts = [
        "photo of sks person smiling",
        "photo of sks person with neutral expression",
        "photo of sks person mouth open eating",
        "photo of sks person looking left",
        "photo of sks person looking right",
        "photo of sks person close-up portrait",
        "photo of sks person soft lighting",
        "photo of sks person dramatic shadows",
        # ... 100+ variations
    ]

    synthetic_images = []

    for prompt in prompts:
        for seed in range(num_samples // len(prompts)):
            image = dreambooth_model(
                prompt=prompt,
                negative_prompt="cartoon, 3d, painting, blur, distorted",
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(seed)
            ).images[0]

            synthetic_images.append(image)

    return synthetic_images
```

**Step 3: Mix Synthetic + Real for Training**
```python
# Training dataset
real_images = load_real_source_images()  # 20 images
synthetic_images = generate_synthetic_training_set(dreambooth_model, 980)

combined_dataset = real_images + synthetic_images  # 1000 total

# Train LoRA on combined dataset
lora_model = train_lora(
    source_data=combined_dataset,
    target_data=target_frames,
    epochs=100
)
```

---

### 4.3 Quality Validation

**Check Generated Images:**
```python
def validate_synthetic_quality(synthetic_images, real_images):
    """
    Ensure synthetic images preserve identity
    """

    # Extract embeddings
    real_embeddings = [extract_embedding(img) for img in real_images]
    synthetic_embeddings = [extract_embedding(img) for img in synthetic_images]

    # Compute mean real embedding
    mean_real_emb = np.mean(real_embeddings, axis=0)

    # Check synthetic similarity
    similarities = [
        np.dot(syn_emb, mean_real_emb)
        for syn_emb in synthetic_embeddings
    ]

    # Filter low-quality synthetics
    threshold = 0.75
    good_synthetics = [
        img for img, sim in zip(synthetic_images, similarities)
        if sim >= threshold
    ]

    print(f"Kept {len(good_synthetics)}/{len(synthetic_images)} synthetic images")
    print(f"Mean similarity: {np.mean(similarities):.3f}")

    return good_synthetics
```

---

## 5. Full Pipeline Integration

### 5.1 Unified Workflow

```python
# watserface/processors/modules/adaptive_swapper.py

class AdaptiveSwapper:
    """
    Intelligent swap pipeline with conditional execution
    """

    def __init__(self):
        self.occlusion_classifier = OcclusionClassifier()
        self.traditional_swapper = FaceSwapper()  # Existing
        self.dkt_handler = TransparencyHandler()   # Phase 2.5
        self.lora_swapper = LoRASwapper()          # Dual-head

    def process_frame(self, source_identity, target_frame):
        """
        Main entry point - routes to appropriate pipeline
        """

        # Step 0: Detect faces
        faces = detect_faces(target_frame)
        if not faces:
            return target_frame  # No faces, return original

        target_face = faces[0]

        # Step 1: Classify occlusion type
        occlusion_type, confidence = self.occlusion_classifier.classify(
            target_frame,
            target_face.bbox
        )

        # Step 2: Route to appropriate pipeline
        if occlusion_type == 'none':
            # Fast path: Traditional swap
            result = self._traditional_pipeline(source_identity, target_frame, target_face)

        elif occlusion_type == 'opaque':
            # Medium path: XSeg + traditional
            result = self._xseg_pipeline(source_identity, target_frame, target_face)

        elif occlusion_type == 'transparent':
            # Slow path: DKT + ControlNet
            result = self._dkt_pipeline(source_identity, target_frame, target_face)

        return result

    def _traditional_pipeline(self, source_identity, target_frame, target_face):
        """Clean swap - no occlusions"""

        if has_lora_model(source_identity):
            # Use custom LoRA (dual-head)
            swapped, confidence_mask = self.lora_swapper(source_identity, target_frame)
        else:
            # Use pretrained swapper
            swapped = self.traditional_swapper(source_identity, target_frame)
            confidence_mask = create_box_mask(target_face)

        # Simple paste
        result = paste_back(target_frame, swapped, confidence_mask, target_face.matrix)

        return result

    def _xseg_pipeline(self, source_identity, target_frame, target_face):
        """Opaque occlusion - use XSeg mask"""

        # Swap
        if has_lora_model(source_identity):
            swapped, predicted_mask = self.lora_swapper(source_identity, target_frame)
            # Use predicted mask
            final_mask = predicted_mask
        else:
            swapped = self.traditional_swapper(source_identity, target_frame)
            # Use XSeg inference
            xseg_mask = create_occlusion_mask(target_frame)
            final_mask = xseg_mask

        # Paste with occlusion mask
        result = paste_back(target_frame, swapped, final_mask, target_face.matrix)

        return result

    def _dkt_pipeline(self, source_identity, target_frame, target_face):
        """Transparent occlusion - full Phase 2.5 pipeline"""

        # Layer 1: Traditional swap
        swapped = self._xseg_pipeline(source_identity, target_frame, target_face)

        # Layer 2: DKT estimation
        depth, normal, alpha = self.dkt_handler.estimate_transparency(
            target_frame,
            temporal_context=self.frame_buffer  # From video processor
        )

        # Layer 3: ControlNet refinement
        if alpha.max() > 0.1:  # Has transparent regions
            result = self.dkt_handler.controlnet_inpaint(
                swapped, depth, normal, alpha
            )
        else:
            result = swapped  # False positive, skip ControlNet

        return result
```

---

### 5.2 Performance Monitoring

```python
class PipelineProfiler:
    """Track which pipelines are used and their performance"""

    def __init__(self):
        self.stats = {
            'traditional': {'count': 0, 'total_time': 0},
            'xseg': {'count': 0, 'total_time': 0},
            'dkt': {'count': 0, 'total_time': 0}
        }

    def log(self, pipeline_type, elapsed_time):
        self.stats[pipeline_type]['count'] += 1
        self.stats[pipeline_type]['total_time'] += elapsed_time

    def report(self):
        total_frames = sum(s['count'] for s in self.stats.values())

        print("Pipeline Usage Report:")
        for pipeline, stats in self.stats.items():
            pct = 100 * stats['count'] / total_frames if total_frames > 0 else 0
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0

            print(f"  {pipeline:12s}: {stats['count']:4d} frames ({pct:5.1f}%) | "
                  f"avg {avg_time*1000:.1f}ms")

        total_time = sum(s['total_time'] for s in self.stats.values())
        print(f"\nTotal processing time: {total_time:.1f}s")

# Example output:
# Pipeline Usage Report:
#   traditional :  800 frames ( 80.0%) | avg 450.2ms
#   xseg        :  150 frames ( 15.0%) | avg 612.8ms
#   dkt         :   50 frames (  5.0%) | avg 2847.3ms
#
# Total processing time: 640.3s
```

---

## 6. Success Criteria

Phase 3 is **COMPLETE** when:

✅ **Occlusion classifier** achieves ≥95% accuracy on validation set
✅ **LoRA dual-head** produces both swapped face and accurate mask in single pass
✅ **DreamBooth pipeline** generates ≥500 usable synthetic images from 20 real
✅ **Adaptive routing** selects correct pipeline ≥90% of time
✅ **Performance** 3-5x faster than always-DKT approach
✅ **Quality** matches or exceeds Phase 2.5 on transparent occlusions
✅ **Backward compatible** - works with existing pretrained models

---

## 7. Timeline Estimate

**Milestone 1: Occlusion Classifier (Week 1)**
- Collect/label training dataset
- Train ResNet18 classifier
- Export to ONNX, integrate

**Milestone 2: LoRA Dual-Head (Week 2)**
- Fix IdentityGenerator architecture (AdaIN injection)
- Add second decoder head (mask)
- Retrain on existing datasets

**Milestone 3: DreamBooth Pipeline (Week 3)**
- Implement SD fine-tuning
- Generate synthetic dataset
- Validate quality

**Milestone 4: Adaptive Pipeline (Week 4)**
- Integrate all components
- Performance profiling
- Quality validation

**Total:** 4 weeks (assuming Phase 2.5 complete)

---

## 8. References

- [Phase 2.5 DKT PoC](PHASE_2.5_DKT_POC.md) - Transparency handling
- [DreamBooth Paper](https://arxiv.org/abs/2208.12242) - Personalization of text-to-image
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Parameter-efficient fine-tuning

---

**Status:** Not Started (blocked by Phase 2.5)
**Dependencies:** Phase 2.5 complete, LoRA architecture fixed
**Owner:** TBD
**Last Updated:** 2026-01-25
