# WatserFace Training Architecture PRD (v2.0.0)

**Version:** 2.0.0  
**Date:** 2026-01-04  
**Status:** Approved for Execution  
**Previous:** v1.0.0 (CLI-first approach - superseded)

## 1. Executive Summary

We are upgrading WatserFace from stubbed training infrastructure to a **production-grade, UI-first application**. The goal is to implement real training pipelines using `diffusers`, `peft`, and `transformers` while maintaining our existing Gradio UI as the primary interface.

**Key Change from v1.0.0:** We preserve the existing 78-file UI infrastructure rather than replacing it with a CLI. The UI will be redesigned for intuitive wizard-like flow, hiding complexity while preserving power-user access.

## 2. User Experience (UI Flow)

### 2.1 Wizard-Style Training Interface

The Training tab will guide users through a step-by-step workflow:

```
┌─────────────────────────────────────────────────────────────┐
│  WatserFace Training Wizard                                 │
├─────────────────────────────────────────────────────────────┤
│  ● Step 1: Source    ○ Step 2: Target    ○ Step 3: Train   │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  WHO do you want to become?                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  📁 Drop video/images here                          │   │
│  │     or click to browse                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [Preview Extracted Faces]              [Next →]            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Step Breakdown

| Step | User Action | System Response |
|------|-------------|-----------------|
| **1. Source** | Upload source video/images | Extract faces, show preview grid, compute embeddings |
| **2. Target** | Upload target video | Extract frames, detect faces, generate initial masks |
| **3. Train** | Click "Start Training" | Run Identity → XSeg → LoRA pipeline with live progress |
| **4. Swap** | (Automatic or manual) | Execute face swap using trained models |
| **5. Review** | View results | QA scan highlights problem frames for optional repair |

### 2.3 Progress & Telemetry

During training, users see:
- **Global Progress Bar:** "Phase 2/5: Training Identity Adapter"
- **Phase Progress:** "Epoch 47/100 | Loss: 0.0234 | ETA: 12m 34s"
- **Live Loss Chart:** Real-time training curve visualization
- **Expandable Logs:** Advanced users can view detailed telemetry

### 2.4 Smart Defaults

| Setting | Default | Why |
|---------|---------|-----|
| Identity Epochs | 100 | Balances quality vs time |
| XSeg Epochs | 50 | Occlusion masks converge faster |
| LoRA Rank | 16 | Good balance of adaptation vs overfitting |
| Batch Size | Auto | Based on available VRAM |

Advanced settings are hidden in a collapsible "Advanced Options" section.

## 3. Technical Architecture

### 3.1 Core Components (Production Stack)

Replace custom stubs with industry-standard libraries:

| Component | Current (Stubs) | New (Production) | Purpose |
|-----------|-----------------|------------------|---------|
| **Identity Training** | Custom adapter | `diffusers.ControlNetModel` | Condition SD on InstantID features |
| **Face Mapping** | Custom LoRA | `peft` (LoRA) | Map Source ID → Target style |
| **Occlusion (XSeg)** | Simple U-Net | `segmentation_models_pytorch` | Robust occlusion masking |
| **Inference** | InsightFace swap | `diffusers.StableDiffusionControlNetInpaintPipeline` | Generative face synthesis |
| **QA / Repair** | None | `mediapipe` + `diffusers` inpainting | Auto-fix problem frames |

### 3.2 Pipeline Steps

#### Step 1: Extraction & Analysis
- **Source:** Extract faces, align landmarks (MediaPipe 478), compute InsightFace embeddings
- **Target:** Extract frames, detect faces, compute initial occlusion masks (convex hull)
- **Output:** Standardized dataset format compatible with HuggingFace `datasets`

#### Step 2: Identity Training (The "Soul")
- **Goal:** Teach the model *who* the source is
- **Method:** Train a ControlNet adapter using IP-Adapter architecture
- **Inputs:** Source face images + keypoints + embeddings
- **Objective:** Minimize reconstruction loss of source identity

#### Step 3: XSeg Training (The "Mask")
- **Goal:** Learn *where* the face is (and isn't) in target video
- **Method:** Train a U-Net (ResNet34 backbone) for binary segmentation
- **Inputs:** Target frames + projected face masks
- **Refinement:** Learn static occlusions (hands, objects, hair)

#### Step 4: LoRA Training (The "Bridge") - Optional
- **Goal:** Overfit to target video's lighting/style while preserving identity
- **Method:** Train LoRA adapter on SD UNet attention layers
- **Constraint:** Freeze Identity ControlNet; train only style adaptation

#### Step 5: Swap Inference
- **Engine:** `StableDiffusionControlNetInpaintPipeline`
- **Conditioning:**
  - Control Image: MediaPipe keypoints (pose/expression from target)
  - Cross-Attention: Source identity embeddings
  - Mask: Trained XSeg output
  - Prompt: "High quality photo of [identity] person..."

#### Step 6: Recursive Review & Repair
- **Scanner:** Iterate through swapped frames
- **Metrics:** Face detection confidence < 0.7 OR identity similarity < 0.6
- **Action:** Re-run inference on flagged frames with adjusted parameters

## 4. Directory Structure

```text
watserface/
├── uis/
│   ├── layouts/
│   │   ├── training.py         # Wizard-style training UI (REDESIGN)
│   │   ├── swap.py             # Main swap interface
│   │   └── modeler.py          # Advanced LoRA training
│   └── components/             # 70+ reusable UI components
├── training/
│   ├── core.py                 # Training orchestrator (FIX)
│   ├── pipeline.py             # NEW: Unified training pipeline
│   ├── trainers/
│   │   ├── identity.py         # NEW: diffusers ControlNet training
│   │   ├── lora.py             # NEW: peft LoRA training
│   │   └── xseg.py             # UPGRADE: smp-based segmentation
│   ├── validators/
│   │   └── quality.py          # NEW: QA & repair logic
│   └── datasets/
│       ├── instantid_dataset.py
│       ├── lora_dataset.py
│       └── xseg_dataset.py
└── models/
    └── ...
```

## 5. Requirements

### New Dependencies
```
diffusers>=0.25.0
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0
segmentation-models-pytorch>=0.3.3
scikit-image>=0.22.0
```

### Existing (Keep)
```
gradio>=4.0.0
torch>=2.0.0
onnxruntime
mediapipe
insightface
```

## 6. Implementation Plan

### Phase 1: Stabilization (Immediate)
- [x] Restore deleted training files
- [x] Fix generator vs path bug in `core.py`
- [ ] Verify app runs without crashes

### Phase 2: Training Backend Upgrade
- [ ] Create `training/trainers/identity.py` with diffusers ControlNet
- [ ] Create `training/trainers/lora.py` with peft
- [ ] Upgrade `training/trainers/xseg.py` to use smp
- [ ] Create `training/pipeline.py` orchestrator
- [ ] Create `training/validators/quality.py` for QA

### Phase 3: UI Redesign
- [ ] Redesign `uis/layouts/training.py` as wizard flow
- [ ] Add step indicators and validation
- [ ] Implement smart defaults with collapsible advanced options
- [ ] Add live training visualization

### Phase 4: Integration & Polish
- [ ] Connect new training backend to UI
- [ ] Implement QA/repair workflow
- [ ] Performance optimization
- [ ] Documentation updates

## 7. Success Criteria

1. **Zero Stubs:** All training loops execute real backpropagation on production models
2. **Intuitive UI:** New users complete first training in < 5 minutes without docs
3. **Visual Feedback:** Users always know what's happening and ETA
4. **Automatic QC:** Output video is "self-healed" by recursive review
5. **Backward Compatible:** Existing trained models continue to work

## 8. Migration Notes

### From v1.0.0 (CLI-first)
The CLI approach was abandoned because:
- Existing UI infrastructure (78 files) represents significant investment
- Users expect GUI for creative tools
- CLI can be added later as optional power-user interface

### Breaking Changes
- Training telemetry format standardized
- Dataset directory structure may change
- Model checkpoint format updated for diffusers compatibility
