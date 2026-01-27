# GitHub Projects Setup Guide

**Purpose:** Track Milestone 0, Phase 2.5, and Phase 3 development with multi-agent collaboration

**Date:** 2026-01-25

---

## Quick Setup

### Option 1: GitHub CLI (Recommended)

```bash
# Install gh CLI if needed
brew install gh

# Login
gh auth login

# Create project
gh project create --owner IlliquidAsset --title "WatserFace PoC Roadmap" --description "Milestone 0 → Phase 2.5 → Phase 3"

# Create milestones
gh milestone create "Milestone 0: Baseline Validation" --due-date 2026-02-15 --description "Validate pixel-perfect swaps on clean faces"
gh milestone create "Phase 2.5: DKT Transparency" --due-date 2026-03-15 --description "Physics-aware transparent occlusion handling"
gh milestone create "Phase 3: Inverted Compositing" --due-date 2026-04-15 --description "Conditional execution pipeline"

# Run the bulk issue creation script below
bash create_poc_issues.sh
```

### Option 2: GitHub Web Interface

1. Go to: https://github.com/IlliquidAsset/watserface/projects
2. Click "New project"
3. Select "Board" template
4. Name: "WatserFace PoC Roadmap"
5. Create columns: Backlog, In Progress, Review, Done
6. Manually create issues from templates below

---

## Project Structure

### Board Columns

```
┌─────────────┬──────────────┬──────────┬──────┐
│  Backlog    │ In Progress  │  Review  │ Done │
├─────────────┼──────────────┼──────────┼──────┤
│ Not started │ Active work  │ Testing  │  ✅  │
│ tasks       │ by agents    │ QA check │      │
└─────────────┴──────────────┴──────────┴──────┘
```

### Milestones

1. **Milestone 0: Baseline Validation** (Due: 2026-02-15)
2. **Phase 2.5: DKT Transparency** (Due: 2026-03-15)
3. **Phase 3: Inverted Compositing** (Due: 2026-04-15)

### Labels

```yaml
# Priority
priority:critical   - Red - Blocking work
priority:high       - Orange - Important
priority:medium     - Yellow - Normal priority
priority:low        - Green - Nice to have

# Type
type:infrastructure - Purple - Setup/tooling
type:feature        - Blue - New functionality
type:bug            - Red - Fix broken behavior
type:docs           - Gray - Documentation
type:test           - Cyan - Testing/validation

# Phase
phase:milestone-0   - Pink
phase:2.5          - Teal
phase:3            - Indigo

# Agent Assignment
agent:gemini       - Agent assigned to Gemini
agent:claude       - Agent assigned to Claude
agent:grok         - Agent assigned to Grok
agent:unassigned   - Needs assignment

# Status
status:blocked     - Yellow - Waiting on dependency
status:ready       - Green - Can start immediately
```

---

## Bulk Issue Creation Script

Save as `create_poc_issues.sh`:

```bash
#!/bin/bash

# Milestone 0: Baseline Validation Issues

gh issue create \
  --title "[M0] Create test dataset (20 clean face pairs)" \
  --body "$(cat <<EOF
## Description
Create validation dataset with clean, non-occluded face pairs for baseline testing.

## Acceptance Criteria
- [ ] 20 pairs of frontal faces (±15° rotation max)
- [ ] Various demographics (skin tone, gender, age)
- [ ] Good lighting, no harsh shadows
- [ ] No occlusions (hands, objects, hair)
- [ ] Similar pose/expression within each pair
- [ ] Organized in \`test_data/clean_pairs/pair_NNN/\` structure

## Files
- source.jpg (face A)
- target.jpg (face B)
- metadata.json (demographics, quality scores)

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section 3.1

## Estimated Time
2 days
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:infrastructure,priority:critical,agent:unassigned"

gh issue create \
  --title "[M0] Implement BaselineValidator class" \
  --body "$(cat <<EOF
## Description
Create automated validation framework for measuring swap quality.

## Acceptance Criteria
- [ ] Identity similarity measurement (cosine similarity)
- [ ] Background preservation (SSIM)
- [ ] Perceptual quality (LPIPS)
- [ ] Color consistency (CIE2000 ΔE)
- [ ] Sharpness measurement (Laplacian variance)
- [ ] Edge artifact detection

## Implementation
\`\`\`python
class BaselineValidator:
    def validate_swap(source, target, output) -> metrics_dict
    def measure_identity_similarity() -> float
    def measure_background_preservation() -> float
    def measure_perceptual_distance() -> float
    def measure_color_consistency() -> float
    def measure_sharpness() -> float
    def detect_edge_artifacts() -> float
\`\`\`

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section 3.3

## Estimated Time
3 days
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:feature,priority:critical,agent:gemini"

gh issue create \
  --title "[M0] Run FaceSwap.dev comparison baseline" \
  --body "$(cat <<EOF
## Description
Run identical test suite on FaceSwap.dev to establish competitive benchmark.

## Acceptance Criteria
- [ ] FaceSwap.dev environment setup
- [ ] Extract faces from 20 test pairs
- [ ] Train or use pretrained FaceSwap model
- [ ] Run conversion on all test cases
- [ ] Collect same metrics as WatserFace
- [ ] Generate comparison report

## Commands
\`\`\`bash
cd /Users/kendrick/Documents/dev/faceswap.dev
python faceswap.py extract -i source.jpg -o extract/source
python faceswap.py extract -i target.jpg -o extract/target
python faceswap.py train -A extract/source -B extract/target -m models/test
python faceswap.py convert -i target.jpg -o output.jpg -m models/test
\`\`\`

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section 3.4

## Estimated Time
1 day
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:test,priority:high,agent:claude"

gh issue create \
  --title "[M0] Fix interpolation artifacts (INTER_LANCZOS4)" \
  --body "$(cat <<EOF
## Description
Improve upsampling quality by using LANCZOS4 instead of INTER_AREA.

## Problem
Current code uses INTER_AREA for all warping (watserface/face_helper.py:139), which causes pixelation on upsampling.

## Solution
\`\`\`python
if bounding_box[2] - bounding_box[0] > crop_size[0]:
    interpolation_method = cv2.INTER_AREA  # Downsampling
else:
    interpolation_method = cv2.INTER_LANCZOS4  # Upsampling
\`\`\`

## Testing
- [ ] Compare before/after on test dataset
- [ ] Measure sharpness improvement
- [ ] Verify no performance regression

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section 4.1

## Estimated Time
1 day
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:bug,priority:high,agent:gemini"

gh issue create \
  --title "[M0] Implement color correction (LAB histogram matching)" \
  --body "$(cat <<EOF
## Description
Add automatic color correction to match swapped face to target lighting.

## Implementation
Location: watserface/face_helper.py

\`\`\`python
def match_target_color(swapped_face, target_face, mask):
    \"\"\"Match color statistics in LAB space\"\"\"
    swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)

    for i in range(3):
        swapped_lab[:,:,i] = (
            (swapped_lab[:,:,i] - swapped_lab[:,:,i].mean()) *
            (target_lab[:,:,i].std() / swapped_lab[:,:,i].std()) +
            target_lab[:,:,i].mean()
        )

    return cv2.cvtColor(swapped_lab, cv2.COLOR_LAB2BGR)
\`\`\`

## Testing
- [ ] Visual inspection (before/after)
- [ ] Measure ΔE improvement
- [ ] Add to paste_back pipeline

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section 4.2

## Estimated Time
2 days
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:feature,priority:high,agent:gemini"

gh issue create \
  --title "[M0] Implement multi-band blending (Laplacian pyramid)" \
  --body "$(cat <<EOF
## Description
Replace linear alpha blending with multi-resolution Laplacian pyramid blending for seamless edges.

## Implementation
Location: watserface/face_helper.py

\`\`\`python
def laplacian_pyramid_blend(img1, img2, mask, levels=6):
    # Build Gaussian pyramids
    # Build Laplacian pyramids
    # Blend each level
    # Reconstruct
    return result
\`\`\`

## Reference
- Burt & Adelson 1983 paper
- docs/architecture/MILESTONE_0_BASELINE.md Section 4.3

## Testing
- [ ] Compare edge quality vs linear blend
- [ ] Measure edge artifact score
- [ ] Performance benchmark

## Estimated Time
3 days
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:feature,priority:medium,agent:gemini"

gh issue create \
  --title "[M0] Add unsharp mask sharpening option" \
  --body "$(cat <<EOF
## Description
Add optional unsharp mask to recover sharpness lost during blending.

## Implementation
\`\`\`python
def apply_unsharp_mask(image, sigma=1.0, strength=0.5):
    blurred = cv2.GaussianBlur(image, (0,0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
\`\`\`

## Configuration
Add UI option:
- Checkbox: "Apply sharpening"
- Slider: Strength (0-100%)

## Testing
- [ ] Visual quality check
- [ ] Measure sharpness metric
- [ ] Ensure no over-sharpening artifacts

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section 4.4

## Estimated Time
1 day
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:feature,priority:low,agent:gemini"

gh issue create \
  --title "[M0] Run full validation suite and generate report" \
  --body "$(cat <<EOF
## Description
Execute complete test suite and validate acceptance criteria met.

## Tasks
- [ ] Run WatserFace on all 20 pairs × 54 configs
- [ ] Run FaceSwap.dev on same dataset
- [ ] Collect all metrics
- [ ] Generate comparison tables
- [ ] Visual inspection (human eval)
- [ ] Create final report

## Acceptance Criteria
✅ Identity Similarity ≥ 0.85
✅ Background SSIM ≥ 0.90
✅ Perceptual LPIPS ≤ 0.15
✅ Color ΔE < 5.0
✅ No visible edge artifacts
✅ Competitive with FaceSwap.dev

## Report Format
Markdown with:
- Metrics summary table
- Before/after images
- FaceSwap.dev comparison
- Pass/fail determination

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section 6

## Estimated Time
2 days

## Dependencies
- All other M0 issues complete
EOF
)" \
  --milestone "Milestone 0: Baseline Validation" \
  --label "phase:milestone-0,type:test,priority:critical,status:blocked,agent:claude"

# Phase 2.5: DKT Transparency Issues

gh issue create \
  --title "[P2.5-M1] Research and select DKT implementation" \
  --body "$(cat <<EOF
## Description
Evaluate DKT implementations and select best option for integration.

## Options to Evaluate
1. **DKT Official** (if released)
2. **Depth-Anything V2** + custom transmission estimation
3. **Marigold** diffusion-based depth
4. **Custom fine-tuning** of SD depth model

## Evaluation Criteria
- Transparency handling quality
- Inference speed
- Model size
- License compatibility
- Documentation/support

## Deliverable
Decision document with:
- Comparison table
- Recommendation
- Integration plan

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Section 4.2

## Estimated Time
2 days

## Dependencies
- Milestone 0 complete
EOF
)" \
  --milestone "Phase 2.5: DKT Transparency" \
  --label "phase:2.5,type:infrastructure,priority:critical,status:blocked,agent:grok"

gh issue create \
  --title "[P2.5-M1] Setup ControlNet pipeline" \
  --body "$(cat <<EOF
## Description
Implement Stable Diffusion ControlNet pipeline for depth + normal conditioning.

## Implementation
\`\`\`python
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth"
)
controlnet_normal = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_normalbae"
)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=[controlnet_depth, controlnet_normal]
).to("mps")  # or "cuda"
\`\`\`

## Testing
- [ ] Download models (~9GB)
- [ ] Test single frame inference
- [ ] Measure performance
- [ ] Validate output quality

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Section 4.1

## Estimated Time
2 days

## Dependencies
- Milestone 0 complete
EOF
)" \
  --milestone "Phase 2.5: DKT Transparency" \
  --label "phase:2.5,type:infrastructure,priority:critical,status:blocked,agent:gemini"

gh issue create \
  --title "[P2.5-M1] Single frame mayonnaise test" \
  --body "$(cat <<EOF
## Description
Prove DKT + ControlNet works on the iconic corn dog + mayo frame.

## Test Case
- Source: Clean face
- Target: Person eating corn dog with mayo dripping
- Expected: Mayo looks realistic over swapped face (not hole)

## Pipeline
1. Layer 1: Traditional swap (region mask)
2. Layer 2: DKT depth/normal estimation
3. Layer 3: ControlNet inpaint with DKT constraints
4. Compare: Original, traditional swap, DKT swap

## Success Criteria
- ✅ Mayo visible and realistic
- ✅ Face identity preserved
- ✅ Lighting consistent
- ✅ No obvious AI artifacts

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Section 4.1 Step 3

## Estimated Time
3 days

## Dependencies
- DKT implementation selected
- ControlNet pipeline setup
EOF
)" \
  --milestone "Phase 2.5: DKT Transparency" \
  --label "phase:2.5,type:test,priority:critical,status:blocked,agent:gemini"

gh issue create \
  --title "[P2.5-M2] Implement DKTEstimator class" \
  --body "$(cat <<EOF
## Description
Create depth/normal/alpha estimation module using selected DKT approach.

## Implementation
Location: watserface/depth/dkt_estimator.py

\`\`\`python
class DKTEstimator:
    def estimate(frames, temporal_window=5):
        \"\"\"
        Returns:
            depth_map: (H, W) float32
            normal_map: (H, W, 3) float32
            alpha_map: (H, W) float32 transmission coefficient
        \"\"\"

    def depth_to_normals(depth): ...
    def variance_to_transmission(variance): ...
\`\`\`

## Features
- Temporal consistency (multi-frame median)
- Normal map generation from depth gradients
- Transmission estimation from depth variance

## Testing
- [ ] Unit tests for each method
- [ ] Visual inspection of outputs
- [ ] Performance benchmark

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Section 4.2

## Estimated Time
5 days

## Dependencies
- DKT implementation selected
- Single frame test validated
EOF
)" \
  --milestone "Phase 2.5: DKT Transparency" \
  --label "phase:2.5,type:feature,priority:critical,status:blocked,agent:gemini"

gh issue create \
  --title "[P2.5-M3] ControlNet hyperparameter optimization" \
  --body "$(cat <<EOF
## Description
Fine-tune ControlNet parameters for optimal face swapping quality.

## Grid Search
\`\`\`python
configs = {
    'denoise_strength': [0.1, 0.2, 0.3, 0.5],
    'controlnet_scale': [0.5, 0.8, 1.0, 1.2],
    'num_steps': [15, 20, 30, 50],
    'guidance_scale': [5.0, 7.5, 10.0]
}
\`\`\`

## Evaluation
- Identity preservation
- Transparency realism (human eval)
- Temporal consistency
- Inference speed

## Deliverable
Best configuration + performance report

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Section 4.3

## Estimated Time
3 days

## Dependencies
- DKTEstimator implemented
- Test dataset ready
EOF
)" \
  --milestone "Phase 2.5: DKT Transparency" \
  --label "phase:2.5,type:test,priority:high,status:blocked,agent:gemini"

gh issue create \
  --title "[P2.5-M4] Implement TransparencyHandler for video" \
  --body "$(cat <<EOF
## Description
Process full videos with DKT transparency handling and temporal coherence.

## Implementation
Location: watserface/processors/modules/transparency_handler.py

\`\`\`python
class TransparencyHandler:
    def __init__(self, temporal_window=5):
        self.dkt = DKTEstimator()
        self.controlnet = load_controlnet_pipeline()
        self.frame_buffer = deque(maxlen=temporal_window)

    def process_video(video_path, source_identity):
        # Temporal buffering
        # DKT estimation
        # ControlNet inpaint (conditional)
        # Output video
\`\`\`

## Features
- Frame buffering for temporal context
- Conditional ControlNet (skip if alpha < threshold)
- Progress tracking
- Checkpoint/resume support

## Testing
- [ ] Full video processing
- [ ] Temporal consistency check (no flicker)
- [ ] Performance profiling

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Section 4.4

## Estimated Time
5 days

## Dependencies
- ControlNet optimized
- DKTEstimator production-ready
EOF
)" \
  --milestone "Phase 2.5: DKT Transparency" \
  --label "phase:2.5,type:feature,priority:critical,status:blocked,agent:gemini"

gh issue create \
  --title "[P2.5] Run full test suite (mayo, glasses, steam)" \
  --body "$(cat <<EOF
## Description
Validate Phase 2.5 on all transparent occlusion scenarios.

## Test Cases
1. **Mayo Test**: Liquid dripping over face
2. **Glasses Test**: Transparent solid refraction
3. **Steam Test**: Volumetric semi-transparent
4. **Negative Test**: Opaque occlusion (verify skip)

## Success Criteria
- ✅ Mayo: Human eval ≥ 4/5
- ✅ Glasses: Face detection ≥ 0.8
- ✅ Steam: Smooth depth gradients
- ✅ Opaque: Performance = traditional
- ✅ Performance: < 3s per frame
- ✅ Temporal: No visible flicker

## Deliverable
Phase 2.5 completion report

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Sections 5, 9

## Estimated Time
3 days

## Dependencies
- All P2.5 milestones complete
EOF
)" \
  --milestone "Phase 2.5: DKT Transparency" \
  --label "phase:2.5,type:test,priority:critical,status:blocked,agent:claude"

# Phase 3: Inverted Compositing Issues

gh issue create \
  --title "[P3] Train occlusion classifier" \
  --body "$(cat <<EOF
## Description
Build lightweight CNN to classify occlusion type for pipeline routing.

## Classes
- 0: none (clean swap)
- 1: opaque (solid objects)
- 2: transparent (glass, liquid, smoke)

## Dataset
Collect/label:
- 1000 clean swaps
- 1000 opaque occlusions
- 1000 transparent occlusions

## Training
- Architecture: ResNet18
- Epochs: 10
- Export: ONNX

## Target
≥95% accuracy on validation set

## Reference
docs/architecture/PHASE_3_INVERTED.md Section 2

## Estimated Time
5 days

## Dependencies
- Phase 2.5 complete
EOF
)" \
  --milestone "Phase 3: Inverted Compositing" \
  --label "phase:3,type:feature,priority:critical,status:blocked,agent:gemini"

gh issue create \
  --title "[P3] Fix LoRA architecture (AdaIN injection)" \
  --body "$(cat <<EOF
## Description
Fix broken IdentityGenerator to actually use source embedding.

## Current Bug
watserface/training/train_instantid.py:77-83
\`\`\`python
def forward(self, x, id_emb):
    x = self.enc(x)
    x = self.res_blocks(x)  # id_emb NEVER USED!
    x = self.dec(x)
    return x
\`\`\`

## Fix: Add AdaIN
\`\`\`python
class IdentityGeneratorDualHead(nn.Module):
    def __init__(self):
        self.enc = ...
        self.adain = AdaptiveInstanceNorm(channels, emb_dim=512)
        self.res_blocks = ...
        self.face_decoder = ...
        self.mask_decoder = ...  # NEW: XSeg head

    def forward(self, target_frame, source_embedding):
        x = self.enc(target_frame)
        x = self.adain(x, source_embedding)  # INJECT IDENTITY
        x = self.res_blocks(x)
        swapped_face = self.face_decoder(x)
        confidence_mask = self.mask_decoder(x)  # DUAL HEAD
        return swapped_face, confidence_mask
\`\`\`

## Testing
- [ ] Train on test dataset
- [ ] Verify identity preservation
- [ ] Validate dual-head outputs

## Reference
docs/architecture/PHASE_3_INVERTED.md Section 3

## Estimated Time
7 days

## Dependencies
- Phase 2.5 complete
EOF
)" \
  --milestone "Phase 3: Inverted Compositing" \
  --label "phase:3,type:bug,priority:critical,status:blocked,agent:gemini"

gh issue create \
  --title "[P3] Implement DreamBooth synthetic data generation" \
  --body "$(cat <<EOF
## Description
Generate unlimited training pairs from limited source images using SD fine-tuning.

## Pipeline
1. Fine-tune SD on source identity (500 epochs)
2. Generate 980 synthetic variations
3. Validate quality (identity similarity ≥ 0.75)
4. Mix with 20 real images for 1000 total dataset

## Implementation
Location: watserface/training/dreambooth_finetune.py

## Prompts
Generate variations:
- Expressions: smiling, neutral, surprised
- Poses: frontal, left/right, up/down
- Lighting: natural, studio, dramatic
- 100+ total variations

## Reference
docs/architecture/PHASE_3_INVERTED.md Section 4

## Estimated Time
7 days

## Dependencies
- Phase 2.5 complete
- LoRA architecture fixed
EOF
)" \
  --milestone "Phase 3: Inverted Compositing" \
  --label "phase:3,type:feature,priority:high,status:blocked,agent:gemini"

gh issue create \
  --title "[P3] Implement AdaptiveSwapper (unified pipeline)" \
  --body "$(cat <<EOF
## Description
Create intelligent swap pipeline with conditional execution.

## Implementation
Location: watserface/processors/modules/adaptive_swapper.py

\`\`\`python
class AdaptiveSwapper:
    def process_frame(source_identity, target_frame):
        # Step 0: Classify occlusion
        occlusion_type = self.classifier.classify(target_frame)

        # Step 1: Route to appropriate pipeline
        if occlusion_type == 'none':
            return self._traditional_pipeline(...)
        elif occlusion_type == 'opaque':
            return self._xseg_pipeline(...)
        elif occlusion_type == 'transparent':
            return self._dkt_pipeline(...)
\`\`\`

## Features
- Automatic pipeline selection
- Performance profiling
- Fallback handling

## Testing
- [ ] Validate routing accuracy ≥90%
- [ ] Measure performance improvement (target: 3-5x)
- [ ] Quality maintained on all occlusion types

## Reference
docs/architecture/PHASE_3_INVERTED.md Section 5

## Estimated Time
5 days

## Dependencies
- Occlusion classifier trained
- Phase 2.5 complete
EOF
)" \
  --milestone "Phase 3: Inverted Compositing" \
  --label "phase:3,type:feature,priority:critical,status:blocked,agent:gemini"

gh issue create \
  --title "[P3] Run full validation and performance benchmarks" \
  --body "$(cat <<EOF
## Description
Validate Phase 3 complete and measure performance gains.

## Tests
1. **Routing Accuracy**: Classifier selects correct pipeline ≥90%
2. **Performance**: 3-5x faster than always-DKT
3. **Quality**: Matches Phase 2.5 on transparent cases
4. **Compatibility**: Works with existing pretrained models

## Benchmarks
Run 1000-frame video with:
- 800 clean frames
- 150 opaque frames
- 50 transparent frames

Measure:
- Total processing time
- Per-pipeline performance
- Quality metrics

## Deliverable
Phase 3 completion report + performance comparison table

## Reference
docs/architecture/PHASE_3_INVERTED.md Sections 6, 7

## Estimated Time
3 days

## Dependencies
- All Phase 3 features complete
EOF
)" \
  --milestone "Phase 3: Inverted Compositing" \
  --label "phase:3,type:test,priority:critical,status:blocked,agent:claude"

echo "✅ All issues created successfully!"
```

---

## Manual Issue Creation (Web Interface)

If you prefer to create issues manually, use these templates:

### Milestone 0 Template

```markdown
Title: [M0] <Task Name>

Labels: phase:milestone-0, type:<feature/bug/test/infrastructure>, priority:<critical/high/medium/low>, agent:unassigned

Milestone: Milestone 0: Baseline Validation

Body:
## Description
<Brief description>

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Reference
docs/architecture/MILESTONE_0_BASELINE.md Section X

## Estimated Time
X days

## Dependencies
- <Other issues that must complete first>
```

### Phase 2.5 Template

```markdown
Title: [P2.5-M<milestone>] <Task Name>

Labels: phase:2.5, type:<type>, priority:<level>, status:blocked, agent:unassigned

Milestone: Phase 2.5: DKT Transparency

Body:
## Description
<Brief description>

## Implementation
<Code snippets or architecture>

## Testing
- [ ] Test 1
- [ ] Test 2

## Reference
docs/architecture/PHASE_2.5_DKT_POC.md Section X

## Estimated Time
X days

## Dependencies
- Milestone 0 complete
- <Other specific dependencies>
```

---

## Agent Assignment Workflow

### 1. Review Backlog

```bash
# List unassigned issues
gh issue list --label "agent:unassigned" --state open

# Or in Projects board: Filter by "agent:unassigned" label
```

### 2. Assign to Agent

```bash
# Assign to Gemini
gh issue edit <ISSUE_NUMBER> --add-label "agent:gemini" --remove-label "agent:unassigned"

# Assign to Claude
gh issue edit <ISSUE_NUMBER> --add-label "agent:claude" --remove-label "agent:unassigned"

# Assign to Grok
gh issue edit <ISSUE_NUMBER> --add-label "agent:grok" --remove-label "agent:unassigned"
```

### 3. Move to In Progress

```bash
# When agent starts work
gh issue edit <ISSUE_NUMBER> --remove-label "status:blocked"
# Move card in Projects board: Backlog → In Progress
```

### 4. Mark Complete

```bash
# When work done
gh issue close <ISSUE_NUMBER>
# Move card in Projects board: In Progress → Review → Done
```

---

## Dependency Tracking

### Critical Path

```
Milestone 0 Complete
    ↓
[P2.5-M1] DKT Selection
[P2.5-M1] ControlNet Setup
[P2.5-M1] Mayo Test
    ↓
[P2.5-M2] DKTEstimator
    ↓
[P2.5-M3] ControlNet Optimization
    ↓
[P2.5-M4] TransparencyHandler
    ↓
Phase 2.5 Complete
    ↓
[P3] Occlusion Classifier
[P3] LoRA Fix
[P3] DreamBooth
[P3] AdaptiveSwapper
    ↓
Phase 3 Complete
```

### Parallel Work Opportunities

**Can work simultaneously after M0:**
- DKT research (Grok)
- ControlNet setup (Gemini)
- Test dataset expansion (Claude)

**Can work simultaneously after P2.5-M2:**
- ControlNet optimization (Gemini)
- Video processor (Gemini)
- Documentation (Claude)

---

## Progress Tracking

### Weekly Reporting

```bash
# Issues completed this week
gh issue list --milestone "Milestone 0: Baseline Validation" --state closed --search "closed:>2026-01-20"

# Issues in progress
gh issue list --label "phase:milestone-0" --state open

# Blocked issues
gh issue list --label "status:blocked" --state open
```

### Project Views

Create custom views in GitHub Projects:

1. **By Agent**: Group by agent label
2. **By Priority**: Sort by priority label
3. **By Phase**: Filter by phase label
4. **Blocked Items**: Filter status:blocked

---

## Next Steps

1. **Run the script** (Option 1) or **manually create issues** (Option 2)
2. **Review and refine** issue descriptions
3. **Assign first batch** to agents:
   - Gemini: Heavy coding (interpolation fix, color correction, blending)
   - Claude: Testing & reporting (FaceSwap comparison, validation)
   - Grok: Research (DKT options, documentation)
4. **Start Milestone 0** work
5. **Track progress** in Projects board

---

**Created:** 2026-01-25
**Status:** Ready to execute
