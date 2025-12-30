# Attribution

This project is a derivative work based on **FaceFusion** by Henry Ruhs.

## Original Project
- **Name**: FaceFusion
- **Author**: Henry Ruhs
- **Repository**: https://github.com/facefusion/facefusion
- **Version at Fork**: 3.3.4
- **License**: OpenRAIL-AS
- **Copyright**: (c) 2025 Henry Ruhs

## Modifications by WatserFace

This derivative work adds significant training infrastructure and removes certain safety features:

### Phase 1 - UI Fixes (5 files modified)
- Fixed missing `Any` type import in training_source.py
- Added video preview galleries with frame sliders
- Created comprehensive training UI components

### Phase 2 - MediaPipe Integration (2 files)
- `dataset_extractor.py` - Extracts frames + 478 landmarks as JSON
- `landmark_smoother.py` - Temporal smoothing for jitter reduction

### Phase 3 - XSeg Training (3 files)
- `models/xseg_unet.py` - U-Net architecture for occlusion masks
- `datasets/xseg_dataset.py` - Dataset loader with auto-mask generation
- `train_xseg.py` - Complete training loop with ONNX export

### Phase 4 - InstantID Training (3 files)
- `models/instantid_adapter.py` - Identity adapter architecture
- `datasets/instantid_dataset.py` - Dataset with face embeddings
- `train_instantid.py` - Identity training with ONNX export

### Phase 5 - Integration (4 files)
- `training/core.py` - Main training orchestration
- `device_utils.py` - MPS/CUDA/CPU auto-detection
- `xseg_annotator.py` - Manual mask annotation UI
- Updated model auto-discovery system

### Safety Modifications
- **NSFW Detection**: Disabled for research and development use (commit 7c24b04b)
- **Tamper Validation**: Bypassed to enable code modifications (commit 7c24b04b)

**⚠️ IMPORTANT**: These modifications change the ethical posture of the software. Users are responsible for ensuring ethical and legal use.

## Copyright Notice

- Original FaceFusion: Copyright (c) 2025 Henry Ruhs
- Training Extensions & Modifications: Copyright (c) 2025 IlliquidAsset
- Combined Work: Licensed under OpenRAIL-AS

## License

This derivative work is distributed under the OpenRAIL-AS license, maintaining all use-based restrictions from the original FaceFusion project.

See LICENSE.md for the full license text.

## Acknowledgments

We thank Henry Ruhs and the FaceFusion community for creating the foundational platform that made this work possible. WatserFace would not exist without their excellent work.

---

**WatserFace** - "Who's that again?"
