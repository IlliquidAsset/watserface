# Changelog

All notable changes to WatserFace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.10.0] - 2025-12-29

### 🎉 Initial Release as WatserFace

**Forked from**: FaceFusion 3.3.4 by Henry Ruhs

This is the first independent release of WatserFace, a derivative of FaceFusion focused on advanced training capabilities and research flexibility.

#### Added

**Training Infrastructure** (26 new files, 2,424+ lines)
- Complete training backend for custom model creation
- MediaPipe 478-landmark detection and persistence
- XSeg U-Net occlusion mask training with ONNX export
- InstantID identity preservation training with ONNX export
- Dataset extraction with frame sampling and validation
- Temporal smoothing for landmark sequences (Savitzky-Golay filter)
- Training UI components with real-time progress tracking
- Device auto-detection (MPS/CUDA/CPU) and optimization
- XSeg mask annotation UI with manual painting
- Model persistence and automatic registration in Swap tab

**UI/UX Enhancements**
- Video preview galleries with frame sliders
- Training progress telemetry with live updates
- Split face swap layout for better organization
- Enhanced face masker and swapper options
- Improved component tooltips and labels

**Performance Optimizations**
- 30% speedup for face prediction (2.90s → 2.03s)
- Optimized center calculation in `predict_next_faces`
- Memory-efficient dataset loading

**Documentation**
- Brand guidelines with WatserFace visual identity
- Generative Inpainting methodology documentation
- Complete attribution to FaceFusion
- Responsible use guidelines

#### Modified

**Safety Features** (⚠️ Breaking Changes)
- NSFW content detection **disabled** for research use
- Tamper validation **bypassed** to enable code modifications
- Users now responsible for ethical and legal compliance

**Core Pipeline**
- Face detection optimized for MediaPipe 478 landmarks
- Processor registration expanded for trained models
- Version bumped to 0.10.0 (independent fork versioning)

**Configuration**
- Config file renamed: `facefusion.ini` → `watserface.ini`
- Main script renamed: `facefusion.py` → `watserface.py`
- Package name retained as `facefusion` (imports unchanged for now)

#### Technical Details

**Requirements**
- Python 3.11+ required
- PyTorch with MPS/CUDA support
- New dependencies: scipy (for smoothing)

**Training Features**
- Identity Training: 1000 frames @ 2-frame intervals
- Occlusion Training: 2000 frames @ 2-frame intervals
- Batch size: 4 (configurable)
- Learning rates: 0.0001 (identity), 0.001 (occlusion)
- ONNX export for production inference
- Auto-mask generation from landmark convex hull

**Model Compatibility**
- All trained models auto-register in `.assets/models/trained/`
- Compatible with existing FaceFusion inference pipeline
- Hash files generated for validator bypass

#### Breaking Changes from FaceFusion 3.3.4

1. **Safety Features Removed**: NSFW detection and tamper checks disabled
2. **Version Reset**: 3.3.4 → 0.10.0 (fork versioning)
3. **Config Files**: Renamed to `watserface.*`
4. **Branding**: UI strings reference WatserFace, not FaceFusion

#### Attribution

This project is based on **FaceFusion** by Henry Ruhs:
- **Original Repository**: https://github.com/facefusion/facefusion
- **Original License**: OpenRAIL-AS
- **Original Copyright**: (c) 2025 Henry Ruhs

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete attribution details.

#### Migration Guide

**From FaceFusion 3.3.4**:
1. Backup your `.assets/models/` directory
2. Update config: `facefusion.ini` → `watserface.ini`
3. Update launch command: `python facefusion.py` → `python watserface.py`
4. Review [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for safety changes
5. All existing models remain compatible

**New Training Features**:
- Access Training tab in UI (`ui_layouts = swap training` in config)
- Upload source videos for identity training
- Upload target video for occlusion training
- Annotate masks with XSeg Annotator (optional)
- Trained models auto-appear in Swap tab after refresh

---

## [3.3.4] - 2025-08-06 (FaceFusion upstream)

Last synchronized version from original FaceFusion repository before fork.

For FaceFusion changelog, see: https://github.com/facefusion/facefusion/releases

---

**WatserFace** - "Who's that again?"

Maintained by IlliquidAsset | Based on FaceFusion by Henry Ruhs
