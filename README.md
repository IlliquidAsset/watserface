---
title: WatserFace
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8080
pinned: false
license: other
license_name: openrail-as
short_description: Advanced face manipulation and training platform
startup_duration_timeout: 5m
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
---

# WatserFace

> "Who's that again?"

**Advanced face manipulation and training platform with comprehensive dataset training capabilities.**

**Based on [FaceFusion](https://github.com/facefusion/facefusion) by Henry Ruhs**

---

[![License](https://img.shields.io/badge/license-OpenRAIL--AS-green)](LICENSE.md)
![Version](https://img.shields.io/badge/version-0.10.0-blue)
[![Original Project](https://img.shields.io/badge/based%20on-FaceFusion-orange)](https://github.com/facefusion/facefusion)

## ⚠️ Important Notice

WatserFace is a derivative of FaceFusion with **modified safety features**:
- **NSFW content detection**: Disabled for research and development use
- **Tamper validation**: Bypassed to enable code customization

**Users are fully responsible for ethical and legal compliance.**
See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for detailed guidelines.

## What is WatserFace?

WatserFace extends the original FaceFusion face manipulation platform with production-ready training infrastructure:

### From Original FaceFusion
- ✅ Industry-leading face swapping technology
- ✅ Real-time preview with multiple processor types
- ✅ Batch processing for videos and images
- ✅ Advanced face selection and masking
- ✅ Hardware acceleration (CUDA/MPS/CPU)

### New in WatserFace v0.10.0
- 🎓 **Training Tab**: Full UI for dataset management and model training
- 📊 **MediaPipe Pipeline**: Extract and smooth 478-landmark facial data
- 🎭 **XSeg Training**: Custom occlusion mask generation with U-Net
- 🆔 **InstantID Training**: Identity-preserving face adaptation
- 🖼️ **Annotation UI**: Manual mask painting with Gradio ImageEditor
- ⚡ **MPS Support**: Optimized for Apple Silicon (30% faster face prediction)
- 📈 **Real-time Progress**: Live telemetry for training workflows

## Features

### Face Swapping
- High-fidelity face replacement with identity preservation
- Multiple swapper models (inswapper, simswap, etc.)
- Face enhancement and expression restoration
- Age modification and frame enhancement
- Lip syncing capabilities

### Training Infrastructure
- **Identity Training**: Train custom InstantID models from your own datasets
- **Occlusion Training**: Train XSeg models for precise masking
- **Dataset Extraction**: Automated frame sampling with MediaPipe landmarks
- **Temporal Smoothing**: Jitter reduction for video sequences
- **ONNX Export**: Production-ready model output
- **Auto-Registration**: Trained models automatically appear in Swap tab

### Advanced Features
- 478-point MediaPipe facial landmarks
- Savitzky-Golay temporal smoothing
- Auto-mask generation from landmark convex hull
- Device auto-detection (MPS/CUDA/CPU)
- Batch processing with job management
- Multiple output formats and resolutions

## Methodology: Generative Inpainting for Occlusion Handling

### Abstract
This section outlines a novel approach to addressing high-frequency occlusion artifacts in face-swapping pipelines. By integrating generative inpainting, we bridge the semantic gap between segmentation targets (XSeg) and source identity fragments. This method specifically targets dynamic mesh deformations—such as lip compression during eating—that are frequently underrepresented in standard training distributions.

### Problem Statement
Traditional face-swapping models rely on static or rigid training data, which often fails to generalize to extreme dynamic expressions or interactions with external objects. A canonical failure case is the "corndog example": when a subject eats, the lips crease and wrinkle around the object. Standard models often treat these dynamic occlusions as noise or fail to render the complex interaction between the lips and the object, resulting in visual artifacts or loss of identity coherence in the deformed region.

### Methodology
Our approach draws inspiration from professional image editing workflows (e.g., Adobe Photoshop), employing a layered compositing strategy down to the pixel level. The pipeline integrates three core components:

1. **XSeg Targets (Occlusion Masks):** High-precision segmentation masks that identify static and dynamic occlusions.
2. **Identity Fragments:** Feature maps extracted from the source identity.
3. **Generative Inpainting Bridge:** A generative model acts as the cohesive layer. Instead of simply overlaying the source face onto the target, the inpainting module synthesizes the boundary regions where the XSeg mask meets the identity fragment. This allows for the hallucination of realistic creasing, lighting, and physics-based interactions (like the lip deformation around food) that are not explicitly present in the source identity's latent space.

### Conclusion
By treating face swapping as a layered composition problem rather than a direct translation task, this methodology effectively bridges the gap between rigid model weights and fluid real-world dynamics. The result is a more robust synthesis capable of handling complex interactions and partial occlusions with high fidelity.

## Installation

### Requirements
- Python 3.11+
- CUDA 12.1+ (NVIDIA) or MPS (Apple Silicon)
- 8GB+ RAM (16GB+ recommended for training)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/IlliquidAsset/facefusion.git
cd facefusion

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For training capabilities
pip install -r requirements-training.txt

# Run the application
bash fast_run.sh
# Or: python watserface.py run
```

### Docker (Coming Soon)
```bash
docker pull illiquidasset/watserface:latest
docker run -p 7860:7860 watserface:latest
```

## Usage

### UI Layouts

WatserFace provides two UI layouts to match your workflow preferences:

#### Streamlined Layout (Default)
**Multi-tab guided workflow** - Best for most users

- **📁 Upload**: Upload source and target media with guided tips
- **🔍 Preview**: Quality presets and smart previewing
- **⚡ Process**: Execute face swap with progress tracking
- **🎓 Advanced**: Fine-tuning controls and model training
- **🎬 Studio**: One-shot workflow for power users (see below)

**Configure**: `ui_layouts = streamlined` in `watserface.ini`

#### Studio Tab
**Compact one-shot interface** - For experienced users

The Studio tab (accessible within Streamlined layout) provides a simplified single-screen workflow where you can:
- Build and train custom identity models
- Resume existing identities or reuse face sets
- Configure target media and occlusion models (XSeg)
- Map faces and execute swaps in one place
- Train advanced occlusion models

**When to use Studio:**
- ✅ You prefer a compact, single-screen workflow
- ✅ You're training custom identity models
- ✅ You need advanced occlusion handling (XSeg)
- ✅ You want manual control over face mapping

**Note**: The streamlined guided workflow is recommended for most users. Studio is designed for power users who want all controls in one view.

### Face Swapping

```bash
# Swap face in single image
python watserface.py run \
  -s path/to/source.jpg \
  -t path/to/target.jpg \
  -o path/to/output.jpg

# Swap face in video
python watserface.py run \
  -s path/to/source.jpg \
  -t path/to/target.mp4 \
  -o path/to/output.mp4
```

### Training Workflow

1. **Launch UI**:
   ```bash
   python watserface.py run
   ```

2. **Navigate to Training Tab** (ensure `ui_layouts = swap training` in `watserface.ini`)

3. **Identity Training**:
   - Upload source videos/images (your face dataset)
   - Set model name and epochs
   - Click "Start Identity Training"
   - Trained model auto-registers in Swap tab

4. **Occlusion Training** (Optional):
   - Upload target video (containing occlusions)
   - Optionally annotate masks with XSeg Annotator
   - Set model name and epochs
   - Click "Start Occlusion Training"

5. **Use Trained Models**:
   - Refresh app or click refresh button in Swap tab
   - Select your custom model from dropdown
   - Perform face swap as usual

## Command Line Options

```bash
python watserface.py run --help

# Common options:
  -s SOURCE_PATHS    Source image(s) containing face to swap
  -t TARGET_PATH     Target image/video to swap face onto
  -o OUTPUT_PATH     Output file path

  --face-detector-model {many,retinaface,scrfd,yolo_face}
  --face-landmarker-model {many,2dfan4,peppa_wutz,mediapipe}
  --face-swapper-model [model_name]  # Use custom trained models

  --execution-device-id DEVICE_ID
  --execution-providers {cpu,cuda,mps}
```

## 🚀 Development Environment (HuggingFace Spaces)

### Quick Start
1. **Launch**: HuggingFace Spaces automatically runs `bash dev_start.sh`
2. **Authenticate**: Run `claude-code auth login` in VS Code terminal
3. **Organization ID**: `7d37921e-6314-4b53-a02d-7ea9040b3afb`
4. **Access VS Code**: Available on port 8080 after startup

### Development Setup
- **Environment**: VS Code web server with Claude Code CLI
- **Installation**: Automatic (Node.js, code-server, Claude Code)
- **Session Tracking**: Progress saved in `.claude-session/` directory
- **Documentation**: See `REBRANDING_PRD.md` for project roadmap

## Brand Guidelines

See [brand guidelines.md](brand%20guidelines.md) for WatserFace visual identity:
- **Colors**: Glitch Magenta (#FF00FF), Deep Blurple (#4D4DFF), Electric Lime (#CCFF00)
- **Typography**: Righteous (display), JetBrains Mono (body)
- **Logo**: Split-face mark with vertical offset

## Attribution

### Original FaceFusion
- **Author**: Henry Ruhs
- **Repository**: https://github.com/facefusion/facefusion
- **Version at Fork**: 3.3.4
- **License**: OpenRAIL-AS
- **Copyright**: (c) 2025 Henry Ruhs

### WatserFace Modifications
- **Maintainer**: IlliquidAsset
- **Training Extensions**: 26 files, 2,400+ lines of code
- **Version**: 0.10.0 (independent fork versioning)
- **License**: OpenRAIL-AS (maintains original restrictions)
- **Copyright**: (c) 2025 IlliquidAsset (modifications only)

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete attribution details.

## License

OpenRAIL-AS (Open Responsible AI License - Academic Scholars)

This derivative work maintains all use-based restrictions from the original FaceFusion project. Users MUST NOT use this software for purposes that:
- Violate fundamental human rights
- Enable discrimination or harassment
- Cause physical or mental harm
- Facilitate illegal activities
- Create non-consensual intimate imagery

See [LICENSE.md](LICENSE.md) for full license text.
See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for ethical guidelines.

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history and release notes
- [ATTRIBUTION.md](ATTRIBUTION.md) - Complete attribution to FaceFusion
- [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) - Ethical use guidelines
- [REBRANDING_PRD.md](REBRANDING_PRD.md) - Rebranding strategy and execution plan
- [brand guidelines.md](brand%20guidelines.md) - Visual identity guidelines

## Community

- **Issues**: https://github.com/IlliquidAsset/facefusion/issues
- **Discussions**: https://github.com/IlliquidAsset/facefusion/discussions
- **Original FaceFusion**: https://github.com/facefusion/facefusion

## Credits

- **Original FaceFusion**: [Henry Ruhs](https://github.com/henryruhs) and contributors
- **Training Extensions**: IlliquidAsset
- **Performance Optimizations**: google-labs-jules (Bolt)
- **Community Contributors**: See commit history

## Support

If you find WatserFace useful, consider:
- ⭐ Starring this repository
- ⭐ Starring the [original FaceFusion repository](https://github.com/facefusion/facefusion)
- 📢 Sharing with the community
- 🐛 Reporting issues and contributing fixes

---

**WatserFace v0.10.0** - "Who's that again?"

*Based on FaceFusion by Henry Ruhs | Maintained by IlliquidAsset*

*Licensed under OpenRAIL-AS | Use Responsibly*
