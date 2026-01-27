# Phase 2.5 Architectural Decisions

## Design Choices

*Architectural decisions made during implementation will be appended here.*

---

## Task 2: ControlNet Pipeline Setup (#87)
**Date:** 2026-01-25

### Architectural Decisions

1. **Model Selection: SDXL ControlNet Models**
   - Chose `diffusers/controlnet-depth-sdxl-1.0-small` for depth (smaller VRAM footprint)
   - Chose `diffusers/controlnet-canny-sdxl-1.0` for edge detection
   - Base model: `stabilityai/stable-diffusion-xl-base-1.0`
   - VAE: `madebyollin/sdxl-vae-fp16-fix` (prevents NaN issues with fp16)

2. **Conditioning Scale: 0.75 Default**
   - Set default `controlnet_conditioning_scale=0.75` (within 0.7-0.8 range)
   - Configurable per-instance for fine-tuning
   - Both depth and canny use same scale by default

3. **Memory Optimization Strategy**
   - `enable_model_cpu_offload()` for <16GB VRAM support
   - `torch.float16` dtype for GPU acceleration
   - Fallback to `torch.float32` on CPU

4. **Class Design: ControlNetPipeline**
   - Added new `ControlNetPipeline` class alongside existing `ControlNetConditioner`
   - `ControlNetConditioner` = SD1.5 depth+normal (existing)
   - `ControlNetPipeline` = SDXL depth+canny (new, for face swap)
   - Factory function `create_controlnet_pipeline()` for easy instantiation

5. **Conditioning Preparation**
   - `prepare_depth_conditioning()`: Uses MiDaS depth estimation, outputs 3-channel normalized depth
   - `prepare_canny_conditioning()`: Uses OpenCV Canny edge detection (100/200 thresholds)
   - Both resize to 512x512 target size

6. **Fallback Behavior**
   - If models fail to load, `process()` returns resized input image
   - Graceful degradation without crashing

