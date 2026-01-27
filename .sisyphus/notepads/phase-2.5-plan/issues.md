# Phase 2.5 Issues and Gotchas

## Problems and Workarounds

*Issues encountered and their solutions will be appended here.*

---

## Task 2: ControlNet Pipeline Setup (#87)
**Date:** 2026-01-25

### Issues Encountered

1. **Context7 API Quota Exceeded**
   - Context7 monthly quota exceeded during research
   - Workaround: Used librarian agent + grep.app GitHub search for documentation
   - Resolution: Successfully gathered all needed info from alternative sources

2. **No Sample Images in .assets/examples/**
   - Task specified `.assets/examples/target-1080p.jpg` but directory was empty
   - Workaround: Used existing face set frames from `models/face_sets/` for testing
   - Resolution: Tests use `models/face_sets/faceset_512e84d4_1768337182/frames/frame_000498.png`

3. **LSP Warnings on Existing Code**
   - Pre-existing LSP errors in `controlnet.py` for diffusers imports
   - These are false positives - diffusers exports these classes at runtime
   - Resolution: Ignored as they don't affect functionality

### Notes for Future Tasks

- Model loading is lazy (not loaded until `load()` called)
- Full pipeline test with actual model loading requires GPU and ~10GB VRAM
- Processing time target (<10s on GPU) cannot be verified without GPU test

