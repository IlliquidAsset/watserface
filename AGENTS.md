# AGENTS.md

## 🧠 Context & Architecture

This file documents the hidden logic, architectural decisions, and future intent of the WatserFace codebase. Use this to understand the "why" behind the code.

### 1. The "2.5D" Normal Map Pipeline
We have shifted from pure 2D warping to a pseudo-3D ("2.5D") pipeline to enable realistic relighting.

*   **Z-Axis Capture:**
    *   In `watserface/face_landmarker.py`, the `detect_with_mediapipe` function now captures the **Z-coordinate** (depth) from MediaPipe.
    *   `landmarks_478` shape is `(478, 3)`.
    *   **Critical:** `landmarks_68` is explicitly stripped to `(68, 2)` to maintain backward compatibility with existing affine transform logic (which expects 2D points). *Do not change this without refactoring the entire warp system.*

*   **Normal Map Generation:**
    *   Located in `watserface/face_helper.py` -> `create_normal_map`.
    *   **Methodology:** We do NOT use a static mesh topology. Instead, we use `scipy.spatial.Delaunay` to dynamically triangulate the 2D points on every frame.
    *   **Why?** This makes the system robust to mesh definition changes and avoids fragile dependencies on `mediapipe.python.solutions` (which has unstable paths in some environments).
    *   **Output:** An RGB image where R=X, G=Y, B=Z normals. This is the "Bridge" to future lighting modules.

### 2. Generative Inpainting ("The Corndog Solution")
The codebase contains scaffolding for a high-fidelity occlusion handler intended to solve dynamic deformation problems (e.g., eating a corndog).

*   **Module:** `watserface/processors/modules/occlusion_inpainter.py`.
*   **Current State:** It is a pass-through (identity) processor.
*   **Future Intent:**
    1.  Receive `occlusion_mask` (from XSeg).
    2.  Receive `normal_map` (from the 2.5D pipeline).
    3.  Use a Diffusion Model (LoRA or ControlNet) to *hallucinate* the boundary pixels between the face and the occlusion, using the Normal Map to guide the 3D shape of the lips/skin around the object.

### 3. UI & Training Internals
*   **Smart Preview:** `watserface/uis/components/smart_preview.py` generates 3 variants (Fast, Balanced, Quality) sequentially. It uses `PresetManager` to toggle global state settings temporarily.
*   **Modeler Stop:** The "Stop" button in the LoRA training tab (`watserface/uis/layouts/modeler.py`) connects to `watserface.training.core.stop_training()`.

### 4. Known Quirks
*   **MediaPipe Imports:** The `mediapipe` package structure varies between environments (pip vs conda vs docker). We rely on top-level imports and avoid deep imports like `mediapipe.python.solutions.face_mesh_connections` where possible.
*   **Thread Safety:** `MEDIAPIPE_FACE_MESH` is global and not thread-safe. Access is protected by `watserface.face_landmarker.THREAD_LOCK`.
