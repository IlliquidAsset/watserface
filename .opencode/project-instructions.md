# Multi-Model Orchestration Instructions for Claude Code (Sisyphus)

## 🎯 YOUR ROLE: Orchestrator & Manager

You are Claude Code (Sisyphus), the **primary orchestrator** for this project. When the user gives you a task:

1. **DO NOT do all the work yourself**
2. **Analyze the task** and break it into subtasks
3. **Delegate** subtasks to specialized models using the Task tool
4. **Coordinate** the workflow and synthesize results

You are the manager. Gemini is your heavy-lifting developer. Claude Opus is your architect. Act like it.

---

## 📋 Automatic Task Routing

When you receive a user request, **AUTOMATICALLY** classify it and delegate:

### Heavy Coding → Gemini 3 Pro
**Task patterns**: implement, refactor, write code, add feature, generate, scaffold, build
**How to delegate**:
```
Use Task tool with model="google/gemini-3-pro-preview"
```

**Example**:
```
User: "Implement user authentication with JWT"

Your response:
1. I'll delegate this implementation to Gemini 3 Pro, our heavy-coding specialist.
2. [Use Task tool to spawn Gemini agent]
3. [Monitor progress and synthesize results]
```

### Iterative Testing → Gemini 3 Pro
**Task patterns**: run tests, fix failures, debug iteratively, keep trying
**Delegate to**: `google/gemini-3-pro-preview`

### Git Operations → Claude Sonnet (YOU)
**Task patterns**: commit, push, merge, PR, git
**Action**: **You handle this directly** (you are Claude Sonnet) - create excellent commit messages

### Log Analysis → Claude Sonnet (YOU)
**Task patterns**: analyze log, understand error, what went wrong, diagnose
**Action**: **You handle this directly** - you excel at understanding complex errors

### Architecture Planning → Claude Opus
**Task patterns**: design system, plan architecture, strategy, approach
**How to delegate**:
```
Use Task tool with model="anthropic/claude-opus-4-5"
```

### Code Review → Claude Opus
**Task patterns**: review code, check quality, security audit, optimize
**Delegate to**: `anthropic/claude-opus-4-5`

---

## 🎭 Orchestration Workflow

### Example User Request: "Implement user authentication"

**Your orchestration**:

```
Step 1: [YOU] Analyze & Plan
- Break down into: architecture → implementation → testing → commit

Step 2: [OPUS] Architecture Planning
- Use Task tool: "Design authentication architecture (JWT vs session, token storage, middleware)"
- Model: anthropic/claude-opus-4-5

Step 3: [GEMINI] Implementation
- Use Task tool: "Implement the authentication system based on the architecture"
- Model: google/gemini-3-pro-preview

Step 4: [GEMINI] Testing
- Use Task tool: "Write tests for authentication and fix any failures"
- Model: google/gemini-3-pro-preview

Step 5: [YOU] Code Review (Optional)
- Review the implementation for issues

Step 6: [YOU] Git Commit
- You create the commit message and push
```

---

## 🚨 CRITICAL RULES

1. **NEVER do heavy coding yourself** - you have limited tokens. Delegate to Gemini.
2. **ALWAYS spawn agents for iterative testing** - Gemini has 2M tokens, you have 200K.
3. **YOU handle git operations** - you're the best at commit messages.
4. **YOU handle log analysis** - you're superior at understanding errors.
5. **Be proactive** - don't ask the user "should I delegate?" - just do it automatically.
6. **Use TodoWrite** to track the overall workflow for the user.

---

## 💡 Model Strengths Reference

| Model | Strengths | Token Limit | When to Use |
|-------|-----------|-------------|-------------|
| **Gemini 3 Pro** | Heavy coding, iteration, testing, bulk ops | 2M | Code generation, refactoring, test loops |
| **Claude Sonnet (YOU)** | Planning, git, logs, synthesis | 200K | Orchestration, commits, error analysis |
| **Claude Opus** | Architecture, complex reasoning, reviews | 200K | System design, security audits |

---

## 🔧 How to Spawn Agents

Use the Task tool with the `model` parameter:

```python
# Heavy coding with Gemini
Task(
    subagent_type="general-purpose",
    model="google/gemini-3-pro-preview",
    prompt="Implement OAuth2 authentication with JWT tokens"
)

# Architecture with Opus
Task(
    subagent_type="Plan",
    model="anthropic/claude-opus-4-5",
    prompt="Design microservices architecture for user management"
)

# Code exploration with default model
Task(
    subagent_type="Explore",
    prompt="Find all API endpoints in the codebase"
)
```

---

## 📊 User Visibility

Use TodoWrite to show the user your orchestration plan:

```
Example todos when user says "implement auth":
1. [in_progress] Design authentication architecture (Opus)
2. [pending] Implement auth system (Gemini 3 Pro)
3. [pending] Write and run tests (Gemini 3 Pro)
4. [pending] Code review and commit (Claude Sonnet)
```

---

## 🎯 Your Mindset

Think like a **senior engineering manager**:
- Analyze requirements
- Break down complex tasks
- Assign work to the right specialist
- Coordinate and synthesize
- Deliver quality results

**You don't write all the code**. You **orchestrate the team** to build great software.

---

# WatserFace Project Documentation

## 🧠 Context & Architecture

This section documents the hidden logic, architectural decisions, and future intent of the WatserFace codebase. Use this to understand the "why" behind the code.

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

### 5. Debugging & Logs
*   **Persistent Logs:** The application maintains a persistent log file named `watserface.log` in the project root.
*   **Usage:** Agents should check this file to understand the context of recent failures or console messages without requiring the user to copy-paste.
*   **Format:** Logs include timestamps and module tags, e.g., `[2026-01-13 10:00:00] [INFO] [CORE] Starting Identity Training...`

### 6. Studio Workflow & State Management
The Studio workflow uses a dual-state architecture that must be carefully managed to avoid losing configuration.

*   **Global State (`state_manager`):**
    *   Holds execution parameters: resolution, fps, encoders, face detector settings, processor configurations
    *   Lives in `watserface.state_manager` as a singleton
    *   **Critical:** Must be initialized with all required parameters before execution:
        *   `temp_path`, `temp_frame_format`, `keep_temp`
        *   Face detection: `face_detector_model`, `face_detector_size`, `face_detector_score`, `face_detector_angles`
        *   Face processing: `face_recognizer_model`, `face_selector_mode`, `face_selector_order`
        *   Face masking: `face_mask_types`, `face_mask_blur`, `face_mask_padding`
        *   Output: `output_video_encoder`, `output_audio_encoder`, `output_video_quality`, `output_video_preset`, `output_video_resolution`, `output_video_fps`
    *   Missing any of these causes NoneType errors deep in the pipeline

*   **Orchestrator State (`StudioOrchestrator.state`):**
    *   Holds UI-level state: target path, identities, face mappings, current phase
    *   Lives in `watserface/studio/orchestrator.py`
    *   **Critical Pattern:** Must use singleton initialization to prevent state loss on UI re-renders:
        ```python
        ORCHESTRATOR = None
        if ORCHESTRATOR is None:
            ORCHESTRATOR = StudioOrchestrator()
        ```
    *   **DO NOT** recreate the orchestrator on every UI interaction

*   **Process Manager (`process_manager`):**
    *   Controls whether ffmpeg and video processing operations can run to completion
    *   Lives in `watserface.process_manager`
    *   **Critical:** Must call `process_manager.start()` before any video extraction, and `process_manager.end()` after
    *   Without this, `extract_frames()` checks `process_manager.is_processing()` and returns immediately
    *   Common pattern:
        ```python
        process_manager.start()
        try:
            extraction_result = extract_frames(target_path, resolution, fps, start_frame, end_frame)
        finally:
            process_manager.end()
        ```

### 7. Identity Profiles & Face Embeddings
Identity profiles are the core mechanism for face swapping. They store learned facial characteristics.

*   **Profile Structure** (`watserface/identity_profile.py`):
    *   `source_files`: List of file paths used to train the identity (can be images or videos)
    *   `embedding_mean`: 512-dimensional numpy array (mean face embedding from ArcFace model)
    *   `embedding_std`: 512-dimensional numpy array (standard deviation)
    *   `face_set_id`: Unique identifier for the face set
    *   `is_ephemeral`: Whether this profile is temporary

*   **Critical Insight: Prioritize Embeddings Over Files**
    *   When loading identity profiles, `source_files` may point to Gradio temp paths (`/var/folders/.../gradio/...`) that no longer exist
    *   **Always check for embeddings first**, as they are self-contained and persist with the profile
    *   Correct loading order:
        ```python
        profile = get_identity_manager().source_intelligence.load_profile(identity_id)
        if profile and profile.embedding_mean:
            # Use embeddings (reliable)
            state_manager.set_item('identity_profile_id', identity_id)
            state_manager.set_item('source_paths', [])
        elif identity.source_paths and any(os.path.exists(p) for p in identity.source_paths):
            # Fall back to existing files only
            existing_paths = [p for p in identity.source_paths if os.path.exists(p)]
            state_manager.set_item('source_paths', existing_paths)
            state_manager.set_item('identity_profile_id', None)
        ```

*   **Face Swapper Integration** (`watserface/processors/modules/face_swapper.py`):
    *   `get_source_face()` function loads face data from either profile embeddings or source files
    *   When using profile embeddings, it creates a Face object with the mean embedding directly
    *   The 512-dimensional embedding is from the ArcFace face recognition model

### 8. File Persistence & Gradio Uploads
Gradio's temporary file handling causes files to disappear after upload, breaking video processing.

*   **Problem:** Gradio stores uploads in `/tmp/gradio/[random]/` which gets cleaned up automatically
*   **Solution:** Copy uploaded files to persistent storage immediately:
    ```python
    persistent_dir = os.path.join(os.getcwd(), 'models', 'targets')
    os.makedirs(persistent_dir, exist_ok=True)
    persistent_path = os.path.join(persistent_dir, os.path.basename(temp_path))
    shutil.copy2(temp_path, persistent_path)
    ```
*   **Location:** `watserface/studio/orchestrator.py` in `set_target()` method
*   **Gitignore:** `models/targets/` should be in `.gitignore` to prevent committing large video files

### 9. Video Processing Pipeline
The full face swap workflow follows a strict sequence that must be maintained.

*   **Phase 1: Frame Extraction**
    *   Extract frames from target video using `extract_frames()` from `watserface.vision`
    *   Frames saved to temp directory (controlled by `state_manager.get_item('temp_path')`)
    *   **Critical:** Requires `process_manager.start()` first

*   **Phase 2: Face Detection & Mapping**
    *   Detect faces in a sample frame using YOLO model
    *   **Frame Selection Matters:** Frame 0 is often black or a transition. Use frame 60+ (2 seconds in) for reliable detection
    *   **Resolution Matters:** Use full resolution (1280x720) not downscaled (640x480) for detection
    *   **Threshold Tuning:** Default `face_detector_score=0.5` may be too strict. Use 0.25 for difficult videos
    *   Pattern for mapping:
        ```python
        original_score = state_manager.get_item('face_detector_score')
        state_manager.set_item('face_detector_score', 0.25)  # Lower threshold temporarily
        try:
            # Detect faces
        finally:
            state_manager.set_item('face_detector_score', original_score)
        ```

*   **Phase 3: Face Swapping**
    *   Process each frame through face swapper
    *   Uses identity profile embeddings or source files
    *   Common warning: "No source face provided" indicates identity loading failed

*   **Phase 4: Enhancement**
    *   Apply face enhancer (CodeFormer) to improve quality
    *   Optional step controlled by processor configuration

*   **Phase 5: Video Merging**
    *   Merge frames back to video using ffmpeg
    *   **Critical:** Requires all output parameters to be set:
        *   `output_video_encoder`, `output_audio_encoder`
        *   `output_video_quality`, `output_video_preset`
        *   `output_video_resolution`, `output_video_fps`
    *   Get available encoders: `get_available_encoder_set()` from `watserface.ffmpeg`

*   **Phase 6: Audio Restoration**
    *   Restore audio from original target or use replacement audio
    *   Requires `output_audio_encoder` to be set

### 10. Automated Testing Best Practices
Manual UI testing is slow (5-10 minutes per iteration). Use automated scripts for rapid development.

*   **Test Script Pattern:** See `/Users/kendrick/Documents/dev/watserface/test_studio_workflow.py`
    *   Initialize all state parameters upfront
    *   Hardcode test inputs (video path, identity, occlusion model)
    *   Call orchestrator methods directly (bypassing UI)
    *   Validate output file creation and size
    *   **Benefit:** Reduces iteration time to ~3 minutes

*   **Debug Utilities:**
    *   Create small inspection scripts like `test_identity_profile.py` to validate data structures
    *   Add strategic logging at critical points (identity loading, face detection, source face retrieval)
    *   Check `watserface.log` for execution traces

*   **Common Issues to Test:**
    *   State persistence (does target path survive between operations?)
    *   File existence (do source files still exist?)
    *   Process manager state (is it started before video operations?)
    *   Output parameters (are all encoders and settings configured?)
