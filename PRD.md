# WatserFace Master PRD

**Product:** WatserFace (formerly FaceFusion fork)
**Current Version:** 0.12.0 (Target)
**Tagline:** "Who's that again?"
**Vibe:** Cheeky, chaotic, digital-native, high-contrast.

---

## 1. Executive Summary & Brand Identity

**Mission:** Modernize FaceFusion into a stable, user-friendly, and visually striking face processing platform.
**Visual Identity:**
-   **Colors:** Glitch Magenta (`#FF00FF`), Deep Blurple (`#4D4DFF`), Electric Lime (`#CCFF00`), Void Black (`#0D0D0D`).
-   **Typography:** Righteous (Headings), JetBrains Mono (Body/UI).
-   **Aesthetic:** "Internet speed test", dark glassmorphism, animated graphs.

---

## 2. Active Development Focus (v0.12.0)
**Priority: HIGH** | **Focus: Training UI Redesign & M4 Mac Optimization**

### 2.1 Problem Statement
1.  **Progress Bar Conflicts:** Text-based telemetry fights with visual progress bars.
2.  **Visuals:** UI lacks the intended "high-tech/speed-test" vibe; missing glassmorphism.
3.  **Performance:** M4 Apple Silicon (4 Performance Cores, 16GB RAM) needs balanced optimization for fanless thermal management.

### 2.2 Success Criteria
-   [ ] **Training Status UI:** Split into 2 columns (Progress Bars left, Real-time Charts right).
-   [ ] **Visuals:** Dark glassmorphism (`backdrop-filter: blur`), animated CSS progress bars.
-   [ ] **Performance:** Default to 4-6 threads + CoreML/MPS. Memory limit capped at 10GB to ensure system stability.
-   [ ] **Telemetry:** Real-time loss graphs using `gradio.LinePlot`.

### 2.3 Technical Specifications
-   **Glassmorphism CSS:** Add to `watserface/uis/assets/overrides.css`.
-   **Training Logic:** Remove `progress()` calls from `train_instantid.py` to fix flickering. Yield rich dictionaries instead of text.
-   **Hardware Detection:** Update `execution.py` and `memory.py` to auto-detect Apple Silicon and set defaults (CoreML, higher thread count).

---

## 3. Core Features (Implemented/In-Progress v0.11.0)

### 3.1 3-Column Swap UI
-   **Column 1 (Source):** Uploads, Identity Profile dropdown.
-   **Column 2 (Target):** Uploads, Trim settings.
-   **Column 3 (Output):** Preview (Top), Settings (Middle), Start/Stop (Bottom).
-   **System Settings:** Moved to a dedicated collapsible section or separate tab to reduce clutter.

### 3.2 Smart Previews
-   **Presets:** Fast, Balanced, Quality.
-   **Function:** Users select a preset to auto-configure complex parameters (detector score, enhancement blend, etc.).

---

## 4. Rebranding & Architecture (Reference)

### 4.1 Package Rename
-   `facefusion` -> `watserface`
-   `facefusion.py` -> `watserface.py`
-   Entry points: `app.py` (wrapper), `deploy_local.sh`.

### 4.2 Licensing & Attribution
-   **License:** OpenRAIL-AS (Strict adherence).
-   **Attribution:** Must credit Henry Ruhs (original author) in `ATTRIBUTION.md`, `README.md`, and Footer.
-   **Safety:** NSFW detection disabled for research/local use; `RESPONSIBLE_USE.md` added.

---

## 5. File Structure
```text
watserface/
├── uis/
│   ├── assets/ (CSS, JS)
│   ├── components/ (Modular UI parts)
│   └── layouts/ (Swap, Training, Webcam)
├── training/
│   ├── core.py (Orchestrator)
│   ├── train_instantid.py (Model logic)
│   └── datasets/ (Data loaders)
└── ...
```
