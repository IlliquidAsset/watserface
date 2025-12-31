# WatserFace v0.11.0 Consolidated PRD

**Project Name:** WatserFace
**Tagline:** "Who's that again?"
**Version:** 0.11.0
**The Vibe:** Cheeky, chaotic, digital-native, high-contrast.

---

## 1. Objectives
1.  **Complete Rebranding:** Rename package to `watserface`, update all imports, and establish visual identity.
2.  **UI Modernization:** Implement a streamlined 3-column layout "above the fold" with a new visual identity.
3.  **Smart Preview System:** Replace technical overwhelm with 3 intelligent presets (Fast, Balanced, Quality).
4.  **Training Integration:** Consolidate the training platform and model management.
5.  **Stability & Deployment:** Ensure zero-error operation and production readiness (HF Spaces).

---

## 2. Division of Labor

### 🤖 Claude Code (Frontend & Visual Specialist)
**Focus:** UI Layout, CSS, Visual Assets, User Experience.

1.  **Visual Identity Implementation:**
    *   Integrate the SVG Logo (`.assets/logo_full.svg`) into `header.py`.
    *   Apply the "WatserFace" color palette (`Glitch Magenta`, `Deep Blurple`, `Electric Lime`) via CSS (`facefusion/uis/assets/overrides.css`).
    *   Update `about.py` to remove legacy titles and subtitles as per user request.

2.  **3-Column Layout Refactor (`swap.py`):**
    *   **Header:** Logo on top, Tab selector below it.
    *   **Column 1 (Source):** Uploads, Custom Identity dropdown, Source settings.
    *   **Column 2 (Target):** Uploads, Trim Frame, Target settings.
    *   **Column 3 (Output & Control):** Preview at the top, Output settings, Workflow selector (Instant/Job), and the main **START** button.
    *   Ensure all sections are properly collapsed and everything is "above the fold".

3.  **System Settings Menu:**
    *   Move Execution, Download, Memory, Temp Frame, Output Options, and Common Options into a dedicated system settings area.

4.  **Footer Implementation:**
    *   Display `v0.11.0`.
    *   Add credits and attribution to Henry Ruhs/FaceFusion.

### 🤖 Gemini (System & Integration Specialist)
**Focus:** Rebranding, Logic, Tests, Versioning.

1.  **Final Package Rebranding:**
    *   Rename `facefusion/` directory to `watserface/`.
    *   Perform global find/replace for `from facefusion` -> `from watserface`.
    *   Rename `watserface.py`, `watserface.ini`.

2.  **Smart Preview Logic:**
    *   Implement the backend logic for `fast`, `balanced`, and `quality` presets.
    *   Link UI selection to these backend configurations.

3.  **Final Verification:**
    *   Fix any errors resulting from Claude's UI refactor.
    *   Bump version to `0.11.0` in `metadata.py`.
    *   Run all tests and verify production readiness.

---

## 3. Visual Specifications (from Brand Guidelines)

*   **Glitch Magenta:** `#FF00FF`
*   **Deep Blurple:** `#4D4DFF`
*   **Electric Lime:** `#CCFF00` (Accents/CTAs)
*   **Void Black:** `#0D0D0D` (Background)
*   **Typography:** Righteous (Headings), JetBrains Mono (Body/UI).

---

## 4. Execution Plan

1.  **[CLAUDE]** UI Layout & Branding:
    *   Modify `header.py`, `footer.py`, `about.py`.
    *   Refactor `swap.py` to the 3-column layout.
    *   Apply color palette in `overrides.css`.
2.  **[GEMINI]** Package Rebranding:
    *   Rename directory and update all imports.
    *   Fix `AttributeError` and other stability bugs.
3.  **[GEMINI]** Smart Preview System:
    *   Implement preset logic in processors.
4.  **[BOTH]** Final testing and +1 Versioning.
