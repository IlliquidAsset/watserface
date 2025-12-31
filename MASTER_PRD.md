# WatserFace v0.11.0 Master PRD

**Version:** 0.11.0
**Status:** Active Development
**Goal:** Modernize UI, consolidate workflows, and stabilize the platform.

## 1. Project Overview
We are refactoring "WatserFace" (formerly FaceFusion fork) to version 0.11.0. This update focuses on a major UI overhaul to improve usability and a cleaner separation of concerns.

## 2. Core Directives

### 2.1 UI Layout Refactoring (Primary Task for Claude)
The "Swap" layout must be redesigned into a 3-column structure to minimize scrolling and expanding sections.

*   **Header:**
    *   **Logo:** Add SVG Logo (Code/File to be implemented).
    *   **Tabs:** Move `Swap` | `Training` tabs *below* the logo.
    *   **Removed:** Remove the large text titles "WatserFace 0.10.0 - Training Edition" and subtitle.
*   **Column 1: Source**
    *   Source Image/Audio inputs.
    *   Source-specific settings (e.g., Face Selector, etc., if tied to source).
    *   Enhanced Source dropdown (Identity Profile).
*   **Column 2: Target**
    *   Target File input.
    *   Trim Frame settings.
    *   Target-specific settings.
*   **Column 3: Output & Control**
    *   **Top:** Preview Component.
    *   **Middle:** Output Path/File settings.
    *   **Bottom:** Workflow Selector (Instant/Job) & **START** Button.

### 2.2 System Settings Menu
Move global configuration options out of the main workflow columns into a dedicated "System Settings" area (e.g., a separate tab, a modal, or a collapsible header section).
*   **Include:** Execution, Download, Memory, Temp Frame, Output Options (Encoder/Quality), Common Options.

### 2.3 Footer
*   **Version:** Display `v0.11.0`.
*   **Credits:** Move "Based on FaceFusion by Henry Ruhs" and other attributions here.

## 3. Technical Requirements

### 3.1 Crash Fixes (Assigned to Gemini - Completed)
*   Fixed `AttributeError: 'NoneType'` in `face_swapper.py` when model options are missing.
*   Verified Identity Loading logic.

### 3.2 Versioning
*   Current Base: 0.10.0
*   Target: 0.11.0 (+1 bump upon completion).

## 4. Work Division

### 🤖 Claude Code (UI & Frontend Specialist)
*   **Objective:** Implement the 3-Column Layout and Header/Footer changes.
*   **Files:** `facefusion/uis/layouts/swap.py`, `facefusion/uis/components/*.py`.
*   **Instructions:**
    1.  Modify `swap.py` to use `gradio.Row` with 3 `gradio.Column` blocks.
    2.  Move components into respective columns.
    3.  Create a new `system_settings.py` component or layout section for the moved global settings.
    4.  Update the Header to accept the SVG logo and reposition tabs.
    5.  Update the Footer with version and credits.

### 🤖 Gemini (Backend & Integration Specialist)
*   **Objective:** Support Claude with logic changes, verify stability, and finalize the release.
*   **Tasks:**
    1.  Ensure `check_syntax.py` passes after UI changes.
    2.  Verify `face_swapper` and `identity_profile` integration remains stable.
    3.  Perform final version bump to `0.11.0` in `metadata.py`.
    4.  Clean up any residual temp files or unused assets.

## 5. Assets
*   **Logo:** SVG Code required (User to provide or we use placeholder).

---
**Execution Order:**
1.  **Claude** starts UI Refactoring.
2.  **Gemini** verifies and runs tests.
3.  **Gemini** bumps version and finalizes.
