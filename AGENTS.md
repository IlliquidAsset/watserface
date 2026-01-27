# Agent Coordination & Activity Log

## 🎯 Mission Control

**All agents working on WatserFace MUST track progress using GitHub Projects.**

### Required Workflow for ALL Agents

1. **Before Starting Work:**
   - Check GitHub Projects board: https://github.com/IlliquidAsset/watserface/projects
   - Find your assigned issue (labeled `agent:gemini`, `agent:claude`, or `agent:grok`)
   - Move issue to "In Progress" column
   - Read linked documentation in issue description

2. **During Work:**
   - Update issue with progress comments every 4-8 hours
   - If blocked, add `status:blocked` label and comment why
   - Ask questions in issue comments (tag @IlliquidAsset)
   - Commit code with issue reference: `git commit -m "[M0] Fix interpolation (#42)"`

3. **After Completion:**
   - Move issue to "Review" column
   - Add comment with:
     - Files changed
     - Testing results
     - Any issues encountered
   - Wait for human validation before marking "Done"

4. **Cross-Agent Coordination:**
   - Read other agents' progress comments
   - If your work depends on another agent, comment on their issue
   - Use issue references: "Blocked by #42" or "Depends on #37"

### Why This Matters

**You are part of a multi-agent team.** Tracking work in GitHub Projects:
- ✅ Prevents duplicate work
- ✅ Makes dependencies visible
- ✅ Enables parallel execution
- ✅ Creates audit trail for debugging
- ✅ Keeps human informed without constant updates

### Setup Guide

See `GITHUB_PROJECT_SETUP.md` for:
- Project structure
- All issues and tasks
- Label meanings
- Assignment workflow

---

## 🔑 GitHub Access Setup (Required for All Agents)

**IMPORTANT:** To track progress in GitHub Projects, you need GitHub CLI access.

### First-Time Setup

If you haven't set up GitHub CLI yet, follow these steps:

1. **Install gh CLI** (if not already installed):
   ```bash
   brew install gh
   ```

2. **Authenticate with GitHub**:
   ```bash
   gh auth login --web
   ```
   - This will provide a one-time code
   - Ask the user to visit https://github.com/login/device
   - User enters the code to authorize

3. **Request project permissions** (if needed):
   ```bash
   gh auth refresh -h github.com -s project,read:project
   ```
   - This will provide another one-time code
   - Ask the user to authorize the additional permissions

4. **Verify access**:
   ```bash
   gh auth status
   gh project list --owner IlliquidAsset
   ```

### When to Request Access

Request GitHub access from the user when:
- You're assigned an issue in GitHub Projects
- You need to update issue status or add comments
- You need to create a pull request
- You need to read project board status

### What to Say

When requesting access, say:
> "To track my progress in GitHub Projects as documented in AGENTS.md, I need to set up GitHub CLI access. I'll need you to authorize two one-time codes during the setup. May I proceed with the authentication setup?"

Then guide them through the authentication steps above.

---

## Phase 2.5: Transparency & Compositing Proof-of-Concept (The "Mayonnaise Strategy")
**Date:** January 21, 2026
**Objective:** Validate a workflow for handling semi-transparent occlusions (fluids, glass) by generating a "dirty" swap (face swapped *under* the occlusion) and compositing the original occlusion back on top using a depth-derived alpha mask.

### 1. The Strategy
*   **Concept:** Use Monocular Depth Estimation (MiDaS) to create a mask of foreground objects (the occlusion).
*   **Dirty Swap:** Force the face swapper to ignore the occlusion and swap the face "through" it.
*   **Composite:** `Final = (Dirty_Swap * (1 - Alpha)) + (Original_Target * Alpha)`

### 2. Execution & Troubleshooting
We encountered significant challenges in generating a high-quality "dirty" swap.

#### Attempt 1: Standard Hyperswap (Box Mask)
*   **Result:** Failure. The swapper's built-in occlusion detection (XSeg) correctly identified the obstruction and refused to swap the face, returning the original image.
*   **Fix:** Switched to `region` mask to force the swap on skin areas, effectively ignoring the "obstruction" logic.

#### Attempt 2: High-Res Configuration (SimSwap 512 + Pixel Boost)
*   **Goal:** Increase resolution to fix "pixelated/garbage" output.
*   **Configuration:** `simswap_unofficial_512` + `pixel_boost 1024x1024`.
*   **Result:** "Monstrosity." The pixel boost (tiling) logic failed to align chunks correctly, resulting in a fragmented, unusable image.

#### Attempt 3: Landmark Robustness (MediaPipe)
*   **Goal:** Fix feature alignment (eye size/shape) using 478 landmarks instead of 68.
*   **Configuration:** `hyperswap` + `mediapipe`.
*   **Result:** Failure. Output was a blurry, low-res copy of the original.
*   **Diagnosis:** The MediaPipe integration appears to have a coordinate mapping or normalization bug in the current codebase, causing it to fail or default to a safe-mode/no-swap state.

#### Attempt 4: Resolution Grid Test
*   **Action:** Generated a 2x2 grid comparing `inswapper_128`, `hyperswap_256`, `simswap_512`, and `simswap_512+Boost`.
*   **Finding:** Confirmed that `simswap_unofficial_512` (Native, No Boost) provides the best baseline resolution.

### 3. Final Solution (The "Golden" Config)
We achieved the best "dirty" swap using this specific combination:
*   **Swapper:** `simswap_unofficial_512` (Highest native resolution).
*   **Landmarker:** `2dfan4` (68 points) - *Note: Reliable, unlike MediaPipe in current state.*
*   **Enhancer:** `gfpgan_1.4` at **80% blend** (Fixes resolution without the "comic book" over-sharpened look).
*   **Mask:** `region` (Crucial for swapping *under* occlusions without box artifacts).
*   **Compositing:** Custom script `test_transparency_composite.py` using a depth threshold of `0.74` (188/255).

### 4. Remaining Issues & Next Steps
*   **Eye Size/Geometry:** The user noted eye size was too small. This is likely a limitation of the 68-point landmarker (`2dfan4`) unable to capture the nuanced eye shape of the source.
*   **Action Item:** The MediaPipe integration (478 landmarks) must be fixed to allow for higher-fidelity geometry warping in future iterations.
