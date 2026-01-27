# Phase 2.5: WatserFace Transparency Handling Plan

## Context

### Original Request
User requested a comprehensive plan for all tasks in Phase 2.5 of the WatserFace project, covering issues #86 to #92 as listed on the GitHub Projects board, with the intent to potentially one-shot the entire project using Ultrawork mode, tasking Gemini and Claude for execution.

### Interview Summary
**Key Discussions**:
- Focus confirmed on Phase 2.5 tasks: Research and select DKT implementation (#86), Setup ControlNet pipeline (#87), Single frame mayonnaise test (#88), Implement DKTEstimator class (#89), ControlNet hyperparameter optimization (#90), Implement TransparencyHandler for video (#91), and Run full test suite (mayo, glasses, steam) (#92).
- GitHub CLI access successfully set up to track progress on the project board.
- User's intent to cover all tasks in a single comprehensive plan without splitting into phases.

**Research Findings**:
- **Codebase (Grep Searches)**: Extensive documentation on DKT as a planned feature for transparency, ControlNet integration with Stable Diffusion for inpainting, focus on mayonnaise test case for occlusion handling, and planned classes like DKTEstimator and TransparencyHandler.
- **Librarian (DKT/Point Tracking)**: DKT clarified as not a standard term; recommended TAPIR and CoTracker3 for point tracking under occlusions, with detailed comparisons and implementation code.
- **Librarian (ControlNet)**: Official Hugging Face Diffusers documentation for pipeline setup, best practices for face swap integration using IP-Adapter FaceID and InstantID, and critical parameters like conditioning scale.
- **Librarian (Transparency/Occlusions)**: Recommendations for FaceMat for uncertainty-aware alpha matting, MiDaS for depth estimation, and temporal consistency techniques for video processing.

### Metis Review
**Identified Gaps (addressed)**:
- **Dependencies**: Added explicit dependency chain and task flow to prevent bottlenecks.
- **Existing Code**: Incorporated checks for current transparency handling code (e.g., `test_transparency_composite.py`) to extend rather than replace.
- **Performance Targets**: Defined latency and memory constraints for video processing.
- **Test Data**: Included tasks to identify or create test assets for mayo/glasses/steam scenarios.
- **Rollback Strategy**: Ensured preservation of the 'Golden Config' as a fallback.
- **Quantitative Metrics**: Added specific acceptance criteria with measurable thresholds (e.g., SSIM, processing time).

---

## Work Objectives

### Core Objective
Develop and validate a transparency handling pipeline for WatserFace to address semi-transparent occlusions (e.g., mayo, glasses, steam) in face swapping, integrating advanced point tracking and ControlNet technologies across single frame and video processing.

### Concrete Deliverables
- Research report on point tracking implementation selection (TAPIR/CoTracker3) for transparency handling (#86).
- Functional ControlNet pipeline integrated with Stable Diffusion for face swap conditioning (#87).
- Single frame test result for corn dog with mayo scenario, validated against quality metrics (#88).
- Implemented `DKTEstimator` class for depth/normal estimation in `watserface/depth/dkt_estimator.py` (#89).
- Optimized ControlNet hyperparameters with documented ablation study (#90).
- Implemented `TransparencyHandler` class for video processing with temporal coherence in `watserface/processors/modules/transparency_handler.py` (#91).
- Full test suite results for mayo, glasses, and steam scenarios, with CI integration (#92).

### Definition of Done
- [ ] All tasks (#86 to #92) have completed deliverables as specified.
- [ ] Single frame mayonnaise test (#88) passes visual QA with SSIM >0.85.
- [ ] Video processing (#91) handles 30fps with temporal consistency score >0.9.
- [ ] Full test suite (#92) validates all scenarios with defined pass/fail thresholds.

### Must Have
- Preservation of existing 'Golden Config' (simswap_512 + gfpgan_1.4 + region mask) as fallback.
- Backward compatibility with current face swap pipeline.
- Quantitative acceptance criteria for each task (e.g., SSIM, processing time).
- Unit tests for new classes (`DKTEstimator`, `TransparencyHandler`).

### Must NOT Have (Guardrails)
- Modification of the existing 'Golden Config' without explicit approval.
- Introduction of new dependencies without justification.
- Deviation from the compositing formula: `Final = (Dirty_Swap * (1 - Alpha)) + (Original_Target * Alpha)`.
- Breaking of existing single-frame transparency workflow while adding video support.
- Real-time processing, model training, audio handling, UI/UX changes, new face swap models, or cloud deployment considerations (explicitly out of scope for Phase 2.5).

---

## Verification Strategy (MANDATORY)

> Test strategy confirmed as a hybrid approach: TDD with pytest for objective metrics and infrastructure, manual QA for subjective quality assessments.

### Test Decision
- **Infrastructure exists**: YES (confirmed by user, codebase uses pytest with `tests/` directory).
- **User wants tests**: YES (TDD with pytest for critical components, manual QA for subjective quality).
- **Framework**: pytest (confirmed by user for test infrastructure).

### TDD Enabled for Objective Metrics
Each TODO for infrastructure and measurable components follows RED-GREEN-REFACTOR:

**Task Structure:**
1. **RED**: Write failing test first
   - Test file: `[path].test.py`
   - Test command: `pytest [file]`
   - Expected: FAIL (test exists, implementation doesn't)
2. **GREEN**: Implement minimum code to pass
   - Command: `pytest [file]`
   - Expected: PASS
3. **REFACTOR**: Clean up while keeping green
   - Command: `pytest [file]`
   - Expected: PASS (still)

### Manual QA for Subjective Quality
**CRITICAL**: For tasks involving subjective quality (e.g., visual realism of transparency), manual verification MUST be exhaustive.

Each TODO includes detailed verification procedures:

**By Deliverable Type:**

| Type | Verification Tool | Procedure |
|------|------------------|-----------|
| **Frontend/UI** | Playwright browser | Navigate, interact, screenshot |
| **TUI/CLI** | interactive_bash (tmux) | Run command, verify output |
| **API/Backend** | curl / httpie | Send request, verify response |
| **Library/Module** | Python REPL | Import, call, verify |
| **Config/Infra** | Shell commands | Apply, verify state |

**Evidence Required:**
- Commands run with actual output
- Screenshots for visual changes
- Response bodies for API changes
- Terminal output for CLI changes

---

## Task Flow

```
#86 (DKT Research) → #87 (ControlNet Setup) → #88 (Mayo Test)
                       ↘ #89 (DKTEstimator) → #90 (Hyperparam Opt) → #91 (TransparencyHandler) → #92 (Full Test Suite)
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| A | #86, #87 | Independent research and setup tasks can start concurrently. |

| Task | Depends On | Reason |
|------|------------|--------|
| #88 | #86, #87 | Single frame test requires DKT selection and ControlNet setup. |
| #89 | #86 | DKTEstimator implementation depends on DKT research outcome. |
| #90 | #87, #89 | Hyperparameter optimization requires ControlNet pipeline and DKTEstimator. |
| #91 | #89, #90 | TransparencyHandler for video needs DKTEstimator and optimized parameters. |
| #92 | #88, #91 | Full test suite depends on single frame validation and video processing capability. |

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> Specify parallelizability for EVERY task.

- [x] 1. Research and Select Point Tracking Implementation (#86)

  **What to do**:
  - Conduct a detailed comparison of point tracking solutions (TAPIR, CoTracker3) for transparency handling under occlusions.
  - Evaluate based on accuracy (e.g., DAVIS AJ score), speed (fps), occlusion handling, and integration ease with WatserFace.
  - Produce a comparison matrix with at least 3 options, a recommended choice with rationale, and sample code snippets for integration.
  - Document findings in a report for team reference.

  **Must NOT do**:
  - Do not implement or integrate any solution at this stage; focus on research only.
  - Avoid scope creep into optical flow or unrelated tracking methods.

  **Parallelizable**: YES (with #87)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - None identified in codebase for point tracking; rely on external research.

  **API/Type References** (contracts to implement against):
  - Research from librarian: TAPIR (https://github.com/google-deepmind/tapnet), CoTracker3 (https://github.com/facebookresearch/co-tracker).

  **Test References** (testing patterns to follow):
  - To be determined based on test infrastructure.

  **Documentation References** (specs and requirements):
  - Librarian findings: TAPIR paper (https://arxiv.org/abs/2306.08637), CoTracker3 paper (https://arxiv.org/abs/2307.07635).
  - Project docs: `docs/architecture/PHASE_2.5_DKT_POC.md` - DKT scope and goals.

  **External References** (libraries and frameworks):
  - Official docs: https://deepmind-tapir.github.io/, https://co-tracker.github.io/.

  **WHY Each Reference Matters**:
  - TAPIR/CoTracker3 repos and papers provide implementation details and performance metrics crucial for selection.
  - Project docs outline the intended use of point tracking for transparency, guiding the research focus.

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Documentation file created: `docs/research/point-tracking-selection.md`.
  - [ ] Report covers comparison matrix with ≥3 options, includes recommendation with rationale, and sample code snippets.
  - [ ] `pytest tests/test_research_doc.py` → PASS (if test infra exists for doc validation).

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Verify report content:
    - Navigate to: `docs/research/point-tracking-selection.md`.
    - Action: Open and review document.
    - Verify: Document contains comparison matrix, recommendation, rationale, and code snippets.
    - Screenshot: Save evidence to `.sisyphus/evidence/task-1-report.png`.

  **Evidence Required:**
  - [ ] Document content summary captured (copy-paste key sections).
  - [ ] Screenshot saved (for visual confirmation of report).

  **Commit**: YES
  - Message: `feat(research): point tracking selection for transparency handling`
  - Files: `docs/research/point-tracking-selection.md`
  - Pre-commit: `pytest tests/test_research_doc.py` (if applicable)

- [x] 2. Setup ControlNet Pipeline (#87)

  **What to do**:
  - Implement a ControlNet pipeline integrated with Stable Diffusion for face swap conditioning, using pre-trained models (e.g., `diffusers/controlnet-depth-sdxl-1.0-small` for depth, `diffusers/controlnet-canny-sdxl-1.0` for edges).
  - Follow Hugging Face Diffusers documentation for setup, ensuring compatibility with existing face swap pipeline.
  - Configure with `controlnet_conditioning_scale=0.7-0.8` for strong facial geometry preservation.
  - Test initial setup on a sample image to confirm functionality.

  **Must NOT do**:
  - Do not engage in model training or fine-tuning; use pre-trained models only.
  - Avoid requiring GPU >16GB VRAM for inference.

  **Parallelizable**: YES (with #86)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `watserface/inpainting/controlnet.py:11-46` - Existing ControlNetConditioner class for loading models.

  **API/Type References** (contracts to implement against):
  - Librarian findings: Hugging Face Diffusers ControlNet pipeline setup (https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet).

  **Test References** (testing patterns to follow):
  - To be determined based on test infrastructure.

  **Documentation References** (specs and requirements):
  - Project docs: `docs/architecture/PHASE_2.5_DKT_POC.md:227-241` - ControlNet pipeline setup instructions.
  - Librarian docs: ControlNet models and parameters.

  **External References** (libraries and frameworks):
  - Official docs: https://github.com/huggingface/diffusers/tree/main/examples/controlnet.

  **WHY Each Reference Matters**:
  - Existing `controlnet.py` provides a pattern for model loading and integration, ensuring consistency.
  - Hugging Face docs offer official guidance on pipeline setup and optimal parameters for face swap use cases.
  - Project docs specify the exact implementation approach for Phase 2.5.

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_controlnet_pipeline.py`.
  - [ ] Test covers: Pipeline initialization and sample image processing.
  - [ ] `pytest tests/test_controlnet_pipeline.py` → PASS (N tests, 0 failures).

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using Python REPL:
    ```
    > from watserface.inpainting.controlnet import ControlNetConditioner
    > conditioner = ControlNetConditioner()
    Expected: No errors, object initializes.
    ```
  - [ ] Request: Run pipeline on sample image (e.g., `.assets/examples/target-1080p.jpg`).
  - [ ] Verify: Output image generated, size 512x512, processing time <10s on GPU.
  - [ ] Screenshot: Save evidence to `.sisyphus/evidence/task-2-output.png`.

  **Evidence Required:**
  - [ ] Command output captured (copy-paste actual terminal output).
  - [ ] Screenshot saved (for visual output verification).

  **Commit**: YES
  - Message: `feat(controlnet): setup pipeline for face swap conditioning`
  - Files: `watserface/inpainting/controlnet.py`
  - Pre-commit: `pytest tests/test_controlnet_pipeline.py` (if applicable)

- [ ] 3. Single Frame Mayonnaise Test (#88)

  **What to do**:
  - Execute a single frame test using the selected point tracking solution and ControlNet pipeline on the iconic corn dog with mayo image.
  - Follow the existing 'Mayonnaise Strategy': generate a 'dirty swap' ignoring occlusion, estimate depth/alpha mask, composite with original occlusion using `Final = (Dirty_Swap * (1 - Alpha)) + (Original_Target * Alpha)`.
  - Compare results against traditional swap (XSeg) and ground truth if available.
  - Validate quality with visual QA and SSIM >0.85 if metrics are feasible.

  **Must NOT do**:
  - Do not deviate from the existing compositing formula.
  - Avoid using test images other than the corn dog with mayo scenario without justification.

  **Parallelizable**: NO (depends on #86, #87)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `test_transparency_composite.py:5-63` - Existing script for compositing with depth threshold 0.74 (188/255).

  **API/Type References** (contracts to implement against):
  - Librarian findings: FaceMat for alpha matting (https://github.com/hyebin-c/FaceMat), MiDaS for depth estimation.

  **Test References** (testing patterns to follow):
  - To be determined based on test infrastructure.

  **Documentation References** (specs and requirements):
  - Project docs: `docs/architecture/PHASE_2.5_DKT_POC.md:219-261` - Single frame test goals and steps.
  - AGENTS.md:106-141 - Details on 'Mayonnaise Strategy' and test case.

  **External References** (libraries and frameworks):
  - Official docs: https://arxiv.org/abs/2508.03055 (FaceMat paper).

  **WHY Each Reference Matters**:
  - `test_transparency_composite.py` provides the exact compositing logic and threshold to replicate for consistency.
  - Project docs and AGENTS.md define the test scenario and expected workflow, ensuring alignment with Phase 2.5 goals.
  - FaceMat and MiDaS references offer advanced techniques for alpha matting and depth estimation to improve results.

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - Not applicable for this task as it focuses on subjective quality; manual QA prioritized.

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using interactive_bash (tmux session):
    - Command: `python test_transparency_composite.py --target .assets/examples/corn-dog-mayo.jpg --output previewtest/mayo-test-result.png`
    - Expected output contains: `Saved previewtest/mayo-test-result.png`
    - Exit code: 0
  - [ ] Verify state:
    - Navigate to: `previewtest/mayo-test-result.png`
    - Action: Open image.
    - Verify: Visual QA passes (face swap under mayo looks natural, no holes).
    - Screenshot: Save evidence to `.sisyphus/evidence/task-3-result.png`.

  **Evidence Required:**
  - [ ] Command output captured (copy-paste actual terminal output).
  - [ ] Screenshot saved (for visual verification).

  **Commit**: YES
  - Message: `test(mayo): single frame transparency test result`
  - Files: `previewtest/mayo-test-result.png`, `docs/test-reports/mayo-single-frame-report.md`
  - Pre-commit: None (manual QA)

- [ ] 4. Implement DKTEstimator Class (#89)

  **What to do**:
  - Develop the `DKTEstimator` class in `watserface/depth/dkt_estimator.py` for depth/normal/alpha estimation using the selected point tracking approach (e.g., CoTracker3 Offline).
  - Ensure it tracks points through occlusions for transparency estimation, following codebase class patterns.
  - Include methods for single frame and batch processing.
  - Target tracking of ≥10 points through 30 frames with <5px drift for accuracy.

  **Must NOT do**:
  - Do not implement optical flow or unrelated tracking methods; focus on point tracking for transparency.
  - Avoid breaking existing depth estimation workflows if they exist.

  **Parallelizable**: NO (depends on #86)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `watserface/depth/estimator.py:8-107` - Existing DepthEstimator class pattern for model loading and estimation.

  **API/Type References** (contracts to implement against):
  - Librarian findings: CoTracker3 implementation (https://github.com/facebookresearch/co-tracker).

  **Test References** (testing patterns to follow):
  - To be determined based on test infrastructure.

  **Documentation References** (specs and requirements):
  - Project docs: `docs/architecture/PHASE_2.5_DKT_POC.md:298-303` - DKTEstimator class definition and goals.

  **External References** (libraries and frameworks):
  - Official docs: https://co-tracker.github.io/ (CoTracker3 project page).

  **WHY Each Reference Matters**:
  - `estimator.py` provides a template for class structure and estimation logic, ensuring consistency with WatserFace architecture.
  - CoTracker3 docs and project page offer implementation details for accurate point tracking under occlusions.
  - Project docs specify the exact requirements for DKTEstimator in Phase 2.5.

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_dkt_estimator.py`.
  - [ ] Test covers: Point tracking through occlusion (≥10 points, 30 frames, <5px drift).
  - [ ] `pytest tests/test_dkt_estimator.py` → PASS (N tests, 0 failures).

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using Python REPL:
    ```
    > from watserface.depth.dkt_estimator import DKTEstimator
    > estimator = DKTEstimator()
    > result = estimator.estimate(sample_frame)
    Expected: No errors, returns depth/normal/alpha data.
    ```
  - [ ] Verify state:
    - Command: `python -m watserface.depth.test_dkt_estimator --frame .assets/examples/corn-dog-mayo.jpg`
    - Expected output contains: `Tracking successful, points tracked: ≥10`
    - Exit code: 0
  - [ ] Screenshot: Save evidence to `.sisyphus/evidence/task-4-test.png`.

  **Evidence Required:**
  - [ ] Command output captured (copy-paste actual terminal output).
  - [ ] Screenshot saved (for visual confirmation if applicable).

  **Commit**: YES
  - Message: `feat(depth): implement DKTEstimator for transparency estimation`
  - Files: `watserface/depth/dkt_estimator.py`
  - Pre-commit: `pytest tests/test_dkt_estimator.py` (if applicable)

- [ ] 5. ControlNet Hyperparameter Optimization (#90)

  **What to do**:
  - Fine-tune ControlNet parameters (e.g., `controlnet_conditioning_scale`, `control_guidance_start/end`, `strength`) for optimal face swapping quality.
  - Conduct an ablation study varying key parameters, documenting impact on output quality (SSIM, visual QA).
  - Define search space bounds before optimization (e.g., scale 0.3-0.9).
  - Target improved face geometry preservation and reduced artifacts.

  **Must NOT do**:
  - Do not engage in full model retraining; focus on parameter tuning only.
  - Avoid undocumented parameter changes; all variations must be logged.

  **Parallelizable**: NO (depends on #87, #89)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `watserface/inpainting/controlnet.py:100-116` - Existing parameter application logic in ControlNetConditioner.

  **API/Type References** (contracts to implement against):
  - Librarian findings: ControlNet parameter tuning (https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet#guess_mode).

  **Test References** (testing patterns to follow):
  - To be determined based on test infrastructure.

  **Documentation References** (specs and requirements):
  - Project docs: `docs/architecture/PHASE_2.5_DKT_POC.md:383-385` - Optimization goals for ControlNet.

  **External References** (libraries and frameworks):
  - Official docs: https://github.com/huggingface/diffusers/tree/main/examples/controlnet.

  **WHY Each Reference Matters**:
  - `controlnet.py` shows how parameters are currently applied, guiding the optimization process.
  - Librarian findings provide recommended ranges and impacts of ControlNet parameters for face swaps.
  - Project docs specify the need for fine-tuning to improve face swapping quality in Phase 2.5.

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_controlnet_optimization.py`.
  - [ ] Test covers: Parameter variations and quality metrics (e.g., SSIM improvement).
  - [ ] `pytest tests/test_controlnet_optimization.py` → PASS (N tests, 0 failures).

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using interactive_bash (tmux session):
    - Command: `python watserface/inpainting/test_controlnet_optimization.py --image .assets/examples/corn-dog-mayo.jpg`
    - Expected output contains: `Optimization complete, optimal scale: [value], SSIM: [>0.85 if measured]`
    - Exit code: 0
  - [ ] Verify state:
    - Navigate to: `docs/test-reports/controlnet-optimization-report.md`
    - Action: Open report.
    - Verify: Ablation study documented with metrics for each parameter set.
    - Screenshot: Save evidence to `.sisyphus/evidence/task-5-report.png`.

  **Evidence Required:**
  - [ ] Command output captured (copy-paste actual terminal output).
  - [ ] Screenshot saved (for visual confirmation of report).

  **Commit**: YES
  - Message: `feat(controlnet): optimize hyperparameters for face swapping`
  - Files: `docs/test-reports/controlnet-optimization-report.md`
  - Pre-commit: `pytest tests/test_controlnet_optimization.py` (if applicable)

- [ ] 6. Implement TransparencyHandler for Video (#91)

  **What to do**:
  - Develop the `TransparencyHandler` class in `watserface/processors/modules/transparency_handler.py` for video processing with temporal coherence.
  - Integrate `DKTEstimator` for depth/normal estimation and use optimized ControlNet parameters.
  - Ensure handling of variable frame rates, processing at 30fps with temporal consistency score >0.9.
  - Implement chunked processing for memory efficiency on long videos.

  **Must NOT do**:
  - Do not implement real-time processing; focus on batch processing only.
  - Avoid breaking existing single-frame transparency workflows.

  **Parallelizable**: NO (depends on #89, #90)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `test_transparency_composite.py:5-63` - Existing single-frame compositing logic to extend for video.

  **API/Type References** (contracts to implement against):
  - Librarian findings: VOIN approach for temporal consistency (https://openaccess.thecvf.com/content/ICCV2021/papers/Ke_Occlusion-Aware_Video_Object_Inpainting_ICCV_2021_paper.pdf).

  **Test References** (testing patterns to follow):
  - To be determined based on test infrastructure.

  **Documentation References** (specs and requirements):
  - Project docs: `docs/architecture/PHASE_2.5_DKT_POC.md:422-440` - TransparencyHandler goals and video processing requirements.

  **External References** (libraries and frameworks):
  - Official docs: https://arxiv.org/abs/2508.03055 (FaceMat for alpha matting).

  **WHY Each Reference Matters**:
  - `test_transparency_composite.py` provides the baseline compositing logic to adapt for video frames.
  - VOIN paper offers techniques for maintaining temporal coherence across frames, critical for video quality.
  - Project docs define the specific requirements for video transparency handling in Phase 2.5.

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_transparency_handler.py`.
  - [ ] Test covers: Video processing at 30fps, temporal consistency score >0.9.
  - [ ] `pytest tests/test_transparency_handler.py` → PASS (N tests, 0 failures).

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using interactive_bash (tmux session):
    - Command: `python watserface/processors/test_transparency_handler.py --video .assets/examples/target-1080p.mp4 --output previewtest/transparent-video-output.mp4`
    - Expected output contains: `Processing complete, frames processed: [count], consistency score: >0.9`
    - Exit code: 0
  - [ ] Verify state:
    - Navigate to: `previewtest/transparent-video-output.mp4`
    - Action: Play video.
    - Verify: Visual QA passes (no flicker, transparency handled naturally).
    - Screenshot: Save evidence to `.sisyphus/evidence/task-6-video.png` (key frames).

  **Evidence Required:**
  - [ ] Command output captured (copy-paste actual terminal output).
  - [ ] Screenshot saved (for visual verification of key frames).

  **Commit**: YES
  - Message: `feat(video): implement TransparencyHandler for temporal coherence`
  - Files: `watserface/processors/modules/transparency_handler.py`
  - Pre-commit: `pytest tests/test_transparency_handler.py` (if applicable)

- [ ] 7. Run Full Test Suite (Mayo, Glasses, Steam) (#92)

  **What to do**:
  - Execute a full test suite validating transparency handling across three scenarios: mayo (liquid), glasses (solid transparent), and steam (diffuse).
  - Identify or create test assets for each scenario if not already available.
  - Run tests using the implemented `TransparencyHandler` and `DKTEstimator`, ensuring all scenarios pass defined thresholds (e.g., SSIM >0.85, temporal consistency >0.9).
  - Integrate with CI if infrastructure exists, documenting pass/fail results.

  **Must NOT do**:
  - Do not expand beyond the three specified scenarios (mayo, glasses, steam).
  - Avoid undocumented test results; all outcomes must be logged.

  **Parallelizable**: NO (depends on #88, #91)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `test_transparency_composite.py:5-63` - Baseline for single-frame testing to adapt for full suite.

  **API/Type References** (contracts to implement against):
  - Librarian findings: FaceMat for alpha matting validation (https://github.com/hyebin-c/FaceMat).

  **Test References** (testing patterns to follow):
  - To be determined based on test infrastructure.

  **Documentation References** (specs and requirements):
  - Project docs: `docs/architecture/PHASE_2.5_DKT_POC.md:638-661` - Full test suite goals and scenarios.

  **External References** (libraries and frameworks):
  - Official docs: https://openaccess.thecvf.com/content/ICCV2021/papers/Ke_Occlusion-Aware_Video_Object_Inpainting_ICCV_2021_paper.pdf (VOIN for temporal consistency).

  **WHY Each Reference Matters**:
  - `test_transparency_composite.py` provides a starting point for test execution logic, adaptable for multiple scenarios.
  - Project docs define the exact test scenarios and expectations for Phase 2.5 validation.
  - Librarian references offer advanced techniques for ensuring quality in transparency handling.

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - Not applicable for this task as it focuses on subjective quality; manual QA prioritized.

  **Manual Execution Verification (ALWAYS include, even with tests):**
  - [ ] Using interactive_bash (tmux session):
    - Command: `python watserface/tests/run_full_transparency_suite.py --output docs/test-reports/full-suite-report.md`
    - Expected output contains: `All scenarios passed, mayo: [SSIM], glasses: [SSIM], steam: [SSIM]`
    - Exit code: 0
  - [ ] Verify state:
    - Navigate to: `docs/test-reports/full-suite-report.md`
    - Action: Open report.
    - Verify: All scenarios documented with metrics, pass/fail status clear.
    - Screenshot: Save evidence to `.sisyphus/evidence/task-7-report.png`.

  **Evidence Required:**
  - [ ] Command output captured (copy-paste actual terminal output).
  - [ ] Screenshot saved (for visual confirmation of report).

  **Commit**: YES
  - Message: `test(suite): full transparency test suite results`
  - Files: `docs/test-reports/full-suite-report.md`
  - Pre-commit: None (manual QA)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(research): point tracking selection for transparency handling` | docs/research/point-tracking-selection.md | pytest tests/test_research_doc.py |
| 2 | `feat(controlnet): setup pipeline for face swap conditioning` | watserface/inpainting/controlnet.py | pytest tests/test_controlnet_pipeline.py |
| 3 | `test(mayo): single frame transparency test result` | previewtest/mayo-test-result.png, docs/test-reports/mayo-single-frame-report.md | None (manual QA) |
| 4 | `feat(depth): implement DKTEstimator for transparency estimation` | watserface/depth/dkt_estimator.py | pytest tests/test_dkt_estimator.py |
| 5 | `feat(controlnet): optimize hyperparameters for face swapping` | docs/test-reports/controlnet-optimization-report.md | pytest tests/test_controlnet_optimization.py |
| 6 | `feat(video): implement TransparencyHandler for temporal coherence` | watserface/processors/modules/transparency_handler.py | pytest tests/test_transparency_handler.py |
| 7 | `test(suite): full transparency test suite results` | docs/test-reports/full-suite-report.md | None (manual QA) |

---

## Success Criteria

### Verification Commands
```bash
pytest tests/  # Expected: All tests pass for implemented components
python watserface/tests/run_full_transparency_suite.py --output docs/test-reports/full-suite-report.md  # Expected: All scenarios pass
```

### Final Checklist
- [ ] All 'Must Have' criteria met (preservation of Golden Config, backward compatibility, unit tests).
- [ ] All 'Must NOT Have' guardrails adhered to (no breaking changes, no scope creep).
- [ ] All tests pass or manual QA verifies functionality for subjective quality tasks.
