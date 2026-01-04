import gradio
import time
from typing import Optional, List, Any

from watserface import state_manager, wording
from watserface.uis.components import about, footer, source, target
from watserface.blob_helper import BlobOrchestrator

# --- Components ---
# We reuse some but wrapper others
BLOB_SOURCE = None
BLOB_TARGET = None
BLOB_MODE_SELECTOR = None
BLOB_ANALYZE_BTN = None
BLOB_PROCESS_BTN = None
BLOB_STATUS = None
BLOB_PREVIEW_HTML = None
BLOB_RESULT_VIDEO = None
BLOB_RESULT_IMAGE = None
BLOB_ORCHESTRATOR = BlobOrchestrator()

def pre_check() -> bool:
    return True

def render() -> gradio.Blocks:
    """Render the Dr. Blob Layout"""
    global BLOB_SOURCE, BLOB_TARGET, BLOB_MODE_SELECTOR, BLOB_ANALYZE_BTN, BLOB_PROCESS_BTN, BLOB_STATUS, BLOB_PREVIEW_HTML
    global BLOB_RESULT_VIDEO, BLOB_RESULT_IMAGE

    with gradio.Blocks(css="watserface/uis/assets/blob_styles.css", elem_id="blob_container") as layout:

        # Header
        with gradio.Row(elem_classes=["blob-header"]):
            with gradio.Column():
                gradio.HTML(
                    """
                    <div class="blob-title">WATSERFACE</div>
                    <div class="blob-subtitle">QUANTUM FIDELITY ENGINE</div>
                    """
                )

        # Main Stage
        with gradio.Row():
            # Source Column
            with gradio.Column(scale=1, elem_classes=["blob-glass"]):
                gradio.Markdown("### 🧬 SOURCE DNA")
                # Re-implementing a simple source input to avoid legacy UI clutter
                BLOB_SOURCE = gradio.File(
                    label="Drop Source Images or Face Set",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".webp", ".txt"], # .txt for face set? No, usually json or internal ID.
                    elem_id="blob_source"
                )

            # Target Column
            with gradio.Column(scale=1, elem_classes=["blob-glass"]):
                gradio.Markdown("### 🎯 TARGET REALITY")
                BLOB_TARGET = gradio.File(
                    label="Drop Target Video or Image",
                    file_count="single",
                    file_types=[".mp4", ".mov", ".png", ".jpg"],
                    elem_id="blob_target"
                )

        # Action Zone
        with gradio.Row(elem_classes=["blob-glass"]):
            with gradio.Column(scale=1):
                gradio.Markdown("### ⚙️ PERFORMANCE MODE")
                BLOB_MODE_SELECTOR = gradio.Radio(
                    choices=["Fast", "Balanced", "Quality"],
                    value="Quality",
                    label="Select Processing Priority",
                    info="Fast: Speed first. Balanced: Best of both. Quality: Maximum fidelity (Training enabled).",
                    elem_id="blob_mode_selector"
                )
            with gradio.Column(scale=2):
                 BLOB_ANALYZE_BTN = gradio.Button(
                     "🔮 ANALYZE & OPTIMIZE",
                     variant="primary",
                     elem_classes=["blob-btn-primary", "blob-pulse"]
                 )

        # Status & Preview
        with gradio.Row(elem_classes=["blob-glass"]):
            with gradio.Column(scale=1):
                gradio.Markdown("### 📡 SYSTEM TELEMETRY")
                BLOB_STATUS = gradio.Textbox(
                    label="Pipeline Status",
                    value="System Idle. Awaiting Inputs.",
                    lines=10,
                    interactive=False,
                    elem_classes=["blob-status-box"]
                )

            with gradio.Column(scale=1):
                gradio.Markdown("### 👁️ MATERIALIZATION")

                BLOB_RESULT_VIDEO = gradio.Video(
                    label="Output Reality",
                    visible=False,
                    interactive=False
                )
                BLOB_RESULT_IMAGE = gradio.Image(
                    label="Output Reality",
                    visible=False,
                    interactive=False
                )

                # Placeholder for initial state
                BLOB_PREVIEW_HTML = gradio.HTML(
                     """<div style='text-align:center; padding: 50px; color: #555;'>Result will materialize here...</div>"""
                )

                BLOB_PROCESS_BTN = gradio.Button(
                    "⚡ ENGAGE REALITY SHIFT ⚡",
                    variant="primary",
                    visible=False, # Hidden until analysis complete
                    elem_classes=["blob-btn-primary"]
                )

        # Footer
        with gradio.Row():
             gradio.HTML("<div style='text-align:center; color: #555; margin-top: 20px;'>Dr. Blob's Lab © 2024</div>")

    return layout

def listen() -> None:
    """Event listeners"""

    # Analyze Click
    BLOB_ANALYZE_BTN.click(
        fn=wrap_analyze,
        inputs=[BLOB_SOURCE, BLOB_TARGET, BLOB_MODE_SELECTOR],
        outputs=[BLOB_STATUS, BLOB_PROCESS_BTN]
    )

    # Process Click
    BLOB_PROCESS_BTN.click(
        fn=wrap_process,
        inputs=[BLOB_SOURCE, BLOB_TARGET],
        outputs=[BLOB_STATUS, BLOB_PREVIEW_HTML, BLOB_RESULT_VIDEO, BLOB_RESULT_IMAGE]
    )

# Wrapper functions for the orchestrator
def wrap_analyze(sources, target, mode):
    # This needs to return a generator
    strategy_cache = {}

    for msg, strategy in BLOB_ORCHESTRATOR.analyze_inputs(sources, target, mode):
        strategy_cache = strategy
        # Format log for textbox
        log_text = "\n".join(BLOB_ORCHESTRATOR.log_history[-10:])
        yield log_text, gradio.update(visible=False)

    # Final yield to show button
    yield "\n".join(BLOB_ORCHESTRATOR.log_history[-10:]), gradio.update(visible=True)

    # Store strategy in state_manager
    state_manager.set_item("blob_strategy", strategy_cache)


def wrap_process(sources, target):
    strategy = state_manager.get_item("blob_strategy")
    if not strategy:
        yield "❌ Error: Strategy lost. Please Analyze again.", gradio.update(visible=True), gradio.update(visible=False), gradio.update(visible=False)
        return

    final_output = None
    for msg, result in BLOB_ORCHESTRATOR.execute_pipeline(strategy, sources, target):
        log_text = "\n".join(BLOB_ORCHESTRATOR.log_history[-10:])
        if result:
            final_output = result
        # Yield progress
        yield log_text, gradio.update(visible=True), gradio.update(visible=False), gradio.update(visible=False)

    # After pipeline finishes
    final_log = "✅ Sequence Complete. \n" + "\n".join(BLOB_ORCHESTRATOR.log_history[-10:])

    if final_output:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(final_output)

        if mime_type and mime_type.startswith('video'):
             yield final_log, gradio.update(visible=False), gradio.update(value=final_output, visible=True), gradio.update(visible=False)
        else:
             yield final_log, gradio.update(visible=False), gradio.update(visible=False), gradio.update(value=final_output, visible=True)
    else:
        yield final_log, gradio.update(visible=True), gradio.update(visible=False), gradio.update(visible=False)

def run(ui: gradio.Blocks) -> None:
    ui.launch(
        inbrowser=state_manager.get_item('open_browser'),
        server_name=state_manager.get_item('server_name'),
        server_port=state_manager.get_item('server_port'),
        show_error=True
    )
