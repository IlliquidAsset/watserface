import gradio
import os
from typing import Optional, List, Any, Dict

from watserface import state_manager, wording
from watserface.studio import StudioOrchestrator, StudioPhase
from watserface.uis.components import about, footer
from watserface.identity_profile import get_identity_manager
from watserface.face_set import list_face_sets

ORCHESTRATOR: Optional[StudioOrchestrator] = None

IDENTITY_NAME_INPUT = None
IDENTITY_FILES_INPUT = None
IDENTITY_RESUME_DROPDOWN = None
FACESET_DROPDOWN = None
IDENTITY_ADD_BTN = None
IDENTITY_TRAIN_BTN = None
IDENTITY_EPOCHS_SLIDER = None
IDENTITY_LIST = None

TARGET_INPUT = None
TARGET_INFO = None
EXECUTION_TARGET_INPUT = None

OCCLUSION_NAME_INPUT = None
OCCLUSION_TRAIN_BTN = None
OCCLUSION_EPOCHS_SLIDER = None

IDENTITY_SELECT = None
OCCLUSION_SELECT = None

MAP_BTN = None
MAPPING_DISPLAY = None

PREVIEW_BTN = None
PREVIEW_IMAGE = None
QUALITY_DISPLAY = None

EXECUTE_BTN = None
OUTPUT_VIDEO = None
OUTPUT_IMAGE = None

STATUS_LOG = None
IDENTITY_STATUS = None
OCCLUSION_STATUS = None
MAPPING_STATUS = None
EXECUTION_STATUS = None

def pre_check() -> bool:
    return True


def render_content() -> None:
    """Render studio content without Blocks wrapper (for embedding in other layouts)"""
    global ORCHESTRATOR
    global IDENTITY_NAME_INPUT, IDENTITY_FILES_INPUT, IDENTITY_RESUME_DROPDOWN, FACESET_DROPDOWN
    global IDENTITY_ADD_BTN, IDENTITY_TRAIN_BTN
    global IDENTITY_EPOCHS_SLIDER, IDENTITY_LIST, IDENTITY_STATUS
    global TARGET_INPUT, TARGET_INFO
    global OCCLUSION_NAME_INPUT, OCCLUSION_TRAIN_BTN, OCCLUSION_EPOCHS_SLIDER, OCCLUSION_STATUS
    global MAP_BTN, MAPPING_DISPLAY, MAPPING_STATUS
    global PREVIEW_BTN, PREVIEW_IMAGE, QUALITY_DISPLAY
    global EXECUTE_BTN, OUTPUT_VIDEO, OUTPUT_IMAGE, EXECUTION_STATUS
    global STATUS_LOG

    # Only create orchestrator once to preserve state across renders
    if ORCHESTRATOR is None:
        ORCHESTRATOR = StudioOrchestrator()

    with gradio.Row():
        with gradio.Column(scale=1):
            gradio.Markdown('### 1. Identity Builder')

            IDENTITY_NAME_INPUT = gradio.Textbox(
                label='Identity Name',
                placeholder='e.g., person_a',
                value='identity_1'
            )

            IDENTITY_FILES_INPUT = gradio.File(
                label='Source Media (images/videos)',
                file_count='multiple',
                file_types=['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov']
            )

            with gradio.Row():
                identity_choices = [""] + [p.id for p in get_identity_manager().source_intelligence.list_profiles()]
                IDENTITY_RESUME_DROPDOWN = gradio.Dropdown(
                    choices=identity_choices,
                    label="Resume Existing Identity (Optional)",
                    info="Continue training an existing identity"
                )

                faceset_choices = [""] + list_face_sets()
                FACESET_DROPDOWN = gradio.Dropdown(
                    choices=faceset_choices,
                    label="Reuse Face Set (Optional)",
                    info="Use pre-extracted frames from a previous session"
                )

            with gradio.Row():
                IDENTITY_ADD_BTN = gradio.Button('Add Media', variant='secondary')
                IDENTITY_TRAIN_BTN = gradio.Button('Train Identity', variant='primary')

            IDENTITY_EPOCHS_SLIDER = gradio.Slider(
                label='Training Epochs',
                minimum=10,
                maximum=500,
                value=100,
                step=10
            )

            IDENTITY_STATUS = gradio.Textbox(
                label='Identity Status',
                value='IDLE',
                interactive=False,
                lines=3
            )

            IDENTITY_LIST = gradio.Textbox(
                label='Identities',
                value='No identities yet',
                lines=3,
                interactive=False
            )

        with gradio.Column(scale=1):
            gradio.Markdown('### 2. Target & Occlusion')

            TARGET_INPUT = gradio.File(
                label='Target (video/image to swap onto)',
                file_count='single',
                file_types=['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov']
            )

            TARGET_INFO = gradio.Textbox(
                label='Target Info',
                value='No target set',
                interactive=False
            )

            gradio.Markdown('#### Occlusion Training (Optional)')

            OCCLUSION_NAME_INPUT = gradio.Textbox(
                label='Occlusion Model Name',
                placeholder='e.g., xseg_custom',
                value='occlusion_1'
            )

            OCCLUSION_EPOCHS_SLIDER = gradio.Slider(
                label='XSeg Epochs',
                minimum=10,
                maximum=200,
                value=50,
                step=10
            )

            OCCLUSION_TRAIN_BTN = gradio.Button('Train Occlusion Model', variant='secondary')

            OCCLUSION_STATUS = gradio.Textbox(
                label='Occlusion Status',
                value='IDLE',
                interactive=False,
                lines=3
            )

    with gradio.Row():
        with gradio.Column(scale=1):
            gradio.Markdown('### 3. Map & Execute')

            with gradio.Row():
                MAP_BTN = gradio.Button('Auto-Map Faces', variant='secondary')
                PREVIEW_BTN = gradio.Button('Preview', variant='secondary')
                EXECUTE_BTN = gradio.Button('Execute', variant='primary')

            MAPPING_DISPLAY = gradio.Textbox(
                label='Face Mappings',
                value='No mappings yet',
                lines=3,
                interactive=False
            )

            MAPPING_STATUS = gradio.Textbox(
                label='Mapping Status',
                value='IDLE',
                interactive=False,
                lines=1
            )

            QUALITY_DISPLAY = gradio.Textbox(
                label='Quality Scores',
                value='',
                lines=2,
                interactive=False
            )

        with gradio.Column(scale=2):
            gradio.Markdown('### Output')

            PREVIEW_IMAGE = gradio.Image(
                label='Preview',
                visible=True,
                interactive=False
            )

            OUTPUT_VIDEO = gradio.Video(
                label='Output Video',
                visible=False,
                interactive=False
            )

            OUTPUT_IMAGE = gradio.Image(
                label='Output Image',
                visible=False,
                interactive=False
            )

            EXECUTION_STATUS = gradio.Textbox(
                label='Execution Status',
                value='IDLE',
                interactive=False,
                lines=1
            )

    with gradio.Row():
        STATUS_LOG = gradio.Textbox(
            label='Pipeline Log',
            value='Ready. Add identity media and target to begin.',
            lines=8,
            interactive=False
        )


def render() -> gradio.Blocks:
    global ORCHESTRATOR
    global IDENTITY_NAME_INPUT, IDENTITY_FILES_INPUT, IDENTITY_RESUME_DROPDOWN, FACESET_DROPDOWN
    global IDENTITY_ADD_BTN, IDENTITY_TRAIN_BTN
    global IDENTITY_EPOCHS_SLIDER, IDENTITY_LIST, IDENTITY_STATUS
    global TARGET_INPUT, TARGET_INFO, EXECUTION_TARGET_INPUT
    global OCCLUSION_NAME_INPUT, OCCLUSION_TRAIN_BTN, OCCLUSION_EPOCHS_SLIDER, OCCLUSION_STATUS
    global IDENTITY_SELECT, OCCLUSION_SELECT
    global MAP_BTN, MAPPING_DISPLAY, MAPPING_STATUS
    global PREVIEW_BTN, PREVIEW_IMAGE, QUALITY_DISPLAY
    global EXECUTE_BTN, OUTPUT_VIDEO, OUTPUT_IMAGE, EXECUTION_STATUS
    global STATUS_LOG

    # Only create orchestrator once to preserve state across renders
    if ORCHESTRATOR is None:
        ORCHESTRATOR = StudioOrchestrator()
    
    with gradio.Blocks(elem_id='studio_container') as layout:

        with gradio.Row():
            with gradio.Column(scale=1):
                gradio.Markdown('### 1. Identity Builder')
                
                IDENTITY_NAME_INPUT = gradio.Textbox(
                    label='Identity Name',
                    placeholder='e.g., person_a',
                    value='identity_1'
                )
                
                IDENTITY_FILES_INPUT = gradio.File(
                    label='Source Media (images/videos)',
                    file_count='multiple',
                    file_types=['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov']
                )

                with gradio.Row():
                    IDENTITY_ADD_BTN = gradio.Button('Add Media', variant='secondary')
                    IDENTITY_TRAIN_BTN = gradio.Button('Train Identity', variant='primary')
                
                IDENTITY_EPOCHS_SLIDER = gradio.Slider(
                    label='Training Epochs',
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10
                )
                
                IDENTITY_STATUS = gradio.Textbox(
                    label='Identity Status',
                    value='IDLE',
                    interactive=False,
                    lines=3
                )
                
                IDENTITY_LIST = gradio.Textbox(
                    label='Identities',
                    value=get_identity_list(),
                    lines=3,
                    interactive=False
                )
            
            with gradio.Column(scale=1):
                gradio.Markdown('### 2. Target & Occlusion')
                
                TARGET_INPUT = gradio.File(
                    label='Target (video/image to swap onto)',
                    file_count='single',
                    file_types=['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov']
                )
                
                TARGET_INFO = gradio.Textbox(
                    label='Target Info',
                    value='No target set',
                    interactive=False
                )
                
                gradio.Markdown('#### Occlusion Training (Optional)')
                
                OCCLUSION_NAME_INPUT = gradio.Textbox(
                    label='Occlusion Model Name',
                    placeholder='e.g., xseg_custom',
                    value='occlusion_1'
                )
                
                OCCLUSION_EPOCHS_SLIDER = gradio.Slider(
                    label='XSeg Epochs',
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10
                )
                
                OCCLUSION_TRAIN_BTN = gradio.Button('Train Occlusion Model', variant='secondary')
                
                OCCLUSION_STATUS = gradio.Textbox(
                    label='Occlusion Status',
                    value='IDLE',
                    interactive=False,
                    lines=3
                )
        
        with gradio.Row():
            with gradio.Column(scale=1):
                gradio.Markdown('### 3. Map & Execute')

                EXECUTION_TARGET_INPUT = gradio.File(
                    label='Target File (Alternative Upload)',
                    file_count='single',
                    file_types=['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov']
                )

                # Add selectors for manual override
                with gradio.Row():
                    identity_choices = [""] + [p.id for p in get_identity_manager().source_intelligence.list_profiles()]
                    IDENTITY_SELECT = gradio.Dropdown(choices=identity_choices, label="Select Identity")
                    
                    # Get trained occlusion models
                    occlusion_choices = [""]
                    trained_models_dir = os.path.join(os.path.dirname(__file__), '../../../.assets/models/trained')
                    if os.path.exists(trained_models_dir):
                        occlusion_choices += [f for f in os.listdir(trained_models_dir) if 'xseg' in f.lower() and f.endswith('.onnx')]
                    
                    OCCLUSION_SELECT = gradio.Dropdown(choices=occlusion_choices, label="Select Occlusion Model (Optional)")

                with gradio.Row():
                    MAP_BTN = gradio.Button('Auto-Map Faces', variant='secondary')
                    PREVIEW_BTN = gradio.Button('Preview', variant='secondary')
                    EXECUTE_BTN = gradio.Button('Execute', variant='primary')
                
                MAPPING_DISPLAY = gradio.Textbox(
                    label='Face Mappings',
                    value='No mappings yet',
                    lines=3,
                    interactive=False
                )

                MAPPING_STATUS = gradio.Textbox(
                    label='Mapping Status',
                    value='IDLE',
                    interactive=False,
                    lines=1
                )
                
                QUALITY_DISPLAY = gradio.Textbox(
                    label='Quality Scores',
                    value='',
                    lines=2,
                    interactive=False
                )
            
            with gradio.Column(scale=2):
                gradio.Markdown('### Output')
                
                PREVIEW_IMAGE = gradio.Image(
                    label='Preview',
                    visible=True,
                    interactive=False
                )
                
                OUTPUT_VIDEO = gradio.Video(
                    label='Output Video',
                    visible=False,
                    interactive=False
                )
                
                OUTPUT_IMAGE = gradio.Image(
                    label='Output Image',
                    visible=False,
                    interactive=False
                )

                EXECUTION_STATUS = gradio.Textbox(
                    label='Execution Status',
                    value='IDLE',
                    interactive=False,
                    lines=1
                )
        
        with gradio.Row():
            STATUS_LOG = gradio.Textbox(
                label='Pipeline Log',
                value='Ready. Add identity media and target to begin.',
                lines=8,
                interactive=False
            )

        with gradio.Row():
            gradio.HTML('''
                <div style="text-align:center; padding: 20px; border-top: 1px solid #eee; margin-top: 20px;">
                    <h3 style="margin:0; color: #444;">WATSERFACE STUDIO</h3>
                    <p style="margin:0; font-size: 0.9em; color: #888;">Unified Face Synthesis Pipeline v1.0.0</p>
                </div>
            ''')

    return layout


def listen() -> None:
    IDENTITY_ADD_BTN.click(
        fn=handle_add_identity,
        inputs=[IDENTITY_NAME_INPUT, IDENTITY_FILES_INPUT],
        outputs=[STATUS_LOG, IDENTITY_LIST]
    )
    
    IDENTITY_TRAIN_BTN.click(
        fn=handle_train_identity,
        inputs=[IDENTITY_NAME_INPUT, IDENTITY_EPOCHS_SLIDER, IDENTITY_FILES_INPUT],
        outputs=[STATUS_LOG, IDENTITY_STATUS, IDENTITY_LIST]
    )
    
    TARGET_INPUT.change(
        fn=handle_set_target,
        inputs=[TARGET_INPUT],
        outputs=[STATUS_LOG, TARGET_INFO]
    )

    EXECUTION_TARGET_INPUT.change(
        fn=handle_set_target,
        inputs=[EXECUTION_TARGET_INPUT],
        outputs=[STATUS_LOG, TARGET_INFO]
    )
    
    OCCLUSION_TRAIN_BTN.click(
        fn=handle_train_occlusion,
        inputs=[OCCLUSION_NAME_INPUT, OCCLUSION_EPOCHS_SLIDER],
        outputs=[STATUS_LOG, OCCLUSION_STATUS]
    )
    
    # Define override handler
    def handle_override(identity, occlusion):
        lines = []
        for msg, _ in ORCHESTRATOR.override_mapping(identity, occlusion):
            lines.append(msg)
        
        # Update display if mapping changed
        mapping_text = []
        for m in ORCHESTRATOR.state.mappings:
            mapping_text.append(f"Face {m.target_face_index} -> {m.source_identity}")
            
        return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), '\n'.join(mapping_text)

    # Listeners for manual override
    IDENTITY_SELECT.change(
        fn=handle_override,
        inputs=[IDENTITY_SELECT, OCCLUSION_SELECT],
        outputs=[STATUS_LOG, MAPPING_DISPLAY]
    )
    
    OCCLUSION_SELECT.change(
        fn=handle_override,
        inputs=[IDENTITY_SELECT, OCCLUSION_SELECT],
        outputs=[STATUS_LOG, MAPPING_DISPLAY]
    )

    MAP_BTN.click(
        fn=handle_auto_map,
        inputs=[],
        outputs=[STATUS_LOG, MAPPING_DISPLAY, MAPPING_STATUS]
    )
    
    PREVIEW_BTN.click(
        fn=handle_preview,
        inputs=[],
        outputs=[STATUS_LOG, PREVIEW_IMAGE, QUALITY_DISPLAY]
    )
    
    EXECUTE_BTN.click(
        fn=handle_execute,
        inputs=[],
        outputs=[STATUS_LOG, EXECUTION_STATUS, OUTPUT_VIDEO, OUTPUT_IMAGE, PREVIEW_IMAGE]
    )


def handle_add_identity(name: str, files: List[Any]):
    if not name or not files:
        return 'Please provide identity name and files', get_identity_list()
    
    log_lines = []
    for msg, data in ORCHESTRATOR.add_identity_media(name, files):
        log_lines.append(msg)
    
    return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), get_identity_list()


def handle_train_identity(name: str, epochs: int, files: List[Any]):
    if not name:
        yield 'Please provide identity name', 'Error: No name provided', get_identity_list()
        return
    
    # Auto-add media if not already added
    if name not in ORCHESTRATOR.state.identities and files:
        yield f'Auto-adding media for {name}...', 'Preparing...', get_identity_list()
        for msg, _ in ORCHESTRATOR.add_identity_media(name, files):
            pass

    yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), 'Starting...', get_identity_list()
    
    success = False
    for msg, telemetry in ORCHESTRATOR.train_identity(name, int(epochs)):
        status_text = "Phase: Training\n"
        
        if telemetry:
            if telemetry.get('status') == 'Complete' or telemetry.get('model_path'):
                success = True
            
            if 'epoch' in telemetry:
                status_text += f"Epoch: {telemetry.get('epoch', '?')}/{telemetry.get('total_epochs', '?')}\n"
            if 'loss' in telemetry:
                status_text += f"Loss: {telemetry['loss']}\n"
            if 'eta' in telemetry:
                status_text += f"ETA: {telemetry['eta']}\n"
            if 'status' in telemetry:
                status_text += f"Status: {telemetry['status']}\n"
            if 'overall_progress' in telemetry:
                 status_text += f"Progress: {telemetry['overall_progress']}"

        yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), status_text, get_identity_list()
    
    if success:
        yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), f"COMPLETE: {name} trained", get_identity_list()
    else:
        yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), f"FAILED: Could not train {name}. Check logs.", get_identity_list()


def handle_set_target(file: Any):
    if not file:
        return 'No target file', 'No target set'
    
    info = {}
    for msg, data in ORCHESTRATOR.set_target(file):
        info = data
    
    info_text = f"Path: {info.get('path', 'N/A')}\nVideo: {info.get('is_video', False)}\nAudio: {info.get('has_audio', False)}"
    return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), info_text


def handle_train_occlusion(name: str, epochs: int):
    if not name:
        yield 'Please provide occlusion model name', 'Error: No name provided'
        return
    
    yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), 'Starting...'
    
    for msg, telemetry in ORCHESTRATOR.train_occlusion(name, int(epochs)):
        status_text = "Phase: Training Occlusion\n"
        
        if telemetry:
            if 'epoch' in telemetry:
                status_text += f"Epoch: {telemetry.get('epoch', '?')}/{telemetry.get('total_epochs', '?')}\n"
            if 'loss' in telemetry:
                status_text += f"Loss: {telemetry['loss']}\n"
            if 'eta' in telemetry:
                status_text += f"ETA: {telemetry['eta']}\n"
            if 'status' in telemetry:
                status_text += f"Status: {telemetry['status']}\n"
            if 'overall_progress' in telemetry:
                 status_text += f"Progress: {telemetry['overall_progress']}"
        
        yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), status_text
    
    yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), f"COMPLETE: {name} trained"


def handle_auto_map():
    mappings_text = []
    yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), '...', 'Scanning...'
    
    for msg, data in ORCHESTRATOR.auto_map_faces():
        yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), '...', 'Scanning...'
    
    for m in ORCHESTRATOR.state.mappings:
        mappings_text.append(f"Face {m.target_face_index} -> {m.source_identity} (conf: {m.confidence:.2f})")
    
    yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), '\n'.join(mappings_text) or 'No mappings', 'Mapped'


def handle_preview():
    preview_frame = None
    for msg, data in ORCHESTRATOR.preview():
        if data is not None:
            preview_frame = data
    
    quality_text = []
    for m in ORCHESTRATOR.state.mappings:
        quality_text.append(f"{m.source_identity}: {m.quality_score:.2f}")
    
    return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), preview_frame, '\n'.join(quality_text)


def handle_execute():
    output_path = None
    yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), 'Starting...', gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True)
    
    for msg, data in ORCHESTRATOR.execute():
        if data is not None:
            output_path = data
        yield '\n'.join(ORCHESTRATOR.state.get_recent_logs()), 'Processing...', gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True)
    
    logs = '\n'.join(ORCHESTRATOR.state.get_recent_logs())
    
    if output_path:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(output_path)
        if mime_type and mime_type.startswith('video'):
            yield logs, 'Complete', gradio.update(value=output_path, visible=True), gradio.update(visible=False), gradio.update(visible=False)
        else:
            yield logs, 'Complete', gradio.update(visible=False), gradio.update(value=output_path, visible=True), gradio.update(visible=False)
    else:
        yield logs, 'Failed', gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True)



def get_identity_list() -> str:
    if not ORCHESTRATOR or not ORCHESTRATOR.state.identities:
        return 'No identities yet'
    
    lines = []
    for name, identity in ORCHESTRATOR.state.identities.items():
        status = 'trained' if identity.model_path else f'{len(identity.source_paths)} files'
        epochs = f' ({identity.epochs_trained} epochs)' if identity.epochs_trained else ''
        lines.append(f"- {name}: {status}{epochs}")
    
    return '\n'.join(lines)


KEYBOARD_SHORTCUTS_JS = """
<script>
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    const keyMap = {
        'KeyA': 'identity_add_btn',
        'KeyT': 'identity_train_btn',
        'KeyM': 'map_btn',
        'KeyP': 'preview_btn',
        'KeyE': 'execute_btn',
        'KeyO': 'occlusion_train_btn'
    };
    
    if (e.ctrlKey || e.metaKey) {
        if (e.key === 's') {
            e.preventDefault();
            const saveEvent = new CustomEvent('studio_save_project');
            document.dispatchEvent(saveEvent);
        }
        if (e.key === 'o') {
            e.preventDefault();
            const loadEvent = new CustomEvent('studio_load_project');
            document.dispatchEvent(loadEvent);
        }
        return;
    }
    
    const btnId = keyMap[e.code];
    if (btnId) {
        e.preventDefault();
        const btn = document.querySelector(`#${btnId} button`) || 
                   document.querySelector(`button[id*="${btnId}"]`);
        if (btn) btn.click();
    }
    
    if (e.code === 'Space' && e.shiftKey) {
        e.preventDefault();
        const previewBtn = document.querySelector('#preview_btn button');
        if (previewBtn) previewBtn.click();
    }
});
</script>
"""


def run(ui: gradio.Blocks) -> None:
    ui.launch(
        inbrowser=state_manager.get_item('open_browser'),
        server_name=state_manager.get_item('server_name'),
        server_port=state_manager.get_item('server_port'),
        show_error=True
    )
