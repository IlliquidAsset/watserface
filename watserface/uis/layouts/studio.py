import gradio
from typing import Optional, List, Any, Dict

from watserface import state_manager, wording
from watserface.studio import StudioOrchestrator, StudioPhase
from watserface.uis.components import about, footer

ORCHESTRATOR: Optional[StudioOrchestrator] = None

IDENTITY_NAME_INPUT = None
IDENTITY_FILES_INPUT = None
IDENTITY_ADD_BTN = None
IDENTITY_TRAIN_BTN = None
IDENTITY_EPOCHS_SLIDER = None
IDENTITY_LIST = None

TARGET_INPUT = None
TARGET_INFO = None

OCCLUSION_NAME_INPUT = None
OCCLUSION_TRAIN_BTN = None
OCCLUSION_EPOCHS_SLIDER = None

MAP_BTN = None
MAPPING_DISPLAY = None

PREVIEW_BTN = None
PREVIEW_IMAGE = None
QUALITY_DISPLAY = None

EXECUTE_BTN = None
OUTPUT_VIDEO = None
OUTPUT_IMAGE = None

STATUS_LOG = None
PHASE_INDICATOR = None


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    global ORCHESTRATOR
    global IDENTITY_NAME_INPUT, IDENTITY_FILES_INPUT, IDENTITY_ADD_BTN, IDENTITY_TRAIN_BTN
    global IDENTITY_EPOCHS_SLIDER, IDENTITY_LIST
    global TARGET_INPUT, TARGET_INFO
    global OCCLUSION_NAME_INPUT, OCCLUSION_TRAIN_BTN, OCCLUSION_EPOCHS_SLIDER
    global MAP_BTN, MAPPING_DISPLAY
    global PREVIEW_BTN, PREVIEW_IMAGE, QUALITY_DISPLAY
    global EXECUTE_BTN, OUTPUT_VIDEO, OUTPUT_IMAGE
    global STATUS_LOG, PHASE_INDICATOR
    
    ORCHESTRATOR = StudioOrchestrator()
    
    with gradio.Blocks(elem_id='studio_container') as layout:
        
        with gradio.Row():
            gradio.HTML('''
                <div style="text-align:center; padding: 10px;">
                    <h1 style="margin:0; font-size: 2em;">WATSERFACE STUDIO</h1>
                    <p style="margin:0; color: #666;">Unified Face Synthesis Pipeline v1.0.0</p>
                </div>
            ''')
        
        with gradio.Row():
            PHASE_INDICATOR = gradio.Textbox(
                label='Current Phase',
                value='IDLE',
                interactive=False,
                scale=1
            )
        
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
        
        with gradio.Row():
            STATUS_LOG = gradio.Textbox(
                label='Pipeline Log',
                value='Ready. Add identity media and target to begin.',
                lines=8,
                interactive=False
            )
    
    return layout


def listen() -> None:
    IDENTITY_ADD_BTN.click(
        fn=handle_add_identity,
        inputs=[IDENTITY_NAME_INPUT, IDENTITY_FILES_INPUT],
        outputs=[STATUS_LOG, IDENTITY_LIST]
    )
    
    IDENTITY_TRAIN_BTN.click(
        fn=handle_train_identity,
        inputs=[IDENTITY_NAME_INPUT, IDENTITY_EPOCHS_SLIDER],
        outputs=[STATUS_LOG, PHASE_INDICATOR, IDENTITY_LIST]
    )
    
    TARGET_INPUT.change(
        fn=handle_set_target,
        inputs=[TARGET_INPUT],
        outputs=[STATUS_LOG, TARGET_INFO]
    )
    
    OCCLUSION_TRAIN_BTN.click(
        fn=handle_train_occlusion,
        inputs=[OCCLUSION_NAME_INPUT, OCCLUSION_EPOCHS_SLIDER],
        outputs=[STATUS_LOG, PHASE_INDICATOR]
    )
    
    MAP_BTN.click(
        fn=handle_auto_map,
        inputs=[],
        outputs=[STATUS_LOG, MAPPING_DISPLAY]
    )
    
    PREVIEW_BTN.click(
        fn=handle_preview,
        inputs=[],
        outputs=[STATUS_LOG, PREVIEW_IMAGE, QUALITY_DISPLAY]
    )
    
    EXECUTE_BTN.click(
        fn=handle_execute,
        inputs=[],
        outputs=[STATUS_LOG, PHASE_INDICATOR, OUTPUT_VIDEO, OUTPUT_IMAGE, PREVIEW_IMAGE]
    )


def handle_add_identity(name: str, files: List[Any]):
    if not name or not files:
        return 'Please provide identity name and files', get_identity_list()
    
    log_lines = []
    for msg, data in ORCHESTRATOR.add_identity_media(name, files):
        log_lines.append(msg)
    
    return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), get_identity_list()


def handle_train_identity(name: str, epochs: int):
    if not name:
        return 'Please provide identity name', 'IDLE', get_identity_list()
    
    for msg, telemetry in ORCHESTRATOR.train_identity(name, int(epochs)):
        pass
    
    return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), ORCHESTRATOR.state.phase.value.upper(), get_identity_list()


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
        return 'Please provide occlusion model name', 'IDLE'
    
    for msg, telemetry in ORCHESTRATOR.train_occlusion(name, int(epochs)):
        pass
    
    return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), ORCHESTRATOR.state.phase.value.upper()


def handle_auto_map():
    mappings_text = []
    for msg, data in ORCHESTRATOR.auto_map_faces():
        pass
    
    for m in ORCHESTRATOR.state.mappings:
        mappings_text.append(f"Face {m.target_face_index} -> {m.source_identity} (conf: {m.confidence:.2f})")
    
    return '\n'.join(ORCHESTRATOR.state.get_recent_logs()), '\n'.join(mappings_text) or 'No mappings'


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
    for msg, data in ORCHESTRATOR.execute():
        if data is not None:
            output_path = data
    
    logs = '\n'.join(ORCHESTRATOR.state.get_recent_logs())
    phase = ORCHESTRATOR.state.phase.value.upper()
    
    if output_path:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(output_path)
        if mime_type and mime_type.startswith('video'):
            return logs, phase, gradio.update(value=output_path, visible=True), gradio.update(visible=False), gradio.update(visible=False)
        else:
            return logs, phase, gradio.update(visible=False), gradio.update(value=output_path, visible=True), gradio.update(visible=False)
    
    return logs, phase, gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True)


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
