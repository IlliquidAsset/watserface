from typing import Optional, Any, Tuple
import os
import gradio
import pandas as pd

from watserface import state_manager
from watserface.filesystem import resolve_relative_path
from watserface.uis.components import about, footer, training_source, training_target, terminal
from watserface.training import core as training_core


def generate_progress_html(label: str, percentage: float, detail_text: str) -> str:
    """Generate HTML for a progress bar using the template"""
    template_path = resolve_relative_path("uis/assets/training_progress.html")
    if not os.path.exists(template_path):
        return f"<div>{label}: {percentage:.1f}% - {detail_text}</div>"
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    return template.format(
        label=label,
        percentage=min(max(percentage, 0), 100),
        detail_text=detail_text
    )


def generate_metrics_html(telemetry: dict) -> str:
    """Generate HTML for the metrics panel"""
    epoch_time = telemetry.get('epoch_time', 'N/A')
    eta = telemetry.get('eta', 'N/A')
    device = telemetry.get('device', 'N/A')
    loss = telemetry.get('loss', 'N/A')
    
    return f"""
    <div class="glass-metrics-panel">
        <div class="metric-row">
            <span class="metric-label">Device</span>
            <span class="metric-value">{device}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Epoch Time</span>
            <span class="metric-value">{epoch_time}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">ETA</span>
            <span class="metric-value">{eta}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Current Loss</span>
            <span class="metric-value">{loss}</span>
        </div>
    </div>
    """


def wrapped_start_identity_training(
    model_name: str,
    epochs: int,
    source_type: str,
    face_set_id: Optional[str] = None,
    source_files: Any = None,
    save_as_face_set: bool = False,
    face_set_name: Optional[str] = None,
    face_set_description: Optional[str] = None
):
    """Wrapper to format identity training output with throttling and telemetry"""
    import time
    last_update = 0
    last_status = None
    loss_history = []
    historical_loaded = False

    # Determine parameters based on source type
    if source_type == "Face Set":
        training_kwargs = {
            'model_name': model_name,
            'epochs': epochs,
            'face_set_id': face_set_id
        }
    else:  # "Upload Files"
        training_kwargs = {
            'model_name': model_name,
            'epochs': epochs,
            'source_files': source_files,
            'save_as_face_set': save_as_face_set,
            'new_face_set_name': face_set_name if save_as_face_set else None
        }

    # Show the telemetry group immediately
    yield "Initializing...", None, generate_progress_html("Overall Progress", 0, "Starting..."), generate_progress_html("Epoch Progress", 0, "Starting..."), generate_progress_html("Batch Progress", 0, "Starting..."), generate_metrics_html({}), gradio.update(visible=True)

    for status_data in training_core.start_identity_training(**training_kwargs):
        current_time = time.time()
        
        # Parse status data
        message = status_data[0] if isinstance(status_data, list) else str(status_data)
        telemetry = status_data[1] if isinstance(status_data, list) and len(status_data) > 1 else {}
        
        # Prepare updates
        plot_update = None
        overall_html = generate_progress_html("Overall Progress", 0, "Initializing...")
        epoch_html = generate_progress_html("Epoch Progress", 0, "Waiting...")
        batch_html = generate_progress_html("Batch Progress", 0, "Waiting...")
        metrics_html = generate_metrics_html({})
        
        if telemetry:
            # Load historical data if available
            if not historical_loaded and 'historical_loss' in telemetry:
                historical_data = telemetry['historical_loss']
                for entry in historical_data:
                    loss_history.append({
                        'epoch': float(entry['epoch']),
                        'loss': float(entry['loss'])
                    })
                historical_loaded = True
                plot_update = pd.DataFrame(loss_history)

            # Update history and plot
            loss_val = telemetry.get('loss') or telemetry.get('current_loss')
            if loss_val is not None:
                try:
                    # If we have epoch progress, use it for X axis (e.g. 1.5 for mid epoch 1)
                    epoch_val = int(telemetry.get('epoch', 0))

                    if 'batch' in telemetry and 'total_batches' in telemetry:
                        batch_fraction = int(telemetry['batch']) / int(telemetry['total_batches'])
                        # Subtract 1 because epoch starts at 1, so epoch 1 + 0.5 is 1.5
                        epoch_val = (epoch_val - 1) + batch_fraction

                    loss_history.append({
                        'epoch': float(epoch_val),
                        'loss': float(loss_val)
                    })
                    plot_update = pd.DataFrame(loss_history)
                except (ValueError, TypeError):
                    pass
            
            # Extract progress values
            def parse_pct(val):
                if isinstance(val, str) and '%' in val:
                    return float(val.strip('%'))
                if isinstance(val, (int, float)) and val <= 1.0:
                    return val * 100
                return float(val)

            overall_pct = parse_pct(telemetry.get('overall_progress', 0))
            epoch_pct = parse_pct(telemetry.get('epoch_progress', 0))
            batch_pct = parse_pct(telemetry.get('batch_progress', 0))
            
            if 'batch' in telemetry and 'total_batches' in telemetry:
                batch_pct = (int(telemetry['batch']) / int(telemetry['total_batches'])) * 100

            # Generate HTML
            overall_html = generate_progress_html("Overall Progress", overall_pct, f"Epoch {telemetry.get('epoch', 0)}/{telemetry.get('total_epochs', '?')}")
            epoch_html = generate_progress_html("Epoch Progress", epoch_pct, f"Loss: {telemetry.get('loss', 'N/A')}")
            batch_html = generate_progress_html("Batch Progress", batch_pct, f"Batch {telemetry.get('batch', 0)}/{telemetry.get('total_batches', '?')}")
            metrics_html = generate_metrics_html(telemetry)

        # Only update UI every 0.5 seconds to avoid flickering
        if current_time - last_update >= 0.5:
            last_update = current_time
            yield message, plot_update, overall_html, epoch_html, batch_html, metrics_html, gradio.update(visible=True)

    # Always yield the final status
    yield "Training Complete", pd.DataFrame(loss_history) if loss_history else None, generate_progress_html("Overall", 100, "Done"), generate_progress_html("Epoch", 100, "Done"), generate_progress_html("Batch", 100, "Done"), metrics_html, gradio.update(visible=True)


def wrapped_start_occlusion_training(model_name: str, epochs: int, target_file: Any):
    import time
    last_update = 0
    for status in training_core.start_occlusion_training(model_name, epochs, target_file):
        current_time = time.time()
        formatted_status = str(status)
        if current_time - last_update >= 0.5:
            last_update = current_time
            yield formatted_status
    yield "Occlusion Training Complete"


def wrapped_stop_training():
    return training_core.stop_training()


# Components
IDENTITY_MODEL_NAME : Optional[gradio.Textbox] = None
IDENTITY_EPOCHS : Optional[gradio.Slider] = None
START_IDENTITY_BUTTON : Optional[gradio.Button] = None
STOP_IDENTITY_BUTTON : Optional[gradio.Button] = None
IDENTITY_STATUS_TEXT : Optional[gradio.Textbox] = None
IDENTITY_TELEMETRY_GROUP : Optional[gradio.Group] = None
IDENTITY_OVERALL_PROGRESS : Optional[gradio.HTML] = None
IDENTITY_EPOCH_PROGRESS : Optional[gradio.HTML] = None
IDENTITY_BATCH_PROGRESS : Optional[gradio.HTML] = None
IDENTITY_METRICS : Optional[gradio.HTML] = None
LOSS_PLOT : Optional[gradio.LinePlot] = None

OCCLUSION_MODEL_NAME : Optional[gradio.Textbox] = None
OCCLUSION_EPOCHS : Optional[gradio.Slider] = None
START_OCCLUSION_BUTTON : Optional[gradio.Button] = None
STOP_OCCLUSION_BUTTON : Optional[gradio.Button] = None
OCCLUSION_STATUS : Optional[gradio.Textbox] = None


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    global IDENTITY_MODEL_NAME, IDENTITY_EPOCHS, START_IDENTITY_BUTTON, STOP_IDENTITY_BUTTON, IDENTITY_STATUS_TEXT
    global IDENTITY_OVERALL_PROGRESS, IDENTITY_EPOCH_PROGRESS, IDENTITY_BATCH_PROGRESS, IDENTITY_METRICS
    global IDENTITY_TELEMETRY_GROUP, LOSS_PLOT
    global OCCLUSION_MODEL_NAME, OCCLUSION_EPOCHS, START_OCCLUSION_BUTTON, STOP_OCCLUSION_BUTTON, OCCLUSION_STATUS

    with gradio.Blocks() as layout:
        about.render()
        
        with gradio.Tabs():
            # === Identity Training Tab ===
            with gradio.Tab("Step 1 & 2: Identities"):
                gradio.Markdown("### 👤 Face Set Extraction & Identity Training")
                
                with gradio.Row():
                    # Config Column
                    with gradio.Column():
                        training_source.render()
                        IDENTITY_MODEL_NAME = gradio.Textbox(label="Identity Model Name", placeholder="e.g. my_actor_v1")
                        IDENTITY_EPOCHS = gradio.Slider(label="Epochs", minimum=1, maximum=1000, value=100, step=1)
                        
                        with gradio.Row():
                            START_IDENTITY_BUTTON = gradio.Button("▶ Start Training", variant="primary", elem_classes=["primary-btn"])
                            STOP_IDENTITY_BUTTON = gradio.Button("⏹ Stop", variant="stop", elem_classes=["stop-btn"])
                        
                        IDENTITY_STATUS_TEXT = gradio.Textbox(label="Current Status", value="Idle", interactive=False, lines=2)

                # NEW: Telemetry and Progress (Divided into columns, hidden initially)
                with gradio.Group(visible=False) as IDENTITY_TELEMETRY_GROUP:
                    gradio.Markdown("### 📊 Live Telemetry & Progress")
                    with gradio.Row():
                        # Column 1: Progress Bars
                        with gradio.Column(scale=1):
                            IDENTITY_OVERALL_PROGRESS = gradio.HTML(value=generate_progress_html("Overall Progress", 0, "Idle"))
                            IDENTITY_EPOCH_PROGRESS = gradio.HTML(value=generate_progress_html("Epoch Progress", 0, "Idle"))
                            IDENTITY_BATCH_PROGRESS = gradio.HTML(value=generate_progress_html("Batch Progress", 0, "Idle"))

                        # Column 2: Charts & Metrics
                        with gradio.Column(scale=1):
                            LOSS_PLOT = gradio.LinePlot(
                                label="Training Loss",
                                x="epoch", y="loss",
                                title="Loss over Time",
                                width=400, height=200,
                                tooltip=["epoch", "loss"],
                                overlay_point=True,
                                elem_classes=["loss-chart-container"]
                            )
                            IDENTITY_METRICS = gradio.HTML(value=generate_metrics_html({}))

            # === Occlusion Training Tab ===
            with gradio.Tab("Occlusion Training"):
                with gradio.Row():
                    with gradio.Column():
                        gradio.Markdown("### 🎭 Train Occlusion Model")
                        training_target.render()
                        OCCLUSION_MODEL_NAME = gradio.Textbox(label="Occlusion Model Name", placeholder="e.g. corndog_scene_mask")
                        OCCLUSION_EPOCHS = gradio.Slider(label="Epochs", minimum=1, maximum=200, value=50, step=1)
                        with gradio.Row():
                            START_OCCLUSION_BUTTON = gradio.Button("Start Occlusion Training", variant="primary")
                            STOP_OCCLUSION_BUTTON = gradio.Button("Stop", variant="stop")
                        OCCLUSION_STATUS = gradio.Textbox(label="Occlusion Status", value="Idle", interactive=False, lines=5)

        with gradio.Row():
            terminal.render()

        footer.render()

    return layout


def listen() -> None:
    training_source.listen()
    training_target.listen()
    terminal.listen()

    START_IDENTITY_BUTTON.click(
        wrapped_start_identity_training,
        inputs=[
            IDENTITY_MODEL_NAME,
            IDENTITY_EPOCHS,
            training_source.SOURCE_TYPE_RADIO,
            training_source.FACE_SET_DROPDOWN,
            training_source.TRAINING_SOURCE_FILE,
            training_source.SAVE_AS_FACE_SET_CHECKBOX,
            training_source.FACE_SET_NAME,
            training_source.FACE_SET_DESCRIPTION
        ],
        outputs=[
            IDENTITY_STATUS_TEXT, 
            LOSS_PLOT, 
            IDENTITY_OVERALL_PROGRESS, 
            IDENTITY_EPOCH_PROGRESS, 
            IDENTITY_BATCH_PROGRESS, 
            IDENTITY_METRICS,
            IDENTITY_TELEMETRY_GROUP
        ]
    )
    STOP_IDENTITY_BUTTON.click(wrapped_stop_training, outputs=[IDENTITY_STATUS_TEXT])

    START_OCCLUSION_BUTTON.click(
        wrapped_start_occlusion_training,
        inputs=[OCCLUSION_MODEL_NAME, OCCLUSION_EPOCHS, training_target.TRAINING_TARGET_FILE],
        outputs=[OCCLUSION_STATUS]
    )
    STOP_OCCLUSION_BUTTON.click(wrapped_stop_training, outputs=[OCCLUSION_STATUS])


def run(ui : gradio.Blocks) -> None:
    ui.launch(inbrowser = state_manager.get_item('open_browser'), server_name = state_manager.get_item('server_name'), server_port = state_manager.get_item('server_port'), show_error = True)
