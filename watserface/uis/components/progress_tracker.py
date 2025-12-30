import gradio
from typing import Optional, Dict, Any
import time

from watserface.uis.core import register_ui_component
from watserface.preset_manager import PresetManager


class ProgressTracker:
    """Enhanced progress tracking with visual indicators"""
    
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        self.start_time = None
        self.current_preset = "balanced"
        self.processing_status = "Ready"
    
    def start_processing(self, preset_name: str, estimated_steps: int = 100):
        """Start progress tracking"""
        self.current_step = 0
        self.total_steps = estimated_steps
        self.start_time = time.time()
        self.current_preset = preset_name
        self.processing_status = "Processing"
    
    def update_progress(self, step: int, status_message: str = ""):
        """Update progress step"""
        self.current_step = min(step, self.total_steps)
        if status_message:
            self.processing_status = status_message
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        if self.total_steps == 0:
            return {
                "percentage": 0,
                "elapsed_time": 0,
                "estimated_total": 0,
                "remaining_time": 0,
                "status": self.processing_status
            }
        
        percentage = (self.current_step / self.total_steps) * 100
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        if percentage > 0:
            estimated_total = elapsed_time / (percentage / 100)
            remaining_time = max(0, estimated_total - elapsed_time)
        else:
            estimated_total = 0
            remaining_time = 0
        
        return {
            "percentage": percentage,
            "elapsed_time": elapsed_time,
            "estimated_total": estimated_total,
            "remaining_time": remaining_time,
            "status": self.processing_status,
            "preset": self.current_preset
        }


# Global progress tracker instance
_progress_tracker = ProgressTracker()

PROGRESS_BAR: Optional[gradio.Progress] = None
PROGRESS_STATUS: Optional[gradio.Textbox] = None
PROGRESS_INFO: Optional[gradio.Markdown] = None


def render() -> None:
    """Render progress tracking components"""
    global PROGRESS_BAR, PROGRESS_STATUS, PROGRESS_INFO
    
    with gradio.Column() as progress_container:
        gradio.Markdown("### 📊 Processing Progress")
        
        # Main progress display
        PROGRESS_STATUS = gradio.Textbox(
            label="Status",
            value="Ready to process",
            interactive=False,
            lines=1,
            elem_id="progress_status"
        )
        
        # Detailed progress information
        PROGRESS_INFO = gradio.Markdown(
            """
            **Ready to start processing**
            
            When processing begins, you'll see:
            - ⏱️ Estimated time remaining
            - 📈 Current processing step
            - 🎯 Quality preset being used
            - 🔄 Real-time status updates
            """,
            elem_id="progress_info"
        )
        
        # Step-by-step progress indicator
        with gradio.Row():
            step_upload = gradio.HTML(
                """
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 24px;">📁</div>
                    <div style="font-size: 12px; margin-top: 5px;">Upload</div>
                    <div style="width: 100%; height: 4px; background: #e0e0e0; margin-top: 5px; border-radius: 2px;">
                        <div id="step-upload-progress" style="width: 100%; height: 100%; background: #4CAF50; border-radius: 2px;"></div>
                    </div>
                </div>
                """,
                elem_id="step_upload_indicator"
            )
            
            step_preview = gradio.HTML(
                """
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 24px;">🔍</div>
                    <div style="font-size: 12px; margin-top: 5px;">Preview</div>
                    <div style="width: 100%; height: 4px; background: #e0e0e0; margin-top: 5px; border-radius: 2px;">
                        <div id="step-preview-progress" style="width: 0%; height: 100%; background: #2196F3; border-radius: 2px;"></div>
                    </div>
                </div>
                """,
                elem_id="step_preview_indicator"
            )
            
            step_process = gradio.HTML(
                """
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 24px;">⚡</div>
                    <div style="font-size: 12px; margin-top: 5px;">Process</div>
                    <div style="width: 100%; height: 4px; background: #e0e0e0; margin-top: 5px; border-radius: 2px;">
                        <div id="step-process-progress" style="width: 0%; height: 100%; background: #FF9800; border-radius: 2px;"></div>
                    </div>
                </div>
                """,
                elem_id="step_process_indicator"
            )
            
            step_complete = gradio.HTML(
                """
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 24px;">✅</div>
                    <div style="font-size: 12px; margin-top: 5px;">Complete</div>
                    <div style="width: 100%; height: 4px; background: #e0e0e0; margin-top: 5px; border-radius: 2px;">
                        <div id="step-complete-progress" style="width: 0%; height: 100%; background: #4CAF50; border-radius: 2px;"></div>
                    </div>
                </div>
                """,
                elem_id="step_complete_indicator"
            )
    
    # Register components
    register_ui_component('progress_container', progress_container)
    register_ui_component('progress_status', PROGRESS_STATUS)
    register_ui_component('progress_info', PROGRESS_INFO)


def listen() -> None:
    """Set up progress tracking event listeners"""
    pass  # Progress updates will be called programmatically


def update_progress_display(
    step: int,
    total_steps: int,
    status_message: str,
    preset_name: str = "balanced"
) -> tuple:
    """Update progress display with current information"""
    
    percentage = (step / total_steps * 100) if total_steps > 0 else 0
    preset_config = PresetManager.get_preset(preset_name)
    
    # Create detailed progress info
    progress_info = f"""
    **🔄 Processing in progress... ({percentage:.1f}%)**
    
    - 🎯 **Quality Preset**: {preset_config.display_name if preset_config else preset_name}
    - 📈 **Step**: {step} of {total_steps}
    - ⏱️ **Estimated Time**: {preset_config.estimated_time if preset_config else "Calculating..."}
    - 📊 **Status**: {status_message}
    
    💡 **Tip**: Processing time depends on your selected quality preset and media size.
    """
    
    return status_message, progress_info


def get_step_indicator_html(step_name: str, is_active: bool, is_complete: bool, progress: float = 0) -> str:
    """Generate HTML for step indicators"""
    
    icons = {
        "upload": "📁",
        "preview": "🔍", 
        "process": "⚡",
        "complete": "✅"
    }
    
    colors = {
        "upload": "#4CAF50",
        "preview": "#2196F3",
        "process": "#FF9800", 
        "complete": "#4CAF50"
    }
    
    icon = icons.get(step_name, "⚪")
    color = colors.get(step_name, "#e0e0e0")
    
    if is_complete:
        progress = 100
        opacity = "1.0"
    elif is_active:
        opacity = "1.0"
    else:
        opacity = "0.5"
        progress = 0
    
    return f"""
    <div style="text-align: center; padding: 10px; opacity: {opacity};">
        <div style="font-size: 24px;">{icon}</div>
        <div style="font-size: 12px; margin-top: 5px; text-transform: capitalize;">{step_name}</div>
        <div style="width: 100%; height: 4px; background: #e0e0e0; margin-top: 5px; border-radius: 2px;">
            <div style="width: {progress}%; height: 100%; background: {color}; border-radius: 2px; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """


def start_processing(preset_name: str = "balanced") -> None:
    """Start processing with progress tracking"""
    global _progress_tracker
    _progress_tracker.start_processing(preset_name)


def update_processing_step(step: int, status: str = "") -> None:
    """Update current processing step"""
    global _progress_tracker
    _progress_tracker.update_progress(step, status)


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance"""
    return _progress_tracker