import gradio
import traceback
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from facefusion.uis.core import register_ui_component


class ErrorType(Enum):
    """Types of errors that can occur in the application"""
    FILE_UPLOAD = "file_upload"
    PROCESSING = "processing"
    MEMORY = "memory"
    MODEL_LOADING = "model_loading"
    NETWORK = "network"
    VALIDATION = "validation"
    SYSTEM = "system"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorHandler:
    """Comprehensive error handling and user feedback system"""
    
    ERROR_MESSAGES = {
        # File upload errors
        "unsupported_format": {
            "title": "❌ Unsupported File Format",
            "message": "The file format you uploaded is not supported.",
            "solution": "Please upload JPG, PNG, MP4, or MOV files.",
            "severity": ErrorSeverity.ERROR
        },
        "file_too_large": {
            "title": "❌ File Too Large",
            "message": "The uploaded file exceeds the maximum size limit.",
            "solution": "Please compress your file to under 100MB or use a shorter video.",
            "severity": ErrorSeverity.ERROR
        },
        "no_face_detected": {
            "title": "⚠️ No Face Detected",
            "message": "No faces were found in the uploaded image.",
            "solution": "Please upload a clear photo with at least one visible face.",
            "severity": ErrorSeverity.WARNING
        },
        "multiple_faces": {
            "title": "⚠️ Multiple Faces Detected",
            "message": "Multiple faces were found. The first detected face will be used.",
            "solution": "For best results, use photos with a single clear face.",
            "severity": ErrorSeverity.WARNING
        },
        "blurry_image": {
            "title": "⚠️ Low Quality Image",
            "message": "The uploaded image appears to be blurry or low quality.",
            "solution": "For better results, use a high-resolution, clear image.",
            "severity": ErrorSeverity.WARNING
        },
        
        # Processing errors
        "out_of_memory": {
            "title": "❌ Out of Memory",
            "message": "Not enough memory to process this media.",
            "solution": "Try reducing video length, resolution, or use Fast preset.",
            "severity": ErrorSeverity.ERROR
        },
        "gpu_unavailable": {
            "title": "⚠️ GPU Unavailable",
            "message": "GPU acceleration is not available. Processing will use CPU.",
            "solution": "Processing will be slower but still functional.",
            "severity": ErrorSeverity.WARNING
        },
        "model_load_failed": {
            "title": "❌ Model Loading Failed",
            "message": "Failed to load required AI models.",
            "solution": "Check internet connection or try restarting the application.",
            "severity": ErrorSeverity.ERROR
        },
        "processing_timeout": {
            "title": "⏱️ Processing Timeout",
            "message": "Processing took too long and was stopped.",
            "solution": "Try using Fast preset or shorter video clips.",
            "severity": ErrorSeverity.ERROR
        },
        "invalid_settings": {
            "title": "⚙️ Invalid Settings",
            "message": "Some settings are not compatible with your system.",
            "solution": "Reset to default settings or use preset configurations.",
            "severity": ErrorSeverity.WARNING
        },
        
        # System errors
        "disk_space_low": {
            "title": "💾 Low Disk Space",
            "message": "Not enough disk space for processing.",
            "solution": "Free up disk space or use shorter videos.",
            "severity": ErrorSeverity.ERROR
        },
        "network_error": {
            "title": "🌐 Network Error",
            "message": "Failed to download required models or resources.",
            "solution": "Check your internet connection and try again.",
            "severity": ErrorSeverity.ERROR
        }
    }
    
    def __init__(self):
        self.error_log: List[Dict] = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup error logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('WatserFace_ErrorHandler')
    
    def handle_error(
        self, 
        error_key: str, 
        details: str = "", 
        error_type: ErrorType = ErrorType.SYSTEM
    ) -> Tuple[str, str, str]:
        """Handle an error and return user-friendly message"""
        
        error_info = self.ERROR_MESSAGES.get(error_key, {
            "title": "❌ Unknown Error",
            "message": "An unexpected error occurred.",
            "solution": "Please try again or restart the application.",
            "severity": ErrorSeverity.ERROR
        })
        
        # Log the error
        error_entry = {
            "key": error_key,
            "type": error_type.value,
            "title": error_info["title"],
            "message": error_info["message"],
            "solution": error_info["solution"],
            "details": details,
            "severity": error_info["severity"].value
        }
        
        self.error_log.append(error_entry)
        self.logger.error(f"{error_key}: {error_info['message']} | Details: {details}")
        
        # Return formatted message for UI
        return self.format_error_message(error_info)
    
    def format_error_message(self, error_info: Dict) -> Tuple[str, str, str]:
        """Format error message for display"""
        severity = error_info["severity"]
        
        # Choose color/style based on severity
        if severity == ErrorSeverity.ERROR:
            status_class = "status-error"
        elif severity == ErrorSeverity.WARNING:
            status_class = "status-warning"
        else:
            status_class = "status-info"
        
        title = error_info["title"]
        message = error_info["message"]
        solution = f"💡 **Solution**: {error_info['solution']}"
        
        return title, message, solution
    
    def get_error_summary(self) -> str:
        """Get summary of recent errors"""
        if not self.error_log:
            return "✅ No errors reported"
        
        recent_errors = self.error_log[-5:]  # Last 5 errors
        summary_lines = ["**Recent Issues:**"]
        
        for error in recent_errors:
            icon = "❌" if error["severity"] == "error" else "⚠️"
            summary_lines.append(f"- {icon} {error['title']}")
        
        return "\n".join(summary_lines)
    
    def validate_upload(self, file_path: str) -> Tuple[bool, str]:
        """Validate uploaded file and return status"""
        try:
            import os
            from facefusion.filesystem import is_image, is_video
            
            if not os.path.exists(file_path):
                return False, self.handle_error("file_not_found")[1]
            
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return False, self.handle_error("file_too_large")[1]
            
            if not (is_image(file_path) or is_video(file_path)):
                return False, self.handle_error("unsupported_format")[1]
            
            return True, "✅ File validation passed"
            
        except Exception as e:
            return False, self.handle_error("validation_error", str(e))[1]
    
    def validate_system_requirements(self) -> List[str]:
        """Validate system requirements and return warnings"""
        warnings = []
        
        try:
            import psutil
            import torch
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB
                warnings.append(self.handle_error("low_memory")[1])
            
            # Check GPU
            if not torch.cuda.is_available():
                warnings.append(self.handle_error("gpu_unavailable")[1])
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free < 2 * 1024 * 1024 * 1024:  # 2GB
                warnings.append(self.handle_error("disk_space_low")[1])
                
        except Exception as e:
            warnings.append(f"⚠️ System check failed: {str(e)}")
        
        return warnings


# Global error handler instance
_error_handler = ErrorHandler()

ERROR_DISPLAY: Optional[gradio.Markdown] = None
ERROR_STATUS: Optional[gradio.Textbox] = None


def render_error_display() -> None:
    """Render error display components"""
    global ERROR_DISPLAY, ERROR_STATUS
    
    with gradio.Column(visible=False) as error_container:
        ERROR_STATUS = gradio.Textbox(
            label="Status",
            value="",
            interactive=False,
            lines=1,
            elem_id="error_status"
        )
        
        ERROR_DISPLAY = gradio.Markdown(
            "",
            elem_id="error_display"
        )
        
        error_dismiss = gradio.Button(
            value="✖️ Dismiss",
            variant="secondary",
            size="sm"
        )
    
    register_ui_component('error_container', error_container)
    register_ui_component('error_status', ERROR_STATUS)
    register_ui_component('error_display', ERROR_DISPLAY)
    register_ui_component('error_dismiss', error_dismiss)
    
    return error_container, ERROR_STATUS, ERROR_DISPLAY, error_dismiss


def show_error(error_key: str, details: str = "") -> Tuple[Any, str, str]:
    """Show error message in UI"""
    title, message, solution = _error_handler.handle_error(error_key, details)
    
    error_content = f"""
    ### {title}
    
    {message}
    
    {solution}
    """
    
    return (
        gradio.update(visible=True),  # Show error container
        f"{title}",  # Status text
        error_content  # Full error message
    )


def hide_error() -> Tuple[Any, str, str]:
    """Hide error message"""
    return (
        gradio.update(visible=False),  # Hide error container
        "",  # Clear status
        ""   # Clear message
    )


def validate_file_upload(file_path: str) -> Tuple[bool, str]:
    """Validate file upload and return status"""
    return _error_handler.validate_upload(file_path)


def check_system_requirements() -> List[str]:
    """Check system requirements and return warnings"""
    return _error_handler.validate_system_requirements()


def setup_error_handling() -> None:
    """Setup global error handling"""
    import sys
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Global exception handler"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _error_handler.logger.error(f"Uncaught exception: {error_msg}")
        
        # Show user-friendly error
        show_error("system", error_msg)
    
    sys.excepthook = handle_exception


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return _error_handler


def listen_error_events() -> None:
    """Setup error handling event listeners"""
    error_dismiss = register_ui_component('error_dismiss', None)
    error_container = register_ui_component('error_container', None)
    error_status = register_ui_component('error_status', None)
    error_display = register_ui_component('error_display', None)
    
    if error_dismiss and all([error_container, error_status, error_display]):
        error_dismiss.click(
            fn=hide_error,
            outputs=[error_container, error_status, error_display]
        )