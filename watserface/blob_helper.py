from typing import Any, List, Dict, Generator, Optional, Tuple
import os
import gradio
import time

from watserface import state_manager, logger, face_analyser, wording
from watserface.filesystem import is_video, is_image, resolve_file_paths, has_audio
from watserface.training import core as training_core
from watserface.face_set import get_face_set_manager

class BlobOrchestrator:
    """
    The Brain of the Dr. Blob interface.
    Automates the decision making for high-quality deepfakes.
    """

    def __init__(self):
        self.log_history = []

    def log(self, message: str, telemetry: Optional[Dict] = None):
        """Add log message and yield it"""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.log_history.append(entry)
        return entry

    def analyze_inputs(self, source_files: Any, target_files: Any, mode: str = "Quality") -> Generator[Tuple[str, Dict], None, None]:
        """
        Step 1: Analyze inputs to determine the best strategy.
        Yields: (Status Message, Strategy Dict)
        """
        # Normalize Paths
        s_files = source_files if isinstance(source_files, list) else [source_files]
        sources = [f.name if hasattr(f, 'name') else f for f in s_files if f is not None]

        t_files = target_files if isinstance(target_files, list) else [target_files]
        targets = [f.name if hasattr(f, 'name') else f for f in t_files if f is not None]

        # Base Strategy
        strategy = {
            "needs_identity_training": False,
            "needs_occlusion_training": False,
            "needs_face_enhancer": False,
            "processors": ["face_swapper"],
            "model_name": None,
            "xseg_model_name": None,
            "source_type": "single_image",
            "mode": mode
        }

        # Apply Mode Defaults
        if mode == "Fast":
            yield self.log("🏎️ Mode: Fast. Speed prioritized. Enhancers disabled."), strategy
        elif mode == "Balanced":
             strategy["needs_face_enhancer"] = True
             strategy["processors"].append("face_enhancer")
             yield self.log("⚖️ Mode: Balanced. Enhancers active. Training skipped."), strategy
        else: # Quality
             strategy["needs_face_enhancer"] = True
             strategy["processors"].append("face_enhancer")
             yield self.log("💎 Mode: Quality. Full Dr. Blob protocol active. Training enabled."), strategy

        telemetry = {}

        yield self.log("🔍 Dr. Blob is analyzing your inputs..."), strategy

        # --- Source Analysis ---
        if not sources:
             yield self.log("⚠️ No source files provided."), strategy
             return

        source_count = len(sources)
        strategy["source_count"] = source_count

        # Check for Face Set (File extension check + ID check fallback)
        is_face_set = False
        first_source = sources[0]

        # If it's a string path that doesn't exist, assume it's an ID (Legacy/Internal)
        if isinstance(first_source, str) and not os.path.exists(first_source):
             is_face_set = True

        if is_face_set:
            strategy["source_type"] = "face_set"
            if mode == "Quality":
                strategy["needs_identity_training"] = True
                strategy["model_name"] = f"blob_faceset_{first_source}"
                yield self.log(f"📂 Face Set ID detected ({first_source}). Will use for Identity Model."), strategy
            else:
                yield self.log(f"📂 Face Set ID detected. Using raw set without training (Mode: {mode})."), strategy
        elif source_count > 1:
            strategy["source_type"] = "multiple_images"
            if mode == "Quality":
                strategy["needs_identity_training"] = True
                strategy["model_name"] = f"blob_auto_{int(time.time())}"
                yield self.log(f"📸 Detected {source_count} source images. Identity Training enabled for max resemblance."), strategy
            else:
                yield self.log(f"📸 Detected {source_count} source images. Training skipped (Mode: {mode})."), strategy
        else:
            strategy["source_type"] = "single_image"
            yield self.log("📸 Single source image detected. Standard swap will be used."), strategy

        # --- Target Analysis ---
        if not targets:
             yield self.log("⚠️ No target files provided."), strategy
             return

        target_path = targets[0]

        if is_video(target_path):
            yield self.log("🎥 Target is a video."), strategy

            # Check for audio to enable lip sync
            if has_audio(target_path):
                if "lip_syncer" not in strategy["processors"]:
                    strategy["processors"].append("lip_syncer")
                yield self.log("🔊 Audio detected. Lip Syncer enabled for realism."), strategy

            if mode == "Quality":
                strategy["needs_occlusion_training"] = True
                strategy["xseg_model_name"] = f"blob_xseg_{int(time.time())}"
                yield self.log("🛡️ Video target detected. Occlusion (XSeg) training scheduled."), strategy
            else:
                 yield self.log(f"🎥 Video target detected. Occlusion training skipped (Mode: {mode})."), strategy
        else:
            yield self.log("🖼️ Target is an image."), strategy
            strategy["needs_occlusion_training"] = False

        yield self.log("✅ Analysis Complete. Strategy formulated."), strategy

    def execute_pipeline(self, strategy: Dict, source_files: Any, target_files: Any) -> Generator[Tuple[str, Any], None, None]:
        """
        Step 2: Execute the formulated strategy.
        Yields: (Status Message, Preview Image/Result)
        """

        # Normalize Paths
        s_files = source_files if isinstance(source_files, list) else [source_files]
        s_paths = [f.name if hasattr(f, 'name') else f for f in s_files if f is not None]

        t_files = target_files if isinstance(target_files, list) else [target_files]
        t_path = t_files[0].name if hasattr(t_files[0], 'name') else t_files[0]

        trained_identity_model_path = None
        trained_xseg_model_path = None

        # 1. Identity Training (if needed)
        if strategy["needs_identity_training"]:
            model_name = strategy["model_name"]
            yield self.log(f"🚀 Starting Identity Training for '{model_name}'..."), None

            # Call training core
            kwargs = {
                "model_name": model_name,
                "epochs": 100, # Good balance for "Blob" quality
            }

            if strategy["source_type"] == "face_set":
                 kwargs["face_set_id"] = s_paths[0]
            else:
                 kwargs["source_files"] = s_paths
                 kwargs["save_as_face_set"] = True # Always save for re-use

            telemetry = {}
            for status in training_core.start_identity_training(**kwargs):
                # Status is a tuple [message, telemetry]
                if isinstance(status, list) and len(status) >= 2:
                    msg = status[0]
                    telemetry = status[1]
                else:
                    msg = str(status)

                yield self.log(f"🧠 [Identity] {msg}"), None

            # Capture model path from telemetry
            if telemetry.get('model_path'):
                 trained_identity_model_path = os.path.basename(telemetry['model_path'])
                 yield self.log(f"🧠 Identity Model captured: {trained_identity_model_path}"), None

        # 2. Occlusion Training (if needed)
        if strategy["needs_occlusion_training"]:
            xseg_name = strategy["xseg_model_name"]
            yield self.log(f"🛡️ Starting Occlusion Training for '{xseg_name}'..."), None

            telemetry = {}
            for status in training_core.start_occlusion_training(
                model_name=xseg_name,
                epochs=50, # Quick fine-tune
                target_file=t_path
            ):
                 if isinstance(status, list) and len(status) >= 2:
                     msg = status[0]
                     telemetry = status[1]
                 else:
                     msg = str(status)
                 yield self.log(f"🎭 [Occlusion] {msg}"), None

            # Capture model path from telemetry
            if telemetry.get('model_path'):
                # Model path is absolute, but face_masker usually expects name or path in assets
                # It copies to .assets/models/trained/
                trained_xseg_model_path = os.path.basename(telemetry['model_path'])
                yield self.log(f"🎭 XSeg Model captured: {trained_xseg_model_path}"), None

        # 3. Configure Processors
        yield self.log(f"⚙️ Configuring Processors ({', '.join(strategy['processors'])})..."), None

        # Set Processors
        state_manager.set_item('processors', strategy['processors'])

        # Set Models based on Strategy
        if "face_enhancer" in strategy['processors']:
            state_manager.set_item('face_enhancer_model', 'codeformer')
            state_manager.set_item('face_enhancer_blend', 100) # Max blend

        # Set Swapper Model (Prioritize Trained Model)
        if trained_identity_model_path:
             state_manager.set_item('face_swapper_model', trained_identity_model_path)
             yield self.log(f"✅ Using Trained Identity Model: {trained_identity_model_path}"), None
        else:
             state_manager.set_item('face_swapper_model', 'inswapper_128')

        # Set Occlusion Model (Prioritize Trained Model)
        if trained_xseg_model_path:
             state_manager.set_item('face_occluder_model', trained_xseg_model_path)
             # Ensure mask type is set to occlusion
             current_mask_types = state_manager.get_item('face_mask_types') or []
             if 'occlusion' not in current_mask_types:
                 current_mask_types.append('occlusion')
                 state_manager.set_item('face_mask_types', current_mask_types)
             yield self.log(f"✅ Using Trained XSeg Model: {trained_xseg_model_path}"), None

        # 4. Prepare Paths
        state_manager.set_item('source_paths', s_paths)
        state_manager.set_item('target_path', t_path)

        # Generate Output Path
        current_output_path = state_manager.get_item('output_path')
        if current_output_path and os.path.isdir(current_output_path):
            output_dir = current_output_path
        elif current_output_path and os.path.isfile(current_output_path):
            output_dir = os.path.dirname(current_output_path)
        else:
            output_dir = "output"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(t_path)
        name, ext = os.path.splitext(base_name)
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"{name}_blob_{timestamp}{ext}")
        state_manager.set_item('output_path', output_path)

        # 5. Execution
        yield self.log(f"🎬 Processing to: {output_path}"), None

        from watserface.core import process_video, process_image
        import time as t_module

        start_time = t_module.time()
        error_code = 0

        try:
            if is_video(t_path):
                error_code = process_video(start_time)
            elif is_image(t_path):
                error_code = process_image(start_time)
            else:
                yield self.log("❌ Error: Target file type not supported."), None
                return
        except Exception as e:
            yield self.log(f"❌ Critical Error: {str(e)}"), None
            import traceback
            traceback.print_exc()
            return

        if error_code == 0:
            yield self.log("✨ Processing Complete!"), output_path
            yield self.log("🎉 Job Done!"), None
        else:
            yield self.log(f"⚠️ Processing finished with error code: {error_code}"), None
