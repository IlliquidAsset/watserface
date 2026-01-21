from typing import Any, Dict, Generator, List, Optional, Tuple
import os
import shutil
import time

from watserface import state_manager, logger
from watserface.filesystem import is_video, is_image, has_audio
from watserface.studio.state import StudioState, StudioPhase, IdentityProfile, OcclusionProfile, FaceMapping
from watserface.studio.identity_builder import IdentityBuilder, create_identity_builder
from watserface.training import core as training_core


class StudioOrchestrator:
    
    def __init__(self):
        self.state = StudioState()
        self.identity_builder = create_identity_builder()
        self._scan_workspace()

    def _scan_workspace(self):
        """Scan workspace for existing face sets and identities."""
        try:
            face_sets = self.identity_builder.list_face_sets()
            for fs_data in face_sets:
                name = fs_data.get('name', 'Unknown')
                fs_id = fs_data.get('id')

                if name not in self.state.identities:
                    identity = self.state.add_identity(name)
                    identity.face_set_id = fs_id

                    # Check for existing trained model
                    set_dir = self.identity_builder.face_set_dir / fs_id
                    if set_dir.exists():
                        # Look for model files
                        potential_models = (
                            list(set_dir.glob("*_controlnet")) +
                            list(set_dir.glob("*.pth")) +
                            list(set_dir.glob("*.safetensors"))
                        )

                        for model_path in potential_models:
                            if model_path.is_dir() or model_path.is_file():
                                identity.model_path = str(model_path)
                                identity.epochs_trained = 100  # Assume trained
                                break

                    self.state.log(f"Recovered identity '{name}' from workspace")
        except Exception as e:
            logger.warn(f"Workspace scan failed: {e}", __name__)

    def add_identity_media(self, name: str, files: List[Any]) -> Generator[Tuple[str, Dict], None, None]:
        paths = self._normalize_paths(files)
        if not paths:
            yield self.state.log('No valid files provided'), {}
            return
        
        identity = self.state.add_identity(name, paths)
        yield self.state.log(f'Added {len(paths)} files to identity "{name}"'), {'identity': name, 'count': len(identity.source_paths)}
    
    def extract_faces(self, name: str) -> Generator[Tuple[str, Dict], None, None]:
        identity = self.state.get_identity(name)
        if not identity:
            yield self.state.log(f'Identity "{name}" not found'), {}
            return
        
        if not identity.source_paths:
            yield self.state.log(f'Identity "{name}" has no source media'), {}
            return
        
        self.state.phase = StudioPhase.IDENTITY_BUILDING
        yield self.state.log(f'Extracting faces for identity "{name}"...'), {}
        
        face_set_id = None
        
        for msg, data in self.identity_builder.extract_from_sources(name, identity.source_paths):
            yield self.state.log(msg), data
            if data.get('face_set_id'):
                face_set_id = data['face_set_id']
        
        if face_set_id:
            identity.face_set_id = face_set_id
            yield self.state.log(f'Face set created: {face_set_id}'), {'face_set_id': face_set_id}
        
        self.state.phase = StudioPhase.IDLE
    
    def train_identity(self, name: str, epochs: int = 100, continue_training: bool = False) -> Generator[Tuple[str, Dict], None, None]:
        identity = self.state.get_identity(name)
        if not identity:
            yield self.state.log(f'Identity "{name}" not found'), {}
            return
        
        if not identity.source_paths and not identity.face_set_id:
            yield self.state.log(f'Identity "{name}" has no source media'), {}
            return
        
        self.state.phase = StudioPhase.IDENTITY_BUILDING
        yield self.state.log(f'Starting identity training for "{name}" ({epochs} epochs)'), {}
        
        start_epoch = identity.epochs_trained if continue_training else 0
        
        kwargs = {
            'model_name': name,
            'epochs': epochs,
        }
        
        if identity.face_set_id:
            kwargs['face_set_id'] = identity.face_set_id
        else:
            kwargs['source_files'] = identity.source_paths
            kwargs['save_as_face_set'] = True
        
        telemetry = {}
        for status in training_core.start_identity_training(**kwargs):
            if isinstance(status, list) and len(status) >= 2:
                msg, telemetry = status[0], status[1]
            else:
                msg = str(status)
            yield self.state.log(msg), telemetry
        
        if telemetry.get('model_path'):
            identity.model_path = telemetry['model_path']
            identity.epochs_trained = start_epoch + epochs
            if telemetry.get('final_loss'):
                identity.last_loss = float(telemetry['final_loss'])
            yield self.state.log(f'Identity "{name}" trained: {identity.model_path}'), telemetry
        
        self.state.phase = StudioPhase.IDLE
    
    def set_target(self, file: Any) -> Generator[Tuple[str, Dict], None, None]:
        paths = self._normalize_paths([file])
        if not paths:
            yield self.state.log('No valid target file'), {}
            return

        temp_path = paths[0]

        # Check if file actually exists
        if not os.path.exists(temp_path):
            yield self.state.log(f'⚠️ Warning: File does not exist: {temp_path}'), {'path': temp_path, 'exists': False}
            return

        # Copy to persistent location to avoid Gradio temp cleanup
        persistent_dir = os.path.join(os.getcwd(), 'models', 'targets')
        os.makedirs(persistent_dir, exist_ok=True)

        filename = os.path.basename(temp_path)
        persistent_path = os.path.join(persistent_dir, filename)

        # Copy file to persistent location
        try:
            shutil.copy2(temp_path, persistent_path)
            self.state.target_path = persistent_path
            yield self.state.log(f'Copied target to persistent storage: {filename}'), {}
        except Exception as e:
            yield self.state.log(f'Failed to copy target: {e}'), {}
            # Fall back to temp path (risky but better than nothing)
            self.state.target_path = temp_path

        info = {'path': self.state.target_path, 'is_video': False, 'has_audio': False, 'exists': True}
        if is_video(self.state.target_path):
            info['is_video'] = True
            info['has_audio'] = has_audio(self.state.target_path)

        yield self.state.log(f'Target set: {os.path.basename(self.state.target_path)}'), info
    
    def train_occlusion(self, name: str, epochs: int = 50) -> Generator[Tuple[str, Dict], None, None]:
        if not self.state.target_path:
            yield self.state.log('No target set for occlusion training'), {}
            return
        
        occlusion = self.state.add_occlusion(name, self.state.target_path)
        self.state.phase = StudioPhase.OCCLUSION_TRAINING
        yield self.state.log(f'Starting occlusion training for "{name}" ({epochs} epochs)'), {}
        
        telemetry = {}
        for status in training_core.start_occlusion_training(
            model_name=name,
            epochs=epochs,
            target_file=self.state.target_path
        ):
            if isinstance(status, list) and len(status) >= 2:
                msg, telemetry = status[0], status[1]
            else:
                msg = str(status)
            yield self.state.log(msg), telemetry
        
        if telemetry.get('model_path'):
            occlusion.model_path = telemetry['model_path']
            occlusion.epochs_trained = epochs
            yield self.state.log(f'Occlusion model "{name}" trained: {occlusion.model_path}'), telemetry
        
        self.state.phase = StudioPhase.IDLE
    
    def auto_map_faces(self) -> Generator[Tuple[str, Dict], None, None]:
        if not self.state.target_path:
            yield self.state.log('No target set'), {}
            return

        if not self.state.identities:
            yield self.state.log('No identities defined'), {}
            return

        # Verify target file exists
        if not os.path.exists(self.state.target_path):
            yield self.state.log(f'❌ Target file no longer exists: {self.state.target_path}'), {}
            yield self.state.log('Please re-upload the target file'), {}
            return

        self.state.phase = StudioPhase.MAPPING
        yield self.state.log(f'Analyzing target for face mapping: {self.state.target_path}'), {}
        
        from watserface.face_analyser import get_many_faces
        from watserface.vision import read_static_image, read_static_images

        # Temporarily lower face detector threshold for initial mapping
        original_score = state_manager.get_item('face_detector_score')
        state_manager.set_item('face_detector_score', 0.25)  # Lower threshold to detect more faces

        if is_video(self.state.target_path):
            from watserface.ffmpeg import extract_frames
            from watserface.temp_helper import get_temp_directory_path, create_temp_directory, clear_temp_directory
            from watserface import process_manager

            create_temp_directory(self.state.target_path)

            # Set process manager state to allow ffmpeg to complete
            # Extract frame from 2 seconds in (frame 60 at 30fps) to avoid blank first frames
            # Use original resolution to preserve face quality for detection
            process_manager.start()
            extraction_result = extract_frames(self.state.target_path, '1280x720', 30.0, 60, 61)
            process_manager.end()

            if not extraction_result:
                yield self.state.log('Could not extract frame from video'), {}
                return

            temp_dir = get_temp_directory_path(self.state.target_path)
            frame_path = os.path.join(temp_dir, '00000001.png')
            if os.path.exists(frame_path):
                frame = read_static_image(frame_path)
                clear_temp_directory(self.state.target_path)
            else:
                yield self.state.log('Could not find extracted frame'), {}
                clear_temp_directory(self.state.target_path)
                return
        else:
            frame = read_static_image(self.state.target_path)
        
        # Try to detect faces with debug info
        faces = get_many_faces([frame])

        # Restore original face detector score
        state_manager.set_item('face_detector_score', original_score)

        if not faces:
            # Log frame info for debugging
            import numpy
            frame_info = f"Frame shape: {frame.shape if hasattr(frame, 'shape') else 'unknown'}"
            yield self.state.log(f'No faces detected in target. {frame_info}'), {}
            yield self.state.log('Try: 1) Use a video with clear, frontal faces 2) Check face detector settings'), {}
            return

        yield self.state.log(f'Detected {len(faces)} face(s) in target'), {'face_count': len(faces)}
        
        self.state.mappings.clear()
        identity_names = list(self.state.identities.keys())
        
        for i, face in enumerate(faces):
            if i < len(identity_names):
                mapping = FaceMapping(
                    source_identity=identity_names[i],
                    target_face_index=i,
                    confidence=float(face.score_set.get('detector', 0.0)) if face.score_set else 0.0
                )
                self.state.mappings.append(mapping)
                yield self.state.log(f'Mapped face {i} -> identity "{identity_names[i]}"'), {}
        
        self.state.phase = StudioPhase.IDLE

    def override_mapping(self, identity_name: str, occlusion_model: str) -> Generator[Tuple[str, Dict], None, None]:
        """Manually override mapping settings."""
        if identity_name:
            # If identity is not in state (e.g. pre-existing but not loaded), try to load it
            if identity_name not in self.state.identities:
                from watserface.identity_profile import get_identity_manager
                profile = get_identity_manager().source_intelligence.load_profile(identity_name)
                if profile:
                    self.state.add_identity(identity_name, profile.source_files)
                    self.state.log(f"Loaded identity '{identity_name}' from storage")
            
            # Update mapping
            self.state.mappings.clear()
            self.state.mappings.append(FaceMapping(
                source_identity=identity_name,
                target_face_index=0,  # Default to first face
                confidence=1.0
            ))
            yield self.state.log(f"Set identity mapping: Face 0 -> {identity_name}"), {}

        if occlusion_model:
            # Set occlusion model in global state
            if occlusion_model.endswith('.onnx'):
                state_manager.set_item('face_occluder_model', occlusion_model)
            else:
                state_manager.set_item('face_occluder_model', occlusion_model.replace('.onnx', '').replace('.hash', ''))
            mask_types = state_manager.get_item('face_mask_types') or []
            if 'occlusion' not in mask_types:
                mask_types.append('occlusion')
                state_manager.set_item('face_mask_types', mask_types)
            yield self.state.log(f"Using occlusion model: {occlusion_model}"), {}
    
    def preview(self, frame_number: int = 0) -> Generator[Tuple[str, Any], None, None]:
        if not self.state.target_path or not self.state.mappings:
            yield self.state.log('Cannot preview: missing target or mappings'), None
            return

        # Verify target file exists
        if not os.path.exists(self.state.target_path):
            yield self.state.log(f'❌ Target file no longer exists: {self.state.target_path}'), None
            yield self.state.log('Please re-upload the target file'), None
            return

        yield self.state.log(f'Generating preview from: {self.state.target_path}'), None
        
        self._configure_processors()
        
        from watserface.vision import read_static_image
        from watserface.core import process_preview_frame
        
        if is_video(self.state.target_path):
            from watserface.ffmpeg import extract_frames
            from watserface.temp_helper import get_temp_directory_path, create_temp_directory, clear_temp_directory
            from watserface import process_manager

            create_temp_directory(self.state.target_path)

            # Set process manager state to allow ffmpeg to complete
            process_manager.start()
            extraction_result = extract_frames(self.state.target_path, '640x480', 30.0, frame_number, frame_number + 1)
            process_manager.end()

            if not extraction_result:
                yield self.state.log('Could not extract preview frame'), None
                return

            temp_dir = get_temp_directory_path(self.state.target_path)
            # When extracting a single frame, it's always saved as 00000001.png (first output frame)
            frame_path = os.path.join(temp_dir, '00000001.png')
            if os.path.exists(frame_path):
                frame = read_static_image(frame_path)
                clear_temp_directory(self.state.target_path)
            else:
                yield self.state.log('Could not find extracted preview frame'), None
                clear_temp_directory(self.state.target_path)
                return
        else:
            frame = read_static_image(self.state.target_path)
        
        result = process_preview_frame(frame)
        
        for mapping in self.state.mappings:
            mapping.quality_score = self._estimate_quality(result, mapping)
        
        low_quality = [m for m in self.state.mappings if m.quality_score < self.state.quality_threshold]
        if low_quality and self.state.auto_train_on_low_quality:
            yield self.state.log(f'Quality below threshold for {len(low_quality)} mapping(s). Consider more training.'), result
        else:
            yield self.state.log('Preview generated'), result
    
    def execute(self) -> Generator[Tuple[str, Any], None, None]:
        if not self.state.target_path or not self.state.mappings:
            yield self.state.log('Cannot execute: missing target or mappings'), None
            return

        # Verify target file exists
        if not os.path.exists(self.state.target_path):
            yield self.state.log(f'❌ Target file no longer exists: {self.state.target_path}'), None
            yield self.state.log('Please re-upload the target file'), None
            return

        self.state.phase = StudioPhase.EXECUTING
        yield self.state.log(f'Starting execution with target: {self.state.target_path}'), None
        
        self._configure_processors()
        
        output_dir = state_manager.get_item('output_path') or 'output'
        if not os.path.isdir(output_dir):
            output_dir = os.path.dirname(output_dir) or 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(self.state.target_path)
        name, ext = os.path.splitext(base_name)
        timestamp = int(time.time())
        self.state.output_path = os.path.join(output_dir, f'{name}_studio_{timestamp}{ext}')
        
        state_manager.set_item('output_path', self.state.output_path)
        
        from watserface.core import process_video, process_image
        
        start_time = time.time()
        
        try:
            if is_video(self.state.target_path):
                error_code = process_video(start_time)
            else:
                error_code = process_image(start_time)
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield self.state.log(f'Execution failed: {e}'), None
            self.state.phase = StudioPhase.IDLE
            return
        
        if error_code == 0:
            self.state.phase = StudioPhase.REVIEWING
            yield self.state.log(f'Execution complete: {self.state.output_path}'), self.state.output_path
        else:
            yield self.state.log(f'Execution finished with error code {error_code}'), None
            self.state.phase = StudioPhase.IDLE
    
    def _normalize_paths(self, files: List[Any]) -> List[str]:
        paths = []
        for f in files:
            if f is None:
                continue
            if hasattr(f, 'name'):
                paths.append(f.name)
            elif isinstance(f, str) and os.path.exists(f):
                paths.append(f)
        return paths
    
    def _configure_processors(self) -> None:
        processors = ['face_swapper']
        
        has_trained_identity = any(i.model_path for i in self.state.identities.values())
        has_trained_occlusion = any(o.model_path for o in self.state.occlusions.values())
        
        if has_trained_identity:
            identity = next((i for i in self.state.identities.values() if i.model_path), None)
            if identity:
                state_manager.set_item('face_swapper_model', os.path.basename(identity.model_path))
        
        if has_trained_occlusion:
            occlusion = next((o for o in self.state.occlusions.values() if o.model_path), None)
            if occlusion:
                state_manager.set_item('face_occluder_model', occlusion.model_path)
                mask_types = state_manager.get_item('face_mask_types') or []
                if 'occlusion' not in mask_types:
                    mask_types.append('occlusion')
                    state_manager.set_item('face_mask_types', mask_types)
        
        processors.append('face_enhancer')
        state_manager.set_item('face_enhancer_model', 'codeformer')
        state_manager.set_item('face_enhancer_blend', 100)
        
        if self.state.target_path and is_video(self.state.target_path) and has_audio(self.state.target_path):
            # Only add lip syncer if we have source audio to sync from
            from watserface.filesystem import filter_audio_paths
            source_paths = state_manager.get_item('source_paths')
            if source_paths and filter_audio_paths(source_paths):
                processors.append('lip_syncer')
        
        state_manager.set_item('processors', processors)
        
        for mapping in self.state.mappings:
            identity = self.state.identities.get(mapping.source_identity)
            if identity:
                # Check if this identity has a saved profile with embeddings
                # If so, use the profile embeddings instead of source files (which may not exist)
                from watserface.identity_profile import get_identity_manager
                profile = get_identity_manager().source_intelligence.load_profile(mapping.source_identity)

                if profile and profile.embedding_mean:
                    # Use identity profile embeddings (more reliable than file paths)
                    state_manager.set_item('identity_profile_id', mapping.source_identity)
                    state_manager.set_item('source_paths', [])
                    logger.info(f'[ORCHESTRATOR] Using identity profile embeddings for: {mapping.source_identity}', __name__)
                elif identity.source_paths and any(os.path.exists(p) for p in identity.source_paths):
                    # Fall back to source files only if they actually exist
                    existing_paths = [p for p in identity.source_paths if os.path.exists(p)]
                    state_manager.set_item('source_paths', existing_paths)
                    state_manager.set_item('identity_profile_id', None)
                    logger.info(f'[ORCHESTRATOR] Using {len(existing_paths)} existing source file(s)', __name__)
                else:
                    logger.warn(f'[ORCHESTRATOR] No valid source for identity {mapping.source_identity}', __name__)
                break

        state_manager.set_item('target_path', self.state.target_path)

        # Set output parameters required for execution
        from watserface.vision import detect_video_resolution, detect_video_fps, detect_image_resolution, pack_resolution

        # Set trim frame parameters only if not already set (e.g. by test script)
        if state_manager.get_item('trim_frame_start') is None:
            state_manager.set_item('trim_frame_start', None)
        if state_manager.get_item('trim_frame_end') is None:
            state_manager.set_item('trim_frame_end', None)

        # Set output quality and encoder parameters
        from watserface.ffmpeg import get_available_encoder_set
        from watserface.common_helper import get_first

        available_encoders = get_available_encoder_set()
        state_manager.set_item('output_image_quality', 80)
        state_manager.set_item('output_audio_encoder', get_first(available_encoders.get('audio')))
        state_manager.set_item('output_audio_quality', 80)
        state_manager.set_item('output_audio_volume', 100)
        state_manager.set_item('output_video_encoder', get_first(available_encoders.get('video')))
        state_manager.set_item('output_video_quality', 80)
        state_manager.set_item('output_video_preset', 'veryfast')

        # Set resolution and fps based on target
        if self.state.target_path:
            if is_video(self.state.target_path):
                video_resolution = detect_video_resolution(self.state.target_path)
                if video_resolution:
                    state_manager.set_item('output_video_resolution', pack_resolution(video_resolution))
                video_fps = detect_video_fps(self.state.target_path)
                if video_fps:
                    state_manager.set_item('output_video_fps', video_fps)
            else:
                image_resolution = detect_image_resolution(self.state.target_path)
                if image_resolution:
                    state_manager.set_item('output_image_resolution', pack_resolution(image_resolution))
    
    def _estimate_quality(self, frame: Any, mapping: FaceMapping) -> float:
        return 0.85
