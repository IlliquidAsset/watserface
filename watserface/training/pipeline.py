"""
Training Pipeline Orchestrator.

Coordinates the full face training workflow:
1. Identity Training (ControlNet)
2. XSeg Training (Occlusion Masks)
3. LoRA Training (Style Adaptation)

Manages dependencies, progress reporting, and model validation.
"""

import os
import time
from typing import Iterator, Tuple, Dict, Any, Optional, List
from pathlib import Path

from watserface import logger


class TrainingPipeline:
    """
    Orchestrates the complete face training pipeline.
    
    Pipeline Steps:
    1. Identity Training: Learn source face identity
    2. XSeg Training: Learn face occlusion patterns  
    3. LoRA Training: Fine-tune style adaptation
    """
    
    def __init__(
        self,
        output_dir: str,
        source_profile_id: str,
        model_name: str,
        epochs_identity: int = 100,
        epochs_xseg: int = 50,
        epochs_lora: int = 50,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        max_frames: int = 1000
    ):
        self.output_dir = Path(output_dir)
        self.source_profile_id = source_profile_id
        self.model_name = model_name
        
        self.epochs = {
            'identity': epochs_identity,
            'xseg': epochs_xseg,
            'lora': epochs_lora
        }
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_frames = max_frames
        
        self.pipeline_steps = [
            'identity',
            'xseg', 
            'lora'
        ]
        
        # Track completed models
        self.trained_models = {}
        
    def run_pipeline(self) -> Iterator[Tuple[str, Dict]]:
        """
        Execute the complete training pipeline.
        
        Yields:
            (status_message, telemetry_dict)
        """
        logger.info(f"[Pipeline] Starting training pipeline for '{self.model_name}'", __name__)
        
        total_steps = len(self.pipeline_steps)
        start_time = time.time()
        
        yield "Initializing training pipeline...", {
            'status': 'Starting',
            'total_steps': total_steps,
            'pipeline': self.pipeline_steps
        }
        
        try:
            for step_idx, step in enumerate(self.pipeline_steps):
                step_start = time.time()
                
                # Calculate overall progress
                overall_progress = (step_idx / total_steps) * 100
                
                yield f"Step {step_idx + 1}/{total_steps}: Starting {step} training", {
                    'status': f'Starting {step}',
                    'current_step': step_idx + 1,
                    'total_steps': total_steps,
                    'overall_progress': f"{overall_progress:.1f}%"
                }
                
                # Run the specific training step
                step_generator = self._run_training_step(step)
                
                for status_msg, telemetry in step_generator:
                    # Update telemetry with pipeline context
                    telemetry.update({
                        'pipeline_step': step,
                        'current_step': step_idx + 1,
                        'total_steps': total_steps,
                        'overall_progress': f"{overall_progress:.1f}%"
                    })
                    
                    yield status_msg, telemetry
                
                # Store the trained model path
                if 'model_path' in telemetry:
                    self.trained_models[step] = telemetry['model_path']
                
                step_time = time.time() - step_start
                logger.info(f"[Pipeline] Completed {step} training in {step_time:.1f}s", __name__)
            
            # Pipeline complete
            total_time = time.time() - start_time
            
            final_report = {
                'status': 'Pipeline Complete',
                'total_time': f"{int(total_time // 60)}m {int(total_time % 60)}s",
                'trained_models': self.trained_models,
                'model_name': self.model_name,
                'output_dir': str(self.output_dir)
            }
            
            yield f"Training pipeline complete! Models saved to {self.output_dir}", final_report
            
        except Exception as e:
            logger.error(f"[Pipeline] Pipeline failed: {e}", __name__)
            yield f"Pipeline failed: {e}", {
                'status': 'Error',
                'error': str(e),
                'completed_steps': list(self.trained_models.keys())
            }
            raise
    
    def _run_training_step(self, step: str) -> Iterator[Tuple[str, Dict]]:
        """Execute a specific training step."""
        if step == 'identity':
            yield from self._run_identity_training()
        elif step == 'xseg':
            yield from self._run_xseg_training()  
        elif step == 'lora':
            yield from self._run_lora_training()
        else:
            raise ValueError(f"Unknown training step: {step}")
    
    def _run_identity_training(self) -> Iterator[Tuple[str, Dict]]:
        """Run identity training using ControlNet."""
        from watserface.training.trainers.identity import train_identity_model
        
        dataset_dir = self.output_dir / "identity_dataset"
        if not dataset_dir.exists():
            raise ValueError(f"Identity dataset not found at {dataset_dir}")
        
        yield from train_identity_model(
            dataset_dir=str(dataset_dir),
            model_name=f"{self.model_name}_identity",
            epochs=self.epochs['identity'],
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            save_interval=max(1, self.epochs['identity'] // 10),
            max_frames=self.max_frames
        )
    
    def _run_xseg_training(self) -> Iterator[Tuple[str, Dict]]:
        """Run XSeg occlusion training."""
        from watserface.training.trainers.xseg import train_xseg_model
        
        dataset_dir = self.output_dir / "xseg_dataset"
        if not dataset_dir.exists():
            raise ValueError(f"XSeg dataset not found at {dataset_dir}")
        
        yield from train_xseg_model(
            dataset_dir=str(dataset_dir),
            model_name=f"{self.model_name}_xseg",
            epochs=self.epochs['xseg'],
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            save_interval=max(1, self.epochs['xseg'] // 10)
        )
    
    def _run_lora_training(self) -> Iterator[Tuple[str, Dict]]:
        """Run LoRA style adaptation training."""
        from watserface.training.trainers.lora import train_lora_model
        
        # LoRA training needs both identity and xseg models as context
        identity_model_path = self.trained_models.get('identity')
        xseg_model_path = self.trained_models.get('xseg')
        
        if not identity_model_path:
            logger.warn("[Pipeline] No identity model found, LoRA training may fail", __name__)
        if not xseg_model_path:
            logger.warn("[Pipeline] No XSeg model found, LoRA training may fail", __name__)
        
        dataset_dir = self.output_dir / "lora_dataset"
        if not dataset_dir.exists():
            raise ValueError(f"LoRA dataset not found at {dataset_dir}")
        
        yield from train_lora_model(
            dataset_dir=str(dataset_dir),
            source_profile_id=self.source_profile_id,
            model_name=f"{self.model_name}_lora",
            epochs=self.epochs['lora'],
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            lora_rank=16,
            save_interval=max(1, self.epochs['lora'] // 10),
            max_frames=self.max_frames
        )
    
    def validate_pipeline_requirements(self) -> List[str]:
        """
        Check that all required datasets and dependencies exist.
        
        Returns:
            List of missing requirements
        """
        missing = []
        
        # Check datasets
        required_datasets = ['identity_dataset', 'xseg_dataset', 'lora_dataset']
        for dataset in required_datasets:
            dataset_path = self.output_dir / dataset
            if not dataset_path.exists():
                missing.append(f"Dataset directory: {dataset_path}")
            else:
                # Check for actual data
                frames = list(dataset_path.glob('*.png'))
                if len(frames) == 0:
                    missing.append(f"Dataset frames in: {dataset_path}")
        
        # Check dependencies
        try:
            import diffusers
        except ImportError:
            missing.append("diffusers package (pip install diffusers)")
        
        try:
            import transformers
        except ImportError:
            missing.append("transformers package (pip install transformers)")
        
        try:
            import peft
        except ImportError:
            missing.append("peft package (pip install peft)")
        
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            missing.append("segmentation-models-pytorch package")
        
        return missing


def run_training_pipeline(
    output_dir: str,
    source_profile_id: str,
    model_name: str,
    epochs_identity: int = 100,
    epochs_xseg: int = 50,
    epochs_lora: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    max_frames: int = 1000,
    progress: Any = None
) -> Iterator[Tuple[str, Dict]]:
    """
    High-level pipeline runner for compatibility with training/core.py.
    
    Args:
        output_dir: Base directory containing datasets and output models
        source_profile_id: Profile ID for the source identity
        model_name: Base name for trained models
        epochs_identity: Epochs for identity training
        epochs_xseg: Epochs for XSeg training  
        epochs_lora: Epochs for LoRA training
        batch_size: Training batch size
        learning_rate: Learning rate
        max_frames: Maximum frames to use per dataset
        progress: Optional progress callback
        
    Yields:
        (status_message, telemetry_dict)
    """
    pipeline = TrainingPipeline(
        output_dir=output_dir,
        source_profile_id=source_profile_id,
        model_name=model_name,
        epochs_identity=epochs_identity,
        epochs_xseg=epochs_xseg,
        epochs_lora=epochs_lora,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_frames=max_frames
    )
    
    # Validate requirements
    missing = pipeline.validate_pipeline_requirements()
    if missing:
        yield f"Missing requirements: {', '.join(missing)}", {
            'status': 'Validation Failed',
            'missing': missing
        }
        return
    
    yield from pipeline.run_pipeline()
