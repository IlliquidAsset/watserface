"""
XSeg Trainer - Occlusion Mask Segmentation using segmentation_models_pytorch.

Uses a pretrained encoder (ResNet34) with U-Net decoder for robust face segmentation.
Handles occlusions like hands, hair, glasses, food, etc.
"""

import os
import time
from typing import Iterator, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

from watserface import logger
from watserface.training.datasets.xseg_dataset import XSegDataset


class XSegTrainer:
    """
    XSeg occlusion mask trainer using segmentation_models_pytorch.
    
    Uses ResNet34 encoder with U-Net decoder for binary face segmentation.
    Pretrained ImageNet weights provide robust feature extraction.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = "sigmoid"
    ):
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation_models_pytorch not installed. "
                "Install with: pip install segmentation-models-pytorch"
            )
        
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        
        self.model = None
        self.device = None
        self.optimizer = None
        self.criterion = None
    
    def _create_model(self) -> nn.Module:
        return smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.classes,
            activation=self.activation
        )
    
    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def train(
        self,
        dataset_dir: str,
        model_name: str,
        epochs: int,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        save_interval: int = 10,
        image_size: int = 256
    ) -> Iterator[Tuple[str, Dict]]:
        """
        Train XSeg model on dataset.
        
        Args:
            dataset_dir: Directory with images and masks
            model_name: Name for saved model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            save_interval: Save checkpoint every N epochs
            image_size: Input image size (square)
            
        Yields:
            (status_message, telemetry_dict)
        """
        logger.info(f"[XSeg] Initializing training for '{model_name}'", __name__)
        
        self.device = self._setup_device()
        logger.info(f"[XSeg] Using device: {self.device}", __name__)
        
        dataset = XSegDataset(dataset_dir, image_size=image_size)
        if len(dataset) == 0:
            raise ValueError(f"No valid samples found in {dataset_dir}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.model = self._create_model().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.criterion = smp.losses.DiceLoss(mode='binary')
        bce_loss = nn.BCEWithLogitsLoss()
        
        checkpoint_path = os.path.join(dataset_dir, f"{model_name}_xseg.pth")
        start_epoch = 0
        
        if os.path.exists(checkpoint_path):
            logger.info(f"[XSeg] Resuming from checkpoint", __name__)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                logger.info(f"[XSeg] Resumed from epoch {start_epoch}", __name__)
            except Exception as e:
                logger.warn(f"[XSeg] Could not load checkpoint: {e}", __name__)
        
        self.model.train()
        start_time = time.time()
        avg_loss = 0.0
        
        total_batches = len(dataloader)
        
        yield f"Starting XSeg training ({len(dataset)} samples)", {
            'status': 'Starting',
            'total_samples': len(dataset),
            'device': str(self.device),
            'encoder': self.encoder_name
        }
        
        for epoch in range(start_epoch, epochs):
            # Check for stop signal at start of epoch
            from watserface.training import core as training_core
            if training_core._training_stopped:
                logger.info(f"[XSeg] Training stopped at epoch {epoch}", __name__)
                self._save_checkpoint(checkpoint_path, epoch, avg_loss)
                yield f"Training stopped at epoch {epoch}. Checkpoint saved.", {
                    'status': 'Stopped',
                    'epoch': epoch,
                    'checkpoint_saved': True
                }
                return

            epoch_loss = 0.0
            epoch_start = time.time()

            for batch_idx, (images, masks) in enumerate(dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device).float()
                
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                
                masks = masks / 255.0 if masks.max() > 1 else masks
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                
                dice_loss = self.criterion(outputs, masks)
                binary_loss = bce_loss(outputs, masks)
                loss = dice_loss + 0.5 * binary_loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / total_batches
            epoch_time = time.time() - epoch_start
            
            elapsed = time.time() - start_time
            completed = epoch - start_epoch + 1
            avg_epoch_time = elapsed / completed
            remaining = epochs - epoch - 1
            eta_seconds = avg_epoch_time * remaining
            
            overall_progress = ((epoch + 1) / epochs) * 100
            
            telemetry = {
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'loss': f"{avg_loss:.4f}",
                'epoch_time': f"{epoch_time:.1f}s",
                'eta': f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s",
                'overall_progress': f"{overall_progress:.1f}%",
                'device': str(self.device),
                'encoder': self.encoder_name
            }
            
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                self._save_checkpoint(checkpoint_path, epoch + 1, avg_loss)
                logger.info(f"[XSeg] Checkpoint saved at epoch {epoch + 1}", __name__)
            
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                logger.info(
                    f"[XSeg] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | ETA: {telemetry['eta']}",
                    __name__
                )
            
            yield f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}", telemetry
        
        yield "Exporting ONNX model...", {'status': 'Exporting'}
        
        onnx_path = self._export_onnx(dataset_dir, model_name)
        
        final_report = {
            'status': 'Complete',
            'model_path': onnx_path,
            'total_epochs': epochs,
            'final_loss': f"{avg_loss:.4f}",
            'total_time': f"{int((time.time() - start_time) // 60)}m"
        }
        
        yield f"Training complete! Model saved to {onnx_path}", final_report
    
    def _save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'encoder_name': self.encoder_name
        }, path)
    
    def _export_onnx(self, output_dir: str, model_name: str) -> str:
        self.model.eval()
        
        output_path = os.path.join(output_dir, f"{model_name}_xseg.onnx")
        
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['image'],
            output_names=['mask'],
            dynamic_axes={
                'image': {0: 'batch', 2: 'height', 3: 'width'},
                'mask': {0: 'batch', 2: 'height', 3: 'width'}
            },
            opset_version=12
        )
        
        logger.info(f"[XSeg] Exported ONNX model to {output_path}", __name__)
        return output_path


def train_xseg_model(
    dataset_dir: str,
    model_name: str,
    epochs: int,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_interval: int = 10,
    progress: Any = None
) -> Iterator[Tuple[str, Dict]]:
    """
    Compatibility wrapper for existing training/core.py interface.
    
    This function maintains backward compatibility with the existing
    training orchestrator while using the new SMP-based trainer.
    """
    trainer = XSegTrainer(
        encoder_name="resnet34",
        encoder_weights="imagenet"
    )
    
    yield from trainer.train(
        dataset_dir=dataset_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_interval=save_interval
    )
