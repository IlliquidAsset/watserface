"""
LoRA Trainer using HuggingFace PEFT library.

Trains lightweight Low-Rank Adaptation layers for style transfer
from source identity to target video lighting/aesthetic.
"""

import os
import time
from typing import Iterator, Tuple, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from watserface import logger


class LoRATrainer:
    """
    LoRA trainer using HuggingFace PEFT for style adaptation.
    
    Trains Low-Rank Adaptation layers on a base model to learn
    source→target style mapping while preserving identity.
    """
    
    DEFAULT_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    def __init__(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft not installed. Install with: pip install peft"
            )
        
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or self.DEFAULT_TARGET_MODULES
        
        self.model = None
        self.device = None
        self.optimizer = None
        self.peft_config = None
    
    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def _create_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def _count_parameters(self, model: nn.Module) -> Tuple[int, int, float]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        percentage = (trainable / total * 100) if total > 0 else 0
        return trainable, total, percentage
    
    def wrap_model_with_lora(self, base_model: nn.Module) -> nn.Module:
        """
        Wrap a base model with LoRA adapters using PEFT.
        
        For models that don't match standard PEFT architectures,
        we manually inject LoRA into Conv2d and Linear layers.
        """
        self.peft_config = self._create_lora_config()
        
        try:
            peft_model = get_peft_model(base_model, self.peft_config)
            return peft_model
        except Exception as e:
            logger.warn(f"[LoRA] PEFT auto-wrap failed: {e}. Using manual injection.", __name__)
            return self._manual_lora_injection(base_model)
    
    def _manual_lora_injection(self, model: nn.Module) -> nn.Module:
        """Manually inject LoRA into Conv2d and Linear layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                
                lora_module = LoRALayer(module, rank=self.rank, alpha=self.alpha)
                setattr(parent, child_name, lora_module)
        
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        
        return model
    
    def train(
        self,
        base_model: nn.Module,
        dataloader: DataLoader,
        model_name: str,
        epochs: int,
        learning_rate: float = 1e-4,
        save_dir: str = ".",
        save_interval: int = 10
    ) -> Iterator[Tuple[str, Dict]]:
        """
        Train LoRA adapters on base model.
        
        Args:
            base_model: The frozen base model to adapt
            dataloader: Training data loader
            model_name: Name for saved model
            epochs: Number of training epochs
            learning_rate: Optimizer learning rate
            save_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
            
        Yields:
            (status_message, telemetry_dict)
        """
        logger.info(f"[LoRA] Initializing training for '{model_name}'", __name__)
        
        self.device = self._setup_device()
        logger.info(f"[LoRA] Using device: {self.device}", __name__)
        
        self.model = self.wrap_model_with_lora(base_model).to(self.device)
        
        trainable, total, pct = self._count_parameters(self.model)
        logger.info(f"[LoRA] Trainable: {trainable:,} / {total:,} ({pct:.2f}%)", __name__)
        
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        
        criterion = nn.L1Loss()
        
        checkpoint_path = os.path.join(save_dir, f"{model_name}_lora.pth")
        start_epoch = 0
        loss_history = []
        
        if os.path.exists(checkpoint_path):
            logger.info(f"[LoRA] Resuming from checkpoint", __name__)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self._load_lora_weights(checkpoint.get('lora_state', {}))
                start_epoch = checkpoint.get('epoch', 0)
                loss_history = checkpoint.get('loss_history', [])
                logger.info(f"[LoRA] Resumed from epoch {start_epoch}", __name__)
            except Exception as e:
                logger.warn(f"[LoRA] Could not load checkpoint: {e}", __name__)
        
        self.model.train()
        start_time = time.time()
        avg_loss = 0.0
        
        yield f"Starting LoRA training (rank={self.rank})", {
            'status': 'Starting',
            'trainable_params': trainable,
            'total_params': total,
            'compression': f"{pct:.2f}%",
            'device': str(self.device)
        }
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            batch_count = 0
            epoch_start = time.time()
            
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get('input', batch.get('source')).to(self.device)
                    targets = batch.get('target', batch.get('output')).to(self.device)
                    conditioning = batch.get('conditioning', batch.get('embedding'))
                    if conditioning is not None:
                        conditioning = conditioning.to(self.device)
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    conditioning = None
                
                self.optimizer.zero_grad()
                
                if conditioning is not None:
                    outputs = self.model(inputs, conditioning)
                else:
                    outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            epoch_time = time.time() - epoch_start
            
            loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})
            
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
                'lora_rank': self.rank,
                'trainable_params': trainable,
                'loss_history': loss_history
            }
            
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                self._save_checkpoint(checkpoint_path, epoch + 1, avg_loss, loss_history)
                logger.info(f"[LoRA] Checkpoint saved at epoch {epoch + 1}", __name__)
            
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                logger.info(
                    f"[LoRA] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | ETA: {telemetry['eta']}",
                    __name__
                )
            
            yield f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}", telemetry
        
        yield "Exporting LoRA weights...", {'status': 'Exporting'}
        
        final_path = self._export_lora(save_dir, model_name)
        
        final_report = {
            'status': 'Complete',
            'model_path': final_path,
            'total_epochs': epochs,
            'final_loss': f"{avg_loss:.4f}",
            'lora_rank': self.rank,
            'trainable_params': f"{trainable:,}"
        }
        
        yield f"Training complete! Model saved to {final_path}", final_report
    
    def _save_checkpoint(self, path: str, epoch: int, loss: float, loss_history: list) -> None:
        lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()
        
        torch.save({
            'epoch': epoch,
            'lora_state': lora_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'loss_history': loss_history,
            'lora_rank': self.rank,
            'lora_alpha': self.alpha
        }, path)
    
    def _load_lora_weights(self, lora_state: dict) -> None:
        model_state = self.model.state_dict()
        for name, param in lora_state.items():
            if name in model_state:
                model_state[name].copy_(param)
    
    def _export_lora(self, output_dir: str, model_name: str) -> str:
        output_path = os.path.join(output_dir, f"{model_name}_lora_weights.pth")
        
        lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()
        
        torch.save({
            'lora_state': lora_state,
            'lora_rank': self.rank,
            'lora_alpha': self.alpha,
            'target_modules': self.target_modules
        }, output_path)
        
        logger.info(f"[LoRA] Exported weights to {output_path}", __name__)
        return output_path


class LoRALayer(nn.Module):
    """
    Manual LoRA layer implementation for Conv2d and Linear.
    
    Implements: output = base_output + (alpha/rank) * (B @ A) @ input
    where A and B are low-rank matrices.
    """
    
    def __init__(self, base_layer: nn.Module, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        elif isinstance(base_layer, nn.Conv2d):
            in_channels = base_layer.in_channels
            out_channels = base_layer.out_channels
            kernel_size = base_layer.kernel_size[0]
            self.lora_A = nn.Parameter(torch.zeros(rank, in_channels * kernel_size * kernel_size))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        
        if isinstance(self.base_layer, nn.Linear):
            lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        elif isinstance(self.base_layer, nn.Conv2d):
            batch, channels, h, w = x.shape
            kernel_size = self.base_layer.kernel_size[0]
            padding = self.base_layer.padding[0]
            stride = self.base_layer.stride[0]
            
            unfolded = nn.functional.unfold(x, kernel_size, padding=padding, stride=stride)
            lora_out = (unfolded.transpose(1, 2) @ self.lora_A.T @ self.lora_B.T).transpose(1, 2)
            
            out_h = (h + 2 * padding - kernel_size) // stride + 1
            out_w = (w + 2 * padding - kernel_size) // stride + 1
            lora_output = lora_out.view(batch, -1, out_h, out_w) * self.scaling
        else:
            lora_output = 0
        
        return base_output + lora_output


def train_lora_model(
    dataset_dir: str,
    source_profile_id: str,
    model_name: str,
    epochs: int,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    save_interval: int = 10,
    progress: Any = None,
    max_frames: int = 1000
) -> Iterator[Tuple[str, Dict]]:
    """
    Compatibility wrapper for existing training/core.py interface.
    
    Maintains backward compatibility while using the new PEFT-based trainer.
    """
    from watserface.training.train_instantid import IdentityGenerator, FaceDataset
    
    base_model = IdentityGenerator()
    
    dataset = FaceDataset(dataset_dir, max_frames=max_frames)
    if len(dataset) == 0:
        raise ValueError(f"No frames found in {dataset_dir}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    class WrappedDataLoader:
        def __init__(self, loader, embedding_dim=512):
            self.loader = loader
            self.embedding_dim = embedding_dim
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        def __iter__(self):
            for batch in self.loader:
                batch_size = batch.shape[0]
                yield {
                    'input': batch,
                    'target': batch,
                    'conditioning': torch.randn(batch_size, self.embedding_dim)
                }
        
        def __len__(self):
            return len(self.loader)
    
    wrapped_loader = WrappedDataLoader(dataloader)
    
    trainer = LoRATrainer(
        rank=lora_rank,
        alpha=float(lora_rank),
        dropout=0.1
    )
    
    yield from trainer.train(
        base_model=base_model,
        dataloader=wrapped_loader,
        model_name=model_name,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=dataset_dir,
        save_interval=save_interval
    )
