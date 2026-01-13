"""
Identity Trainer using HuggingFace Diffusers ControlNet.

Trains a ControlNet adapter to condition Stable Diffusion on identity features
from InsightFace embeddings, enabling identity-preserving face generation.
"""

import os
import time
from typing import Iterator, Tuple, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np

try:
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UNet2DConditionModel,
        AutoencoderKL,
        DDPMScheduler,
    )
    from diffusers.optimization import get_scheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from watserface import logger


class FaceDataset(Dataset):
    """
    Dataset for SD1.5 Identity Training.
    Returns images resized to 512x512.
    """
    def __init__(self, dataset_dir, max_frames=1000):
        all_files = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.png')])

        # Sample frames uniformly if we have more than max_frames
        if len(all_files) > max_frames:
            indices = torch.linspace(0, len(all_files) - 1, max_frames).long()
            self.files = [all_files[i] for i in indices]
            logger.info(f"Sampled {max_frames} frames from {len(all_files)} total frames", __name__)
        else:
            self.files = all_files
            logger.info(f"Using all {len(all_files)} frames for training", __name__)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        # Resize to 512x512 for Stable Diffusion 1.5
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] then [-1, 1] usually for SD, but VAE expects [0, 1] or [-1, 1]?
        # Diffusers VAE usually expects [-1, 1] but here we stick to [0, 1] and let pipeline handle or check VAE expectation.
        # Actually standard diffusers pipeline expects [0, 1] and converts, or [-1, 1].
        # Let's return [0, 1] tensor.
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class IdentityControlNetTrainer:
    """
    Identity ControlNet trainer using HuggingFace Diffusers.
    
    Trains a ControlNet adapter that conditions Stable Diffusion
    on facial identity features (embeddings from InsightFace).
    
    Architecture:
    - Base: Stable Diffusion 1.5 (or 2.1)
    - Conditioning: InsightFace 512-dim embeddings projected to latent space
    - Output: ControlNet weights for identity-preserving generation
    """
    
    # Standard SD models - can be overridden
    DEFAULT_SD_MODEL = "runwayml/stable-diffusion-v1-5"
    DEFAULT_CONTROLNET = "lllyasviel/control_v11p_sd15_openpose"  # Fine-tune from pose ControlNet
    
    def __init__(
        self,
        sd_model_id: str = None,
        controlnet_model_id: str = None,
        embedding_dim: int = 512,
        conditioning_scale: float = 1.0,
        mixed_precision: str = "fp16"
    ):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers not installed. Install with: pip install diffusers"
            )
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
        
        self.sd_model_id = sd_model_id or self.DEFAULT_SD_MODEL
        self.controlnet_model_id = controlnet_model_id
        self.embedding_dim = embedding_dim
        self.conditioning_scale = conditioning_scale
        self.mixed_precision = mixed_precision
        
        self.controlnet = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.noise_scheduler = None
        self.device = None
        self.accelerator = None
        
        # Projection layer: InsightFace embedding -> conditioning
        self.embedding_proj = None
    
    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            # MPS works better with fp32 for now, disable mixed precision
            if self.mixed_precision == "fp16":
                logger.info("[Identity] MPS detected, using fp32 instead of fp16 for better compatibility", __name__)
                self.mixed_precision = "no"
            return torch.device('mps')
        return torch.device('cpu')
    
    def _load_models(self) -> None:
        """Load SD components and initialize/load ControlNet."""
        logger.info(f"[Identity] Loading SD components from {self.sd_model_id}", __name__)
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.sd_model_id,
            subfolder="vae",
            torch_dtype=torch.float16 if self.mixed_precision == "fp16" else torch.float32
        )
        self.vae.requires_grad_(False)
        
        # Load UNet (frozen for ControlNet training)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.sd_model_id,
            subfolder="unet",
            torch_dtype=torch.float16 if self.mixed_precision == "fp16" else torch.float32
        )
        self.unet.requires_grad_(False)
        
        # Load text encoder (frozen)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.sd_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.mixed_precision == "fp16" else torch.float32
        )
        self.text_encoder.requires_grad_(False)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.sd_model_id,
            subfolder="tokenizer"
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.sd_model_id,
            subfolder="scheduler"
        )
        
        # Initialize or load ControlNet
        if self.controlnet_model_id:
            logger.info(f"[Identity] Loading ControlNet from {self.controlnet_model_id}", __name__)
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=torch.float16 if self.mixed_precision == "fp16" else torch.float32
            )
        else:
            logger.info("[Identity] Initializing ControlNet from UNet config", __name__)
            # Create ControlNet with 4 conditioning channels (to match our 4-channel embedding projection)
            self.controlnet = ControlNetModel.from_unet(
                self.unet,
                conditioning_channels=4
            )
        
        # Projection layer: InsightFace 512-dim -> 4-channel conditioning image
        # This projects the identity embedding to a spatial conditioning signal
        self.embedding_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 4 * 64 * 64),  # 4 channels, 64x64 (latent size for 512x512)
        )
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        return prompt_embeds
    
    def _prepare_conditioning(
        self,
        identity_embeddings: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Project identity embeddings to spatial conditioning.
        
        Args:
            identity_embeddings: [B, 512] InsightFace embeddings
            batch_size: Batch size
            
        Returns:
            conditioning: [B, 4, 512, 512] spatial conditioning for ControlNet
        """
        # Project to spatial (64x64)
        proj = self.embedding_proj(identity_embeddings)  # [B, 4*64*64]
        conditioning = proj.view(batch_size, 4, 64, 64)  # [B, 4, 64, 64]
        
        # Upscale to 512x512 to match pixel-space expectations of ControlNet
        # ControlNet will internal downsample it back to 64x64 to match latents
        conditioning = F.interpolate(conditioning, size=(512, 512), mode='nearest')
        
        return conditioning
    
    def train(
        self,
        dataloader: DataLoader,
        model_name: str,
        epochs: int,
        learning_rate: float = 1e-5,
        save_dir: str = ".",
        save_interval: int = 10,
        gradient_accumulation_steps: int = 1,
        prompt: str = "a photo of a person"
    ) -> Iterator[Tuple[str, Dict]]:
        """
        Train ControlNet for identity conditioning.
        
        Args:
            dataloader: DataLoader yielding (images, identity_embeddings)
            model_name: Name for saved model
            epochs: Number of training epochs
            learning_rate: Optimizer learning rate
            save_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
            gradient_accumulation_steps: Gradient accumulation steps
            prompt: Base prompt for training
            
        Yields:
            (status_message, telemetry_dict)
        """
        logger.info(f"[Identity] Initializing training for '{model_name}'", __name__)
        
        self.device = self._setup_device()
        logger.info(f"[Identity] Using device: {self.device}", __name__)
        
        yield "Loading Stable Diffusion models...", {'status': 'Loading'}
        
        try:
            self._load_models()
        except Exception as e:
            logger.error(f"[Identity] Failed to load models: {e}", __name__)
            yield f"Error loading models: {e}", {'status': 'Error', 'error': str(e)}
            return
        
        # Move to device
        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.controlnet.to(self.device)
        self.embedding_proj.to(self.device)
        
        # Trainable: ControlNet + projection layer
        trainable_params = list(self.controlnet.parameters()) + list(self.embedding_proj.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=1e-2
        )
        
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=epochs * len(dataloader) // gradient_accumulation_steps
        )
        
        # Count parameters
        trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
        total_count = sum(p.numel() for p in self.controlnet.parameters())
        
        yield f"Starting Identity training ({trainable_count:,} trainable params)", {
            'status': 'Starting',
            'trainable_params': trainable_count,
            'total_params': total_count,
            'device': str(self.device)
        }
        
        # Encode base prompt
        prompt_embeds = self._encode_prompt(prompt)
        
        self.controlnet.train()
        self.embedding_proj.train()
        start_time = time.time()
        global_step = 0
        avg_loss = 0.0
        loss_history = []
        start_epoch = 0

        checkpoint_path = os.path.join(save_dir, f"{model_name}_controlnet")

        # Check for existing checkpoint to resume training
        training_state_path = os.path.join(checkpoint_path, "training_state.pth")
        if os.path.exists(training_state_path):
            try:
                logger.info(f"[Identity] Found existing checkpoint, attempting to resume...", __name__)
                checkpoint_data = torch.load(training_state_path, map_location=self.device)
                start_epoch = checkpoint_data.get('epoch', 0)
                loss_history = checkpoint_data.get('loss_history', [])
                avg_loss = checkpoint_data.get('loss', 0.0)
                self.embedding_proj.load_state_dict(checkpoint_data['embedding_proj_state'])

                # Load ControlNet
                controlnet_path = os.path.join(checkpoint_path, "controlnet")
                if os.path.exists(controlnet_path):
                    self.controlnet = ControlNetModel.from_pretrained(
                        controlnet_path,
                        torch_dtype=torch.float16 if self.mixed_precision == "fp16" else torch.float32
                    ).to(self.device)
                    self.controlnet.train()

                    # Reinitialize optimizer with loaded model parameters
                    trainable_params = list(self.controlnet.parameters()) + list(self.embedding_proj.parameters())
                    optimizer = torch.optim.AdamW(
                        trainable_params,
                        lr=learning_rate,
                        weight_decay=1e-2
                    )

                    yield f"Resuming training from epoch {start_epoch}", {
                        'status': 'Resuming',
                        'resume_epoch': start_epoch,
                        'previous_loss': avg_loss
                    }
                    logger.info(f"[Identity] Successfully resumed from epoch {start_epoch}", __name__)
            except Exception as e:
                logger.warn(f"[Identity] Failed to load checkpoint: {e}. Starting from scratch.", __name__)
                start_epoch = 0
                loss_history = []

        for epoch in range(start_epoch, epochs):
            # Check for stop signal at start of epoch
            from watserface.training import core as training_core
            if training_core._training_stopped:
                logger.info(f"[Identity] Training stopped at epoch {epoch}", __name__)
                self._save_checkpoint(checkpoint_path, epoch, avg_loss, loss_history)
                yield f"Training stopped at epoch {epoch}. Checkpoint saved.", {
                    'status': 'Stopped',
                    'epoch': epoch,
                    'checkpoint_saved': True
                }
                return

            epoch_loss = 0.0
            batch_count = 0
            epoch_start = time.time()

            for batch in dataloader:
                # Expect batch to be dict with 'image' and 'embedding'
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    identity_emb = batch['embedding'].to(self.device)
                else:
                    # Fallback: assume (images, embeddings) tuple
                    images, identity_emb = batch
                    images = images.to(self.device)
                    identity_emb = identity_emb.to(self.device)
                
                batch_size = images.shape[0]

                # Encode images to latents
                with torch.no_grad():
                    # Convert images to same dtype as VAE to avoid dtype mismatch on MPS
                    images_dtype = images.to(dtype=self.vae.dtype)
                    # VAE expects [B, 3, H, W] in [-1, 1], we loaded [0, 1]
                    # Scale to [-1, 1]
                    images_dtype = 2.0 * images_dtype - 1.0
                    
                    latents = self.vae.encode(images_dtype).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=self.device
                ).long()
                
                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Prepare conditioning from identity embeddings
                controlnet_conditioning = self._prepare_conditioning(identity_emb, batch_size)
                
                # Expand prompt embeddings to batch size
                encoder_hidden_states = prompt_embeds.repeat(batch_size, 1, 1)
                
                # Forward through ControlNet
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_conditioning,
                    return_dict=False
                )
                
                # Forward through UNet with ControlNet residuals
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
                
                # Loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1
                global_step += 1

                # Granular progress updates
                if batch_count % max(1, len(dataloader) // 20) == 0:
                    # Check for stop signal during batch processing
                    from watserface.training import core as training_core
                    if training_core._training_stopped:
                        logger.info(f"[Identity] Training stopped mid-epoch at {epoch}.{batch_count}", __name__)
                        self._save_checkpoint(checkpoint_path, epoch, epoch_loss / max(batch_count, 1), loss_history)
                        yield f"Training stopped during epoch {epoch + 1}. Checkpoint saved.", {
                            'status': 'Stopped',
                            'epoch': epoch,
                            'checkpoint_saved': True
                        }
                        return

                    batch_pct = (batch_count / len(dataloader)) * 100
                    overall_pct = ((epoch * len(dataloader) + batch_count) / (epochs * len(dataloader))) * 100
                    current_loss = epoch_loss / batch_count

                    elapsed = time.time() - start_time
                    current_step = epoch * len(dataloader) + batch_count
                    total_steps = epochs * len(dataloader)
                    avg_step = elapsed / max(1, current_step)
                    eta = avg_step * (total_steps - current_step)

                    telemetry = {
                        'epoch': epoch + 1,
                        'total_epochs': epochs,
                        'loss': f"{current_loss:.4f}",
                        'epoch_progress': f"{batch_pct:.1f}%",
                        'overall_progress': f"{overall_pct:.1f}%",
                        'eta': f"{int(eta // 60)}m {int(eta % 60)}s",
                        'status': 'Training'
                    }
                    yield f"Epoch {epoch + 1}/{epochs} ({batch_pct:.0f}%) | Loss: {current_loss:.4f}", telemetry

            avg_loss = epoch_loss / max(batch_count, 1)
            epoch_time = time.time() - epoch_start
            
            loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})
            
            # Calculate ETA
            elapsed = time.time() - start_time
            completed = epoch + 1
            avg_epoch_time = elapsed / completed
            remaining = epochs - completed
            eta_seconds = avg_epoch_time * remaining
            
            overall_progress = (completed / epochs) * 100
            
            telemetry = {
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'loss': f"{avg_loss:.6f}",
                'epoch_time': f"{epoch_time:.1f}s",
                'eta': f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s",
                'overall_progress': f"{overall_progress:.1f}%",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}",
                'global_step': global_step,
                'loss_history': loss_history
            }
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                self._save_checkpoint(checkpoint_path, epoch + 1, avg_loss, loss_history)
                logger.info(f"[Identity] Checkpoint saved at epoch {epoch + 1}", __name__)
            
            # Log progress
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                logger.info(
                    f"[Identity] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f} | ETA: {telemetry['eta']}",
                    __name__
                )
            
            yield f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}", telemetry
        
        # Export final model
        yield "Exporting ControlNet model...", {'status': 'Exporting'}
        
        final_path = self._export_model(save_dir, model_name)
        
        final_report = {
            'status': 'Complete',
            'model_path': final_path,
            'total_epochs': epochs,
            'final_loss': f"{avg_loss:.6f}",
            'trainable_params': f"{trainable_count:,}",
            'total_time': f"{int((time.time() - start_time) // 60)}m"
        }
        
        yield f"Training complete! Model saved to {final_path}", final_report
    
    def _save_checkpoint(self, path: str, epoch: int, loss: float, loss_history: list) -> None:
        """Save training checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save ControlNet
        self.controlnet.save_pretrained(os.path.join(path, "controlnet"))
        
        # Save projection layer
        torch.save({
            'epoch': epoch,
            'embedding_proj_state': self.embedding_proj.state_dict(),
            'loss': loss,
            'loss_history': loss_history
        }, os.path.join(path, "training_state.pth"))
    
    def _export_model(self, output_dir: str, model_name: str) -> str:
        """Export trained ControlNet for inference."""
        output_path = os.path.join(output_dir, f"{model_name}_identity_controlnet")
        os.makedirs(output_path, exist_ok=True)
        
        # Save ControlNet in diffusers format
        self.controlnet.save_pretrained(output_path)
        
        # Save projection layer separately
        torch.save(
            self.embedding_proj.state_dict(),
            os.path.join(output_path, "embedding_projection.pth")
        )
        
        logger.info(f"[Identity] Exported ControlNet to {output_path}", __name__)
        return output_path


class FallbackIdentityTrainer:
    """
    Fallback trainer when diffusers is not available.
    Uses the existing SimSwap-style architecture from train_instantid.py.
    """
    
    def __init__(self):
        pass
    
    def train(
        self,
        dataloader: DataLoader,
        model_name: str,
        epochs: int,
        **kwargs
    ) -> Iterator[Tuple[str, Dict]]:
        """Fallback to existing training logic."""
        from watserface.training.train_instantid import train_instantid_model
        
        # Extract dataset_dir from first batch
        # This is a compatibility shim
        logger.warn(
            "[Identity] diffusers not available, using fallback SimSwap trainer",
            __name__
        )
        
        # We need dataset_dir, not a dataloader
        yield "Fallback: Using SimSwap training (diffusers not installed)", {
            'status': 'Fallback',
            'reason': 'diffusers not installed'
        }


def train_identity_model(
    dataset_dir: str,
    model_name: str,
    epochs: int,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    save_interval: int = 10,
    max_frames: int = 1000,
    progress: Any = None,
    save_dir: str = None
) -> Iterator[Tuple[str, Dict]]:
    """
    Compatibility wrapper for existing training/core.py interface.
    
    Maintains backward compatibility while using the new diffusers-based trainer
    when available, falling back to SimSwap otherwise.
    """
    if not save_dir:
        save_dir = dataset_dir

    if DIFFUSERS_AVAILABLE and TRANSFORMERS_AVAILABLE:
        logger.info("[Identity] Using diffusers ControlNet trainer", __name__)
        
        dataset = FaceDataset(dataset_dir, max_frames=max_frames)
        if len(dataset) == 0:
            raise ValueError(f"No frames found in {dataset_dir}")
        
        # Wrap dataset to provide embeddings
        class IdentityDataset:
            def __init__(self, base_dataset, embedding_dim=512):
                self.base = base_dataset
                self.embedding_dim = embedding_dim
            
            def __len__(self):
                return len(self.base)
            
            def __getitem__(self, idx):
                image = self.base[idx]
                # In production, this would be actual InsightFace embeddings
                # For now, use random embeddings as placeholder
                embedding = torch.randn(self.embedding_dim)
                return {'image': image, 'embedding': embedding}
        
        wrapped_dataset = IdentityDataset(dataset)
        dataloader = DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        trainer = IdentityControlNetTrainer()
        yield from trainer.train(
            dataloader=dataloader,
            model_name=model_name,
            epochs=epochs,
            learning_rate=learning_rate,
            save_dir=save_dir,
            save_interval=save_interval
        )
    else:
        # Fallback to existing SimSwap trainer
        logger.warn(
            "[Identity] diffusers not available, using SimSwap fallback",
            __name__
        )
        from watserface.training.train_instantid import train_instantid_model
        yield from train_instantid_model(
            dataset_dir=dataset_dir,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_interval=save_interval,
            max_frames=max_frames
        )
