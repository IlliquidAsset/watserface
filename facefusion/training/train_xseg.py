import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import onnx
from typing import Any

from facefusion import logger
from facefusion.training.datasets.xseg_dataset import XSegDataset

# --- Model Definition (Simplified U-Net) ---
class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.conv(x)

class SimpleUNet(nn.Module):
	def __init__(self, n_channels=3, n_classes=1): # Binary mask
		super().__init__()
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
		self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
		self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(256 + 128, 128))
		self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(128 + 64, 64))
		self.outc = nn.Conv2d(64, n_classes, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x = self.up1(torch.cat([x3, nn.Upsample(scale_factor=0.5)(x2)], dim=1)) # Simplified skip
		# Correct U-Net has skip connections of same size
		# Let's use proper upsampling
		x = self.up1(torch.cat([nn.Upsample(scale_factor=2)(x3), x2], dim=1)) 
		x = self.up2(torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1))
		logits = self.outc(x)
		return self.sigmoid(logits)

# Correct U-Net logic for skip connections
class UNet(nn.Module):
	def __init__(self, n_channels=3, n_classes=1):
		super(UNet, self).__init__()
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = DoubleConv(64, 128)
		self.pool1 = nn.MaxPool2d(2)
		self.down2 = DoubleConv(128, 256)
		self.pool2 = nn.MaxPool2d(2)
		
		self.up1 = DoubleConv(256 + 128, 128)
		self.up2 = DoubleConv(128 + 64, 64)
		self.outc = nn.Conv2d(64, n_classes, 1)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(self.pool1(x1))
		x3 = self.down2(self.pool2(x2))
		
		x = torch.cat([nn.functional.interpolate(x3, scale_factor=2), x2], dim=1)
		x = self.up1(x)
		x = torch.cat([nn.functional.interpolate(x, scale_factor=2), x1], dim=1)
		x = self.up2(x)
		logits = self.outc(x)
		return torch.sigmoid(logits)


def train_xseg_model(dataset_dir: str, model_name: str, epochs: int, batch_size: int, learning_rate: float, save_interval: int, progress: Any) -> str:
	logger.info(f"Initializing XSeg training for {model_name}...", __name__)
	
	# Device
	device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
	logger.debug(f"Training on device: {device}", __name__)

	# Dataset
	dataset = XSegDataset(dataset_dir)
	if len(dataset) == 0:
		raise ValueError("Dataset is empty.")
	
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # Workers=0 for safety

	# Model
	model = UNet().to(device)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.BCELoss() # Binary Cross Entropy

	# Training Loop
	model.train()
	
	for epoch in progress.tqdm(range(epochs), desc="Training XSeg"):
		epoch_loss = 0
		for images, masks in dataloader:
			images = images.to(device)
			masks = masks.to(device).float().unsqueeze(1) # [B, 1, H, W]
			
			# Normalize masks to 0-1 if they are 255
			masks = masks / 255.0 if masks.max() > 1 else masks

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, masks)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
		
		avg_loss = epoch_loss / len(dataloader)
		if epoch % 5 == 0:
			logger.debug(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}", __name__)

	# Export to ONNX
	output_path = os.path.join(dataset_dir, f"{model_name}.onnx")
	dummy_input = torch.randn(1, 3, 256, 256).to(device)
	
	logger.info(f"Exporting model to {output_path}...", __name__)
	torch.onnx.export(
		model, 
		dummy_input, 
		output_path, 
		input_names=['input'], 
		output_names=['output'],
		dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
	)
	
	return output_path
