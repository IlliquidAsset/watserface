"""
XSeg U-Net Architecture for Occlusion Mask Prediction.

Input: (B, 3, 256, 256) RGB image
Output: (B, 1, 256, 256) Binary mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
	"""Double convolution block: (Conv -> BN -> ReLU) * 2"""

	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
		super().__init__()

		# Use bilinear upsampling or transposed convolutions
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels)
		else:
			self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)

		# Handle size mismatch if input is not evenly divisible
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])

		# Concatenate skip connection
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class XSegUNet(nn.Module):
	"""
	U-Net for XSeg Occlusion Mask Prediction.

	Architecture:
	- Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512)
	- Bottleneck: 1024 channels
	- Decoder: 4 upsampling blocks with skip connections (512 -> 256 -> 128 -> 64)
	- Output: 1 channel sigmoid for binary mask

	Input: (B, 3, 256, 256) RGB image
	Output: (B, 1, 256, 256) Binary mask (values in [0, 1])
	"""

	def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
		super(XSegUNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		# Encoder
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)

		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)

		# Decoder
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)

		# Output layer
		self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

	def forward(self, x):
		# Encoder with skip connections
		x1 = self.inc(x)      # 64 channels
		x2 = self.down1(x1)   # 128 channels
		x3 = self.down2(x2)   # 256 channels
		x4 = self.down3(x3)   # 512 channels
		x5 = self.down4(x4)   # 1024 (or 512) channels

		# Decoder with skip connections
		x = self.up1(x5, x4)  # 512 -> 256 channels
		x = self.up2(x, x3)   # 256 -> 128 channels
		x = self.up3(x, x2)   # 128 -> 64 channels
		x = self.up4(x, x1)   # 64 channels

		# Output layer with sigmoid
		logits = self.outc(x)
		output = torch.sigmoid(logits)

		return output


def test_xseg_unet():
	"""Test XSegUNet architecture."""
	model = XSegUNet(n_channels=3, n_classes=1, bilinear=True)

	# Test forward pass
	batch_size = 2
	dummy_input = torch.randn(batch_size, 3, 256, 256)

	with torch.no_grad():
		output = model(dummy_input)

	print(f"Input shape: {dummy_input.shape}")
	print(f"Output shape: {output.shape}")
	print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

	# Count parameters
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")

	assert output.shape == (batch_size, 1, 256, 256), "Output shape mismatch!"
	assert output.min() >= 0 and output.max() <= 1, "Output not in [0, 1] range!"

	print("XSegUNet test passed!")


if __name__ == '__main__':
	test_xseg_unet()
