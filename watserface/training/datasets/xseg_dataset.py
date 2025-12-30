import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from watserface import logger

class XSegDataset(Dataset):
	def __init__(self, dataset_dir: str, transform=None):
		self.dataset_dir = dataset_dir
		self.transform = transform
		self.frames = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.png') and not f.startswith('mask_')])
		
	def __len__(self):
		return len(self.frames)

	def __getitem__(self, idx):
		frame_name = self.frames[idx]
		frame_path = os.path.join(self.dataset_dir, frame_name)
		
		# Load Image
		image = cv2.imread(frame_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		# Load or Generate Mask
		# Check for manual mask first
		mask_name = f"mask_{frame_name}"
		mask_path = os.path.join(self.dataset_dir, mask_name)
		
		if os.path.exists(mask_path):
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		else:
			# Auto-generate from landmarks (Weak Supervision)
			json_path = os.path.join(self.dataset_dir, frame_name.replace('.png', '.json'))
			mask = self._generate_mask_from_json(json_path, image.shape[:2])

		# Preprocess
		if self.transform:
			# Apply transforms (resize, normalize)
			pass # Implement later or assume fixed size
			
		# Resize to 256x256 (standard XSeg)
		image = cv2.resize(image, (256, 256))
		mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
		
		# To Tensor
		image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
		mask = torch.from_numpy(mask).long()
		
		return image, mask

	def _generate_mask_from_json(self, json_path: str, shape: Tuple[int, int]) -> np.ndarray:
		mask = np.zeros(shape, dtype=np.uint8)
		if not os.path.exists(json_path):
			return mask
			
		with open(json_path, 'r') as f:
			data = json.load(f)
		
		# Use 478 landmarks if available
		landmarks = None
		if '478' in data:
			landmarks = np.array(data['478'])
		elif '68' in data:
			landmarks = np.array(data['68'])
			
		if landmarks is not None:
			# Create convex hull mask
			hull = cv2.convexHull(landmarks.astype(np.int32))
			cv2.fillConvexPoly(mask, hull, 255)
			
		return mask

def check_dataset_masks(dataset_path: str) -> Dict[str, float]:
	frames = [f for f in os.listdir(dataset_path) if f.endswith('.png') and not f.startswith('mask_')]
	masks = [f for f in os.listdir(dataset_path) if f.startswith('mask_')]
	
	if not frames:
		return {'mask_coverage': 0.0}
		
	return {'mask_coverage': (len(masks) / len(frames)) * 100}
