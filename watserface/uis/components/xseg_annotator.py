"""
XSeg Mask Annotator UI Component.

Allows users to manually paint masks for XSeg training.
"""

from typing import Optional
from pathlib import Path
import gradio
import cv2
import numpy as np

from watserface import state_manager
from watserface.uis.core import register_ui_component
from watserface.training.dataset_extractor import load_dataset_manifest

# Global components
XSEG_DATASET_PATH : Optional[gradio.Textbox] = None
XSEG_SAMPLE_SLIDER : Optional[gradio.Slider] = None
XSEG_IMAGE_EDITOR : Optional[gradio.ImageEditor] = None
XSEG_SAVE_BUTTON : Optional[gradio.Button] = None
XSEG_STATUS : Optional[gradio.Textbox] = None


def render() -> None:
	"""Render XSeg Annotator component."""
	global XSEG_DATASET_PATH, XSEG_SAMPLE_SLIDER, XSEG_IMAGE_EDITOR, XSEG_SAVE_BUTTON, XSEG_STATUS

	with gradio.Accordion("🎨 XSeg Mask Annotator", open=False):
		gradio.Markdown(
			"Paint masks for occlusion training. "
			"**White = Keep (face visible)**, **Black = Remove (occluded)**, "
			"**Gray = Soft transition**."
		)

		XSEG_DATASET_PATH = gradio.Textbox(
			label="Dataset Directory",
			placeholder="/path/to/training_dataset_occlusion",
			value=state_manager.get_item('training_dataset_path') or ""
		)

		with gradio.Row():
			XSEG_SAMPLE_SLIDER = gradio.Slider(
				label="Sample",
				minimum=0,
				maximum=100,
				step=1,
				value=0
			)

		XSEG_IMAGE_EDITOR = gradio.ImageEditor(
			label="Paint Mask",
			type="numpy",
			brush=gradio.Brush(
				colors=["#FFFFFF", "#000000", "#808080"],
				default_color="#FFFFFF",
				color_mode="fixed"
			),
			height=512
		)

		with gradio.Row():
			XSEG_SAVE_BUTTON = gradio.Button("Save Mask", variant="primary")
			load_dataset_button = gradio.Button("Load Dataset")

		XSEG_STATUS = gradio.Textbox(
			label="Status",
			value="No dataset loaded",
			interactive=False
		)

		# Register components
		register_ui_component('xseg_image_editor', XSEG_IMAGE_EDITOR)
		register_ui_component('xseg_status', XSEG_STATUS)

		# Event handlers
		load_dataset_button.click(
			fn=load_dataset,
			inputs=[XSEG_DATASET_PATH],
			outputs=[XSEG_SAMPLE_SLIDER, XSEG_IMAGE_EDITOR, XSEG_STATUS]
		)

		XSEG_SAMPLE_SLIDER.change(
			fn=load_sample,
			inputs=[XSEG_DATASET_PATH, XSEG_SAMPLE_SLIDER],
			outputs=[XSEG_IMAGE_EDITOR, XSEG_STATUS]
		)

		XSEG_SAVE_BUTTON.click(
			fn=save_mask,
			inputs=[XSEG_DATASET_PATH, XSEG_SAMPLE_SLIDER, XSEG_IMAGE_EDITOR],
			outputs=[XSEG_STATUS]
		)


def listen() -> None:
	"""Setup event listeners (already done in render())."""
	pass


def load_dataset(dataset_path: str):
	"""
	Load dataset and update UI.

	Returns:
		(slider_update, editor_update, status_message)
	"""
	try:
		if not dataset_path or not Path(dataset_path).exists():
			return (
				gradio.Slider(maximum=0, value=0),
				gradio.ImageEditor(value=None),
				"❌ Invalid dataset path"
			)

		# Load manifest
		manifest = load_dataset_manifest(dataset_path)
		num_samples = len(manifest['samples'])

		# Load first sample
		first_sample = load_sample(dataset_path, 0)

		return (
			gradio.Slider(minimum=0, maximum=num_samples-1, value=0),
			first_sample[0],  # image_editor update
			f"✅ Loaded {num_samples} samples"
		)

	except Exception as e:
		return (
			gradio.Slider(maximum=0, value=0),
			gradio.ImageEditor(value=None),
			f"❌ Error loading dataset: {str(e)}"
		)


def load_sample(dataset_path: str, sample_idx: int):
	"""
	Load a sample and existing mask if available.

	Returns:
		(image_editor_update, status_message)
	"""
	try:
		if not dataset_path or not Path(dataset_path).exists():
			return (
				gradio.ImageEditor(value=None),
				"❌ Invalid dataset path"
			)

		dataset_dir = Path(dataset_path)
		manifest = load_dataset_manifest(dataset_path)

		if sample_idx >= len(manifest['samples']):
			return (
				gradio.ImageEditor(value=None),
				"❌ Invalid sample index"
			)

		sample = manifest['samples'][sample_idx]

		# Load image
		image_path = dataset_dir / sample['image_path']
		image = cv2.imread(str(image_path))
		if image is None:
			return (
				gradio.ImageEditor(value=None),
				f"❌ Failed to load image: {image_path}"
			)

		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Try to load existing mask
		masks_dir = dataset_dir / 'masks'
		mask_path = masks_dir / f"{sample['name']}_mask.png"

		if mask_path.exists():
			mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
			mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

			# Combine image and mask (mask as overlay)
			editor_value = {
				'background': image_rgb,
				'layers': [mask_rgb],
				'composite': None
			}
			status = f"Sample {sample_idx}: {sample['name']} (has existing mask)"
		else:
			editor_value = {'background': image_rgb, 'layers': [], 'composite': None}
			status = f"Sample {sample_idx}: {sample['name']} (no mask yet)"

		return (
			gradio.ImageEditor(value=editor_value),
			status
		)

	except Exception as e:
		return (
			gradio.ImageEditor(value=None),
			f"❌ Error loading sample: {str(e)}"
		)


def save_mask(dataset_path: str, sample_idx: int, editor_data):
	"""
	Save painted mask to dataset.

	Returns:
		status_message
	"""
	try:
		if not dataset_path or not Path(dataset_path).exists():
			return "❌ Invalid dataset path"

		if editor_data is None:
			return "❌ No image data"

		dataset_dir = Path(dataset_path)
		manifest = load_dataset_manifest(dataset_path)

		if sample_idx >= len(manifest['samples']):
			return "❌ Invalid sample index"

		sample = manifest['samples'][sample_idx]

		# Create masks directory
		masks_dir = dataset_dir / 'masks'
		masks_dir.mkdir(exist_ok=True)

		# Extract mask from editor
		# The editor returns a dict with 'background', 'layers', and 'composite'
		if isinstance(editor_data, dict):
			if 'composite' in editor_data and editor_data['composite'] is not None:
				# Use composite if available
				mask_rgb = editor_data['composite']
			elif 'layers' in editor_data and len(editor_data['layers']) > 0:
				# Use first layer
				mask_rgb = editor_data['layers'][0]
			else:
				return "❌ No mask data in editor"
		else:
			# Direct numpy array
			mask_rgb = editor_data

		# Convert to grayscale mask
		if len(mask_rgb.shape) == 3:
			mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
		else:
			mask_gray = mask_rgb

		# Save mask
		mask_path = masks_dir / f"{sample['name']}_mask.png"
		cv2.imwrite(str(mask_path), mask_gray)

		return f"✅ Mask saved: {mask_path.name}"

	except Exception as e:
		return f"❌ Error saving mask: {str(e)}"
