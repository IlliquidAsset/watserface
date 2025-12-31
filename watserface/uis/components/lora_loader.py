"""
LoRA Loader Component - Load existing LoRA checkpoints for continued training
"""
from typing import Optional, Tuple, List
import os
import torch
import gradio

# Components
LORA_MODE_RADIO: Optional[gradio.Radio] = None
EXISTING_LORA_DROPDOWN: Optional[gradio.Dropdown] = None


def render() -> None:
	"""Render the LoRA loader component"""
	global LORA_MODE_RADIO, EXISTING_LORA_DROPDOWN

	with gradio.Row():
		LORA_MODE_RADIO = gradio.Radio(
			choices=["New LoRA", "Continue Training"],
			value="New LoRA",
			label="Training Mode",
			scale=1
		)

		# Get existing LoRA checkpoints
		lora_choices = get_existing_loras()

		EXISTING_LORA_DROPDOWN = gradio.Dropdown(
			choices=lora_choices,
			label="Select LoRA to Continue",
			visible=False,
			scale=2
		)


def listen() -> None:
	"""Set up event listeners"""
	LORA_MODE_RADIO.change(
		toggle_lora_mode,
		inputs=[LORA_MODE_RADIO],
		outputs=[EXISTING_LORA_DROPDOWN]
	)


def toggle_lora_mode(mode: str) -> gradio.Dropdown:
	"""Toggle visibility of existing LoRA dropdown"""
	if mode == "Continue Training":
		# Refresh the list and show dropdown
		lora_choices = get_existing_loras()
		return gradio.Dropdown(choices=lora_choices, visible=True, value=None)
	else:
		return gradio.Dropdown(visible=False, value=None)


def get_existing_loras() -> List[Tuple[str, str]]:
	"""Get list of existing LoRA checkpoints"""
	lora_choices = []

	# Check both training directory and deployed models
	search_paths = [
		os.path.abspath('.jobs/training_dataset_lora'),
		os.path.abspath('.assets/models/trained')
	]

	seen_models = set()

	for base_path in search_paths:
		if not os.path.exists(base_path):
			continue

		for filename in os.listdir(base_path):
			if filename.endswith('_lora.pth'):
				model_name = filename.replace('_lora.pth', '')

				# Skip duplicates
				if model_name in seen_models:
					continue
				seen_models.add(model_name)

				checkpoint_path = os.path.join(base_path, filename)

				try:
					# Load checkpoint to get training info
					checkpoint = torch.load(checkpoint_path, map_location='cpu')

					if isinstance(checkpoint, dict):
						epoch = checkpoint.get('epoch', '?')
						loss = checkpoint.get('loss', '?')
						lora_rank = checkpoint.get('lora_rank', '?')

						# Format: "model_name (Epoch epoch, Loss loss, Rank rank)"
						if isinstance(loss, float):
							display_name = f"{model_name} (Epoch {epoch}, Loss {loss:.4f}, Rank {lora_rank})"
						else:
							display_name = f"{model_name} (Epoch {epoch}, Rank {lora_rank})"

						lora_choices.append((display_name, model_name))
					else:
						# Old format or unrecognized
						lora_choices.append((model_name, model_name))

				except Exception as e:
					# If we can't load checkpoint, still show it
					lora_choices.append((model_name, model_name))

	return sorted(lora_choices, key=lambda x: x[0])
