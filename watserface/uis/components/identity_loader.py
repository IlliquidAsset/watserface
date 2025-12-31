"""
Identity Loader Component - Load existing identity profiles for enrichment
"""
from typing import Optional, Tuple, List
import os
import gradio

from watserface.identity_profile import SourceIntelligence

# Components
IDENTITY_MODE_RADIO: Optional[gradio.Radio] = None
EXISTING_IDENTITY_DROPDOWN: Optional[gradio.Dropdown] = None


def render() -> None:
	"""Render the identity loader component"""
	global IDENTITY_MODE_RADIO, EXISTING_IDENTITY_DROPDOWN

	IDENTITY_MODE_RADIO = gradio.Radio(
		choices=["New Identity", "Load Existing"],
		value="New Identity",
		label="Training Mode"
	)

	# Get existing identities
	identity_choices = get_existing_identities()

	EXISTING_IDENTITY_DROPDOWN = gradio.Dropdown(
		choices=identity_choices,
		label="Select Identity to Enrich",
		visible=False,
		info="Select an existing identity to add more training data and improve it"
	)


def listen() -> None:
	"""Set up event listeners"""
	IDENTITY_MODE_RADIO.change(
		toggle_identity_mode,
		inputs=[IDENTITY_MODE_RADIO],
		outputs=[EXISTING_IDENTITY_DROPDOWN]
	)


def toggle_identity_mode(mode: str) -> gradio.Dropdown:
	"""Toggle visibility of existing identity dropdown"""
	if mode == "Load Existing":
		# Refresh the list and show dropdown
		identity_choices = get_existing_identities()
		return gradio.Dropdown(choices=identity_choices, visible=True, value=None)
	else:
		return gradio.Dropdown(visible=False, value=None)


def get_existing_identities() -> List[Tuple[str, str]]:
	"""Get list of existing identity profiles"""
	identities_path = os.path.abspath('models/identities')

	if not os.path.exists(identities_path):
		return []

	identity_choices = []

	for identity_id in os.listdir(identities_path):
		profile_path = os.path.join(identities_path, identity_id, 'profile.json')

		if os.path.exists(profile_path):
			try:
				# Load profile to get name and stats
				intel = SourceIntelligence()
				profile = intel.load_source_profile(identity_id)

				if profile:
					# Get training stats
					history = profile.get('training_history', [])
					total_epochs = sum(entry.get('epochs', 0) for entry in history)
					sessions = len(history)

					# Format: "Name (ID sessions, total epochs)"
					display_name = f"{profile.get('name', identity_id)} ({sessions} session{'s' if sessions != 1 else ''}, {total_epochs} epochs)"
					identity_choices.append((display_name, identity_id))
			except Exception as e:
				# If we can't load profile, skip it
				continue

	return sorted(identity_choices, key=lambda x: x[0])
