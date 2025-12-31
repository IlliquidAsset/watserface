"""
Modeler Source Component - Identity Profile Selector
"""
from typing import Optional, Tuple
import gradio

from watserface import state_manager, wording, logger
from watserface.identity_profile import get_identity_manager
from watserface.uis.core import register_ui_component


MODELER_SOURCE_PROFILE_DROPDOWN: Optional[gradio.Dropdown] = None
MODELER_SOURCE_REFRESH_BUTTON: Optional[gradio.Button] = None
MODELER_SOURCE_STATUS: Optional[gradio.Textbox] = None
MODELER_SOURCE_INFO: Optional[gradio.Markdown] = None


def render() -> None:
	"""Render identity profile selector for Modeler tab"""
	global MODELER_SOURCE_PROFILE_DROPDOWN, MODELER_SOURCE_REFRESH_BUTTON, MODELER_SOURCE_STATUS, MODELER_SOURCE_INFO

	with gradio.Column():
		# Get available identity profiles
		profiles = get_identity_manager().source_intelligence.list_profiles()
		profile_choices = [(p.name, p.id) for p in profiles]

		with gradio.Row():
			MODELER_SOURCE_PROFILE_DROPDOWN = gradio.Dropdown(
				label="🎯 Select Identity Profile",
				choices=profile_choices,
				value=state_manager.get_item('source_profile_id_for_modeler'),
				interactive=True,
				elem_id="modeler_source_profile_dropdown",
				info="Choose the source identity to train against the target. ❓ Only profiles trained in the Training tab will appear here. For best results, use high-quality identity profiles (3+ source images).",
				scale=4
			)
			MODELER_SOURCE_REFRESH_BUTTON = gradio.Button(
				"🔄",
				variant="secondary",
				elem_id="modeler_source_refresh_button",
				scale=1
			)

		# Status display
		MODELER_SOURCE_STATUS = gradio.Textbox(
			label="Status",
			value="👆 Select a source identity profile to continue",
			interactive=False,
			lines=2,
			elem_id="modeler_source_status"
		)

		# Profile information display
		MODELER_SOURCE_INFO = gradio.Markdown(
			"",
			visible=False,
			elem_id="modeler_source_info"
		)

	# Register components
	register_ui_component('modeler_source_profile_dropdown', MODELER_SOURCE_PROFILE_DROPDOWN)
	register_ui_component('modeler_source_refresh_button', MODELER_SOURCE_REFRESH_BUTTON)
	register_ui_component('modeler_source_status', MODELER_SOURCE_STATUS)
	register_ui_component('modeler_source_info', MODELER_SOURCE_INFO)


def listen() -> None:
	"""Set up event listeners"""
	if MODELER_SOURCE_PROFILE_DROPDOWN and MODELER_SOURCE_STATUS and MODELER_SOURCE_INFO:
		MODELER_SOURCE_PROFILE_DROPDOWN.change(
			update_source_profile,
			inputs=[MODELER_SOURCE_PROFILE_DROPDOWN],
			outputs=[MODELER_SOURCE_STATUS, MODELER_SOURCE_INFO]
		)
	
	if MODELER_SOURCE_REFRESH_BUTTON:
		MODELER_SOURCE_REFRESH_BUTTON.click(
			refresh_source_profiles,
			outputs=[MODELER_SOURCE_PROFILE_DROPDOWN]
		)


def refresh_source_profiles() -> gradio.Dropdown:
	"""Refresh the list of identity profiles"""
	try:
		manager = get_identity_manager()
		profiles = manager.source_intelligence.list_profiles()
		profile_choices = [(p.name, p.id) for p in profiles]
		return gradio.Dropdown(choices=profile_choices)
	except Exception as e:
		logger.error(f"Failed to refresh profiles: {e}", __name__)
		return gradio.Dropdown()


def update_source_profile(profile_id: str = None) -> Tuple[str, str]:
	"""Update state when source profile is selected"""
	if not profile_id:
		state_manager.set_item('source_profile_id_for_modeler', None)
		return (
			"👆 Select a source identity profile to continue",
			""
		)

	manager = get_identity_manager()
	profile = manager.source_intelligence.load_profile(profile_id)

	if profile:
		# Set the global state for modeler source profile
		state_manager.set_item('source_profile_id_for_modeler', profile.id)

		status_msg = f"✅ Loaded source identity: {profile.name}"

		# Profile Info
		stats = profile.quality_stats
		profile_info = f"""
### 🆔 Source Profile: {profile.name}

- **Created**: {profile.created_at}
- **Sources**: {stats.get('total_processed', 0)} files
- **Embeddings**: {stats.get('final_embedding_count', 0)}
- **Quality**: {'High' if stats.get('final_embedding_count', 0) > 10 else 'Medium' if stats.get('final_embedding_count', 0) > 3 else 'Low'}

**✅ Ready for paired training**
"""

		return (status_msg, profile_info)
	else:
		state_manager.set_item('source_profile_id_for_modeler', None)
		return ("❌ Failed to load profile", "")