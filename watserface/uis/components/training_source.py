from typing import Any, List, Optional, Tuple
import os

import gradio

from watserface import state_manager, wording
from watserface.face_store import clear_reference_faces, clear_static_faces
from watserface.filesystem import is_image, is_video
from watserface.uis.core import register_ui_component
from watserface.uis.types import ComponentOptions, File
from watserface.vision import count_video_frame_total
from watserface.face_set import get_face_set_manager

# Source Mode Selection
SOURCE_TYPE_RADIO : Optional[gradio.Radio] = None
FACE_SET_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SET_PREVIEW_GALLERY : Optional[gradio.Gallery] = None
SAVE_AS_FACE_SET_CHECKBOX : Optional[gradio.Checkbox] = None
FACE_SET_NAME : Optional[gradio.Textbox] = None
FACE_SET_DESCRIPTION : Optional[gradio.Textbox] = None

# Face Set Management
FACE_SET_MANAGE_ACCORDION : Optional[gradio.Accordion] = None
FACE_SET_EDIT_NAME : Optional[gradio.Textbox] = None
FACE_SET_EDIT_DESCRIPTION : Optional[gradio.Textbox] = None
FACE_SET_EDIT_TAGS : Optional[gradio.Textbox] = None
FACE_SET_UPDATE_BUTTON : Optional[gradio.Button] = None
FACE_SET_DELETE_BUTTON : Optional[gradio.Button] = None
FACE_SET_REFRESH_BUTTON : Optional[gradio.Button] = None
FACE_SET_EXPORT_BUTTON : Optional[gradio.Button] = None
FACE_SET_IMPORT_FILE : Optional[gradio.File] = None
FACE_SET_IMPORT_BUTTON : Optional[gradio.Button] = None
FACE_SET_MANAGE_STATUS : Optional[gradio.Textbox] = None

# Original Upload Components
TRAINING_SOURCE_FILE : Optional[gradio.File] = None
TRAINING_SOURCE_IMAGE : Optional[gradio.Image] = None
TRAINING_SOURCE_VIDEO : Optional[gradio.Video] = None
TRAINING_SOURCE_GALLERY : Optional[gradio.Gallery] = None
TRAINING_SOURCE_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global SOURCE_TYPE_RADIO, FACE_SET_DROPDOWN, FACE_SET_PREVIEW_GALLERY
	global SAVE_AS_FACE_SET_CHECKBOX, FACE_SET_NAME, FACE_SET_DESCRIPTION
	global FACE_SET_MANAGE_ACCORDION, FACE_SET_EDIT_NAME, FACE_SET_EDIT_DESCRIPTION
	global FACE_SET_EDIT_TAGS, FACE_SET_UPDATE_BUTTON, FACE_SET_DELETE_BUTTON
	global FACE_SET_REFRESH_BUTTON, FACE_SET_EXPORT_BUTTON, FACE_SET_IMPORT_FILE
	global FACE_SET_IMPORT_BUTTON, FACE_SET_MANAGE_STATUS
	global TRAINING_SOURCE_FILE
	global TRAINING_SOURCE_IMAGE
	global TRAINING_SOURCE_VIDEO
	global TRAINING_SOURCE_GALLERY
	global TRAINING_SOURCE_SLIDER

	# Source Type Selector
	SOURCE_TYPE_RADIO = gradio.Radio(
		choices=["Face Set", "Upload Files"],
		value="Upload Files",
		label="Data Source"
	)

	# Face Set Selection (initially hidden)
	face_set_manager = get_face_set_manager()
	face_sets = face_set_manager.list_face_sets()
	face_set_choices = [(f"{fs.name} ({fs.frame_count} frames)", fs.id) for fs in face_sets]

	FACE_SET_DROPDOWN = gradio.Dropdown(
		choices=face_set_choices,
		label="Select Face Set",
		visible=False
	)

	FACE_SET_PREVIEW_GALLERY = gradio.Gallery(
		label="Face Set Preview",
		columns=4,
		rows=1,
		height="auto",
		object_fit="cover",
		visible=False
	)

	# Face Set Management Accordion (initially hidden)
	with gradio.Accordion("Manage Face Sets", open=False, visible=False) as FACE_SET_MANAGE_ACCORDION:
		with gradio.Row():
			FACE_SET_REFRESH_BUTTON = gradio.Button("Refresh List", size="sm")

		gradio.Markdown("### Edit Metadata")
		FACE_SET_EDIT_NAME = gradio.Textbox(
			label="Edit Name",
			placeholder="Enter new name"
		)
		FACE_SET_EDIT_DESCRIPTION = gradio.Textbox(
			label="Edit Description",
			placeholder="Enter description"
		)
		FACE_SET_EDIT_TAGS = gradio.Textbox(
			label="Tags (comma-separated)",
			placeholder="e.g. training, high-quality"
		)

		with gradio.Row():
			FACE_SET_UPDATE_BUTTON = gradio.Button("Update Face Set", variant="primary")
			FACE_SET_DELETE_BUTTON = gradio.Button("Delete Face Set", variant="stop")

		gradio.Markdown("### Export/Import")
		with gradio.Row():
			FACE_SET_EXPORT_BUTTON = gradio.Button("Export Face Set", variant="secondary")

		FACE_SET_IMPORT_FILE = gradio.File(
			label="Import Face Set (upload .zip file)",
			file_types=[".zip"]
		)
		FACE_SET_IMPORT_BUTTON = gradio.Button("Import Uploaded File", variant="secondary")

		FACE_SET_MANAGE_STATUS = gradio.Textbox(
			label="Status",
			interactive=False,
			value=""
		)

	# Upload Mode Components
	is_training_source_image = is_image(state_manager.get_item('training_source_path'))
	is_training_source_video = is_video(state_manager.get_item('training_source_path'))
	TRAINING_SOURCE_FILE = gradio.File(
		label = wording.get('uis.source_file'),
		file_types = [ 'image', 'video' ],
		file_count = 'multiple',
		value = state_manager.get_item('training_source_path') if is_training_source_image or is_training_source_video else None,
		visible=True  # Visible by default (Upload Files mode)
	)

	# Save as Face Set Option (for Upload mode)
	SAVE_AS_FACE_SET_CHECKBOX = gradio.Checkbox(
		label="Save as Face Set for reuse",
		value=False,
		visible=True
	)

	FACE_SET_NAME = gradio.Textbox(
		label="Face Set Name (optional - auto-generated if empty)",
		placeholder="e.g. actor_training_set",
		visible=False
	)

	FACE_SET_DESCRIPTION = gradio.Textbox(
		label="Description (optional)",
		placeholder="e.g. High-quality actor footage",
		visible=False
	)

	training_source_image_options : ComponentOptions =\
	{
		'show_label': False,
		'visible': False
	}
	training_source_video_options : ComponentOptions =\
	{
		'show_label': False,
		'visible': False
	}
	training_source_gallery_options : ComponentOptions =\
	{
		'label': 'Source Files',
		'object_fit': 'cover',
		'columns': 4,
		'rows': 2,
		'height': 'auto',
		'visible': False
	}
	training_source_slider_options : ComponentOptions =\
	{
		'label': 'Preview Frame',
		'minimum': 0,
		'step': 1,
		'value': 0,
		'visible': False
	}

	if is_training_source_image:
		training_source_image_options['value'] = TRAINING_SOURCE_FILE.value.get('path') if isinstance(TRAINING_SOURCE_FILE.value, dict) else TRAINING_SOURCE_FILE.value
		training_source_image_options['visible'] = True
	if is_training_source_video:
		training_source_video_options['value'] = TRAINING_SOURCE_FILE.value.get('path') if isinstance(TRAINING_SOURCE_FILE.value, dict) else TRAINING_SOURCE_FILE.value
		training_source_video_options['visible'] = True
		# Enable slider for video
		video_path = training_source_video_options['value']
		if video_path:
			frame_total = count_video_frame_total(video_path)
			training_source_slider_options['maximum'] = frame_total - 1
			training_source_slider_options['visible'] = True

	TRAINING_SOURCE_IMAGE = gradio.Image(**training_source_image_options)
	TRAINING_SOURCE_VIDEO = gradio.Video(**training_source_video_options)
	TRAINING_SOURCE_GALLERY = gradio.Gallery(**training_source_gallery_options)
	TRAINING_SOURCE_SLIDER = gradio.Slider(**training_source_slider_options)

	register_ui_component('training_source_image', TRAINING_SOURCE_IMAGE)
	register_ui_component('training_source_video', TRAINING_SOURCE_VIDEO)
	register_ui_component('training_source_gallery', TRAINING_SOURCE_GALLERY)
	register_ui_component('training_source_slider', TRAINING_SOURCE_SLIDER)


def listen() -> None:
	# Source type toggle
	SOURCE_TYPE_RADIO.change(
		toggle_source_type,
		inputs=[SOURCE_TYPE_RADIO],
		outputs=[
			FACE_SET_DROPDOWN,
			FACE_SET_PREVIEW_GALLERY,
			FACE_SET_MANAGE_ACCORDION,
			TRAINING_SOURCE_FILE,
			SAVE_AS_FACE_SET_CHECKBOX,
			FACE_SET_NAME,
			FACE_SET_DESCRIPTION
		]
	)

	# Face Set selection
	FACE_SET_DROPDOWN.change(
		load_face_set_preview,
		inputs=[FACE_SET_DROPDOWN],
		outputs=[
			FACE_SET_PREVIEW_GALLERY,
			FACE_SET_EDIT_NAME,
			FACE_SET_EDIT_DESCRIPTION,
			FACE_SET_EDIT_TAGS
		]
	)

	# Face Set management actions
	FACE_SET_UPDATE_BUTTON.click(
		update_face_set,
		inputs=[
			FACE_SET_DROPDOWN,
			FACE_SET_EDIT_NAME,
			FACE_SET_EDIT_DESCRIPTION,
			FACE_SET_EDIT_TAGS
		],
		outputs=[FACE_SET_MANAGE_STATUS, FACE_SET_DROPDOWN]
	)

	FACE_SET_DELETE_BUTTON.click(
		delete_face_set,
		inputs=[FACE_SET_DROPDOWN],
		outputs=[FACE_SET_MANAGE_STATUS, FACE_SET_DROPDOWN]
	)

	FACE_SET_REFRESH_BUTTON.click(
		refresh_face_set_list,
		outputs=[FACE_SET_DROPDOWN, FACE_SET_MANAGE_STATUS]
	)

	FACE_SET_EXPORT_BUTTON.click(
		export_face_set,
		inputs=[FACE_SET_DROPDOWN],
		outputs=[FACE_SET_MANAGE_STATUS]
	)

	FACE_SET_IMPORT_BUTTON.click(
		import_face_set,
		inputs=[FACE_SET_IMPORT_FILE],
		outputs=[FACE_SET_MANAGE_STATUS, FACE_SET_DROPDOWN]
	)

	# Save as Face Set checkbox
	SAVE_AS_FACE_SET_CHECKBOX.change(
		toggle_face_set_inputs,
		inputs=[SAVE_AS_FACE_SET_CHECKBOX],
		outputs=[FACE_SET_NAME, FACE_SET_DESCRIPTION]
	)

	# Original upload handler
	TRAINING_SOURCE_FILE.change(update, inputs = TRAINING_SOURCE_FILE, outputs = [ TRAINING_SOURCE_IMAGE, TRAINING_SOURCE_VIDEO, TRAINING_SOURCE_GALLERY, TRAINING_SOURCE_SLIDER ])


def toggle_source_type(source_type: str) -> Tuple[gradio.Dropdown, gradio.Gallery, gradio.Accordion, gradio.File, gradio.Checkbox, gradio.Textbox, gradio.Textbox]:
	"""Toggle visibility based on source type selection"""
	if source_type == "Face Set":
		# Show Face Set components, hide Upload components
		return (
			gradio.Dropdown(visible=True),  # FACE_SET_DROPDOWN
			gradio.Gallery(visible=True),    # FACE_SET_PREVIEW_GALLERY
			gradio.Accordion(visible=True),  # FACE_SET_MANAGE_ACCORDION
			gradio.File(visible=False),      # TRAINING_SOURCE_FILE
			gradio.Checkbox(visible=False),  # SAVE_AS_FACE_SET_CHECKBOX
			gradio.Textbox(visible=False),   # FACE_SET_NAME
			gradio.Textbox(visible=False)    # FACE_SET_DESCRIPTION
		)
	else:  # "Upload Files"
		# Show Upload components, hide Face Set components
		return (
			gradio.Dropdown(visible=False),  # FACE_SET_DROPDOWN
			gradio.Gallery(visible=False),   # FACE_SET_PREVIEW_GALLERY
			gradio.Accordion(visible=False), # FACE_SET_MANAGE_ACCORDION
			gradio.File(visible=True),       # TRAINING_SOURCE_FILE
			gradio.Checkbox(visible=True),   # SAVE_AS_FACE_SET_CHECKBOX
			gradio.Textbox(visible=False),   # FACE_SET_NAME (hidden until checkbox)
			gradio.Textbox(visible=False)    # FACE_SET_DESCRIPTION (hidden until checkbox)
		)


def load_face_set_preview(face_set_id: Optional[str]) -> Tuple[gradio.Gallery, gradio.Textbox, gradio.Textbox, gradio.Textbox]:
	"""Load preview gallery and metadata for selected Face Set"""
	if not face_set_id:
		return (
			gradio.Gallery(value=None, visible=False),
			gradio.Textbox(value=""),
			gradio.Textbox(value=""),
			gradio.Textbox(value="")
		)

	face_set_manager = get_face_set_manager()
	face_set = face_set_manager.load_face_set(face_set_id)

	if not face_set:
		return (
			gradio.Gallery(value=None, visible=True),
			gradio.Textbox(value=""),
			gradio.Textbox(value=""),
			gradio.Textbox(value="")
		)

	frames_path = face_set_manager.get_face_set_frames_path(face_set_id)

	# Get first 4 frames for preview
	frame_files = []
	if os.path.exists(frames_path):
		frame_files = sorted([
			os.path.join(frames_path, f)
			for f in os.listdir(frames_path)
			if f.endswith(('.png', '.jpg'))
		])[:4]

	# Populate edit fields with current metadata
	tags_str = ", ".join(face_set.tags) if face_set.tags else ""

	return (
		gradio.Gallery(value=frame_files, visible=True),
		gradio.Textbox(value=face_set.name),
		gradio.Textbox(value=face_set.description or ""),
		gradio.Textbox(value=tags_str)
	)


def toggle_face_set_inputs(save_as_face_set: bool) -> Tuple[gradio.Textbox, gradio.Textbox]:
	"""Toggle Face Set name/description inputs based on checkbox"""
	if save_as_face_set:
		return (
			gradio.Textbox(visible=True),  # FACE_SET_NAME
			gradio.Textbox(visible=True)   # FACE_SET_DESCRIPTION
		)
	else:
		return (
			gradio.Textbox(visible=False),  # FACE_SET_NAME
			gradio.Textbox(visible=False)   # FACE_SET_DESCRIPTION
		)


def update_face_set(face_set_id: Optional[str], name: str, description: str, tags_str: str) -> Tuple[gradio.Textbox, gradio.Dropdown]:
	"""Update Face Set metadata"""
	if not face_set_id:
		return (
			gradio.Textbox(value="❌ No Face Set selected"),
			gradio.Dropdown()
		)

	if not name or not name.strip():
		return (
			gradio.Textbox(value="❌ Name cannot be empty"),
			gradio.Dropdown()
		)

	try:
		face_set_manager = get_face_set_manager()

		# Parse tags
		tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []

		# Update metadata
		success = face_set_manager.update_face_set_metadata(
			face_set_id=face_set_id,
			name=name.strip(),
			description=description.strip() if description else None,
			tags=tags
		)

		if success:
			# Refresh dropdown with updated name
			face_sets = face_set_manager.list_face_sets()
			face_set_choices = [(f"{fs.name} ({fs.frame_count} frames)", fs.id) for fs in face_sets]

			return (
				gradio.Textbox(value=f"✅ Face Set '{name}' updated successfully"),
				gradio.Dropdown(choices=face_set_choices, value=face_set_id)
			)
		else:
			return (
				gradio.Textbox(value="❌ Failed to update Face Set"),
				gradio.Dropdown()
			)

	except Exception as e:
		return (
			gradio.Textbox(value=f"❌ Error: {str(e)}"),
			gradio.Dropdown()
		)


def delete_face_set(face_set_id: Optional[str]) -> Tuple[gradio.Textbox, gradio.Dropdown]:
	"""Delete selected Face Set"""
	if not face_set_id:
		return (
			gradio.Textbox(value="❌ No Face Set selected"),
			gradio.Dropdown()
		)

	try:
		face_set_manager = get_face_set_manager()
		face_set = face_set_manager.load_face_set(face_set_id)

		if not face_set:
			return (
				gradio.Textbox(value="❌ Face Set not found"),
				gradio.Dropdown()
			)

		face_set_name = face_set.name

		# Delete Face Set
		success = face_set_manager.delete_face_set(face_set_id)

		if success:
			# Refresh dropdown
			face_sets = face_set_manager.list_face_sets()
			face_set_choices = [(f"{fs.name} ({fs.frame_count} frames)", fs.id) for fs in face_sets]

			return (
				gradio.Textbox(value=f"✅ Face Set '{face_set_name}' deleted successfully"),
				gradio.Dropdown(choices=face_set_choices, value=None)
			)
		else:
			return (
				gradio.Textbox(value="❌ Failed to delete Face Set"),
				gradio.Dropdown()
			)

	except Exception as e:
		return (
			gradio.Textbox(value=f"❌ Error: {str(e)}"),
			gradio.Dropdown()
		)


def refresh_face_set_list() -> Tuple[gradio.Dropdown, gradio.Textbox]:
	"""Refresh the Face Set dropdown list"""
	try:
		face_set_manager = get_face_set_manager()
		face_sets = face_set_manager.list_face_sets()
		face_set_choices = [(f"{fs.name} ({fs.frame_count} frames)", fs.id) for fs in face_sets]

		return (
			gradio.Dropdown(choices=face_set_choices, value=None),
			gradio.Textbox(value=f"✅ Refreshed - {len(face_sets)} Face Set(s) found")
		)

	except Exception as e:
		return (
			gradio.Dropdown(),
			gradio.Textbox(value=f"❌ Error: {str(e)}")
		)


def export_face_set(face_set_id: Optional[str]) -> gradio.Textbox:
	"""Export selected Face Set as zip archive"""
	if not face_set_id:
		return gradio.Textbox(value="❌ No Face Set selected")

	try:
		face_set_manager = get_face_set_manager()
		face_set = face_set_manager.load_face_set(face_set_id)

		if not face_set:
			return gradio.Textbox(value="❌ Face Set not found")

		# Export Face Set
		zip_path = face_set_manager.export_face_set(face_set_id)

		if zip_path:
			return gradio.Textbox(value=f"✅ Face Set exported to: {zip_path}")
		else:
			return gradio.Textbox(value="❌ Export failed")

	except Exception as e:
		return gradio.Textbox(value=f"❌ Error: {str(e)}")


def import_face_set(import_file: Any) -> Tuple[gradio.Textbox, gradio.Dropdown]:
	"""Import Face Set from uploaded zip file"""
	if not import_file:
		return (
			gradio.Textbox(value="❌ No file uploaded"),
			gradio.Dropdown()
		)

	try:
		file_path = import_file.name if hasattr(import_file, 'name') else import_file

		face_set_manager = get_face_set_manager()

		# Import Face Set
		face_set = face_set_manager.import_face_set(file_path)

		if face_set:
			# Refresh dropdown
			face_sets = face_set_manager.list_face_sets()
			face_set_choices = [(f"{fs.name} ({fs.frame_count} frames)", fs.id) for fs in face_sets]

			return (
				gradio.Textbox(value=f"✅ Face Set '{face_set.name}' imported successfully ({face_set.frame_count} frames)"),
				gradio.Dropdown(choices=face_set_choices, value=face_set.id)
			)
		else:
			return (
				gradio.Textbox(value="❌ Import failed"),
				gradio.Dropdown()
			)

	except Exception as e:
		return (
			gradio.Textbox(value=f"❌ Error: {str(e)}"),
			gradio.Dropdown()
		)


def update(files : Any) -> Tuple[gradio.Image, gradio.Video, gradio.Gallery, gradio.Slider]:
	# Handle multiple files or single file
	file = files[0] if isinstance(files, list) and files else files

	if file and is_image(file.name):
		state_manager.set_item('training_source_path', file.name)
		# Show image, hide video and slider, show gallery if multiple files
		gallery_visible = isinstance(files, list) and len(files) > 1
		gallery_value = [f.name for f in files] if gallery_visible else None
		return (
			gradio.Image(value = file.name, visible = True),
			gradio.Video(value = None, visible = False),
			gradio.Gallery(value = gallery_value, visible = gallery_visible),
			gradio.Slider(visible = False)
		)

	if file and is_video(file.name):
		state_manager.set_item('training_source_path', file.name)
		# Show video and slider, hide image
		frame_total = count_video_frame_total(file.name)
		return (
			gradio.Image(value = None, visible = False),
			gradio.Video(value = file.name, visible = True),
			gradio.Gallery(visible = False),
			gradio.Slider(minimum = 0, maximum = frame_total - 1, value = 0, visible = True)
		)

	state_manager.clear_item('training_source_path')
	return (
		gradio.Image(value = None, visible = False),
		gradio.Video(value = None, visible = False),
		gradio.Gallery(visible = False),
		gradio.Slider(visible = False)
	)
