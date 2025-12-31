"""
Purge Tab Layout - Data Management and Cleanup
"""
from typing import Optional, List, Dict, Any
import os
import json
import shutil
import zipfile
from pathlib import Path
import gradio

from watserface import state_manager, logger
from watserface.uis.components import about, footer


# UI Components
PURGE_REFRESH_BUTTON: Optional[gradio.Button] = None
PURGE_TOTAL_SIZE: Optional[gradio.Markdown] = None

# Identity Profiles Section
IDENTITY_PROFILES_DATAFRAME: Optional[gradio.Dataframe] = None
IDENTITY_SELECT_ALL_BUTTON: Optional[gradio.Button] = None
IDENTITY_SELECT_NONE_BUTTON: Optional[gradio.Button] = None
IDENTITY_DELETE_BUTTON: Optional[gradio.Button] = None
IDENTITY_DOWNLOAD_BUTTON: Optional[gradio.Button] = None
IDENTITY_SECTION_SIZE: Optional[gradio.Markdown] = None

# LoRA Models Section
LORA_DATAFRAME: Optional[gradio.Dataframe] = None
LORA_SELECT_ALL_BUTTON: Optional[gradio.Button] = None
LORA_SELECT_NONE_BUTTON: Optional[gradio.Button] = None
LORA_DELETE_BUTTON: Optional[gradio.Button] = None
LORA_DOWNLOAD_BUTTON: Optional[gradio.Button] = None
LORA_SECTION_SIZE: Optional[gradio.Markdown] = None

# Training Datasets Section
DATASET_DATAFRAME: Optional[gradio.Dataframe] = None
DATASET_SELECT_ALL_BUTTON: Optional[gradio.Button] = None
DATASET_SELECT_NONE_BUTTON: Optional[gradio.Button] = None
DATASET_DELETE_BUTTON: Optional[gradio.Button] = None
DATASET_SECTION_SIZE: Optional[gradio.Markdown] = None

# Face Sets Section
FACESET_DATAFRAME: Optional[gradio.Dataframe] = None
FACESET_SELECT_ALL_BUTTON: Optional[gradio.Button] = None
FACESET_SELECT_NONE_BUTTON: Optional[gradio.Button] = None
FACESET_DELETE_BUTTON: Optional[gradio.Button] = None
FACESET_SECTION_SIZE: Optional[gradio.Markdown] = None

# Downloaded Models Section
DOWNLOADED_MODELS_DATAFRAME: Optional[gradio.Dataframe] = None
DOWNLOADED_SELECT_ALL_BUTTON: Optional[gradio.Button] = None
DOWNLOADED_SELECT_NONE_BUTTON: Optional[gradio.Button] = None
DOWNLOADED_DELETE_BUTTON: Optional[gradio.Button] = None
DOWNLOADED_SECTION_SIZE: Optional[gradio.Markdown] = None


def get_directory_size(path: str) -> int:
	"""Calculate total size of directory in bytes"""
	total_size = 0
	if not os.path.exists(path):
		return 0

	if os.path.isfile(path):
		return os.path.getsize(path)

	for dirpath, dirnames, filenames in os.walk(path):
		for filename in filenames:
			filepath = os.path.join(dirpath, filename)
			if os.path.exists(filepath):
				total_size += os.path.getsize(filepath)
	return total_size


def format_size(bytes_size: int) -> str:
	"""Format bytes to human-readable size"""
	for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
		if bytes_size < 1024.0:
			return f"{bytes_size:.2f} {unit}"
		bytes_size /= 1024.0
	return f"{bytes_size:.2f} PB"


def scan_identity_profiles() -> tuple[List[List[Any]], str]:
	"""Scan identity profiles and return data for table"""
	profiles_dir = 'models/identities'
	if not os.path.exists(profiles_dir):
		return ([], "**Identity Profiles**: 0 items (0 B)")

	data = []
	total_size = 0

	for profile_id in os.listdir(profiles_dir):
		profile_path = os.path.join(profiles_dir, profile_id)
		if not os.path.isdir(profile_path):
			continue

		# Load profile metadata
		profile_json = os.path.join(profile_path, 'profile.json')
		name = profile_id
		created = "Unknown"

		if os.path.exists(profile_json):
			try:
				with open(profile_json, 'r') as f:
					metadata = json.load(f)
					name = metadata.get('name', profile_id)
					created = metadata.get('created_at', 'Unknown')
			except:
				pass

		# Calculate size
		size = get_directory_size(profile_path)
		total_size += size

		data.append([
			False,  # Selected checkbox
			name,
			profile_id,
			created,
			format_size(size),
			profile_path
		])

	section_info = f"**Identity Profiles**: {len(data)} items ({format_size(total_size)})"
	return (data, section_info)


def scan_lora_models() -> tuple[List[List[Any]], str]:
	"""Scan LoRA models and return data for table"""
	data = []
	total_size = 0

	# Check both training directory and deployed models
	search_paths = [
		('.jobs/training_dataset_lora', 'Training'),
		('.assets/models/trained', 'Deployed')
	]

	seen_models = set()

	for base_path, location in search_paths:
		if not os.path.exists(base_path):
			continue

		for filename in os.listdir(base_path):
			if not filename.endswith('_lora.pth'):
				continue

			model_name = filename.replace('_lora.pth', '')

			# Skip duplicates
			if model_name in seen_models:
				continue
			seen_models.add(model_name)

			checkpoint_path = os.path.join(base_path, filename)

			# Load checkpoint metadata
			epoch = "Unknown"
			loss = "Unknown"
			rank = "Unknown"

			try:
				import torch
				checkpoint = torch.load(checkpoint_path, map_location='cpu')
				if isinstance(checkpoint, dict):
					epoch = checkpoint.get('epoch', '?')
					loss_val = checkpoint.get('loss', '?')
					if isinstance(loss_val, float):
						loss = f"{loss_val:.4f}"
					rank = checkpoint.get('lora_rank', '?')
			except:
				pass

			# Calculate size (checkpoint + ONNX if exists)
			size = os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0

			onnx_path = checkpoint_path.replace('_lora.pth', '_lora.onnx')
			if os.path.exists(onnx_path):
				size += os.path.getsize(onnx_path)

			hash_path = checkpoint_path.replace('_lora.pth', '_lora.hash')
			if os.path.exists(hash_path):
				size += os.path.getsize(hash_path)

			total_size += size

			data.append([
				False,  # Selected checkbox
				model_name,
				location,
				f"Epoch {epoch}",
				f"Loss: {loss}",
				f"Rank {rank}",
				format_size(size),
				os.path.dirname(checkpoint_path)
			])

	section_info = f"**LoRA Models**: {len(data)} items ({format_size(total_size)})"
	return (data, section_info)


def scan_training_datasets() -> tuple[List[List[Any]], str]:
	"""Scan training dataset directories"""
	jobs_dir = '.jobs'
	if not os.path.exists(jobs_dir):
		return ([], "**Training Datasets**: 0 items (0 B)")

	data = []
	total_size = 0

	for item in os.listdir(jobs_dir):
		item_path = os.path.join(jobs_dir, item)
		if not os.path.isdir(item_path):
			continue

		if not 'training_dataset' in item:
			continue

		# Count frames
		frames = [f for f in os.listdir(item_path) if f.endswith('.png')] if os.path.exists(item_path) else []
		frame_count = len(frames)

		# Calculate size
		size = get_directory_size(item_path)
		total_size += size

		# Get modification time
		try:
			mtime = os.path.getmtime(item_path)
			import datetime
			modified = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
		except:
			modified = "Unknown"

		data.append([
			False,  # Selected checkbox
			item,
			f"{frame_count} frames",
			modified,
			format_size(size),
			item_path
		])

	section_info = f"**Training Datasets**: {len(data)} items ({format_size(total_size)})"
	return (data, section_info)


def scan_face_sets() -> tuple[List[List[Any]], str]:
	"""Scan face sets directories"""
	face_sets_dir = 'models/face_sets'
	if not os.path.exists(face_sets_dir):
		return ([], "**Face Sets**: 0 items (0 B)")

	data = []
	total_size = 0

	for item in os.listdir(face_sets_dir):
		item_path = os.path.join(face_sets_dir, item)
		if not os.path.isdir(item_path):
			continue

		# Load metadata
		metadata_path = os.path.join(item_path, 'faceset.json')
		name = item
		created = "Unknown"
		frames = 0

		if os.path.exists(metadata_path):
			try:
				with open(metadata_path, 'r') as f:
					metadata = json.load(f)
					name = metadata.get('name', item)
					created = metadata.get('created_at', 'Unknown')
					frames = metadata.get('frame_count', 0)
			except:
				pass

		# Calculate size
		size = get_directory_size(item_path)
		total_size += size

		data.append([
			False,  # Selected checkbox
			name,
			item,
			f"{frames} frames",
			created,
			format_size(size),
			item_path
		])

	section_info = f"**Face Sets**: {len(data)} items ({format_size(total_size)})"
	return (data, section_info)


def scan_downloaded_models() -> tuple[List[List[Any]], str]:
	"""Scan downloaded models in .assets/models"""
	models_dir = '.assets/models'
	if not os.path.exists(models_dir):
		return ([], "**Downloaded Models**: 0 items (0 B)")

	data = []
	total_size = 0

	# Exclude trained/ subdirectory
	for filename in os.listdir(models_dir):
		file_path = os.path.join(models_dir, filename)

		# Skip directories and trained models
		if os.path.isdir(file_path):
			if filename == 'trained':
				continue
			# Skip subdirectories like iperov
			size = get_directory_size(file_path)
			total_size += size
			data.append([
				False,
				filename + '/',
				'Directory',
				format_size(size),
				file_path
			])
			continue

		# Only process .onnx files
		if not filename.endswith('.onnx'):
			continue

		# Calculate size
		size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

		# Also include .data file if exists
		data_file = file_path + '.data'
		if os.path.exists(data_file):
			size += os.path.getsize(data_file)

		# Include hash file if exists
		hash_file = os.path.splitext(file_path)[0] + '.hash'
		if os.path.exists(hash_file):
			size += os.path.getsize(hash_file)

		total_size += size

		model_name = filename.replace('.onnx', '')

		data.append([
			False,  # Selected checkbox
			model_name,
			'Downloaded',
			format_size(size),
			file_path
		])

	section_info = f"**Downloaded Models**: {len(data)} items ({format_size(total_size)})"
	return (data, section_info)


def refresh_all_data() -> tuple:
	"""Refresh all data tables and size displays"""
	identity_data, identity_size = scan_identity_profiles()
	lora_data, lora_size = scan_lora_models()
	dataset_data, dataset_size = scan_training_datasets()

	# Calculate total size
	total = 0
	for row in identity_data:
		# Parse size from formatted string (e.g., "1.23 MB")
		size_str = row[4]  # Size column
		# This is approximate - we already have the total from scan functions

	# Recalculate properly
	identity_total = get_directory_size('models/identities')
	lora_total = get_directory_size('.assets/models/trained') + get_directory_size('.jobs/training_dataset_lora')
	dataset_total = sum([get_directory_size(row[5]) for row in dataset_data])

	total = identity_total + lora_total + dataset_total
	total_info = f"## Total Storage Used: {format_size(total)}"

	return (
		gradio.Dataframe(value=identity_data),
		identity_size,
		gradio.Dataframe(value=lora_data),
		lora_size,
		gradio.Dataframe(value=dataset_data),
		dataset_size,
		total_info
	)


def select_all_identity(current_data):
	"""Select all identity profiles"""
	if not current_data:
		return current_data
	new_data = []
	for row in current_data:
		new_row = row.copy()
		new_row[0] = True
		new_data.append(new_row)
	return new_data


def select_none_identity(current_data):
	"""Deselect all identity profiles"""
	if not current_data:
		return current_data
	new_data = []
	for row in current_data:
		new_row = row.copy()
		new_row[0] = False
		new_data.append(new_row)
	return new_data


def delete_selected_identities(current_data):
	"""Delete selected identity profiles"""
	if not current_data:
		return "No items to delete", *refresh_all_data()

	deleted_count = 0
	errors = []

	for row in current_data:
		if row[0]:  # If selected
			profile_path = row[5]  # Path column
			try:
				if os.path.exists(profile_path):
					shutil.rmtree(profile_path)
					deleted_count += 1
					logger.info(f"Deleted identity profile: {profile_path}", __name__)
			except Exception as e:
				errors.append(f"Failed to delete {row[1]}: {str(e)}")

	message = f"✅ Deleted {deleted_count} identity profile(s)"
	if errors:
		message += f"\n\n❌ Errors:\n" + "\n".join(errors)

	return (message, *refresh_all_data())


def select_all_lora(current_data):
	"""Select all LoRA models"""
	if not current_data:
		return current_data
	new_data = []
	for row in current_data:
		new_row = row.copy()
		new_row[0] = True
		new_data.append(new_row)
	return new_data


def select_none_lora(current_data):
	"""Deselect all LoRA models"""
	if not current_data:
		return current_data
	new_data = []
	for row in current_data:
		new_row = row.copy()
		new_row[0] = False
		new_data.append(new_row)
	return new_data


def delete_selected_loras(current_data):
	"""Delete selected LoRA models"""
	if not current_data:
		return "No items to delete", *refresh_all_data()

	deleted_count = 0
	errors = []

	for row in current_data:
		if row[0]:  # If selected
			model_name = row[1]
			base_path = row[7]  # Directory path

			# Delete all related files (.pth, .onnx, .hash)
			for ext in ['_lora.pth', '_lora.onnx', '_lora.hash']:
				file_path = os.path.join(base_path, model_name + ext)
				try:
					if os.path.exists(file_path):
						os.remove(file_path)
						logger.info(f"Deleted: {file_path}", __name__)
				except Exception as e:
					errors.append(f"Failed to delete {file_path}: {str(e)}")

			deleted_count += 1

	message = f"✅ Deleted {deleted_count} LoRA model(s)"
	if errors:
		message += f"\n\n❌ Errors:\n" + "\n".join(errors)

	return (message, *refresh_all_data())


def select_all_datasets(current_data):
	"""Select all datasets"""
	if not current_data:
		return current_data
	new_data = []
	for row in current_data:
		new_row = row.copy()
		new_row[0] = True
		new_data.append(new_row)
	return new_data


def select_none_datasets(current_data):
	"""Deselect all datasets"""
	if not current_data:
		return current_data
	new_data = []
	for row in current_data:
		new_row = row.copy()
		new_row[0] = False
		new_data.append(new_row)
	return new_data


def delete_selected_datasets(current_data):
	"""Delete selected datasets"""
	if not current_data:
		return "No items to delete", *refresh_all_data()

	deleted_count = 0
	errors = []

	for row in current_data:
		if row[0]:  # If selected
			dataset_path = row[5]  # Path column
			try:
				if os.path.exists(dataset_path):
					shutil.rmtree(dataset_path)
					deleted_count += 1
					logger.info(f"Deleted dataset: {dataset_path}", __name__)
			except Exception as e:
				errors.append(f"Failed to delete {row[1]}: {str(e)}")

	message = f"✅ Deleted {deleted_count} dataset(s)"
	if errors:
		message += f"\n\n❌ Errors:\n" + "\n".join(errors)

	return (message, *refresh_all_data())


def pre_check() -> bool:
	"""Pre-check for purge tab"""
	return True


def render() -> gradio.Blocks:
	"""Render the Purge tab layout"""
	global PURGE_REFRESH_BUTTON, PURGE_TOTAL_SIZE
	global IDENTITY_PROFILES_DATAFRAME, IDENTITY_SELECT_ALL_BUTTON, IDENTITY_SELECT_NONE_BUTTON
	global IDENTITY_DELETE_BUTTON, IDENTITY_SECTION_SIZE
	global LORA_DATAFRAME, LORA_SELECT_ALL_BUTTON, LORA_SELECT_NONE_BUTTON
	global LORA_DELETE_BUTTON, LORA_SECTION_SIZE
	global DATASET_DATAFRAME, DATASET_SELECT_ALL_BUTTON, DATASET_SELECT_NONE_BUTTON
	global DATASET_DELETE_BUTTON, DATASET_SECTION_SIZE

	# Initial data
	identity_data, identity_size_str = scan_identity_profiles()
	lora_data, lora_size_str = scan_lora_models()
	dataset_data, dataset_size_str = scan_training_datasets()

	# Calculate total
	identity_total = get_directory_size('models/identities')
	lora_total = get_directory_size('.assets/models/trained') + get_directory_size('.jobs/training_dataset_lora')
	dataset_total = sum([get_directory_size(row[5]) for row in dataset_data])
	total = identity_total + lora_total + dataset_total

	with gradio.Blocks() as layout:
		about.render()

		gradio.Markdown("## 🗑️ Data Management & Cleanup")

		with gradio.Row():
			PURGE_TOTAL_SIZE = gradio.Markdown(f"### Total Storage Used: {format_size(total)}")
			PURGE_REFRESH_BUTTON = gradio.Button("🔄 Refresh All", variant="secondary")

		# Identity Profiles Section
		with gradio.Accordion("👤 Identity Profiles", open=True):
			IDENTITY_SECTION_SIZE = gradio.Markdown(identity_size_str)

			IDENTITY_PROFILES_DATAFRAME = gradio.Dataframe(
				value=identity_data,
				headers=["Select", "Name", "ID", "Created", "Size", "Path"],
				datatype=["bool", "str", "str", "str", "str", "str"],
				col_count=(6, "fixed"),
				interactive=True,
				wrap=True
			)

			with gradio.Row():
				IDENTITY_SELECT_ALL_BUTTON = gradio.Button("Select All", size="sm")
				IDENTITY_SELECT_NONE_BUTTON = gradio.Button("Select None", size="sm")
				IDENTITY_DELETE_BUTTON = gradio.Button("🗑️ Delete Selected", variant="stop", size="sm")

		# LoRA Models Section
		with gradio.Accordion("🎯 LoRA Models", open=True):
			LORA_SECTION_SIZE = gradio.Markdown(lora_size_str)

			LORA_DATAFRAME = gradio.Dataframe(
				value=lora_data,
				headers=["Select", "Name", "Location", "Epoch", "Loss", "Rank", "Size", "Path"],
				datatype=["bool", "str", "str", "str", "str", "str", "str", "str"],
				col_count=(8, "fixed"),
				interactive=True,
				wrap=True
			)

			with gradio.Row():
				LORA_SELECT_ALL_BUTTON = gradio.Button("Select All", size="sm")
				LORA_SELECT_NONE_BUTTON = gradio.Button("Select None", size="sm")
				LORA_DELETE_BUTTON = gradio.Button("🗑️ Delete Selected", variant="stop", size="sm")

		# Training Datasets Section
		with gradio.Accordion("📂 Training Datasets", open=True):
			DATASET_SECTION_SIZE = gradio.Markdown(dataset_size_str)

			DATASET_DATAFRAME = gradio.Dataframe(
				value=dataset_data,
				headers=["Select", "Name", "Frames", "Modified", "Size", "Path"],
				datatype=["bool", "str", "str", "str", "str", "str"],
				col_count=(6, "fixed"),
				interactive=True,
				wrap=True
			)

			with gradio.Row():
				DATASET_SELECT_ALL_BUTTON = gradio.Button("Select All", size="sm")
				DATASET_SELECT_NONE_BUTTON = gradio.Button("Select None", size="sm")
				DATASET_DELETE_BUTTON = gradio.Button("🗑️ Delete Selected", variant="stop", size="sm")

		# Status output
		status_output = gradio.Textbox(label="Status", value="", visible=False)

		footer.render()

	return layout


def listen() -> None:
	"""Set up event listeners"""
	# Refresh all data
	PURGE_REFRESH_BUTTON.click(
		refresh_all_data,
		outputs=[
			IDENTITY_PROFILES_DATAFRAME, IDENTITY_SECTION_SIZE,
			LORA_DATAFRAME, LORA_SECTION_SIZE,
			DATASET_DATAFRAME, DATASET_SECTION_SIZE,
			PURGE_TOTAL_SIZE
		]
	)

	# Identity profile actions
	IDENTITY_SELECT_ALL_BUTTON.click(
		select_all_identity,
		inputs=[IDENTITY_PROFILES_DATAFRAME],
		outputs=[IDENTITY_PROFILES_DATAFRAME]
	)

	IDENTITY_SELECT_NONE_BUTTON.click(
		select_none_identity,
		inputs=[IDENTITY_PROFILES_DATAFRAME],
		outputs=[IDENTITY_PROFILES_DATAFRAME]
	)

	IDENTITY_DELETE_BUTTON.click(
		delete_selected_identities,
		inputs=[IDENTITY_PROFILES_DATAFRAME],
		outputs=[
			PURGE_TOTAL_SIZE,  # Status message goes in total size temporarily
			IDENTITY_PROFILES_DATAFRAME, IDENTITY_SECTION_SIZE,
			LORA_DATAFRAME, LORA_SECTION_SIZE,
			DATASET_DATAFRAME, DATASET_SECTION_SIZE,
			PURGE_TOTAL_SIZE
		]
	)

	# LoRA model actions
	LORA_SELECT_ALL_BUTTON.click(
		select_all_lora,
		inputs=[LORA_DATAFRAME],
		outputs=[LORA_DATAFRAME]
	)

	LORA_SELECT_NONE_BUTTON.click(
		select_none_lora,
		inputs=[LORA_DATAFRAME],
		outputs=[LORA_DATAFRAME]
	)

	LORA_DELETE_BUTTON.click(
		delete_selected_loras,
		inputs=[LORA_DATAFRAME],
		outputs=[
			PURGE_TOTAL_SIZE,
			IDENTITY_PROFILES_DATAFRAME, IDENTITY_SECTION_SIZE,
			LORA_DATAFRAME, LORA_SECTION_SIZE,
			DATASET_DATAFRAME, DATASET_SECTION_SIZE,
			PURGE_TOTAL_SIZE
		]
	)

	# Dataset actions
	DATASET_SELECT_ALL_BUTTON.click(
		select_all_datasets,
		inputs=[DATASET_DATAFRAME],
		outputs=[DATASET_DATAFRAME]
	)

	DATASET_SELECT_NONE_BUTTON.click(
		select_none_datasets,
		inputs=[DATASET_DATAFRAME],
		outputs=[DATASET_DATAFRAME]
	)

	DATASET_DELETE_BUTTON.click(
		delete_selected_datasets,
		inputs=[DATASET_DATAFRAME],
		outputs=[
			PURGE_TOTAL_SIZE,
			IDENTITY_PROFILES_DATAFRAME, IDENTITY_SECTION_SIZE,
			LORA_DATAFRAME, LORA_SECTION_SIZE,
			DATASET_DATAFRAME, DATASET_SECTION_SIZE,
			PURGE_TOTAL_SIZE
		]
	)


def run(ui: gradio.Blocks) -> None:
	"""Run the purge UI"""
	ui.launch(
		inbrowser=state_manager.get_item('open_browser'),
		server_name=state_manager.get_item('server_name'),
		server_port=state_manager.get_item('server_port'),
		show_error=True
	)
