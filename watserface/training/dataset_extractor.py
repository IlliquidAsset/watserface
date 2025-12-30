import os
import cv2
import json
import numpy
from typing import List, Dict, Any, Optional

from watserface import logger, state_manager
from watserface.filesystem import is_video
from watserface.vision import read_video_frame, count_video_frame_total, read_image, write_image
from watserface.face_analyser import get_many_faces

def extract_training_dataset(
	source_paths: List[str],
	output_dir: str,
	frame_interval: int = 1,
	max_frames: int = 100,
	progress: Any = None
):
	"""
	Extracts frames and landmarks from source paths (video/image) for training.
	Yields stats dictionary on every update.
	"""
	stats = {'frames_extracted': 0, 'landmarks_saved': 0, 'current_file': ''}
	
	for source_path in source_paths:
		stats['current_file'] = os.path.basename(source_path)
		if is_video(source_path):
			yield from _extract_video_gen(source_path, output_dir, frame_interval, max_frames, stats, progress)
		else:
			yield from _extract_image_gen(source_path, output_dir, stats, progress)
			
	yield stats

def _extract_video_gen(video_path: str, output_dir: str, interval: int, max_frames: int, stats: Dict[str, Any], progress: Any):
	total_frames = count_video_frame_total(video_path)
	step = max(1, interval)
	frames_to_extract = range(0, total_frames, step)
	
	if len(frames_to_extract) > max_frames:
		step = total_frames // max_frames
		frames_to_extract = range(0, total_frames, step)

	for frame_number in frames_to_extract:
		if progress is not None:
			progress((frame_number / total_frames) * 0.3, desc=f"Processing {stats['current_file']} frame {frame_number}/{total_frames}")
			
		frame = read_video_frame(video_path, frame_number)
		if frame is not None:
			faces = get_many_faces([frame])
			if faces:
				frame_filename = f"frame_{stats['frames_extracted']:06d}.png"
				write_image(os.path.join(output_dir, frame_filename), frame)
				stats['frames_extracted'] += 1
				
				face = faces[0]
				landmark_filename = f"frame_{stats['frames_extracted']-1:06d}.json"
				
				landmarks = {}
				for key, val in face.landmark_set.items():
					if val is not None:
						landmarks[key] = val.tolist()
				
				with open(os.path.join(output_dir, landmark_filename), 'w') as f:
					json.dump(landmarks, f)
				stats['landmarks_saved'] += 1
				if stats['frames_extracted'] % 5 == 0:
					yield stats

def _extract_image_gen(image_path: str, output_dir: str, stats: Dict[str, Any], progress: Any):
	frame = read_image(image_path)
	if frame is not None:
		faces = get_many_faces([frame])
		if faces:
			frame_filename = f"frame_{stats['frames_extracted']:06d}.png"
			write_image(os.path.join(output_dir, frame_filename), frame)
			stats['frames_extracted'] += 1
			
			face = faces[0]
			landmark_filename = f"frame_{stats['frames_extracted']-1:06d}.json"
			landmarks = {}
			for key, val in face.landmark_set.items():
				if val is not None:
					landmarks[key] = val.tolist()
			
			with open(os.path.join(output_dir, landmark_filename), 'w') as f:
				json.dump(landmarks, f)
			stats['landmarks_saved'] += 1
			yield stats

def _extract_video(video_path: str, output_dir: str, interval: int, max_frames: int, stats: Dict[str, int], progress: Any):
	total_frames = count_video_frame_total(video_path)
	step = max(1, interval)
	
	# Limit frames extracted
	frames_to_extract = range(0, total_frames, step)
	if len(frames_to_extract) > max_frames:
		step = total_frames // max_frames
		frames_to_extract = range(0, total_frames, step)

	for frame_number in frames_to_extract:
		if progress is not None:
			progress((frame_number / total_frames) * 0.3, desc=f"Processing frame {frame_number}/{total_frames}")
			
		frame = read_video_frame(video_path, frame_number)
		if frame is not None:
			# Detect faces
			faces = get_many_faces([frame])
			if faces:
				# Save frame
				frame_filename = f"frame_{stats['frames_extracted']:06d}.png"
				write_image(os.path.join(output_dir, frame_filename), frame)
				stats['frames_extracted'] += 1
				
				# Save landmarks (first face)
				face = faces[0]
				landmark_filename = f"frame_{stats['frames_extracted']-1:06d}.json"
				
				# Convert numpy arrays to list for JSON
				landmarks = {}
				for key, val in face.landmark_set.items():
					if val is not None:
						landmarks[key] = val.tolist()
				
				with open(os.path.join(output_dir, landmark_filename), 'w') as f:
					json.dump(landmarks, f)
				stats['landmarks_saved'] += 1

def _extract_image(image_path: str, output_dir: str, stats: Dict[str, int], progress: Any):
	frame = read_image(image_path)
	if frame is not None:
		faces = get_many_faces([frame])
		if faces:
			frame_filename = f"frame_{stats['frames_extracted']:06d}.png"
			write_image(os.path.join(output_dir, frame_filename), frame)
			stats['frames_extracted'] += 1
			
			face = faces[0]
			landmark_filename = f"frame_{stats['frames_extracted']-1:06d}.json"
			landmarks = {}
			for key, val in face.landmark_set.items():
				if val is not None:
					landmarks[key] = val.tolist()
			
			with open(os.path.join(output_dir, landmark_filename), 'w') as f:
				json.dump(landmarks, f)
			stats['landmarks_saved'] += 1