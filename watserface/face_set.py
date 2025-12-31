"""
Face Set Manager - Reusable Collections of Extracted Frames

This module provides the Face Set abstraction for separating frame extraction
from identity training. Face Sets are reusable collections of extracted frames
with landmarks that can be trained multiple times without re-uploading videos.

Pattern follows: watserface/identity_profile.py (SourceIntelligence)
"""

import os
import shutil
import hashlib
import json
import time
import cv2
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Iterator, Tuple

from watserface import logger
from watserface.training.dataset_extractor import extract_training_dataset
from watserface.training.landmark_smoother import apply_smoothing_to_dataset


# ==================== DATACLASSES ====================

@dataclass
class FaceSetConfig:
    """Configuration for face set extraction"""
    frame_interval: int = 2
    max_frames: int = 1000
    min_face_confidence: float = 0.5
    auto_smooth: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceSetConfig':
        return cls(**data)


@dataclass
class ExtractionStats:
    """Statistics from extraction process"""
    total_frames_processed: int = 0
    frames_with_faces: int = 0
    frames_extracted: int = 0
    landmarks_saved: int = 0
    avg_face_confidence: float = 0.0
    processing_time: float = 0.0
    source_videos: List[str] = field(default_factory=list)
    source_images: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionStats':
        return cls(**data)


@dataclass
class FaceSetMetadata:
    """Metadata for a Face Set - stored as faceset.json"""
    id: str
    name: str
    created_at: str
    last_used: Optional[str] = None
    description: Optional[str] = None

    # Source information
    source_files: List[str] = field(default_factory=list)
    source_count: int = 0

    # Extraction configuration
    extraction_config: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)

    # Frame information
    frame_count: int = 0
    frames_directory: str = "frames/"
    landmarks_directory: str = "landmarks/"

    # Quality metrics
    smoothing_applied: bool = False
    thumbnail_path: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    training_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceSetMetadata':
        return cls(**data)


# ==================== FACE SET MANAGER ====================

class FaceSetManager:
    """Manager for Face Set creation, loading, and lifecycle management"""

    def __init__(self, config: Optional[FaceSetConfig] = None):
        self.config = config or FaceSetConfig()
        self.face_sets_dir = Path("models/face_sets")
        self.face_sets_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FaceSetManager initialized. Face sets directory: {self.face_sets_dir}", __name__)

    # ==================== CREATION OPERATIONS ====================

    def create_face_set(
        self,
        source_paths: List[str],
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        progress: Any = None
    ) -> FaceSetMetadata:
        """
        Create a new Face Set from source files (videos/images)

        This extracts frames and landmarks, applies smoothing, and saves metadata.
        Yields progress updates during extraction.

        Args:
            source_paths: List of video/image file paths to extract from
            name: User-friendly name for the Face Set
            description: Optional description
            tags: Optional list of tags for organization
            progress: Optional Gradio progress callback

        Returns:
            FaceSetMetadata for the created Face Set

        Raises:
            ValueError: If no frames were extracted
        """
        logger.info(f"Creating Face Set '{name}' from {len(source_paths)} sources", __name__)

        # Generate unique ID
        source_hash = hashlib.md5(str(sorted(source_paths)).encode()).hexdigest()[:8]
        timestamp = int(time.time())
        face_set_id = f"faceset_{source_hash}_{timestamp}"

        # Create directories
        face_set_dir = self.face_sets_dir / face_set_id
        frames_dir = face_set_dir / "frames"
        landmarks_dir = face_set_dir / "landmarks"

        frames_dir.mkdir(parents=True, exist_ok=True)
        landmarks_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created Face Set directory: {face_set_dir}", __name__)

        # Extract frames using existing dataset_extractor
        start_time = time.time()
        extraction_stats = {'frames_extracted': 0, 'landmarks_saved': 0}

        for stats in extract_training_dataset(
            source_paths=source_paths,
            output_dir=str(frames_dir),  # Extract directly to Face Set directory
            frame_interval=self.config.frame_interval,
            max_frames=self.config.max_frames,
            progress=progress
        ):
            extraction_stats = stats
            logger.debug(f"Extraction progress: {stats.get('frames_extracted', 0)} frames", __name__)

        # Move landmarks to separate directory
        self._reorganize_landmarks(frames_dir, landmarks_dir)

        # Apply smoothing if configured
        if self.config.auto_smooth:
            logger.info("Applying landmark smoothing...", __name__)
            apply_smoothing_to_dataset(str(landmarks_dir))

        # Create thumbnail from first frame
        thumbnail_path = self._create_thumbnail(frames_dir, face_set_dir)

        # Count final frames
        frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix == '.png'])
        if len(frame_files) == 0:
            logger.error("No frames extracted - Face Set creation failed", __name__)
            shutil.rmtree(face_set_dir)
            raise ValueError("No frames were extracted from source files")

        # Create metadata
        metadata = FaceSetMetadata(
            id=face_set_id,
            name=name,
            created_at=datetime.now().isoformat(),
            description=description,
            source_files=source_paths,
            source_count=len(source_paths),
            extraction_config=self.config.to_dict(),
            stats={
                'total_frames_processed': extraction_stats.get('frames_extracted', 0),
                'frames_extracted': len(frame_files),
                'landmarks_saved': len([f for f in landmarks_dir.iterdir() if f.suffix == '.json']),
                'processing_time': time.time() - start_time
            },
            frame_count=len(frame_files),
            frames_directory="frames/",
            landmarks_directory="landmarks/",
            smoothing_applied=self.config.auto_smooth,
            thumbnail_path="thumbnail.png" if thumbnail_path else None,
            tags=tags or [],
            training_history=[]
        )

        # Save metadata
        self._save_metadata(metadata)

        logger.info(f"✅ Face Set created: {face_set_id} ({metadata.frame_count} frames)", __name__)
        return metadata

    def create_from_existing_dataset(
        self,
        dataset_path: str,
        name: str,
        source_files: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> FaceSetMetadata:
        """
        Create Face Set from existing .jobs/training_dataset_identity/
        Used for migration of legacy datasets.

        Args:
            dataset_path: Path to existing dataset directory
            name: Name for the Face Set
            source_files: Optional list of source files (may be unknown for legacy)
            description: Optional description

        Returns:
            FaceSetMetadata for the created Face Set

        Raises:
            ValueError: If dataset_path doesn't exist or contains no frames
        """
        logger.info(f"Migrating legacy dataset from {dataset_path} to Face Set '{name}'", __name__)

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        # Generate unique ID
        timestamp = int(time.time())
        face_set_id = f"faceset_migrated_{timestamp}"

        # Create directories
        face_set_dir = self.face_sets_dir / face_set_id
        frames_dir = face_set_dir / "frames"
        landmarks_dir = face_set_dir / "landmarks"

        face_set_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        landmarks_dir.mkdir(parents=True, exist_ok=True)

        # Copy frames and reorganize landmarks
        legacy_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(('.png', '.json'))])

        frames_copied = 0
        landmarks_copied = 0

        for file in legacy_files:
            src = os.path.join(dataset_path, file)

            if file.endswith('.png'):
                # Copy frame
                dst = frames_dir / file
                shutil.copy(src, dst)
                frames_copied += 1

            elif file.endswith('.json'):
                # Copy landmark
                dst = landmarks_dir / file
                shutil.copy(src, dst)
                landmarks_copied += 1

        if frames_copied == 0:
            logger.error("No frames found in legacy dataset", __name__)
            shutil.rmtree(face_set_dir)
            raise ValueError("No frames found in legacy dataset")

        # Create thumbnail
        thumbnail_path = self._create_thumbnail(frames_dir, face_set_dir)

        # Create metadata
        metadata = FaceSetMetadata(
            id=face_set_id,
            name=name,
            created_at=datetime.now().isoformat(),
            description=description or "Migrated from legacy training dataset",
            source_files=source_files or [],
            source_count=len(source_files) if source_files else 0,
            extraction_config=self.config.to_dict(),
            stats={
                'total_frames_processed': frames_copied,
                'frames_extracted': frames_copied,
                'landmarks_saved': landmarks_copied,
                'processing_time': 0.0  # Unknown for migration
            },
            frame_count=frames_copied,
            frames_directory="frames/",
            landmarks_directory="landmarks/",
            smoothing_applied=False,  # Unknown for legacy
            thumbnail_path="thumbnail.png" if thumbnail_path else None,
            tags=["migrated"],
            training_history=[]
        )

        # Save metadata
        self._save_metadata(metadata)

        logger.info(f"✅ Legacy dataset migrated to Face Set: {face_set_id} ({metadata.frame_count} frames)", __name__)
        return metadata

    # ==================== LOADING OPERATIONS ====================

    def load_face_set(self, face_set_id: str) -> Optional[FaceSetMetadata]:
        """
        Load Face Set metadata by ID

        Args:
            face_set_id: Unique Face Set ID

        Returns:
            FaceSetMetadata if found, None otherwise
        """
        metadata_path = self.face_sets_dir / face_set_id / "faceset.json"

        if not metadata_path.exists():
            logger.warn(f"Face Set metadata not found: {face_set_id}", __name__)
            return None

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)

            metadata = FaceSetMetadata.from_dict(data)

            # Update last_used
            metadata.last_used = datetime.now().isoformat()
            self._save_metadata(metadata)

            logger.debug(f"Loaded Face Set: {face_set_id}", __name__)
            return metadata

        except Exception as e:
            logger.error(f"Failed to load Face Set metadata: {e}", __name__)
            return None

    def get_face_set_frames_path(self, face_set_id: str) -> str:
        """Get absolute path to frames directory for a Face Set"""
        return str((self.face_sets_dir / face_set_id / "frames").resolve())

    def get_face_set_landmarks_path(self, face_set_id: str) -> str:
        """Get absolute path to landmarks directory for a Face Set"""
        return str((self.face_sets_dir / face_set_id / "landmarks").resolve())

    # ==================== LISTING AND SEARCH ====================

    def list_face_sets(
        self,
        tags: Optional[List[str]] = None,
        sort_by: str = 'created_at'  # 'created_at', 'last_used', 'name', 'frame_count'
    ) -> List[FaceSetMetadata]:
        """
        List all Face Sets, optionally filtered by tags

        Args:
            tags: Optional list of tags to filter by (any match)
            sort_by: Sort criterion

        Returns:
            List of FaceSetMetadata, sorted according to sort_by
        """
        face_sets = []

        for face_set_dir in self.face_sets_dir.iterdir():
            if not face_set_dir.is_dir():
                continue

            metadata_path = face_set_dir / "faceset.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)

                metadata = FaceSetMetadata.from_dict(data)

                # Filter by tags if specified
                if tags:
                    if not any(tag in metadata.tags for tag in tags):
                        continue

                face_sets.append(metadata)

            except Exception as e:
                logger.error(f"Failed to load Face Set from {face_set_dir}: {e}", __name__)
                continue

        # Sort
        if sort_by == 'name':
            face_sets.sort(key=lambda fs: fs.name.lower())
        elif sort_by == 'last_used':
            face_sets.sort(key=lambda fs: fs.last_used or '', reverse=True)
        elif sort_by == 'frame_count':
            face_sets.sort(key=lambda fs: fs.frame_count, reverse=True)
        else:  # created_at
            face_sets.sort(key=lambda fs: fs.created_at, reverse=True)

        logger.debug(f"Listed {len(face_sets)} Face Sets (filtered by {tags})", __name__)
        return face_sets

    def search_face_sets(self, query: str) -> List[FaceSetMetadata]:
        """
        Search Face Sets by name or description

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching FaceSetMetadata
        """
        query_lower = query.lower()
        all_face_sets = self.list_face_sets()

        matches = [
            fs for fs in all_face_sets
            if query_lower in fs.name.lower() or
            (fs.description and query_lower in fs.description.lower())
        ]

        logger.debug(f"Found {len(matches)} Face Sets matching '{query}'", __name__)
        return matches

    # ==================== UPDATE OPERATIONS ====================

    def update_face_set_metadata(
        self,
        face_set_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update Face Set metadata (name, description, tags)

        Args:
            face_set_id: Face Set ID to update
            name: New name (optional)
            description: New description (optional)
            tags: New tags list (optional)

        Returns:
            True if successful, False otherwise
        """
        metadata = self.load_face_set(face_set_id)
        if not metadata:
            logger.error(f"Cannot update: Face Set not found: {face_set_id}", __name__)
            return False

        # Update fields
        if name is not None:
            metadata.name = name
        if description is not None:
            metadata.description = description
        if tags is not None:
            metadata.tags = tags

        # Save
        self._save_metadata(metadata)
        logger.info(f"Updated Face Set metadata: {face_set_id}", __name__)
        return True

    def record_training_use(
        self,
        face_set_id: str,
        model_name: str,
        epochs: int
    ) -> None:
        """
        Record that this Face Set was used for training

        Args:
            face_set_id: Face Set ID
            model_name: Name of model trained
            epochs: Number of epochs trained
        """
        metadata = self.load_face_set(face_set_id)
        if not metadata:
            logger.warn(f"Cannot record training use: Face Set not found: {face_set_id}", __name__)
            return

        # Add to training history
        metadata.training_history.append({
            'model_name': model_name,
            'trained_at': datetime.now().isoformat(),
            'epochs': epochs
        })

        # Save
        self._save_metadata(metadata)
        logger.debug(f"Recorded training use for Face Set {face_set_id}: {model_name} ({epochs} epochs)", __name__)

    # ==================== DELETION OPERATIONS ====================

    def delete_face_set(self, face_set_id: str) -> bool:
        """
        Delete a Face Set and all associated files

        Args:
            face_set_id: Face Set ID to delete

        Returns:
            True if successful, False otherwise
        """
        face_set_dir = self.face_sets_dir / face_set_id

        if not face_set_dir.exists():
            logger.warn(f"Cannot delete: Face Set not found: {face_set_id}", __name__)
            return False

        try:
            shutil.rmtree(face_set_dir)
            logger.info(f"✅ Deleted Face Set: {face_set_id}", __name__)
            return True

        except Exception as e:
            logger.error(f"Failed to delete Face Set {face_set_id}: {e}", __name__)
            return False

    def cleanup_unused_face_sets(self, days_unused: int = 90) -> int:
        """
        Clean up Face Sets not used in training for N days

        Args:
            days_unused: Delete Face Sets not used in this many days

        Returns:
            Number of Face Sets deleted
        """
        logger.info(f"Cleaning up Face Sets not used in {days_unused} days...", __name__)

        cutoff = datetime.now().timestamp() - (days_unused * 24 * 60 * 60)
        deleted_count = 0

        for metadata in self.list_face_sets():
            last_used_timestamp = datetime.fromisoformat(metadata.last_used or metadata.created_at).timestamp()

            if last_used_timestamp < cutoff:
                if self.delete_face_set(metadata.id):
                    deleted_count += 1

        logger.info(f"✅ Cleaned up {deleted_count} unused Face Sets", __name__)
        return deleted_count

    # ==================== EXPORT/IMPORT OPERATIONS ====================

    def export_face_set(self, face_set_id: str, export_path: Optional[str] = None) -> Optional[str]:
        """
        Export Face Set as a portable zip archive

        Args:
            face_set_id: Face Set ID to export
            export_path: Optional custom export path (default: ./face_set_exports/)

        Returns:
            Path to exported zip file if successful, None otherwise
        """
        metadata = self.load_face_set(face_set_id)
        if not metadata:
            logger.error(f"Cannot export: Face Set not found: {face_set_id}", __name__)
            return None

        face_set_dir = self.face_sets_dir / face_set_id

        # Create export directory
        if export_path:
            export_dir = Path(export_path)
        else:
            export_dir = Path("face_set_exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Create zip filename
        safe_name = metadata.name.replace(" ", "_").replace("/", "_")
        timestamp = int(datetime.now().timestamp())
        zip_filename = f"{safe_name}_{face_set_id}_{timestamp}.zip"
        zip_path = export_dir / zip_filename

        try:
            import zipfile

            logger.info(f"Exporting Face Set '{metadata.name}' to {zip_path}...", __name__)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from Face Set directory
                for file_path in face_set_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(face_set_dir)
                        zipf.write(file_path, arcname)

            logger.info(f"✅ Face Set exported successfully: {zip_path}", __name__)
            return str(zip_path)

        except Exception as e:
            logger.error(f"Failed to export Face Set: {e}", __name__)
            return None

    def import_face_set(self, zip_path: str, new_name: Optional[str] = None) -> Optional[FaceSetMetadata]:
        """
        Import Face Set from exported zip archive

        Args:
            zip_path: Path to exported Face Set zip file
            new_name: Optional new name for imported Face Set (keeps original if not provided)

        Returns:
            FaceSetMetadata if successful, None otherwise
        """
        if not os.path.exists(zip_path):
            logger.error(f"Cannot import: File not found: {zip_path}", __name__)
            return None

        try:
            import zipfile

            logger.info(f"Importing Face Set from {zip_path}...", __name__)

            # Create temporary directory for extraction
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    zipf.extractall(temp_path)

                # Load metadata
                metadata_path = temp_path / "faceset.json"
                if not metadata_path.exists():
                    logger.error("Invalid Face Set archive: faceset.json not found", __name__)
                    return None

                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)

                # Generate new ID to avoid conflicts
                original_id = metadata_dict['id']
                new_id = self._generate_face_set_id(metadata_dict['name'])
                metadata_dict['id'] = new_id

                # Update name if provided
                if new_name:
                    metadata_dict['name'] = new_name

                # Reset timestamps and history
                metadata_dict['created_at'] = datetime.now().isoformat()
                metadata_dict['last_used'] = None
                metadata_dict['training_history'] = []

                # Create new Face Set directory
                new_face_set_dir = self.face_sets_dir / new_id
                if new_face_set_dir.exists():
                    logger.error(f"Face Set already exists: {new_id}", __name__)
                    return None

                new_face_set_dir.mkdir(parents=True, exist_ok=True)

                # Copy all files to new directory
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(temp_path)
                        dest_path = new_face_set_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest_path)

                # Save updated metadata
                new_metadata = FaceSetMetadata.from_dict(metadata_dict)
                self._save_metadata(new_metadata)

                logger.info(f"✅ Face Set imported successfully: {new_metadata.name} ({new_id})", __name__)
                return new_metadata

        except Exception as e:
            logger.error(f"Failed to import Face Set: {e}", __name__)
            import traceback
            traceback.print_exc()
            return None

    # ==================== UTILITY OPERATIONS ====================

    def get_face_set_summary(self, face_set: FaceSetMetadata) -> str:
        """Get human-readable summary of Face Set"""
        return (
            f"{face_set.name}\n"
            f"  ID: {face_set.id}\n"
            f"  Frames: {face_set.frame_count}\n"
            f"  Created: {face_set.created_at}\n"
            f"  Tags: {', '.join(face_set.tags) if face_set.tags else 'None'}\n"
            f"  Used in {len(face_set.training_history)} trainings"
        )

    # ==================== PRIVATE HELPERS ====================

    def _save_metadata(self, metadata: FaceSetMetadata) -> None:
        """Save Face Set metadata to disk"""
        face_set_dir = self.face_sets_dir / metadata.id
        metadata_path = face_set_dir / "faceset.json"

        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.debug(f"Saved Face Set metadata: {metadata.id}", __name__)

    def _reorganize_landmarks(self, frames_dir: Path, landmarks_dir: Path) -> None:
        """
        Move landmark JSON files from frames directory to landmarks directory

        The dataset_extractor saves frames and landmarks together. This helper
        reorganizes them into separate directories for cleaner organization.
        """
        logger.debug("Reorganizing landmarks into separate directory...", __name__)

        landmark_files = sorted([f for f in frames_dir.iterdir() if f.suffix == '.json'])

        for landmark_file in landmark_files:
            dst = landmarks_dir / landmark_file.name
            shutil.move(str(landmark_file), str(dst))

        logger.debug(f"Moved {len(landmark_files)} landmark files", __name__)

    def _create_thumbnail(self, frames_dir: Path, face_set_dir: Path) -> Optional[str]:
        """
        Create thumbnail from first frame

        Args:
            frames_dir: Directory containing frames
            face_set_dir: Face Set root directory

        Returns:
            Thumbnail filename if successful, None otherwise
        """
        logger.debug("Creating Face Set thumbnail...", __name__)

        frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix == '.png'])
        if not frame_files:
            logger.warn("No frames available for thumbnail creation", __name__)
            return None

        try:
            # Read first frame
            first_frame = cv2.imread(str(frame_files[0]))

            # Resize to thumbnail size (256x256)
            height, width = first_frame.shape[:2]
            size = min(height, width)
            x = (width - size) // 2
            y = (height - size) // 2
            cropped = first_frame[y:y+size, x:x+size]
            thumbnail = cv2.resize(cropped, (256, 256))

            # Save thumbnail
            thumbnail_path = face_set_dir / "thumbnail.png"
            cv2.imwrite(str(thumbnail_path), thumbnail)

            logger.debug(f"Created thumbnail: {thumbnail_path}", __name__)
            return "thumbnail.png"

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}", __name__)
            return None


# ==================== MIGRATION & CLEANUP ====================

def migrate_and_cleanup_legacy_datasets(face_set_manager: 'FaceSetManager', jobs_path: str = '.jobs') -> None:
    """
    Migrate legacy training datasets to Face Sets and clean up old failed trainings.

    This function is called once on first deployment to:
    1. Migrate existing Samantha dataset to "Samantha_Migrated" Face Set
    2. Clean up old failed training datasets from .jobs/

    Args:
        face_set_manager: FaceSetManager instance to use
        jobs_path: Path to .jobs directory (default: '.jobs')
    """
    # Check if migration already done
    migration_marker = Path(jobs_path) / '.face_set_migration_complete'
    if migration_marker.exists():
        logger.debug("Migration already completed, skipping...", __name__)
        return

    logger.info("Starting Face Set migration and cleanup...", __name__)

    # Ensure .jobs exists
    if not os.path.exists(jobs_path):
        logger.info(f"No {jobs_path} directory found, skipping migration", __name__)
        os.makedirs(jobs_path, exist_ok=True)
        migration_marker.touch()
        return

    migrated_count = 0

    # 1. Find training_dataset_identity directory (current Samantha dataset)
    current_dataset = os.path.join(jobs_path, 'training_dataset_identity')

    if os.path.exists(current_dataset):
        frames = [f for f in os.listdir(current_dataset) if f.endswith('.png')]

        if len(frames) > 0:
            logger.info(f"Found existing dataset with {len(frames)} frames, migrating to Face Set...", __name__)

            try:
                # Migrate to Face Set
                face_set = face_set_manager.create_from_existing_dataset(
                    dataset_path=current_dataset,
                    name="Samantha_Migrated",
                    source_files=[],  # Unknown source
                    description="Migrated from existing Samantha training dataset"
                )

                logger.info(f"✅ Migrated Samantha dataset to Face Set: {face_set.id} ({face_set.frame_count} frames)", __name__)
                migrated_count += 1

                # Delete old directory
                shutil.rmtree(current_dataset)
                logger.info(f"🗑️  Cleaned up legacy dataset: {current_dataset}", __name__)

            except Exception as e:
                logger.error(f"Failed to migrate Samantha dataset: {e}", __name__)

    # 2. Clean up other old .jobs/ datasets from failed trainings
    cleanup_count = 0
    for item in os.listdir(jobs_path):
        item_path = os.path.join(jobs_path, item)

        # Remove old training dataset directories
        if os.path.isdir(item_path) and 'training_dataset' in item:
            try:
                logger.info(f"🗑️  Removing old training dataset: {item_path}", __name__)
                shutil.rmtree(item_path)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {item_path}: {e}", __name__)

    logger.info(f"✅ Migration complete! Migrated: {migrated_count}, Cleaned up: {cleanup_count}", __name__)

    # Mark migration as complete
    migration_marker.touch()


# ==================== SINGLETON INSTANCE ====================

_face_set_manager_instance: Optional[FaceSetManager] = None


def get_face_set_manager(config: Optional[FaceSetConfig] = None) -> FaceSetManager:
    """Get or create singleton FaceSetManager instance"""
    global _face_set_manager_instance

    if _face_set_manager_instance is None:
        _face_set_manager_instance = FaceSetManager(config)

        # Run migration on first initialization
        try:
            from watserface import state_manager
            jobs_path = state_manager.get_item('jobs_path')
            if jobs_path:
                migrate_and_cleanup_legacy_datasets(_face_set_manager_instance, jobs_path)
        except Exception as e:
            logger.error(f"Failed to run migration: {e}", __name__)

    return _face_set_manager_instance
