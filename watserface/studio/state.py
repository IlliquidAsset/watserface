from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class StudioPhase(Enum):
    IDLE = 'idle'
    IDENTITY_BUILDING = 'identity_building'
    OCCLUSION_TRAINING = 'occlusion_training'
    MAPPING = 'mapping'
    EXECUTING = 'executing'
    REVIEWING = 'reviewing'


@dataclass
class IdentityProfile:
    name: str
    source_paths: List[str] = field(default_factory=list)
    face_set_id: Optional[str] = None
    model_path: Optional[str] = None
    epochs_trained: int = 0
    last_loss: float = 0.0
    embedding: Optional[Any] = None


@dataclass
class OcclusionProfile:
    name: str
    target_path: Optional[str] = None
    model_path: Optional[str] = None
    epochs_trained: int = 0
    last_loss: float = 0.0
    has_depth: bool = False
    depth_map_path: Optional[str] = None


@dataclass
class FaceMapping:
    source_identity: str
    target_face_index: int
    confidence: float = 0.0
    quality_score: float = 0.0


@dataclass
class StudioState:
    phase: StudioPhase = StudioPhase.IDLE
    
    identities: Dict[str, IdentityProfile] = field(default_factory=dict)
    occlusions: Dict[str, OcclusionProfile] = field(default_factory=dict)
    mappings: List[FaceMapping] = field(default_factory=list)
    
    target_path: Optional[str] = None
    output_path: Optional[str] = None
    
    quality_threshold: float = 0.7
    auto_train_on_low_quality: bool = True
    
    log_history: List[str] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def log(self, message: str) -> str:
        timestamp = time.strftime('%H:%M:%S')
        entry = f'[{timestamp}] {message}'
        self.log_history.append(entry)
        self.updated_at = time.time()
        return entry
    
    def get_identity(self, name: str) -> Optional[IdentityProfile]:
        return self.identities.get(name)
    
    def add_identity(self, name: str, source_paths: Optional[List[str]] = None) -> IdentityProfile:
        if name not in self.identities:
            self.identities[name] = IdentityProfile(name=name, source_paths=source_paths or [])
        elif source_paths:
            self.identities[name].source_paths.extend(source_paths)
        return self.identities[name]
    
    def get_occlusion(self, name: str) -> Optional[OcclusionProfile]:
        return self.occlusions.get(name)
    
    def add_occlusion(self, name: str, target_path: Optional[str] = None) -> OcclusionProfile:
        if name not in self.occlusions:
            self.occlusions[name] = OcclusionProfile(name=name, target_path=target_path)
        return self.occlusions[name]
    
    def needs_more_training(self) -> bool:
        for mapping in self.mappings:
            if mapping.quality_score < self.quality_threshold:
                return True
        return False
    
    def get_recent_logs(self, count: int = 10) -> List[str]:
        return self.log_history[-count:]
