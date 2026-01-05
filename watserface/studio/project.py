from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import time

from watserface.studio.state import StudioState, StudioPhase


@dataclass
class ProjectMetadata:
    name: str
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    version: str = '1.0.0'
    description: str = ''


@dataclass 
class ProjectData:
    metadata: ProjectMetadata
    state: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    file_references: Dict[str, str] = field(default_factory=dict)


class ProjectManager:
    
    def __init__(self, projects_dir: str = '.projects'):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.current_project: Optional[ProjectData] = None
        self.current_path: Optional[Path] = None
    
    def create(self, name: str, description: str = '') -> ProjectData:
        metadata = ProjectMetadata(
            name=name,
            description=description
        )
        
        self.current_project = ProjectData(metadata=metadata)
        return self.current_project
    
    def save(self, path: Optional[str] = None) -> bool:
        if self.current_project is None:
            return False
        
        if path:
            save_path = Path(path)
        elif self.current_path:
            save_path = self.current_path
        else:
            safe_name = self.current_project.metadata.name.replace(' ', '_').lower()
            save_path = self.projects_dir / f'{safe_name}.wfproj'
        
        self.current_project.metadata.modified_at = time.time()
        
        try:
            data = {
                'metadata': asdict(self.current_project.metadata),
                'state': self.current_project.state,
                'settings': self.current_project.settings,
                'file_references': self.current_project.file_references
            }
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.current_path = save_path
            return True
            
        except Exception as e:
            print(f'Failed to save project: {e}')
            return False
    
    def load(self, path: str) -> Optional[ProjectData]:
        load_path = Path(path)
        
        if not load_path.exists():
            return None
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            metadata = ProjectMetadata(**data.get('metadata', {}))
            
            self.current_project = ProjectData(
                metadata=metadata,
                state=data.get('state', {}),
                settings=data.get('settings', {}),
                file_references=data.get('file_references', {})
            )
            
            self.current_path = load_path
            return self.current_project
            
        except Exception as e:
            print(f'Failed to load project: {e}')
            return None
    
    def save_studio_state(self, state: StudioState) -> None:
        if self.current_project is None:
            return
        
        state_dict = {
            'phase': state.phase.value,
            'identities': {},
            'mappings': [],
            'target_path': state.target_path,
            'output_path': state.output_path,
            'occlusion_model_path': state.occlusion_model_path
        }
        
        for name, identity in state.identities.items():
            state_dict['identities'][name] = {
                'name': identity.name,
                'source_paths': identity.source_paths,
                'embedding_path': identity.embedding_path,
                'model_path': identity.model_path,
                'epochs_trained': identity.epochs_trained
            }
        
        for mapping in state.mappings:
            state_dict['mappings'].append({
                'source_identity': mapping.source_identity,
                'target_face_index': mapping.target_face_index,
                'confidence': mapping.confidence,
                'quality_score': mapping.quality_score
            })
        
        self.current_project.state = state_dict
    
    def restore_studio_state(self) -> Optional[Dict[str, Any]]:
        if self.current_project is None:
            return None
        
        return self.current_project.state
    
    def add_file_reference(self, key: str, path: str) -> None:
        if self.current_project is None:
            return
        
        abs_path = str(Path(path).absolute())
        self.current_project.file_references[key] = abs_path
    
    def get_file_reference(self, key: str) -> Optional[str]:
        if self.current_project is None:
            return None
        
        path = self.current_project.file_references.get(key)
        if path and Path(path).exists():
            return path
        return None
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        if self.current_project is None:
            return
        
        self.current_project.settings.update(settings)
    
    def get_settings(self) -> Dict[str, Any]:
        if self.current_project is None:
            return {}
        
        return self.current_project.settings.copy()
    
    def list_projects(self) -> List[Dict[str, Any]]:
        projects = []
        
        for proj_file in self.projects_dir.glob('*.wfproj'):
            try:
                with open(proj_file, 'r') as f:
                    data = json.load(f)
                
                projects.append({
                    'path': str(proj_file),
                    'name': data.get('metadata', {}).get('name', proj_file.stem),
                    'modified_at': data.get('metadata', {}).get('modified_at', 0),
                    'description': data.get('metadata', {}).get('description', '')
                })
            except Exception:
                continue
        
        projects.sort(key=lambda x: x['modified_at'], reverse=True)
        return projects
    
    def delete_project(self, path: str) -> bool:
        try:
            Path(path).unlink()
            
            if self.current_path and str(self.current_path) == path:
                self.current_project = None
                self.current_path = None
            
            return True
        except Exception:
            return False
    
    def export_project(self, output_path: str, include_files: bool = False) -> bool:
        if self.current_project is None:
            return False
        
        try:
            export_data = {
                'metadata': asdict(self.current_project.metadata),
                'state': self.current_project.state,
                'settings': self.current_project.settings
            }
            
            if include_files:
                export_data['file_references'] = self.current_project.file_references
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception:
            return False


def create_project_manager(projects_dir: str = '.projects') -> ProjectManager:
    return ProjectManager(projects_dir)
