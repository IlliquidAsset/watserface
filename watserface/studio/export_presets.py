from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PresetCategory(Enum):
    SOCIAL_MEDIA = 'social_media'
    BROADCAST = 'broadcast'
    WEB = 'web'
    ARCHIVE = 'archive'
    CUSTOM = 'custom'


@dataclass
class ExportPreset:
    name: str
    category: PresetCategory
    resolution: Tuple[int, int]
    fps: int
    video_codec: str
    video_bitrate: str
    audio_codec: str
    audio_bitrate: str
    container: str
    pixel_format: str = 'yuv420p'
    crf: Optional[int] = None
    preset_speed: str = 'medium'
    additional_flags: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_flags is None:
            self.additional_flags = {}
    
    def to_ffmpeg_args(self) -> List[str]:
        args = []
        
        args.extend(['-s', f'{self.resolution[0]}x{self.resolution[1]}'])
        args.extend(['-r', str(self.fps)])
        args.extend(['-c:v', self.video_codec])
        
        if self.crf is not None:
            args.extend(['-crf', str(self.crf)])
        else:
            args.extend(['-b:v', self.video_bitrate])
        
        args.extend(['-c:a', self.audio_codec])
        args.extend(['-b:a', self.audio_bitrate])
        args.extend(['-pix_fmt', self.pixel_format])
        
        if self.video_codec in ('libx264', 'libx265'):
            args.extend(['-preset', self.preset_speed])
        
        for key, value in self.additional_flags.items():
            args.extend([f'-{key}', str(value)])
        
        return args


PRESET_INSTAGRAM_REELS = ExportPreset(
    name='Instagram Reels',
    category=PresetCategory.SOCIAL_MEDIA,
    resolution=(1080, 1920),
    fps=30,
    video_codec='libx264',
    video_bitrate='8M',
    audio_codec='aac',
    audio_bitrate='128k',
    container='mp4',
    crf=23,
    preset_speed='fast',
    additional_flags={'profile:v': 'high', 'level': '4.0'}
)

PRESET_INSTAGRAM_FEED = ExportPreset(
    name='Instagram Feed',
    category=PresetCategory.SOCIAL_MEDIA,
    resolution=(1080, 1080),
    fps=30,
    video_codec='libx264',
    video_bitrate='6M',
    audio_codec='aac',
    audio_bitrate='128k',
    container='mp4',
    crf=23,
    preset_speed='fast'
)

PRESET_TIKTOK = ExportPreset(
    name='TikTok',
    category=PresetCategory.SOCIAL_MEDIA,
    resolution=(1080, 1920),
    fps=30,
    video_codec='libx264',
    video_bitrate='10M',
    audio_codec='aac',
    audio_bitrate='192k',
    container='mp4',
    crf=20,
    preset_speed='fast'
)

PRESET_YOUTUBE_1080P = ExportPreset(
    name='YouTube 1080p',
    category=PresetCategory.SOCIAL_MEDIA,
    resolution=(1920, 1080),
    fps=30,
    video_codec='libx264',
    video_bitrate='12M',
    audio_codec='aac',
    audio_bitrate='192k',
    container='mp4',
    crf=18,
    preset_speed='slow',
    additional_flags={'profile:v': 'high', 'bf': '2', 'g': '30'}
)

PRESET_YOUTUBE_4K = ExportPreset(
    name='YouTube 4K',
    category=PresetCategory.SOCIAL_MEDIA,
    resolution=(3840, 2160),
    fps=30,
    video_codec='libx264',
    video_bitrate='45M',
    audio_codec='aac',
    audio_bitrate='256k',
    container='mp4',
    crf=18,
    preset_speed='slow'
)

PRESET_TWITTER = ExportPreset(
    name='Twitter/X',
    category=PresetCategory.SOCIAL_MEDIA,
    resolution=(1280, 720),
    fps=30,
    video_codec='libx264',
    video_bitrate='5M',
    audio_codec='aac',
    audio_bitrate='128k',
    container='mp4',
    crf=23,
    preset_speed='fast'
)

PRESET_BROADCAST_HD = ExportPreset(
    name='Broadcast HD (1080i)',
    category=PresetCategory.BROADCAST,
    resolution=(1920, 1080),
    fps=30,
    video_codec='libx264',
    video_bitrate='25M',
    audio_codec='pcm_s16le',
    audio_bitrate='1536k',
    container='mov',
    preset_speed='slow',
    additional_flags={'profile:v': 'high', 'level': '4.2'}
)

PRESET_BROADCAST_4K = ExportPreset(
    name='Broadcast 4K',
    category=PresetCategory.BROADCAST,
    resolution=(3840, 2160),
    fps=30,
    video_codec='libx264',
    video_bitrate='80M',
    audio_codec='pcm_s16le',
    audio_bitrate='1536k',
    container='mov',
    preset_speed='slow'
)

PRESET_PRORES_422 = ExportPreset(
    name='ProRes 422',
    category=PresetCategory.BROADCAST,
    resolution=(1920, 1080),
    fps=30,
    video_codec='prores_ks',
    video_bitrate='',
    audio_codec='pcm_s16le',
    audio_bitrate='1536k',
    container='mov',
    pixel_format='yuv422p10le',
    additional_flags={'profile:v': '2'}
)

PRESET_WEB_720P = ExportPreset(
    name='Web 720p',
    category=PresetCategory.WEB,
    resolution=(1280, 720),
    fps=30,
    video_codec='libx264',
    video_bitrate='3M',
    audio_codec='aac',
    audio_bitrate='128k',
    container='mp4',
    crf=23,
    preset_speed='fast'
)

PRESET_WEB_480P = ExportPreset(
    name='Web 480p',
    category=PresetCategory.WEB,
    resolution=(854, 480),
    fps=30,
    video_codec='libx264',
    video_bitrate='1.5M',
    audio_codec='aac',
    audio_bitrate='96k',
    container='mp4',
    crf=25,
    preset_speed='fast'
)

PRESET_WEBM_VP9 = ExportPreset(
    name='WebM VP9',
    category=PresetCategory.WEB,
    resolution=(1920, 1080),
    fps=30,
    video_codec='libvpx-vp9',
    video_bitrate='4M',
    audio_codec='libopus',
    audio_bitrate='128k',
    container='webm',
    additional_flags={'deadline': 'good', 'cpu-used': '4'}
)

PRESET_ARCHIVE_LOSSLESS = ExportPreset(
    name='Archive (Lossless)',
    category=PresetCategory.ARCHIVE,
    resolution=(1920, 1080),
    fps=30,
    video_codec='libx264',
    video_bitrate='',
    audio_codec='flac',
    audio_bitrate='',
    container='mkv',
    crf=0,
    preset_speed='veryslow',
    pixel_format='yuv444p'
)

PRESET_ARCHIVE_HIGH_QUALITY = ExportPreset(
    name='Archive (High Quality)',
    category=PresetCategory.ARCHIVE,
    resolution=(1920, 1080),
    fps=30,
    video_codec='libx265',
    video_bitrate='',
    audio_codec='flac',
    audio_bitrate='',
    container='mkv',
    crf=18,
    preset_speed='slow'
)

PRESET_GIF = ExportPreset(
    name='GIF',
    category=PresetCategory.WEB,
    resolution=(480, 480),
    fps=15,
    video_codec='gif',
    video_bitrate='',
    audio_codec='',
    audio_bitrate='',
    container='gif'
)


ALL_PRESETS = {
    'instagram_reels': PRESET_INSTAGRAM_REELS,
    'instagram_feed': PRESET_INSTAGRAM_FEED,
    'tiktok': PRESET_TIKTOK,
    'youtube_1080p': PRESET_YOUTUBE_1080P,
    'youtube_4k': PRESET_YOUTUBE_4K,
    'twitter': PRESET_TWITTER,
    'broadcast_hd': PRESET_BROADCAST_HD,
    'broadcast_4k': PRESET_BROADCAST_4K,
    'prores_422': PRESET_PRORES_422,
    'web_720p': PRESET_WEB_720P,
    'web_480p': PRESET_WEB_480P,
    'webm_vp9': PRESET_WEBM_VP9,
    'archive_lossless': PRESET_ARCHIVE_LOSSLESS,
    'archive_high_quality': PRESET_ARCHIVE_HIGH_QUALITY,
    'gif': PRESET_GIF
}


class PresetManager:
    
    def __init__(self):
        self.presets: Dict[str, ExportPreset] = ALL_PRESETS.copy()
        self.custom_presets: Dict[str, ExportPreset] = {}
    
    def get_preset(self, name: str) -> Optional[ExportPreset]:
        return self.presets.get(name) or self.custom_presets.get(name)
    
    def get_presets_by_category(self, category: PresetCategory) -> List[ExportPreset]:
        result = []
        for preset in self.presets.values():
            if preset.category == category:
                result.append(preset)
        for preset in self.custom_presets.values():
            if preset.category == category:
                result.append(preset)
        return result
    
    def list_preset_names(self) -> List[str]:
        return list(self.presets.keys()) + list(self.custom_presets.keys())
    
    def add_custom_preset(self, key: str, preset: ExportPreset) -> None:
        preset.category = PresetCategory.CUSTOM
        self.custom_presets[key] = preset
    
    def remove_custom_preset(self, key: str) -> bool:
        if key in self.custom_presets:
            del self.custom_presets[key]
            return True
        return False
    
    def create_custom_preset(
        self,
        name: str,
        resolution: Tuple[int, int],
        fps: int = 30,
        video_codec: str = 'libx264',
        video_bitrate: str = '8M',
        audio_codec: str = 'aac',
        audio_bitrate: str = '192k',
        container: str = 'mp4',
        crf: Optional[int] = 23
    ) -> ExportPreset:
        preset = ExportPreset(
            name=name,
            category=PresetCategory.CUSTOM,
            resolution=resolution,
            fps=fps,
            video_codec=video_codec,
            video_bitrate=video_bitrate,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate,
            container=container,
            crf=crf
        )
        
        key = name.lower().replace(' ', '_')
        self.custom_presets[key] = preset
        return preset
    
    def get_recommended_preset(self, target_platform: str) -> Optional[ExportPreset]:
        platform_map = {
            'instagram': PRESET_INSTAGRAM_REELS,
            'tiktok': PRESET_TIKTOK,
            'youtube': PRESET_YOUTUBE_1080P,
            'twitter': PRESET_TWITTER,
            'x': PRESET_TWITTER,
            'web': PRESET_WEB_720P,
            'broadcast': PRESET_BROADCAST_HD,
            'tv': PRESET_BROADCAST_HD,
            'archive': PRESET_ARCHIVE_HIGH_QUALITY
        }
        
        target_lower = target_platform.lower()
        for key, preset in platform_map.items():
            if key in target_lower:
                return preset
        
        return PRESET_YOUTUBE_1080P


def create_preset_manager() -> PresetManager:
    return PresetManager()


def get_preset(name: str) -> Optional[ExportPreset]:
    return ALL_PRESETS.get(name)


def list_presets() -> List[str]:
    return list(ALL_PRESETS.keys())
