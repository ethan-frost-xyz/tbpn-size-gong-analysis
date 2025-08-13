"""YouTube utilities for downloading, caching, and processing audio.

This package provides modular YouTube functionality including:
- Audio downloading with yt-dlp
- Dual-cache management (raw + preprocessed)
- Audio processing and trimming
- Metadata extraction and formatting
"""

# Import all public functions to maintain backward compatibility
from .audio_processor import convert_and_trim_audio, trim_from_preprocessed
from .cache_manager import ensure_full_preprocessed_from_raw, save_raw_to_cache
from .downloader import (
    download_and_process_youtube_audio,
    download_youtube_audio,
    get_cookies_path,
)
from .metadata_utils import (
    create_folder_name_from_date,
    create_folder_name_from_title,
    sanitize_title_for_folder,
    video_id_from_url,
)

# Maintain backward compatibility with the original function name
download_and_trim_youtube_audio = download_and_process_youtube_audio

__all__ = [
    # Downloader functions
    "download_youtube_audio",
    "download_and_process_youtube_audio",
    "download_and_trim_youtube_audio",  # Backward compatibility alias
    "get_cookies_path",
    # Cache management
    "save_raw_to_cache",
    "ensure_full_preprocessed_from_raw",
    # Audio processing
    "trim_from_preprocessed",
    "convert_and_trim_audio",
    # Metadata utilities
    "video_id_from_url",
    "create_folder_name_from_title",
    "create_folder_name_from_date",
    "sanitize_title_for_folder",
]
