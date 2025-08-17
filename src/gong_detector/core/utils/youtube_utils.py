"""YouTube download and file management utilities.

DEPRECATED: This module is being refactored into smaller, focused modules.
New code should import from:
- gong_detector.core.utils.youtube for YouTube functionality
- gong_detector.core.utils.loudness for LUFS/True Peak analysis
- gong_detector.core.utils.file_utils for file operations

This module maintains backward compatibility by re-exporting all functions.
"""

import logging

# Import all functions from the new modular structure
from .file_utils import (
    cleanup_old_temp_files,
    create_temp_audio_path,
    setup_directories,
)
from .loudness import (
    compute_batch_weighted_dbtp,
    compute_batch_weighted_lufs,
)
from .youtube import (
    create_folder_name_from_date,
    create_folder_name_from_title,
    download_and_process_youtube_audio,
    sanitize_title_for_folder,
    video_id_from_url,
)

# Configure logging
logger = logging.getLogger(__name__)

# Maintain backward compatibility with the original function name
download_and_trim_youtube_audio = download_and_process_youtube_audio

# Re-export all functions for backward compatibility
__all__ = [
    # YouTube functions
    "video_id_from_url",
    "download_and_trim_youtube_audio",
    "download_and_process_youtube_audio",
    "create_folder_name_from_title",
    "create_folder_name_from_date",
    "sanitize_title_for_folder",
    # File utilities
    "cleanup_old_temp_files",
    "create_temp_audio_path",
    "setup_directories",
    # Loudness analysis (batch processing only - use unified_analyzer for individual analysis)
    "compute_batch_weighted_lufs",
    "compute_batch_weighted_dbtp",
]
