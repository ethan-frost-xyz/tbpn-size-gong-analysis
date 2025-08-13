"""Utility functions for audio processing, YouTube operations, and results handling."""

# New modular imports (for future migration)
from . import file_utils, loudness, youtube
from .audio_utils import (
    DEFAULT_SAMPLE_RATE,
    SILENCE_FLOOR_DBFS,
    analyze_audio_slice_levels,
    compute_audio_levels,
    compute_peak_dbfs,
    compute_rms_dbfs,
    extract_audio_slice,
    get_audio_stats,
    get_slice_around_timestamp,
    is_silent,
    normalize_waveform,
)
from .convert_audio import convert_youtube_audio, get_audio_info, validate_audio_file
from .results_utils import (
    format_time,
    print_summary,
    save_positive_samples,
    save_results_to_csv,
)
from .youtube_utils import (
    cleanup_old_temp_files,
    create_folder_name_from_date,
    create_folder_name_from_title,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    sanitize_title_for_folder,
    setup_directories,
)

__all__ = [
    # Audio utilities
    "DEFAULT_SAMPLE_RATE",
    "SILENCE_FLOOR_DBFS",
    "compute_peak_dbfs",
    "compute_rms_dbfs",
    "compute_audio_levels",
    "extract_audio_slice",
    "get_slice_around_timestamp",
    "analyze_audio_slice_levels",
    "is_silent",
    "normalize_waveform",
    "get_audio_stats",
    # Audio conversion
    "convert_youtube_audio",
    "validate_audio_file",
    "get_audio_info",
    # YouTube utilities
    "download_and_trim_youtube_audio",
    "cleanup_old_temp_files",
    "create_folder_name_from_date",
    "create_folder_name_from_title",
    "create_temp_audio_path",
    "sanitize_title_for_folder",
    "setup_directories",
    # Results utilities
    "format_time",
    "print_summary",
    "save_positive_samples",
    "save_results_to_csv",
]
