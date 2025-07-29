"""
TBPN Gong Detection Package.

This package provides tools for detecting gong sounds in audio files
using YAMNet and custom audio processing pipelines.
"""

# Import core functionality
from .core import (
    CSVManager,
    DetectionRecord,
    YAMNetGongDetector,
    detect_from_youtube_comprehensive,
    collect_negative_samples,
    # Audio utilities
    convert_youtube_audio,
    validate_audio_file,
    get_audio_info,
    compute_peak_dbfs,
    compute_rms_dbfs,
    compute_audio_levels,
    extract_audio_slice,
    get_slice_around_timestamp,
    analyze_audio_slice_levels,
    is_silent,
    normalize_waveform,
    get_audio_stats,
    # YouTube utilities
    download_and_trim_youtube_audio,
    format_time,
    print_summary,
    cleanup_old_temp_files,
    create_folder_name_from_date,
    create_folder_name_from_title,
    create_temp_audio_path,
    sanitize_title_for_folder,
    setup_directories,
    # Results utilities
    save_positive_samples,
    save_results_to_csv,
    # Constants
    SILENCE_FLOOR_DBFS,
    DEFAULT_SAMPLE_RATE,
)

__version__ = "1.0.0"

__all__ = [
    # Main detector class
    "YAMNetGongDetector",
    # Comprehensive detection and CSV functionality
    "detect_from_youtube_comprehensive",
    "CSVManager",
    "DetectionRecord",
    # Audio conversion functions
    "convert_youtube_audio",
    "validate_audio_file",
    "get_audio_info",
    # Audio utility functions
    "compute_peak_dbfs",
    "compute_rms_dbfs",
    "compute_audio_levels",
    "extract_audio_slice",
    "get_slice_around_timestamp",
    "analyze_audio_slice_levels",
    "is_silent",
    "normalize_waveform",
    "get_audio_stats",
    # YouTube detection utilities
    "download_and_trim_youtube_audio",
    "format_time",
    "print_summary",
    # YouTube utilities
    "cleanup_old_temp_files",
    "create_folder_name_from_date",
    "create_folder_name_from_title",
    "create_temp_audio_path",
    "sanitize_title_for_folder",
    "setup_directories",
    # Results utilities
    "save_positive_samples",
    "save_results_to_csv",
    # Negative sample collection
    "collect_negative_samples",
    # Constants
    "SILENCE_FLOOR_DBFS",
    "DEFAULT_SAMPLE_RATE",
    # Version info
    "__version__",
] 