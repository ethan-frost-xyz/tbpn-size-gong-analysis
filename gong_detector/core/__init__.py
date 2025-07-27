"""
Core functionality for TBPN gong detection.

This package contains the essential modules for audio processing,
YAMNet integration, and gong detection capabilities.
"""

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
from .comprehensive_csv import ComprehensiveCSVManager, DetectionRecord
from .convert_audio import convert_youtube_audio, get_audio_info, validate_audio_file
from .detect_from_youtube import detect_from_youtube_comprehensive
from .negative_sample_collector import collect_negative_samples
from .results_utils import (
    format_time,
    print_summary,
    save_positive_samples,
    save_results_to_csv,
)
from .yamnet_runner import YAMNetGongDetector
from .youtube_utils import (
    cleanup_old_temp_files,
    create_folder_name_from_date,
    create_folder_name_from_title,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    sanitize_title_for_folder,
    setup_directories,
)

__version__ = "1.0.0"

# Public API - these are the main components users should import
__all__ = [
    # Main detector class
    "YAMNetGongDetector",
    # Comprehensive detection and CSV functionality
    "detect_from_youtube_comprehensive",
    "ComprehensiveCSVManager",
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
