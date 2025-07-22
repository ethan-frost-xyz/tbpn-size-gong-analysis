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
from .convert_audio import convert_youtube_audio, get_audio_info, validate_audio_file
from .detect_from_youtube import (
    download_and_trim_youtube_audio,
    format_time,
    print_summary,
)
from .yamnet_runner import YAMNetGongDetector

__version__ = "1.0.0"

# Public API - these are the main components users should import
__all__ = [
    # Main detector class
    "YAMNetGongDetector",
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
    # Constants
    "SILENCE_FLOOR_DBFS",
    "DEFAULT_SAMPLE_RATE",
    # Version info
    "__version__",
]
