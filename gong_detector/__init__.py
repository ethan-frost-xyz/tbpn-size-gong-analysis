"""Gong Detector package for audio classification using YAMNet."""

from .convert_audio import convert_youtube_audio
from .audio_utils import (
    compute_peak_dbfs,
    compute_rms_dbfs,
    compute_audio_levels,
    extract_audio_slice,
    get_slice_around_timestamp,
    analyze_audio_slice_levels,
    is_silent,
    SILENCE_FLOOR_DBFS,
    MIN_AMPLITUDE
)
from .export_snippets import GongSnippetExporter, export_gong_snippets

__version__ = "0.1.0"
__author__ = "Ethan Frost"
__email__ = "ethanfrostbvt@gmail.com"

__all__ = [
    "convert_youtube_audio",
    "compute_peak_dbfs", 
    "compute_rms_dbfs",
    "compute_audio_levels",
    "extract_audio_slice",
    "get_slice_around_timestamp", 
    "analyze_audio_slice_levels",
    "is_silent",
    "SILENCE_FLOOR_DBFS",
    "MIN_AMPLITUDE",
    "GongSnippetExporter",
    "export_gong_snippets"
] 