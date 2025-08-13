"""Loudness analysis utilities for LUFS and True Peak measurements.

This package provides EBU R128 compliant loudness analysis including:
- LUFS (Loudness Units relative to Full Scale) measurement
- True Peak (dBTP) measurement
- Batch weighting across multiple videos
- BS.1770-4 K-weighting and gating
"""

# Import all public functions to maintain backward compatibility
from .batch_processor import compute_batch_weighted_dbtp, compute_batch_weighted_lufs
from .lufs_analyzer import compute_lufs_segments
from .true_peak_analyzer import compute_true_peak_segments

__all__ = [
    # LUFS analysis
    "compute_lufs_segments",
    "compute_batch_weighted_lufs",
    # True Peak analysis
    "compute_true_peak_segments",
    "compute_batch_weighted_dbtp",
]
