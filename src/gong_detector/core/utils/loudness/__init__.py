"""Loudness analysis utilities for LUFS and True Peak measurements.

This package provides EBU R128 compliant loudness analysis including:
- LUFS (Loudness Units relative to Full Scale) measurement
- True Peak (dBTP) measurement
- Batch weighting across multiple videos
- BS.1770-4 K-weighting and gating
"""

# Import all public functions
from .batch_processor import compute_batch_weighted_dbtp, compute_batch_weighted_lufs
from .unified_analyzer import compute_all_loudness_metrics

# Note: compute_lufs_segments and compute_true_peak_segments have been removed
# Use compute_all_loudness_metrics for optimal performance and no deprecation warnings

__all__ = [
    # Unified analysis (recommended - uses soundfile-only, no librosa warnings)
    "compute_all_loudness_metrics",
    # Batch analysis
    "compute_batch_weighted_lufs",
    "compute_batch_weighted_dbtp",
]
