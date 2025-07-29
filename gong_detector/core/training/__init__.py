"""Training utilities for gong detection."""

from .negative_collector import collect_negative_samples
from .manual_collector import process_single_sample

__all__ = ["collect_negative_samples", "process_single_sample"] 