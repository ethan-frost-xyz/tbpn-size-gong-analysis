"""Training utilities for gong detection."""

from .manual_collector import process_single_sample
from .negative_collector import collect_negative_samples

__all__ = ["collect_negative_samples", "process_single_sample"]
