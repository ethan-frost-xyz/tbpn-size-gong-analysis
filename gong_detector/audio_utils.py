"""Audio utilities for decibel estimation and waveform manipulation.

This module provides simple functions for computing audio levels in dBFS
and extracting audio slices around specific timestamps. All functions work
directly with numpy waveform arrays for maximum flexibility and reuse.
"""

from typing import Tuple
import numpy as np


# Constants
SILENCE_FLOOR_DBFS = -80.0  # Silence floor threshold in dBFS
MIN_AMPLITUDE = 1e-8  # Minimum amplitude to avoid log(0) errors


def compute_peak_dbfs(waveform: np.ndarray) -> float:
    """Compute peak dBFS level from a numpy waveform.
    
    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]
        
    Returns:
        Peak dBFS value, or SILENCE_FLOOR_DBFS if waveform is silent
    """
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS
        
    # Find peak amplitude
    peak_amplitude = np.max(np.abs(waveform))
    
    # Handle silence case
    if peak_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS
        
    # Convert to dBFS: 20 * log10(amplitude)
    peak_dbfs = 20.0 * np.log10(peak_amplitude)
    
    return float(peak_dbfs)


def compute_rms_dbfs(waveform: np.ndarray) -> float:
    """Compute RMS dBFS level from a numpy waveform.
    
    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]
        
    Returns:
        RMS dBFS value, or SILENCE_FLOOR_DBFS if waveform is silent
    """
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS
        
    # Compute RMS amplitude
    rms_amplitude = np.sqrt(np.mean(waveform ** 2))
    
    # Handle silence case
    if rms_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS
        
    # Convert to dBFS: 20 * log10(amplitude)
    rms_dbfs = 20.0 * np.log10(rms_amplitude)
    
    return float(rms_dbfs)


def compute_audio_levels(waveform: np.ndarray) -> Tuple[float, float]:
    """Compute both peak and RMS dBFS levels from a waveform.
    
    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]
        
    Returns:
        Tuple of (peak_dbfs, rms_dbfs)
    """
    peak_dbfs = compute_peak_dbfs(waveform)
    rms_dbfs = compute_rms_dbfs(waveform)
    
    return peak_dbfs, rms_dbfs


def extract_audio_slice(
    waveform: np.ndarray,
    timestamp: float,
    duration_before: float = 20.0,
    duration_after: float = 5.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Extract an audio slice around a specific timestamp.
    
    Args:
        waveform: Full audio waveform as numpy array
        timestamp: Center timestamp in seconds
        duration_before: How many seconds before timestamp to include
        duration_after: How many seconds after timestamp to include
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio slice as numpy array, zero-padded if needed at boundaries
    """
    if len(waveform) == 0:
        return np.array([])
        
    # Convert times to sample indices
    start_time = timestamp - duration_before
    end_time = timestamp + duration_after
    
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Handle boundaries with zero-padding
    slice_length = end_sample - start_sample
    audio_slice = np.zeros(slice_length, dtype=waveform.dtype)
    
    # Calculate valid range within original waveform
    waveform_start = max(0, start_sample)
    waveform_end = min(len(waveform), end_sample)
    
    # Calculate corresponding range in output slice
    slice_start = max(0, -start_sample)
    slice_end = slice_start + (waveform_end - waveform_start)
    
    # Copy valid audio data
    if waveform_start < waveform_end and slice_start < slice_end:
        audio_slice[slice_start:slice_end] = waveform[waveform_start:waveform_end]
    
    return audio_slice


def get_slice_around_timestamp(
    waveform: np.ndarray,
    timestamp: float,
    context_seconds: float = 20.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Simple wrapper to extract audio slice centered around a timestamp.
    
    Args:
        waveform: Full audio waveform as numpy array  
        timestamp: Center timestamp in seconds
        context_seconds: Total context duration (split equally before/after)
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio slice as numpy array
    """
    half_context = context_seconds / 2.0
    return extract_audio_slice(
        waveform=waveform,
        timestamp=timestamp,
        duration_before=half_context,
        duration_after=half_context,
        sample_rate=sample_rate
    )


def analyze_audio_slice_levels(
    waveform: np.ndarray,
    timestamp: float,
    context_seconds: float = 20.0,
    sample_rate: int = 16000
) -> Tuple[float, float]:
    """Extract audio slice and compute its dBFS levels.
    
    Args:
        waveform: Full audio waveform as numpy array
        timestamp: Center timestamp in seconds  
        context_seconds: Context duration around timestamp
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (peak_dbfs, rms_dbfs) for the extracted slice
    """
    audio_slice = get_slice_around_timestamp(
        waveform, timestamp, context_seconds, sample_rate
    )
    
    return compute_audio_levels(audio_slice)


def is_silent(waveform: np.ndarray, threshold_dbfs: float = SILENCE_FLOOR_DBFS + 10) -> bool:
    """Check if an audio waveform is effectively silent.
    
    Args:
        waveform: Audio waveform as numpy array
        threshold_dbfs: Silence threshold in dBFS
        
    Returns:
        True if audio is below silence threshold
    """
    if len(waveform) == 0:
        return True
        
    rms_dbfs = compute_rms_dbfs(waveform)
    return rms_dbfs <= threshold_dbfs 