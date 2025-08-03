"""Audio utilities for decibel estimation and waveform manipulation.

This module provides simple functions for computing audio levels in dBFS
and extracting audio slices around specific timestamps. All functions work
directly with numpy waveform arrays for maximum flexibility and reuse.
"""

import numpy as np

# Constants
SILENCE_FLOOR_DBFS = -80.0  # Silence floor threshold in dBFS
MIN_AMPLITUDE = 1e-8  # Minimum amplitude to avoid log(0) errors
DEFAULT_SAMPLE_RATE = 16000  # Default sample rate for audio processing


def compute_peak_dbfs(waveform: np.ndarray) -> float:
    """Compute peak dBFS level from a numpy waveform.

    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]

    Returns:
        Peak dBFS value, or SILENCE_FLOOR_DBFS if waveform is silent
    """
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS

    peak_amplitude = np.max(np.abs(waveform))

    if peak_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS

    return float(20.0 * np.log10(peak_amplitude))


def compute_rms_dbfs(waveform: np.ndarray) -> float:
    """Compute RMS dBFS level from a numpy waveform.

    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]

    Returns:
        RMS dBFS value, or SILENCE_FLOOR_DBFS if waveform is silent
    """
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS

    rms_amplitude = np.sqrt(np.mean(waveform**2))

    if rms_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS

    return float(20.0 * np.log10(rms_amplitude))


def compute_crest_factor(waveform: np.ndarray) -> float:
    """Compute crest factor (peak-to-RMS ratio) from a waveform.

    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]

    Returns:
        Crest factor as a linear ratio (1.0 = pure sine, >10 = very peaky)
        Returns 1.0 for silence to avoid division by zero
    """
    if len(waveform) == 0:
        return 1.0

    abs_waveform = np.abs(waveform)
    peak_amplitude = np.max(abs_waveform)
    rms_amplitude = np.sqrt(np.mean(waveform**2))

    # Handle silence or very quiet signals
    if rms_amplitude < MIN_AMPLITUDE:
        return 1.0

    return float(peak_amplitude / rms_amplitude)


def compute_loudness_metrics(waveform: np.ndarray) -> dict[str, float]:
    """Compute comprehensive loudness metrics optimized for clipped gong detection.

    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]

    Returns:
        Dictionary with keys: peak_dbfs, rms_dbfs, crest_factor, likely_clipped
    """
    if len(waveform) == 0:
        return {
            "peak_dbfs": SILENCE_FLOOR_DBFS,
            "rms_dbfs": SILENCE_FLOOR_DBFS,
            "crest_factor": 1.0,
            "likely_clipped": False,
        }

    # Compute all metrics in single pass for efficiency
    abs_waveform = np.abs(waveform)
    peak_amplitude = np.max(abs_waveform)
    rms_amplitude = np.sqrt(np.mean(waveform**2))

    # Handle silence case
    if peak_amplitude < MIN_AMPLITUDE:
        return {
            "peak_dbfs": SILENCE_FLOOR_DBFS,
            "rms_dbfs": SILENCE_FLOOR_DBFS,
            "crest_factor": 1.0,
            "likely_clipped": False,
        }

    # Compute dBFS values
    peak_dbfs = 20.0 * np.log10(peak_amplitude)
    rms_dbfs = (
        20.0 * np.log10(rms_amplitude)
        if rms_amplitude >= MIN_AMPLITUDE
        else SILENCE_FLOOR_DBFS
    )

    # Compute crest factor
    crest_factor = (
        peak_amplitude / rms_amplitude if rms_amplitude >= MIN_AMPLITUDE else 1.0
    )

    # Simple clipping detection: peak near 0 dBFS with low crest factor
    likely_clipped = peak_dbfs > -1.0 and crest_factor < 4.0

    return {
        "peak_dbfs": float(peak_dbfs),
        "rms_dbfs": float(rms_dbfs),
        "crest_factor": float(crest_factor),
        "likely_clipped": likely_clipped,
    }


def compute_audio_levels(waveform: np.ndarray) -> tuple[float, float]:
    """Compute both peak and RMS dBFS levels from a waveform.

    Args:
        waveform: Audio waveform as numpy array, normalized to [-1, 1]

    Returns:
        Tuple of (peak_dbfs, rms_dbfs)
    """
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS

    # Compute both levels efficiently in single pass
    abs_waveform = np.abs(waveform)
    peak_amplitude = np.max(abs_waveform)
    rms_amplitude = np.sqrt(np.mean(waveform**2))

    # Handle silence case
    if peak_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS

    peak_dbfs = 20.0 * np.log10(peak_amplitude)

    if rms_amplitude < MIN_AMPLITUDE:
        rms_dbfs = SILENCE_FLOOR_DBFS
    else:
        rms_dbfs = 20.0 * np.log10(rms_amplitude)

    return float(peak_dbfs), float(rms_dbfs)


def extract_audio_slice(
    waveform: np.ndarray,
    timestamp: float,
    duration_before: float = 20.0,
    duration_after: float = 5.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
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
    start_sample = int((timestamp - duration_before) * sample_rate)
    end_sample = int((timestamp + duration_after) * sample_rate)
    slice_length = end_sample - start_sample

    # Initialize output array
    audio_slice = np.zeros(slice_length, dtype=waveform.dtype)

    # Calculate valid ranges
    waveform_start = max(0, start_sample)
    waveform_end = min(len(waveform), end_sample)
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
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Extract audio slice centered around a timestamp.

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
        sample_rate=sample_rate,
    )


def analyze_audio_slice_levels(
    waveform: np.ndarray,
    timestamp: float,
    context_seconds: float = 20.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[float, float]:
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


def is_silent(
    waveform: np.ndarray, threshold_dbfs: float = SILENCE_FLOOR_DBFS + 10
) -> bool:
    """Check if an audio waveform is effectively silent.

    Args:
        waveform: Audio waveform as numpy array
        threshold_dbfs: Silence threshold in dBFS

    Returns:
        True if audio is below silence threshold
    """
    if len(waveform) == 0:
        return True

    # Use RMS for silence detection as it's more representative
    rms_amplitude = np.sqrt(np.mean(waveform**2))

    if rms_amplitude < MIN_AMPLITUDE:
        return True

    rms_dbfs = 20.0 * np.log10(rms_amplitude)
    return rms_dbfs < threshold_dbfs  # Changed from <= to < for more precise threshold


def normalize_waveform(
    waveform: np.ndarray, target_level_dbfs: float = -3.0
) -> np.ndarray:
    """Normalize waveform to a target peak level in dBFS.

    Args:
        waveform: Audio waveform as numpy array
        target_level_dbfs: Target peak level in dBFS (default: -3.0 dBFS)

    Returns:
        Normalized waveform as numpy array
    """
    if len(waveform) == 0:
        return waveform.copy()

    peak_amplitude = np.max(np.abs(waveform))

    if peak_amplitude < MIN_AMPLITUDE:
        return waveform.copy()

    # Calculate required gain
    target_amplitude = 10 ** (target_level_dbfs / 20.0)
    gain = target_amplitude / peak_amplitude

    # Use double precision for more accurate normalization
    return (waveform * gain).astype(np.float32)


def get_audio_stats(waveform: np.ndarray) -> dict:
    """Get comprehensive audio statistics for a waveform.

    Args:
        waveform: Audio waveform as numpy array

    Returns:
        Dictionary with audio statistics including peak, RMS, duration info
    """
    if len(waveform) == 0:
        return {
            "length": 0,
            "peak_dbfs": SILENCE_FLOOR_DBFS,
            "rms_dbfs": SILENCE_FLOOR_DBFS,
            "is_silent": True,
            "peak_amplitude": 0.0,
            "rms_amplitude": 0.0,
        }

    peak_dbfs, rms_dbfs = compute_audio_levels(waveform)
    peak_amplitude = np.max(np.abs(waveform))
    rms_amplitude = np.sqrt(np.mean(waveform**2))

    return {
        "length": len(waveform),
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "is_silent": is_silent(waveform),
        "peak_amplitude": float(peak_amplitude),
        "rms_amplitude": float(rms_amplitude),
    }
