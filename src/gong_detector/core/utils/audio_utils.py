"""Audio utilities for decibel estimation and waveform manipulation.

Provides functions for computing audio levels in dBFS and extracting audio slices
around specific timestamps. Optimized for gong detection with numpy waveform arrays.

Key Features:
- dBFS level calculations (peak and RMS)
- Audio slice extraction with context
- Loudness metrics for gong detection
- Waveform normalization and analysis
- Silence detection utilities
"""

from typing import Union

import numpy as np

# Constants
SILENCE_FLOOR_DBFS: float = -80.0  # Silence floor threshold in dBFS
MIN_AMPLITUDE: float = 1e-8  # Minimum amplitude to avoid log(0) errors
DEFAULT_SAMPLE_RATE: int = 16000  # Default sample rate for audio processing
DEFAULT_TARGET_LEVEL_DBFS: float = -3.0  # Default normalization target
SILENCE_THRESHOLD_DBFS: float = SILENCE_FLOOR_DBFS + 10  # Silence detection threshold


def compute_peak_dbfs(waveform: np.ndarray) -> float:
    """Calculate the peak level of a waveform in dBFS.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to the range [-1, 1].

    Returns
    -------
    float
        Peak dBFS value. Returns `SILENCE_FLOOR_DBFS` for empty or silent inputs.
    """
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS

    peak_amplitude = np.max(np.abs(waveform))

    if peak_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS

    return float(20.0 * np.log10(peak_amplitude))


def compute_rms_dbfs(waveform: np.ndarray) -> float:
    """Calculate the RMS level of a waveform in dBFS.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to the range [-1, 1].

    Returns
    -------
    float
        RMS dBFS level. Returns `SILENCE_FLOOR_DBFS` when no meaningful signal exists.
    """
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS

    rms_amplitude = np.sqrt(np.mean(waveform**2))

    if rms_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS

    return float(20.0 * np.log10(rms_amplitude))


def compute_crest_factor(waveform: np.ndarray) -> float:
    """Estimate the peak-to-RMS ratio (crest factor) of a waveform.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to the range [-1, 1].

    Returns
    -------
    float
        Linear crest factor. Silence returns a neutral value of `1.0`.
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


def compute_loudness_metrics(waveform: np.ndarray) -> dict[str, Union[float, bool]]:
    """Derive peak, RMS, crest factor, and clipping heuristics for a waveform.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to the range [-1, 1].

    Returns
    -------
    dict[str, float | bool]
        Dictionary containing `peak_dbfs`, `rms_dbfs`, `crest_factor`,
        `likely_clipped`, `peak_amplitude`, and `rms_amplitude`.
    """
    if len(waveform) == 0:
        return {
            "peak_dbfs": SILENCE_FLOOR_DBFS,
            "rms_dbfs": SILENCE_FLOOR_DBFS,
            "crest_factor": 1.0,
            "likely_clipped": False,
            "peak_amplitude": 0.0,
            "rms_amplitude": 0.0,
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
            "peak_amplitude": 0.0,
            "rms_amplitude": 0.0,
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
        "peak_amplitude": float(peak_amplitude),
        "rms_amplitude": float(rms_amplitude),
    }


def compute_audio_levels(waveform: np.ndarray) -> tuple[float, float]:
    """Return both peak and RMS levels for a waveform.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to the range [-1, 1].

    Returns
    -------
    tuple[float, float]
        Pair containing `(peak_dbfs, rms_dbfs)`.
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
    """Extract a padded audio segment around a timestamp.

    Parameters
    ----------
    waveform : numpy.ndarray
        Full waveform array.
    timestamp : float
        Center point in seconds from which to slice.
    duration_before : float, default=20.0
        Amount of context to include before the timestamp.
    duration_after : float, default=5.0
        Amount of context to include after the timestamp.
    sample_rate : int, default=DEFAULT_SAMPLE_RATE
        Sample rate used to translate seconds into samples.

    Returns
    -------
    numpy.ndarray
        Slice padded with zeros when the requested window falls outside the waveform.
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
    """Retrieve a symmetric audio slice centered on a timestamp.

    Parameters
    ----------
    waveform : numpy.ndarray
        Full waveform array.
    timestamp : float
        Center timestamp in seconds.
    context_seconds : float, default=20.0
        Total duration of the slice; distributed equally before and after `timestamp`.
    sample_rate : int, default=DEFAULT_SAMPLE_RATE
        Sample rate used to translate seconds into samples.

    Returns
    -------
    numpy.ndarray
        Audio slice extracted with zero padding at the boundaries.
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
    """Compute peak and RMS levels for a contextual audio slice.

    Parameters
    ----------
    waveform : numpy.ndarray
        Full waveform array.
    timestamp : float
        Center timestamp in seconds.
    context_seconds : float, default=20.0
        Total analysis window around the timestamp.
    sample_rate : int, default=DEFAULT_SAMPLE_RATE
        Sample rate used for sample conversion.

    Returns
    -------
    tuple[float, float]
        Tuple containing `(peak_dbfs, rms_dbfs)` for the extracted slice.
    """
    audio_slice = get_slice_around_timestamp(
        waveform, timestamp, context_seconds, sample_rate
    )

    return compute_audio_levels(audio_slice)


def is_silent(
    waveform: np.ndarray, threshold_dbfs: float = SILENCE_THRESHOLD_DBFS
) -> bool:
    """Determine whether a waveform falls below the silence threshold.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to [-1, 1].
    threshold_dbfs : float, default=SILENCE_THRESHOLD_DBFS
        RMS level below which the signal is considered silent.

    Returns
    -------
    bool
        `True` when the RMS level is below `threshold_dbfs`.
    """
    if len(waveform) == 0:
        return True

    # Use RMS for silence detection as it's more representative
    rms_amplitude = np.sqrt(np.mean(waveform**2))

    if rms_amplitude < MIN_AMPLITUDE:
        return True

    rms_dbfs = 20.0 * np.log10(rms_amplitude)
    return rms_dbfs < threshold_dbfs


def normalize_waveform(
    waveform: np.ndarray, target_level_dbfs: float = DEFAULT_TARGET_LEVEL_DBFS
) -> np.ndarray:
    """Scale a waveform so its peak matches the desired dBFS level.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to [-1, 1].
    target_level_dbfs : float, default=DEFAULT_TARGET_LEVEL_DBFS
        Desired peak level expressed in dBFS.

    Returns
    -------
    numpy.ndarray
        Normalized waveform copied to a new array.
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


def get_audio_stats(waveform: np.ndarray) -> dict[str, Union[int, float, bool]]:
    """Summarize key statistics for a waveform.

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio data normalized to [-1, 1].

    Returns
    -------
    dict[str, int | float | bool]
        Dictionary containing length (samples), `peak_dbfs`, `rms_dbfs`, `is_silent`,
        `peak_amplitude`, and `rms_amplitude`.
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
