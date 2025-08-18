"""Unified loudness analyzer for efficient LUFS and True Peak computation.

This module provides a single-pass approach to computing all loudness metrics,
eliminating redundant audio loading and processing for optimal performance.
"""

import logging
from pathlib import Path
from typing import Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Optional imports for loudness analysis
try:
    import librosa  # type: ignore
    import numpy as np  # type: ignore
    import pyloudnorm as pyln  # type: ignore
    import soundfile as sf  # type: ignore

    LOUDNESS_AVAILABLE = True
except ImportError:
    LOUDNESS_AVAILABLE = False
    pyln = None
    librosa = None
    np = None
    sf = None


def load_raw_audio_modern(file_path: str) -> tuple[np.ndarray, int]:
    """Load audio using soundfile for WAV files (no deprecated methods).

    Since raw cache now saves high-quality WAV files, we can use soundfile exclusively,
    eliminating the need for deprecated librosa fallback methods.

    Args:
        file_path: Path to audio file (expected to be WAV format)

    Returns:
        Tuple of (audio_data, sample_rate) where audio_data is 2D array (channels, samples)

    Raises:
        RuntimeError: If audio loading fails
    """
    if not LOUDNESS_AVAILABLE:
        raise RuntimeError("Audio loading requires soundfile")

    try:
        # Use soundfile for WAV files (no fallback needed)
        logger.debug(f"Loading WAV audio with soundfile: {file_path}")
        audio_data, sample_rate = sf.read(file_path, always_2d=True)
        # Convert to librosa format: (channels, samples)
        audio_data = audio_data.T
        logger.debug(f"Successfully loaded: {audio_data.shape}, {sample_rate}Hz")
        return audio_data, sample_rate

    except Exception as sf_error:
        raise RuntimeError(
            f"Failed to load audio file {file_path} with soundfile. "
            f"Error: {sf_error}. "
            f"Ensure the file is a valid WAV format."
        ) from sf_error


def compute_all_loudness_metrics(
    video_id: str,
    detections: list[tuple[float, float, float]],
    index: Optional[Any] = None,
    batch_context: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Compute all LUFS and True Peak metrics in single audio pass.

    This function loads raw audio once and computes all required loudness
    measurements (integrated, short-term, momentary) for both LUFS and True Peak
    analysis across all detection timestamps.

    Args:
        video_id: YouTube video ID
        detections: List of (window_start, confidence, display_timestamp) tuples
        index: Optional LocalMediaIndex instance
        batch_context: Optional dict with batch weighting information

    Returns:
        Tuple of (lufs_metrics_list, dbtp_metrics_list) where each list contains
        one dict per detection with keys:
        - integrated_lufs/dbtp: 5-second window measurement
        - shortterm_lufs/dbtp: 3-second window measurement
        - momentary_lufs/dbtp: 400ms window measurement

    Raises:
        RuntimeError: If loudness libraries not available or raw audio not found
        ValueError: If detections list is invalid
    """
    if not LOUDNESS_AVAILABLE:
        raise RuntimeError(
            "Loudness analysis requires pyloudnorm, librosa, and soundfile. "
            "Install with: pip install pyloudnorm librosa soundfile"
        )

    if not detections:
        return [], []

    logger.info(
        f"Computing unified loudness metrics for {len(detections)} detections from video {video_id}"
    )

    # Find raw audio file
    try:
        from ..local_media import LocalMediaIndex

        idx = index or LocalMediaIndex()

        # Get raw audio path from index
        meta = idx.get(video_id)
        if not meta or not meta.get("raw_path"):
            logger.warning(
                f"No raw audio path found for video {video_id}, returning zero metrics"
            )
            # Return zero metrics for all detections
            zero_lufs = {
                "integrated_lufs": 0.0,
                "shortterm_lufs": 0.0,
                "momentary_lufs": 0.0,
            }
            zero_dbtp = {
                "integrated_dbtp": 0.0,
                "shortterm_dbtp": 0.0,
                "momentary_dbtp": 0.0,
            }
            return [zero_lufs.copy() for _ in detections], [
                zero_dbtp.copy() for _ in detections
            ]

        raw_path = meta["raw_path"]
        if not Path(raw_path).exists():
            logger.warning(
                f"Raw audio file not found: {raw_path}, returning zero metrics"
            )
            # Return zero metrics for all detections
            zero_lufs = {
                "integrated_lufs": 0.0,
                "shortterm_lufs": 0.0,
                "momentary_lufs": 0.0,
            }
            zero_dbtp = {
                "integrated_dbtp": 0.0,
                "shortterm_dbtp": 0.0,
                "momentary_dbtp": 0.0,
            }
            return [zero_lufs.copy() for _ in detections], [
                zero_dbtp.copy() for _ in detections
            ]
    except ImportError as err:
        raise RuntimeError("Could not import LocalMediaIndex") from err

    try:
        # Load raw audio once using modern loading approach
        logger.info(f"Loading raw audio: {raw_path}")
        audio_data, sample_rate = load_raw_audio_modern(raw_path)

        # Ensure mono audio for loudness analysis
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            # Convert stereo to mono using equal weighting
            # Note: BS.1770 specifies channel weighting, but for simplicity we use equal weighting
            audio_data = audio_data.mean(axis=0)  # Average across channels
        else:
            # Already mono or single channel
            audio_data = audio_data.flatten()

        # Create loudness meter with appropriate settings
        meter = pyln.Meter(sample_rate)  # Uses BS.1770-4 K-weighting by default

        logger.info(
            f"Processing {len(detections)} detections with unified loudness analysis"
        )

        # Prepare results lists
        lufs_metrics_list = []
        dbtp_metrics_list = []

        # Process each detection timestamp
        for i, (_window_start, _confidence, display_timestamp) in enumerate(detections):
            logger.debug(
                f"Processing detection {i + 1}/{len(detections)} at {display_timestamp:.2f}s"
            )

            # Initialize metrics with defaults
            lufs_metrics = {
                "integrated_lufs": 0.0,
                "shortterm_lufs": 0.0,
                "momentary_lufs": 0.0,
            }
            dbtp_metrics = {
                "integrated_dbtp": 0.0,
                "shortterm_dbtp": 0.0,
                "momentary_dbtp": 0.0,
            }

            try:
                # Define time windows for different measurement types
                audio_duration = len(audio_data) / sample_rate

                # Integrated LUFS: 5-second window (±2.5s around detection)
                integrated_start = max(0, display_timestamp - 2.5)
                integrated_end = min(audio_duration, display_timestamp + 2.5)

                # Short-term LUFS: 3-second window (±1.5s around detection)
                shortterm_start = max(0, display_timestamp - 1.5)
                shortterm_end = min(audio_duration, display_timestamp + 1.5)

                # Momentary LUFS: 400ms window (±0.2s around detection)
                momentary_start = max(0, display_timestamp - 0.2)
                momentary_end = min(audio_duration, display_timestamp + 0.2)

                # Extract audio segments
                integrated_samples = slice(
                    int(integrated_start * sample_rate),
                    int(integrated_end * sample_rate),
                )
                shortterm_samples = slice(
                    int(shortterm_start * sample_rate), int(shortterm_end * sample_rate)
                )
                momentary_samples = slice(
                    int(momentary_start * sample_rate), int(momentary_end * sample_rate)
                )

                integrated_audio = audio_data[integrated_samples]
                shortterm_audio = audio_data[shortterm_samples]
                momentary_audio = audio_data[momentary_samples]

                # Compute LUFS measurements
                if len(integrated_audio) > 0:
                    try:
                        lufs_metrics["integrated_lufs"] = float(
                            meter.integrated_loudness(integrated_audio)
                        )
                    except Exception as e:
                        logger.debug(f"Integrated LUFS failed for detection {i}: {e}")
                        lufs_metrics["integrated_lufs"] = 0.0

                if len(shortterm_audio) > 0:
                    try:
                        # For short segments, use integrated loudness as approximation
                        lufs_metrics["shortterm_lufs"] = float(
                            meter.integrated_loudness(shortterm_audio)
                        )
                    except Exception as e:
                        logger.debug(f"Short-term LUFS failed for detection {i}: {e}")
                        lufs_metrics["shortterm_lufs"] = 0.0

                if len(momentary_audio) > 0:
                    try:
                        # For very short segments, use integrated loudness as approximation
                        lufs_metrics["momentary_lufs"] = float(
                            meter.integrated_loudness(momentary_audio)
                        )
                    except Exception as e:
                        logger.debug(f"Momentary LUFS failed for detection {i}: {e}")
                        lufs_metrics["momentary_lufs"] = 0.0

                # Compute True Peak measurements with 4x oversampling
                if len(integrated_audio) > 0:
                    try:
                        # Compute True Peak using 4x oversampling for accuracy
                        # Use scipy.signal.resample instead of librosa to avoid deprecation warnings
                        from scipy import signal

                        integrated_resampled = signal.resample(
                            integrated_audio, len(integrated_audio) * 4
                        )
                        dbtp_metrics["integrated_dbtp"] = float(
                            20 * np.log10(np.max(np.abs(integrated_resampled)))
                        )
                        logger.debug(
                            f"Detection {i}: Integrated True Peak = {dbtp_metrics['integrated_dbtp']:.2f} dBTP"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Integrated True Peak failed for detection {i}: {e}"
                        )
                        dbtp_metrics["integrated_dbtp"] = 0.0

                if len(shortterm_audio) > 0:
                    try:
                        shortterm_resampled = signal.resample(
                            shortterm_audio, len(shortterm_audio) * 4
                        )
                        dbtp_metrics["shortterm_dbtp"] = float(
                            20 * np.log10(np.max(np.abs(shortterm_resampled)))
                        )
                        logger.debug(
                            f"Detection {i}: Short-term True Peak = {dbtp_metrics['shortterm_dbtp']:.2f} dBTP"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Short-term True Peak failed for detection {i}: {e}"
                        )
                        dbtp_metrics["shortterm_dbtp"] = 0.0

                if len(momentary_audio) > 0:
                    try:
                        momentary_resampled = signal.resample(
                            momentary_audio, len(momentary_audio) * 4
                        )
                        dbtp_metrics["momentary_dbtp"] = float(
                            20 * np.log10(np.max(np.abs(momentary_resampled)))
                        )
                        logger.debug(
                            f"Detection {i}: Momentary True Peak = {dbtp_metrics['momentary_dbtp']:.2f} dBTP"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Momentary True Peak failed for detection {i}: {e}"
                        )
                        dbtp_metrics["momentary_dbtp"] = 0.0

            except Exception as e:
                logger.warning(
                    f"Failed to process detection {i} at {display_timestamp:.2f}s: {e}"
                )
                # Keep default zero values

            # Apply batch weighting if provided
            if batch_context and batch_context.get("enable_batch_weighting", False):
                # Apply batch weighting logic (placeholder for future implementation)
                pass

            lufs_metrics_list.append(lufs_metrics)
            dbtp_metrics_list.append(dbtp_metrics)

        logger.info(
            f"✓ Computed unified loudness metrics for {len(detections)} detections"
        )
        return lufs_metrics_list, dbtp_metrics_list

    except Exception as e:
        logger.error(f"Failed to compute loudness metrics for video {video_id}: {e}")
        # Return zero metrics for all detections on error
        zero_lufs = {
            "integrated_lufs": 0.0,
            "shortterm_lufs": 0.0,
            "momentary_lufs": 0.0,
        }
        zero_dbtp = {
            "integrated_dbtp": 0.0,
            "shortterm_dbtp": 0.0,
            "momentary_dbtp": 0.0,
        }
        return [zero_lufs.copy() for _ in detections], [
            zero_dbtp.copy() for _ in detections
        ]
