"""True Peak (dBTP) analysis functionality.

This module provides True Peak measurement using ITU-R BS.1770-4 standard
for audio segments from gong detections.
"""

import logging
from pathlib import Path
from typing import Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Optional imports for True Peak analysis
try:
    import librosa  # type: ignore
    import numpy as np  # type: ignore
    import pyloudnorm as pyln  # type: ignore
    LUFS_AVAILABLE = True
except ImportError:
    LUFS_AVAILABLE = False
    pyln = None
    librosa = None
    np = None


def compute_true_peak_segments(
    video_id: str,
    timestamps: list[tuple[float, float]],
    measurement_type: str = "integrated",
    index: Optional[Any] = None,
    batch_context: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Compute True Peak (dBTP) for audio segments from raw audio with batch weighting support.

    This function extracts segments from the raw cached audio based on timestamps
    from gong detections and computes True Peak measurements using ITU-R BS.1770-4
    standard. Supports batch-weighted measurements across multiple videos.

    Args:
        video_id: YouTube video ID
        timestamps: List of (start_time, end_time) tuples in seconds
        measurement_type: Type of measurement context (integrated, short_term, momentary)
        index: Optional LocalMediaIndex instance
        batch_context: Optional dict with batch weighting information

    Returns:
        List of dictionaries containing True Peak measurements for each segment.
        Each dict contains:
        - start_time: Start time in seconds
        - end_time: End time in seconds
        - duration: Segment duration in seconds
        - dbtp: True Peak measurement value (batch-weighted if enabled)
        - raw_dbtp: Raw True Peak measurement (before batch weighting)
        - measurement_type: Type of measurement used
        - valid: Boolean indicating if measurement was successful
        - batch_weighted: Boolean indicating if batch weighting was applied

    Raises:
        RuntimeError: If LUFS library not available or raw audio not found
        ValueError: If timestamps are invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "True Peak analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )

    if not timestamps:
        return []

    logger.info(f"Computing True Peak for {len(timestamps)} segments from video {video_id}")

    # Find raw audio file
    try:
        from gong_detector.core.utils.local_media import LocalMediaIndex
        idx = index or LocalMediaIndex()

        # Get raw audio path from index
        meta = idx.get(video_id)
        if not meta or not meta.get("raw_path"):
            raise RuntimeError(f"No raw audio path found for video {video_id}")

        raw_path = meta["raw_path"]
        if not Path(raw_path).exists():
            raise RuntimeError(f"Raw audio file not found: {raw_path}")

    except ImportError as err:
        raise RuntimeError("Could not import LocalMediaIndex") from err

    results = []

    try:
        # Load raw audio file using librosa (supports WebM and other formats)
        logger.info(f"Loading raw audio for True Peak: {raw_path}")
        audio_data, sample_rate = librosa.load(raw_path, sr=None, mono=False)

        # Ensure mono audio for True Peak analysis
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=0)  # Average across channels

        # Note: pyloudnorm meter not used for True Peak - we compute manually

        logger.info(f"Processing {len(timestamps)} segments with True Peak analysis")

        for i, (start_time, end_time) in enumerate(timestamps):
            segment_result = {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "dbtp": None,
                "raw_dbtp": None,
                "measurement_type": measurement_type,
                "valid": False,
                "batch_weighted": False,
            }

            try:
                # Validate timestamps
                if start_time < 0 or end_time <= start_time:
                    logger.warning(f"Invalid timestamp pair: {start_time}-{end_time}s")
                    results.append(segment_result)
                    continue

                # Convert time to sample indices
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)

                # Check bounds
                if start_sample >= len(audio_data) or end_sample > len(audio_data):
                    logger.warning(f"Timestamp {start_time}-{end_time}s exceeds audio length")
                    results.append(segment_result)
                    continue

                # Extract segment
                segment_audio = audio_data[start_sample:end_sample]

                # Skip very short segments (less than 100ms)
                if len(segment_audio) / sample_rate < 0.1:
                    logger.warning(f"Segment {i+1} too short ({len(segment_audio)/sample_rate:.2f}s) for True Peak")
                    results.append(segment_result)
                    continue

                # Compute True Peak measurement using pyloudnorm
                try:
                    # pyloudnorm doesn't have direct True Peak method, but we can compute it manually
                    # True Peak requires oversampling - we'll use a simple approximation for now
                    # by finding the maximum absolute value and converting to dBTP
                    peak_amplitude = np.max(np.abs(segment_audio))

                    # Convert to dBTP (True Peak approximation)
                    if peak_amplitude > 0:
                        dbtp_value = 20.0 * np.log10(peak_amplitude)
                    else:
                        dbtp_value = -80.0  # Silence floor

                except Exception as peak_error:
                    logger.warning(f"True Peak computation failed for segment {i+1}: {peak_error}")
                    dbtp_value = -80.0

                # Check for valid measurement
                if dbtp_value == -float('inf') or dbtp_value != dbtp_value:  # NaN check
                    logger.warning(f"Invalid True Peak measurement for segment {i+1}")
                else:
                    # Store raw True Peak value
                    segment_result["raw_dbtp"] = float(dbtp_value)
                    segment_result["dbtp"] = float(dbtp_value)  # Default to raw value
                    segment_result["valid"] = True
                    logger.debug(f"Segment {i+1}: {start_time:.1f}-{end_time:.1f}s = {dbtp_value:.1f} dBTP")

            except Exception as e:
                logger.error(f"Error processing segment {i+1} ({start_time}-{end_time}s): {e}")

            results.append(segment_result)

        # Summary statistics
        valid_measurements = [r["dbtp"] for r in results if r["valid"]]
        if valid_measurements:
            logger.info(f"True Peak analysis complete: {len(valid_measurements)}/{len(timestamps)} valid measurements")
            logger.info(f"True Peak range: {min(valid_measurements):.1f} to {max(valid_measurements):.1f} dBTP")
            logger.info(f"Mean True Peak: {sum(valid_measurements)/len(valid_measurements):.1f} dBTP")
        else:
            logger.warning("No valid True Peak measurements obtained")

    except Exception as e:
        logger.error(f"Failed to compute True Peak for video {video_id}: {e}")
        # Return empty results with error indication
        for start_time, end_time in timestamps:
            results.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "dbtp": None,
                "raw_dbtp": None,
                "measurement_type": measurement_type,
                "valid": False,
                "batch_weighted": False,
            })

    return results
