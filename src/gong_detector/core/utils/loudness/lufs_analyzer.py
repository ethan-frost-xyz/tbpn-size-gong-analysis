"""LUFS (Loudness Units relative to Full Scale) analysis functionality.

This module provides LUFS loudness measurement using BS.1770-4 K-weighting
and EBU R128 gating for audio segments from gong detections.
"""

import logging
from pathlib import Path
from typing import Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Optional imports for LUFS analysis
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


def compute_lufs_segments(
    video_id: str,
    timestamps: list[tuple[float, float]],
    measurement_type: str = "integrated",
    index: Optional[Any] = None,
    batch_context: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Compute LUFS loudness for audio segments from raw audio with batch weighting support.

    This function extracts segments from the raw cached audio based on timestamps
    from gong detections and computes LUFS loudness measurements using BS.1770-4
    K-weighting and EBU R128 gating. Supports batch-weighted measurements across
    multiple videos for proper relative loudness analysis.

    Args:
        video_id: YouTube video ID
        timestamps: List of (start_time, end_time) tuples in seconds
        measurement_type: Type of LUFS measurement:
            - "integrated": Integrated loudness over entire segment
            - "short_term": Short-term loudness (3s sliding window)
            - "momentary": Momentary loudness (400ms sliding window)
        index: Optional LocalMediaIndex instance
        batch_context: Optional dict with batch weighting information:
            - "all_segments": List of all audio segments from all videos
            - "reference_lufs": Reference LUFS level for batch normalization
            - "enable_batch_weighting": Boolean to enable batch weighting

    Returns:
        List of dictionaries containing LUFS measurements for each segment.
        Each dict contains:
        - start_time: Start time in seconds
        - end_time: End time in seconds
        - duration: Segment duration in seconds
        - lufs: LUFS measurement value (batch-weighted if enabled)
        - raw_lufs: Raw LUFS measurement (before batch weighting)
        - measurement_type: Type of measurement used
        - valid: Boolean indicating if measurement was successful
        - batch_weighted: Boolean indicating if batch weighting was applied

    Raises:
        RuntimeError: If LUFS library not available or raw audio not found
        ValueError: If timestamps are invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "LUFS analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )

    if not timestamps:
        return []

    # Validate measurement type
    valid_types = ["integrated", "short_term", "momentary"]
    if measurement_type not in valid_types:
        raise ValueError(f"measurement_type must be one of {valid_types}")

    logger.info(f"Computing LUFS for {len(timestamps)} segments from video {video_id}")

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
        logger.info(f"Loading raw audio: {raw_path}")
        audio_data, sample_rate = librosa.load(raw_path, sr=None, mono=False)

        # Ensure mono audio for LUFS analysis
        if len(audio_data.shape) > 1:
            # Convert stereo to mono using equal weighting
            # Note: BS.1770 specifies channel weighting, but for simplicity we use equal weighting
            audio_data = audio_data.mean(axis=0)  # Average across channels (axis 0)

        # Create loudness meter with appropriate settings
        meter = pyln.Meter(sample_rate)  # Uses BS.1770-4 K-weighting by default

        logger.info(f"Processing {len(timestamps)} segments with {measurement_type} LUFS")

        for i, (start_time, end_time) in enumerate(timestamps):
            segment_result = {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "lufs": None,
                "raw_lufs": None,
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

                # Skip very short segments (less than 400ms for momentary, 3s for short-term)
                min_duration = 0.4 if measurement_type == "momentary" else 3.0 if measurement_type == "short_term" else 0.1
                if len(segment_audio) / sample_rate < min_duration:
                    logger.warning(f"Segment {i+1} too short ({len(segment_audio)/sample_rate:.2f}s) for {measurement_type} LUFS")
                    results.append(segment_result)
                    continue

                # Compute LUFS measurement
                if measurement_type == "integrated":
                    # Integrated loudness over entire segment with EBU R128 gating
                    lufs_value = meter.integrated_loudness(segment_audio)
                elif measurement_type == "short_term":
                    # Short-term loudness (3s sliding window)
                    # pyloudnorm doesn't have separate short_term method, use integrated for now
                    # This is a simplified implementation - in production you'd implement proper 3s sliding window
                    lufs_value = meter.integrated_loudness(segment_audio)
                    logger.warning("Short-term LUFS approximated using integrated loudness")
                else:  # momentary
                    # Momentary loudness (400ms sliding window)
                    # pyloudnorm doesn't have separate momentary method, use integrated for now
                    # This is a simplified implementation - in production you'd implement proper 400ms sliding window
                    lufs_value = meter.integrated_loudness(segment_audio)
                    logger.warning("Momentary LUFS approximated using integrated loudness")

                # Check for valid measurement
                if lufs_value == -float('inf') or lufs_value != lufs_value:  # NaN check
                    logger.warning(f"Invalid LUFS measurement for segment {i+1}")
                else:
                    # Store raw LUFS value
                    segment_result["raw_lufs"] = float(lufs_value)
                    segment_result["lufs"] = float(lufs_value)  # Default to raw value
                    segment_result["valid"] = True
                    logger.debug(f"Segment {i+1}: {start_time:.1f}-{end_time:.1f}s = {lufs_value:.1f} LUFS")

            except Exception as e:
                logger.error(f"Error processing segment {i+1} ({start_time}-{end_time}s): {e}")

            results.append(segment_result)

        # Apply batch weighting if provided
        if batch_context and batch_context.get("enable_batch_weighting", False):
            logger.info("Applying batch weighting to LUFS measurements...")
            reference_lufs = batch_context.get("reference_lufs", -23.0)  # EBU R128 reference

            # Get all valid measurements for batch statistics
            valid_results = [r for r in results if r["valid"]]
            if valid_results:
                # Calculate batch statistics
                all_lufs = [r["raw_lufs"] for r in valid_results]
                batch_mean = sum(all_lufs) / len(all_lufs)

                # Apply batch weighting: adjust relative to batch mean and reference
                batch_offset = reference_lufs - batch_mean

                for result in valid_results:
                    # Apply batch weighting
                    result["lufs"] = result["raw_lufs"] + batch_offset
                    result["batch_weighted"] = True

                logger.info(f"Batch weighting applied: offset = {batch_offset:.1f} dB")
                logger.info(f"Batch mean: {batch_mean:.1f} LUFS → Reference: {reference_lufs:.1f} LUFS")

        # Summary statistics
        valid_measurements = [r["lufs"] for r in results if r["valid"]]
        if valid_measurements:
            logger.info(f"LUFS analysis complete: {len(valid_measurements)}/{len(timestamps)} valid measurements")
            logger.info(f"LUFS range: {min(valid_measurements):.1f} to {max(valid_measurements):.1f} LUFS")
            logger.info(f"Mean LUFS: {sum(valid_measurements)/len(valid_measurements):.1f} LUFS")

            # Show batch weighting info if applied
            batch_weighted_count = sum(1 for r in results if r.get("batch_weighted", False))
            if batch_weighted_count > 0:
                logger.info(f"Batch weighting applied to {batch_weighted_count} measurements")
        else:
            logger.warning("No valid LUFS measurements obtained")

    except Exception as e:
        logger.error(f"Failed to compute LUFS for video {video_id}: {e}")
        # Return empty results with error indication
        for start_time, end_time in timestamps:
            results.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "lufs": None,
                "measurement_type": measurement_type,
                "valid": False,
            })

    return results
