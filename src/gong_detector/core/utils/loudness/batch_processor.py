"""Batch processing for loudness measurements across multiple videos.

This module provides batch-weighted LUFS and True Peak analysis across
multiple videos for proper relative loudness analysis.
"""

import logging
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

# Optional imports for loudness analysis
try:
    import pyloudnorm as pyln  # type: ignore
    LUFS_AVAILABLE = True
except ImportError:
    LUFS_AVAILABLE = False
    pyln = None


def compute_batch_weighted_lufs(
    all_video_data: list[dict[str, Any]],
    measurement_type: str = "integrated",
    reference_lufs: float = -23.0,
) -> dict[str, list[dict[str, Any]]]:
    """Compute batch-weighted LUFS across all videos for proper relative analysis.

    This function processes all detection segments from multiple videos together,
    computes LUFS measurements, and applies batch weighting to normalize loudness
    measurements relative to the entire dataset rather than individual videos.

    Args:
        all_video_data: List of video data dicts, each containing:
            - "video_id": YouTube video ID
            - "timestamps": List of (start_time, end_time) tuples
            - "result": Detection result dict
        measurement_type: Type of LUFS measurement (integrated, short_term, momentary)
        reference_lufs: Reference LUFS level for batch normalization (default: -23.0 LUFS)

    Returns:
        Dictionary mapping video_id to list of LUFS measurement dicts for that video.
        Each measurement dict contains batch-weighted LUFS values.

    Raises:
        RuntimeError: If LUFS library not available
        ValueError: If video data is invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "LUFS analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )

    if not all_video_data:
        return {}

    logger.info(f"Computing batch-weighted LUFS for {len(all_video_data)} videos")

    # Step 1: Collect all raw LUFS measurements across all videos
    all_raw_lufs = []
    video_results = {}

    try:
        from gong_detector.core.utils.local_media import LocalMediaIndex
        index = LocalMediaIndex()
    except ImportError:
        logger.warning("Could not import LocalMediaIndex, using None")
        index = None

    # Import the unified analyzer (optimized, soundfile-only)
    from .unified_analyzer import compute_all_loudness_metrics

    # Process each video to get raw LUFS measurements
    for video_data in all_video_data:
        video_id = video_data["video_id"]
        timestamps = video_data["timestamps"]

        logger.info(f"Processing video {video_id} with {len(timestamps)} detection segments")

        # Convert timestamps to detection format for unified analyzer
        # Each timestamp is (start_time, end_time), convert to (window_start, confidence, display_timestamp)
        mock_detections = []
        for start_time, end_time in timestamps:
            # Use center of segment as display timestamp
            display_timestamp = (start_time + end_time) / 2.0
            mock_detections.append((start_time, 1.0, display_timestamp))  # confidence=1.0 for batch processing

        try:
            # Compute all LUFS metrics using unified analyzer (single audio load)
            lufs_metrics, _ = compute_all_loudness_metrics(
                video_id=video_id,
                detections=mock_detections,
                index=index
            )

            # Convert unified analyzer output to batch processor format
            lufs_results = []
            for i, (start_time, end_time) in enumerate(timestamps):
                if i < len(lufs_metrics):
                    # Extract the appropriate measurement type
                    if measurement_type == "integrated":
                        raw_lufs = lufs_metrics[i]["integrated_lufs"]
                    elif measurement_type == "short_term":
                        raw_lufs = lufs_metrics[i]["shortterm_lufs"]
                    elif measurement_type == "momentary":
                        raw_lufs = lufs_metrics[i]["momentary_lufs"]
                    else:
                        raw_lufs = lufs_metrics[i]["integrated_lufs"]  # fallback

                    lufs_results.append({
                        "valid": raw_lufs != 0,  # Unified analyzer returns 0 for failures
                        "raw_lufs": raw_lufs if raw_lufs != 0 else None,
                        "start_time": start_time,
                        "end_time": end_time,
                        "measurement_type": measurement_type
                    })
                else:
                    # Fallback for missing results
                    lufs_results.append({
                        "valid": False,
                        "raw_lufs": None,
                        "start_time": start_time,
                        "end_time": end_time,
                        "measurement_type": measurement_type
                    })
        except Exception as e:
            logger.warning(f"Unified LUFS analysis failed for video {video_id}: {e}")
            # Fallback to empty results
            lufs_results = []
            for start_time, end_time in timestamps:
                lufs_results.append({
                    "valid": False,
                    "raw_lufs": None,
                    "start_time": start_time,
                    "end_time": end_time,
                    "measurement_type": measurement_type
                })

        # Store results for this video
        video_results[video_id] = lufs_results

        # Collect valid raw LUFS measurements for batch statistics
        valid_lufs = [r["raw_lufs"] for r in lufs_results if r["valid"] and r["raw_lufs"] is not None]
        all_raw_lufs.extend(valid_lufs)

    # Step 2: Calculate batch statistics
    if not all_raw_lufs:
        logger.warning("No valid LUFS measurements found across all videos")
        return video_results

    batch_mean_lufs = sum(all_raw_lufs) / len(all_raw_lufs)
    batch_offset = reference_lufs - batch_mean_lufs

    logger.info("Batch LUFS statistics:")
    logger.info(f"  Total measurements: {len(all_raw_lufs)}")
    logger.info(f"  Batch mean: {batch_mean_lufs:.1f} LUFS")
    logger.info(f"  Reference level: {reference_lufs:.1f} LUFS")
    logger.info(f"  Batch offset: {batch_offset:.1f} dB")
    logger.info(f"  LUFS range: {min(all_raw_lufs):.1f} to {max(all_raw_lufs):.1f} LUFS")

    # Step 3: Apply batch weighting to all measurements
    total_weighted = 0
    for _video_id, lufs_results in video_results.items():
        for result in lufs_results:
            if result["valid"] and result["raw_lufs"] is not None:
                # Apply batch weighting
                result["lufs"] = result["raw_lufs"] + batch_offset
                result["batch_weighted"] = True
                total_weighted += 1

    logger.info(f"Applied batch weighting to {total_weighted} measurements across {len(video_results)} videos")

    return video_results


def compute_batch_weighted_dbtp(
    all_video_data: list[dict[str, Any]],
    measurement_type: str = "integrated",
    reference_dbtp: float = -1.0,
) -> dict[str, list[dict[str, Any]]]:
    """Compute batch-weighted True Peak across all videos for proper relative analysis.

    This function processes all detection segments from multiple videos together,
    computes True Peak measurements, and applies batch weighting to normalize
    measurements relative to the entire dataset.

    Args:
        all_video_data: List of video data dicts, each containing:
            - "video_id": YouTube video ID
            - "timestamps": List of (start_time, end_time) tuples
            - "result": Detection result dict
        measurement_type: Type of measurement context (integrated, short_term, momentary)
        reference_dbtp: Reference True Peak level for batch normalization (default: -1.0 dBTP)

    Returns:
        Dictionary mapping video_id to list of True Peak measurement dicts for that video.
        Each measurement dict contains batch-weighted True Peak values.

    Raises:
        RuntimeError: If LUFS library not available
        ValueError: If video data is invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "True Peak analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )

    if not all_video_data:
        return {}

    logger.info(f"Computing batch-weighted True Peak for {len(all_video_data)} videos")

    # Step 1: Collect all raw True Peak measurements across all videos
    all_raw_dbtp = []
    video_results = {}

    try:
        from gong_detector.core.utils.local_media import LocalMediaIndex
        index = LocalMediaIndex()
    except ImportError:
        logger.warning("Could not import LocalMediaIndex, using None")
        index = None

    # Import the unified analyzer (optimized, soundfile-only)
    from .unified_analyzer import compute_all_loudness_metrics

    # Process each video to get raw True Peak measurements
    for video_data in all_video_data:
        video_id = video_data["video_id"]
        timestamps = video_data["timestamps"]

        logger.info(f"Processing video {video_id} with {len(timestamps)} detection segments")

        # Convert timestamps to detection format for unified analyzer
        # Each timestamp is (start_time, end_time), convert to (window_start, confidence, display_timestamp)
        mock_detections = []
        for start_time, end_time in timestamps:
            # Use center of segment as display timestamp
            display_timestamp = (start_time + end_time) / 2.0
            mock_detections.append((start_time, 1.0, display_timestamp))  # confidence=1.0 for batch processing

        try:
            # Compute all True Peak metrics using unified analyzer (single audio load)
            _, dbtp_metrics = compute_all_loudness_metrics(
                video_id=video_id,
                detections=mock_detections,
                index=index
            )

            # Convert unified analyzer output to batch processor format
            dbtp_results = []
            for i, (start_time, end_time) in enumerate(timestamps):
                if i < len(dbtp_metrics):
                    # Extract the appropriate measurement type
                    if measurement_type == "integrated":
                        raw_dbtp = dbtp_metrics[i]["integrated_dbtp"]
                    elif measurement_type == "short_term":
                        raw_dbtp = dbtp_metrics[i]["shortterm_dbtp"]
                    elif measurement_type == "momentary":
                        raw_dbtp = dbtp_metrics[i]["momentary_dbtp"]
                    else:
                        raw_dbtp = dbtp_metrics[i]["integrated_dbtp"]  # fallback

                    dbtp_results.append({
                        "valid": raw_dbtp != 0,  # Unified analyzer returns 0 for failures
                        "raw_dbtp": raw_dbtp if raw_dbtp != 0 else None,
                        "start_time": start_time,
                        "end_time": end_time,
                        "measurement_type": measurement_type
                    })
                else:
                    # Fallback for missing results
                    dbtp_results.append({
                        "valid": False,
                        "raw_dbtp": None,
                        "start_time": start_time,
                        "end_time": end_time,
                        "measurement_type": measurement_type
                    })
        except Exception as e:
            logger.warning(f"Unified True Peak analysis failed for video {video_id}: {e}")
            # Fallback to empty results
            dbtp_results = []
            for start_time, end_time in timestamps:
                dbtp_results.append({
                    "valid": False,
                    "raw_dbtp": None,
                    "start_time": start_time,
                    "end_time": end_time,
                    "measurement_type": measurement_type
                })

        # Store results for this video
        video_results[video_id] = dbtp_results

        # Collect valid raw True Peak measurements for batch statistics
        valid_dbtp = [r["raw_dbtp"] for r in dbtp_results if r["valid"] and r["raw_dbtp"] is not None]
        all_raw_dbtp.extend(valid_dbtp)

    # Step 2: Calculate batch statistics
    if not all_raw_dbtp:
        logger.warning("No valid True Peak measurements found across all videos")
        return video_results

    batch_mean_dbtp = sum(all_raw_dbtp) / len(all_raw_dbtp)
    batch_offset = reference_dbtp - batch_mean_dbtp

    logger.info("Batch True Peak statistics:")
    logger.info(f"  Total measurements: {len(all_raw_dbtp)}")
    logger.info(f"  Batch mean: {batch_mean_dbtp:.1f} dBTP")
    logger.info(f"  Reference level: {reference_dbtp:.1f} dBTP")
    logger.info(f"  Batch offset: {batch_offset:.1f} dB")
    logger.info(f"  True Peak range: {min(all_raw_dbtp):.1f} to {max(all_raw_dbtp):.1f} dBTP")

    # Step 3: Apply batch weighting to all measurements
    total_weighted = 0
    for _video_id, dbtp_results in video_results.items():
        for result in dbtp_results:
            if result["valid"] and result["raw_dbtp"] is not None:
                # Apply batch weighting
                result["dbtp"] = result["raw_dbtp"] + batch_offset
                result["batch_weighted"] = True
                total_weighted += 1

    logger.info(f"Applied batch weighting to {total_weighted} measurements across {len(video_results)} videos")

    return video_results
