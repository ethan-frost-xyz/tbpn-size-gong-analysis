"""Results handling utilities for gong detection.

This module provides functions for saving detection results,
managing positive samples, formatting output, and consolidating
overlapping detections for the gong detection pipeline.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

from ..detector.yamnet_runner import YAMNetGongDetector
from .audio_utils import extract_audio_slice

# Constants
DEFAULT_CONSOLIDATION_WINDOW: float = 3.0  # Default time window in seconds

# Configure logging
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_time_for_filename(seconds: float) -> str:
    """Format seconds as HH_MM_SS string for filenames.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string in HH_MM_SS format (underscores instead of colons)
    """
    return format_time(seconds).replace(":", "_")


def print_summary(
    detections: list[tuple[float, float, float]],
    total_duration: float,
) -> None:
    """Print detection summary.

    Args:
        detections: List of (window_start, confidence, display_timestamp) tuples
        total_duration: Total audio duration in seconds
    """
    count = len(detections)
    start_time = format_time(0.0)
    end_time = format_time(total_duration)

    print(f"\nSUMMARY: Detected {count} gongs between {start_time} and {end_time}")

    if count > 0:
        confidences = [d[1] for d in detections]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Maximum confidence: {max_confidence:.3f}")


def save_results_to_csv(
    detections: list[tuple[float, float, float]], csv_filename: str, csv_dir: str
) -> None:
    """Save detection results to CSV file.

    Args:
        detections: List of (window_start, confidence, display_timestamp) tuples
        csv_filename: Name of the CSV file
        csv_dir: Directory to save CSV files
    """
    if not csv_filename.endswith(".csv"):
        csv_filename += ".csv"

    # Ensure directory exists
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, csv_filename)
    detector = YAMNetGongDetector()
    df = detector.detections_to_dataframe(detections)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")


def save_positive_samples(
    detections: list[tuple[float, float, float]],
    audio_path: str,
    positive_dir: Path,
    upload_date: str = "",
    video_title: str = "",
) -> None:
    """Save detected gong segments to positive samples folder.

    Args:
        detections: List of (window_start, confidence, display_timestamp) tuples
        audio_path: Path to source audio file
        positive_dir: Base directory for positive samples (will create date-based subfolder)
        upload_date: YouTube upload date (YYYYMMDD format) for proper folder naming
        video_title: Video title from YouTube for date-based folder naming
    """
    if not detections:
        print("No gong detections to save")
        return

    # Create date-based folder name using video title if available, otherwise use upload_date
    if video_title:
        from .youtube_utils import create_folder_name_from_title

        folder_name = create_folder_name_from_title(video_title)
        final_positive_dir = positive_dir / folder_name
    elif upload_date:
        from .youtube_utils import create_folder_name_from_date

        folder_name = create_folder_name_from_date(upload_date)
        final_positive_dir = positive_dir / folder_name
    else:
        # Fallback to using the passed directory directly (for backwards compatibility)
        final_positive_dir = positive_dir

    # Load audio waveform
    detector = YAMNetGongDetector()
    waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)

    # Create positive directory if it doesn't exist
    final_positive_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for i, (_window_start, confidence, display_timestamp) in enumerate(detections):
        try:
            # Extract 3-second segment around detection (use display timestamp for center)
            segment = extract_audio_slice(
                waveform,
                display_timestamp,
                duration_before=0.75,
                duration_after=2.25,
                sample_rate=sample_rate,
            )

            # Save segment with descriptive filename (use display timestamp for filename)
            filename = f"at_{format_time_for_filename(display_timestamp)}_s_conf_{confidence:.3f}_{i + 1}.wav"
            output_path = final_positive_dir / filename

            # Convert numpy array to WAV file
            import soundfile as sf

            sf.write(str(output_path), segment, sample_rate)

            saved_count += 1
            print(f"✓ Saved: {filename}")

        except Exception as e:
            print(f"✗ Failed to save segment {i + 1}: {e}")

    print(f"\nSaved {saved_count} positive samples to: {final_positive_dir}")


def consolidate_overlapping_detections(
    detections: list[tuple[float, float, float]],
    consolidation_window: Optional[float] = None,
) -> list[tuple[float, float, float]]:
    """Consolidate overlapping detections using improved clustering algorithm.

    Uses a more robust approach that properly handles overlapping detection windows
    by clustering detections that are close in time, regardless of processing order.

    Args:
        detections: List of detection tuples in format:
            (window_start, confidence, display_timestamp)
            where display_timestamp is used for temporal grouping
        consolidation_window: Time window in seconds for grouping detections.
            Detections within this window are considered part of the same
            gong event. Defaults to DEFAULT_CONSOLIDATION_WINDOW (3.0s)

    Returns:
        List of consolidated detection tuples with one detection per gong event,
        each representing the peak confidence detection from its temporal cluster.

    Raises:
        ValueError: If consolidation_window is negative or zero
        TypeError: If detections contain invalid tuple structure

    Examples:
        >>> detections = [
        ...     (10.0, 0.95, 10.5),  # Gong 1: cluster
        ...     (10.5, 0.97, 11.0),  # Gong 1: peak
        ...     (11.0, 0.94, 11.5),  # Gong 1: trailing
        ...     (20.0, 0.92, 20.5),  # Gong 2: isolated
        ... ]
        >>> consolidated = consolidate_overlapping_detections(detections, 3.0)
        >>> len(consolidated)
        2
        >>> consolidated[0][1]  # First detection confidence
        0.97
    """
    if consolidation_window is None:
        consolidation_window = DEFAULT_CONSOLIDATION_WINDOW

    # Input validation
    if consolidation_window <= 0:
        raise ValueError(
            f"consolidation_window must be positive, got {consolidation_window}"
        )

    if not detections:
        logger.debug("No detections provided for consolidation")
        return []

    try:
        # Validate detection format and sort by display_timestamp (3rd element)
        sorted_detections = sorted(detections, key=lambda x: x[2])
        logger.debug(
            f"Consolidating {len(detections)} detections with "
            f"{consolidation_window}s window"
        )

    except (IndexError, TypeError) as e:
        raise TypeError(
            "Invalid detection format. Expected list of "
            "(window_start, confidence, display_timestamp) tuples"
        ) from e

    # Improved clustering algorithm using Union-Find approach
    clusters: list[list[tuple[float, float, float]]] = []

    for detection in sorted_detections:
        try:
            window_start, confidence, display_timestamp = detection

            # Find if this detection belongs to any existing cluster
            merged_with_cluster = False

            for cluster in clusters:
                # Check if detection is within window of ANY detection in cluster
                for cluster_detection in cluster:
                    time_diff = abs(display_timestamp - cluster_detection[2])
                    if time_diff <= consolidation_window:
                        cluster.append(detection)
                        merged_with_cluster = True
                        break

                if merged_with_cluster:
                    break

            # If not merged with any cluster, create new cluster
            if not merged_with_cluster:
                clusters.append([detection])

        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid detection {detection}: {e}")
            continue

    # Merge clusters that may have become connected through new detections
    merged_clusters = _merge_connected_clusters(clusters, consolidation_window)

    # Select peak detection from each final cluster
    consolidated: list[tuple[float, float, float]] = []
    for cluster in merged_clusters:
        if cluster:  # Safety check
            peak_detection = max(cluster, key=lambda x: x[1])
            consolidated.append(peak_detection)

    # Sort final results by timestamp
    consolidated.sort(key=lambda x: x[2])

    reduction_count = len(detections) - len(consolidated)
    logger.info(
        f"Consolidated {len(detections)} → {len(consolidated)} detections "
        f"(removed {reduction_count} overlaps)"
    )

    return consolidated


def _merge_connected_clusters(
    clusters: list[list[tuple[float, float, float]]], window: float
) -> list[list[tuple[float, float, float]]]:
    """Merge clusters that became connected through intermediate detections.

    Args:
        clusters: List of detection clusters
        window: Time window for connecting clusters

    Returns:
        List of merged clusters
    """
    if len(clusters) <= 1:
        return clusters

    merged = True
    while merged:
        merged = False
        new_clusters = []
        processed = set()

        for i, cluster_a in enumerate(clusters):
            if i in processed:
                continue

            current_cluster = cluster_a[:]

            for j, cluster_b in enumerate(clusters[i + 1 :], i + 1):
                if j in processed:
                    continue

                # Check if any detection in cluster_a is within window of any in cluster_b
                should_merge = False
                for det_a in cluster_a:
                    for det_b in cluster_b:
                        if abs(det_a[2] - det_b[2]) <= window:
                            should_merge = True
                            break
                    if should_merge:
                        break

                if should_merge:
                    current_cluster.extend(cluster_b)
                    processed.add(j)
                    merged = True

            new_clusters.append(current_cluster)
            processed.add(i)

        clusters = new_clusters

    return clusters
