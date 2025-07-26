"""Results handling utilities for gong detection.

This module provides functions for saving detection results,
managing positive samples, and formatting output for the
gong detection pipeline.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .audio_utils import extract_audio_slice
from .yamnet_runner import YAMNetGongDetector


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
    detections: list[tuple[float, float, float]], audio_path: str, positive_dir: Path
) -> None:
    """Save detected gong segments to positive samples folder.

    Args:
        detections: List of (window_start, confidence, display_timestamp) tuples
        audio_path: Path to source audio file
        positive_dir: Directory to save positive samples
    """
    if not detections:
        print("No gong detections to save")
        return

    # Load audio waveform
    detector = YAMNetGongDetector()
    waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)

    # Create positive directory if it doesn't exist
    positive_dir.mkdir(parents=True, exist_ok=True)

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
            output_path = positive_dir / filename

            # Convert numpy array to WAV file
            import soundfile as sf

            sf.write(str(output_path), segment, sample_rate)

            saved_count += 1
            print(f"✓ Saved: {filename}")

        except Exception as e:
            print(f"✗ Failed to save segment {i + 1}: {e}")

    print(f"\nSaved {saved_count} positive samples to: {positive_dir}")
