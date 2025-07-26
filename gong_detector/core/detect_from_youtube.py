#!/usr/bin/env python3
"""Simple YouTube gong detector script.

Download YouTube videos, optionally trim audio segments, and detect gongs
using YAMNet. Designed for testing on real podcast episodes.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .results_utils import (
    print_summary,
    save_positive_samples,
    save_results_to_csv,
)
from .yamnet_runner import YAMNetGongDetector
from .youtube_utils import (
    cleanup_old_temp_files,
    create_folder_name_from_date,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    setup_directories,
)


def process_audio_with_yamnet(
    temp_audio: str, threshold: float
) -> tuple[list[tuple[float, float, float]], float, float]:
    """Process audio file with YAMNet detector.

    Args:
        temp_audio: Path to temporary audio file
        threshold: Confidence threshold for detection

    Returns:
        Tuple of (detections, total_duration, max_gong_confidence)
    """
    # Initialize YAMNet detector
    print("\nStep 2: Loading YAMNet model...")
    detector = YAMNetGongDetector()
    detector.load_model()

    # Process audio
    print("\nStep 3: Processing audio...")
    waveform, sample_rate = detector.load_and_preprocess_audio(temp_audio)

    # Run inference
    print("\nStep 4: Running gong detection...")
    scores, _, _ = detector.run_inference(waveform)

    # Detect gongs with duration validation
    total_duration = len(waveform) / sample_rate
    detections = detector.detect_gongs(
        scores=scores, confidence_threshold=threshold, audio_duration=total_duration
    )

    # Print results using detector's formatted output
    detector.print_detections(detections)

    max_gong_confidence = scores[:, 172].max()

    return detections, total_duration, max_gong_confidence


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Detect gongs in YouTube videos using YAMNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --start_time 5680 --duration 20
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --threshold 0.3 --save_csv results.csv
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --save_positive_samples
        """,
    )

    parser.add_argument("youtube_url", help="YouTube URL to process")
    parser.add_argument(
        "--start_time", type=int, help="Start time in seconds (optional)"
    )
    parser.add_argument("--duration", type=int, help="Duration in seconds (optional)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for gong detection (default: 0.4)",
    )
    parser.add_argument("--save_csv", help="Save results to CSV file (optional)")
    parser.add_argument(
        "--keep_audio",
        action="store_true",
        help="Keep temporary audio file for training data extraction",
    )
    parser.add_argument(
        "--save_positive_samples",
        action="store_true",
        help="Save detected gong segments to training data folder for human review",
    )

    return parser


def detect_from_youtube_comprehensive(
    youtube_url: str,
    threshold: float = 0.4,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
    save_positive_samples: bool = False,
    keep_audio: bool = False,
) -> dict[str, Any]:
    """Run YouTube gong detection and return comprehensive metadata.

    This function provides a programmatic interface for bulk processing,
    returning structured data instead of just printing results.

    Args:
        youtube_url: YouTube URL to process
        threshold: Confidence threshold for detection
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
        save_positive_samples: Whether to save detected segments
        keep_audio: Whether to keep temporary audio file

    Returns:
        Dictionary containing all detection metadata:
        - video_url: Original YouTube URL
        - video_title: Video title
        - upload_date: Upload date (YYYYMMDD format)
        - video_duration: Total video duration in seconds
        - max_confidence: Maximum confidence score in video
        - threshold: Detection threshold used
        - detections: List of detection tuples
        - detection_count: Number of detections found
        - success: Whether processing was successful
        - error_message: Error message if failed
    """
    # Setup directories
    temp_audio_dir, csv_results_dir = setup_directories()
    cleanup_old_temp_files(temp_audio_dir)

    # Create temporary audio file path
    temp_audio = create_temp_audio_path(temp_audio_dir)

    try:
        # Step 1: Download and process audio
        temp_audio, video_title, upload_date = download_and_trim_youtube_audio(
            url=youtube_url,
            output_path=temp_audio,
            start_time=start_time,
            duration=duration,
        )

        # Step 2-4: Process with YAMNet
        detections, total_duration, max_gong_confidence = process_audio_with_yamnet(
            temp_audio, threshold
        )

        # Save positive samples if requested
        if save_positive_samples and detections:
            # Create dated folder within positive samples directory
            folder_name = create_folder_name_from_date(upload_date)
            project_root = Path(__file__).parent.parent.parent
            positive_base_dir = (
                project_root
                / "gong_detector"
                / "training"
                / "data"
                / "raw_samples"
                / "positive"
            )
            positive_dir = positive_base_dir / folder_name
            save_positive_samples(detections, temp_audio, positive_dir)

        # Return comprehensive metadata
        return {
            "video_url": youtube_url,
            "video_title": video_title,
            "upload_date": upload_date,
            "video_duration": total_duration,
            "max_confidence": max_gong_confidence,
            "threshold": threshold,
            "detections": detections,
            "detection_count": len(detections),
            "success": True,
            "error_message": "",
        }

    except Exception as e:
        return {
            "video_url": youtube_url,
            "video_title": "",
            "upload_date": "",
            "video_duration": 0.0,
            "max_confidence": 0.0,
            "threshold": threshold,
            "detections": [],
            "detection_count": 0,
            "success": False,
            "error_message": str(e),
        }

    finally:
        # Clean up temporary file
        if not keep_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)


def main() -> None:
    """Run YouTube gong detection."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup directories
    temp_audio_dir, csv_results_dir = setup_directories()
    cleanup_old_temp_files(temp_audio_dir)

    # Create temporary audio file path
    temp_audio = create_temp_audio_path(temp_audio_dir)

    try:
        # Step 1: Download and process audio
        print("Step 1: Downloading and processing audio...")
        temp_audio, video_title, upload_date = download_and_trim_youtube_audio(
            url=args.youtube_url,
            output_path=temp_audio,
            start_time=args.start_time,
            duration=args.duration,
        )

        # Step 2-4: Process with YAMNet
        detections, total_duration, max_gong_confidence = process_audio_with_yamnet(
            temp_audio, args.threshold
        )

        # Save to CSV if requested
        if args.save_csv:
            save_results_to_csv(detections, args.save_csv, csv_results_dir)

        # Save positive samples if requested
        if args.save_positive_samples and detections:
            # Create dated folder within positive samples directory
            folder_name = create_folder_name_from_date(upload_date)
            # Use absolute path resolution to find the existing positive samples directory
            project_root = Path(__file__).parent.parent.parent
            positive_base_dir = (
                project_root
                / "gong_detector"
                / "training"
                / "data"
                / "raw_samples"
                / "positive"
            )
            positive_dir = positive_base_dir / folder_name
            print(f"\nSaving positive samples to: {positive_dir}")
            save_positive_samples(detections, temp_audio, positive_dir)

        # Print summary and max confidence
        print_summary(detections, total_duration)
        print(f"Max gong confidence: {max_gong_confidence:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Clean up temporary file
        if not args.keep_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
            print(f"Cleaned up temporary file: {temp_audio}")
        elif args.keep_audio and os.path.exists(temp_audio):
            print(f"Temporary file preserved for training: {temp_audio}")


if __name__ == "__main__":
    main()
