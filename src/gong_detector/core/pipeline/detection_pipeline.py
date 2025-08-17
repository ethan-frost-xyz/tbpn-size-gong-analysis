#!/usr/bin/env python3
"""Simple YouTube gong detector script.

Download YouTube videos, optionally trim audio segments, and detect gongs
using YAMNet. Designed for testing on real podcast episodes.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from ..detector.yamnet_runner import YAMNetGongDetector
from ..utils.local_media import (
    ensure_preprocessed_audio,
)
from ..utils.results_utils import (
    consolidate_overlapping_detections,
    print_summary,
    save_positive_samples,
    save_results_to_csv,
)
from ..utils.youtube_utils import (
    cleanup_old_temp_files,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    setup_directories,
    video_id_from_url,
)


def process_audio_with_yamnet(
    temp_audio: str,
    threshold: float,
    max_threshold: Optional[float] = None,
    use_version_one: bool = False,
    batch_size: int = 4000,
    consolidate_detections: bool = True,
) -> tuple[list[tuple[float, float, float]], float, float]:
    """Process audio file with YAMNet detector.

    Args:
        temp_audio: Path to temporary audio file
        threshold: Confidence threshold for detection
        max_threshold: Maximum confidence threshold for detection (optional)
        use_version_one: Whether to use the trained classifier for enhanced detection
        batch_size: Batch size for classifier predictions (larger = faster but more memory)
        consolidate_detections: Whether to consolidate overlapping detections (default: True)

    Returns:
        Tuple of (consolidated_detections, total_duration, max_gong_confidence)
        Note: detections are automatically consolidated to remove sliding-window overlaps
    """
    # Initialize YAMNet detector with optimized settings
    print("\nStep 2: Loading YAMNet model...")
    detector = YAMNetGongDetector(
        use_trained_classifier=use_version_one, batch_size=batch_size
    )
    detector.load_model()

    if use_version_one:
        detector.load_trained_classifier()
        # Print performance configuration
        perf_info = detector.get_performance_info()
        print(
            f"[OK] Performance optimized: {perf_info['tensorflow_threads']['inter_op']} inter-op threads, {perf_info['tensorflow_threads']['intra_op']} intra-op threads"
        )
        print(f"[OK] Batch processing: {perf_info['batch_size']} embeddings per batch")
        print(f"[OK] Hardware: {perf_info['available_devices']['cpu']} CPU(s), {perf_info['available_devices']['gpu']} GPU(s)")
        if perf_info['available_devices']['gpu'] > 0:
            print(f"[OK] GPU acceleration: {perf_info['mixed_precision']} precision")

    # Process audio
    print("\nStep 3: Processing audio...")
    waveform, sample_rate = detector.load_and_preprocess_audio(temp_audio)

    # Run inference
    print("\nStep 4: Running gong detection...")
    scores, embeddings, _ = detector.run_inference(waveform)

    # Detect gongs with duration validation
    total_duration = len(waveform) / sample_rate

    if use_version_one:
        detections = detector.detect_gongs_with_classifier(
            embeddings=embeddings,
            confidence_threshold=threshold,
            max_confidence_threshold=max_threshold,
            audio_duration=total_duration,
        )
    else:
        detections = detector.detect_gongs(
            scores=scores,
            confidence_threshold=threshold,
            max_confidence_threshold=max_threshold,
            audio_duration=total_duration,
        )

    # Print results using detector's formatted output
    detector.print_detections(detections)

    # Configure logging for this module
    logger = logging.getLogger(__name__)

    # Filter out detections within first 180 seconds (intro content)
    print("\nStep 5a: Filtering early detections (first 180 seconds)...")
    early_threshold = 180.0  # 3 minutes
    original_count = len(detections)

    # Filter detections based on display timestamp (3rd element)
    detections = [
        detection for detection in detections
        if detection[2] >= early_threshold
    ]

    filtered_count = len(detections)
    early_removed = original_count - filtered_count

    if early_removed > 0:
        print(f"[OK] Filtered out {early_removed} early detections (< {early_threshold}s)")
    else:
        print("[OK] No early detections found to filter")

    # Consolidate overlapping detections (remove duplicates from sliding window)
    if consolidate_detections:
        print("\nStep 5b: Consolidating overlapping detections...")
        pre_consolidation_count = len(detections)

        try:
            detections = consolidate_overlapping_detections(detections)
            consolidated_count = len(detections)

            if pre_consolidation_count != consolidated_count:
                overlap_reduction = pre_consolidation_count - consolidated_count
                print(
                    f"✓ Consolidated {pre_consolidation_count} → {consolidated_count} "
                    f"detections (removed {overlap_reduction} overlaps)"
                )
                # Print final consolidated results
                print("\nFinal processed detections:")
                detector.print_detections(detections)
            else:
                print("[OK] No overlapping detections found")

        except (ValueError, TypeError) as e:
            logger.error(f"Consolidation failed: {e}")
            print(f"[WARNING] Consolidation failed, using filtered detections: {e}")
    else:
        print("\nStep 5b: Skipping detection consolidation (disabled)")

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
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --threshold 0.3 --max_threshold 0.8
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
        default=0.94,
        help="Confidence threshold for gong detection (default: 0.94)",
    )
    parser.add_argument(
        "--max_threshold",
        type=float,
        default=None,
        help="Maximum confidence threshold for gong detection (optional)",
    )
    parser.add_argument(
        "--use_version_one",
        action="store_true",
        help="Use the trained classifier for enhanced gong detection",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4000,
        help="Batch size for classifier predictions (larger = faster but more memory, default: 4000)",
    )
    parser.add_argument(
        "--no_consolidate",
        action="store_true",
        help="Disable consolidation of overlapping detections (keep all raw detections)",
    )

    return parser


def detect_from_youtube_comprehensive(
    youtube_url: str,
    threshold: float = 0.94,
    max_threshold: Optional[float] = None,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
    should_save_positive_samples: bool = False,
    keep_audio: bool = False,
    use_version_one: bool = False,
    batch_size: int = 4000,
    consolidate_detections: bool = True,
    use_local_media: bool = False,
    local_only: bool = False,
) -> dict[str, Any]:
    """Run YouTube gong detection and return comprehensive metadata.

    This function provides a programmatic interface for bulk processing,
    returning structured data instead of just printing results.

    Args:
        youtube_url: YouTube URL to process
        threshold: Confidence threshold for detection
        max_threshold: Maximum confidence threshold for detection (optional)
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
        should_save_positive_samples: Whether to save detected segments
        keep_audio: Whether to keep temporary audio file
        use_version_one: Whether to use the trained classifier for enhanced detection

    Returns:
        Dictionary containing all detection metadata:
        - video_url: Original YouTube URL
        - video_title: Video title
        - upload_date: Upload date (YYYYMMDD format)
        - video_duration: Total video duration in seconds
        - max_confidence: Maximum confidence score in video
        - threshold: Detection threshold used
        - max_threshold: Maximum threshold used (if any)
        - detections: List of detection tuples
        - detection_count: Number of detections found
        - success: Whether processing was successful
        - error_message: Error message if failed
        - video_loudness_metrics: Video-level loudness metrics
        - detection_loudness_metrics: Detection-level loudness metrics
        - detection_lufs_metrics: Detection-level LUFS metrics
        - detection_dbtp_metrics: Detection-level True Peak metrics
    """
    # Setup directories
    temp_audio_dir, csv_results_dir = setup_directories()
    cleanup_old_temp_files(temp_audio_dir)

    # Create temporary audio file path (used when downloading/transcoding)
    temp_audio = create_temp_audio_path(temp_audio_dir)
    local_audio_used = False

    try:
        # Step 1: Get audio path via local cache or download
        video_title = ""
        upload_date = ""

        if use_local_media or local_only:
            vid = video_id_from_url(youtube_url)
            if not vid:
                raise ValueError("Could not extract video_id from URL for local cache")
            local_path, meta = ensure_preprocessed_audio(
                video_id=vid,
                url=youtube_url,
                start=start_time,
                duration=duration,
                local_only=local_only,
            )
            # Prefer index metadata if available
            video_title = meta.get("video_title", "")
            upload_date = meta.get("upload_date", "")
            temp_audio = local_path
            local_audio_used = True
        else:
            temp_audio, video_title, upload_date = download_and_trim_youtube_audio(
                url=youtube_url,
                output_path=temp_audio,
                start_time=start_time,
                duration=duration,
            )

        # Step 2-5: Process with YAMNet
        detections, total_duration, max_gong_confidence = process_audio_with_yamnet(
            temp_audio,
            threshold,
            max_threshold,
            use_version_one,
            batch_size,
            consolidate_detections,
        )

        # Compute loudness metrics

        from ..utils.audio_utils import (
            analyze_audio_slice_levels,
            compute_loudness_metrics,
        )

        # Load audio for loudness analysis
        detector = YAMNetGongDetector(
            use_trained_classifier=use_version_one, batch_size=batch_size
        )
        waveform, sample_rate = detector.load_and_preprocess_audio(temp_audio)

        # Compute video-level loudness metrics
        video_loudness_metrics = compute_loudness_metrics(waveform)

        # Compute detection-level loudness metrics
        detection_loudness_metrics = []
        for _window_start, _confidence, display_timestamp in detections:
            # Analyze audio slice around detection timestamp
            peak_dbfs, rms_dbfs = analyze_audio_slice_levels(
                waveform,
                display_timestamp,
                context_seconds=10.0,
                sample_rate=sample_rate,
            )

            # Extract slice for detailed analysis
            from ..utils.audio_utils import get_slice_around_timestamp

            audio_slice = get_slice_around_timestamp(
                waveform,
                display_timestamp,
                context_seconds=10.0,
                sample_rate=sample_rate,
            )

            # Compute comprehensive metrics for the slice
            slice_metrics = compute_loudness_metrics(audio_slice)

            detection_loudness_metrics.append(
                {
                    "peak_dbfs": peak_dbfs,
                    "rms_dbfs": rms_dbfs,
                    "crest_factor": slice_metrics["crest_factor"],
                    "likely_clipped": slice_metrics["likely_clipped"],
                    "peak_amplitude": slice_metrics.get("peak_amplitude", 0.0),
                    "rms_amplitude": slice_metrics.get("rms_amplitude", 0.0),
                }
            )

        # Compute LUFS and True Peak metrics for each detection
        detection_lufs_metrics = []
        detection_dbtp_metrics = []

        # Import unified loudness analyzer (optimized single-pass processing)
        from ..utils.loudness import compute_all_loudness_metrics

        if detections:
            try:
                # Get video ID for local media access
                video_id = video_id_from_url(youtube_url)
                if video_id:
                    # Compute all LUFS and True Peak metrics in single audio pass
                    # This replaces 6 separate function calls with 1 optimized call
                    detection_lufs_metrics, detection_dbtp_metrics = compute_all_loudness_metrics(
                        video_id=video_id,
                        detections=detections
                    )
                    print(f"[OK] Computed LUFS and True Peak for {len(detections)} detections")
                else:
                    # Fallback to zeros if no video ID
                    detection_lufs_metrics = [{"integrated_lufs": 0, "shortterm_lufs": 0, "momentary_lufs": 0} for _ in detections]
                    detection_dbtp_metrics = [{"integrated_dbtp": 0, "shortterm_dbtp": 0, "momentary_dbtp": 0} for _ in detections]

            except Exception as e:
                print(f"[WARNING] LUFS computation failed: {e}")
                # Fallback to zeros on error
                detection_lufs_metrics = [{"integrated_lufs": 0, "shortterm_lufs": 0, "momentary_lufs": 0} for _ in detections]
                detection_dbtp_metrics = [{"integrated_dbtp": 0, "shortterm_dbtp": 0, "momentary_dbtp": 0} for _ in detections]
        else:
            # No detections, empty metrics
            detection_lufs_metrics = []
            detection_dbtp_metrics = []

        # Save positive samples if requested
        if should_save_positive_samples and detections:
            # Use date-based folder naming for consistency
            project_root = Path(__file__).parent.parent.parent.parent
            positive_base_dir = (
                project_root
                / "gong_detector"
                / "training"
                / "data"
                / "raw_samples"
                / "positive"
            )
            save_positive_samples(
                detections, temp_audio, positive_base_dir, upload_date, video_title
            )

        # Return comprehensive metadata
        return {
            "video_url": youtube_url,
            "video_title": video_title,
            "upload_date": upload_date,
            "video_duration": total_duration,
            "max_confidence": max_gong_confidence,
            "threshold": threshold,
            "max_threshold": max_threshold,
            "detections": detections,
            "detection_count": len(detections),
            "success": True,
            "error_message": "",
            "video_loudness_metrics": video_loudness_metrics,
            "detection_loudness_metrics": detection_loudness_metrics,
            "detection_lufs_metrics": detection_lufs_metrics,
            "detection_dbtp_metrics": detection_dbtp_metrics,
        }

    except Exception as e:
        return {
            "video_url": youtube_url,
            "video_title": "",
            "upload_date": "",
            "video_duration": 0.0,
            "max_confidence": 0.0,
            "threshold": threshold,
            "max_threshold": max_threshold,
            "detections": [],
            "detection_count": 0,
            "success": False,
            "error_message": str(e),
            "video_loudness_metrics": {},
            "detection_loudness_metrics": [],
            "detection_lufs_metrics": [],
            "detection_dbtp_metrics": [],
        }

    finally:
        # Clean up temporary file. Never delete when using a local cached file.
        if not local_audio_used and not keep_audio and os.path.exists(temp_audio):
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

        # Step 2-5: Process with YAMNet
        detections, total_duration, max_gong_confidence = process_audio_with_yamnet(
            temp_audio,
            args.threshold,
            args.max_threshold,
            args.use_version_one,
            args.batch_size,
            not args.no_consolidate,  # Invert the flag: no_consolidate=False means consolidate=True
        )

        # Save to CSV if requested
        if args.save_csv:
            save_results_to_csv(detections, args.save_csv, csv_results_dir)

        # Save positive samples if requested
        if args.save_positive_samples and detections:
            # Use date-based folder naming for consistency
            project_root = Path(__file__).parent.parent.parent.parent
            positive_base_dir = (
                project_root
                / "gong_detector"
                / "training"
                / "data"
                / "raw_samples"
                / "positive"
            )
            print("\nSaving positive samples using date-based folder naming...")
            save_positive_samples(
                detections, temp_audio, positive_base_dir, upload_date, video_title
            )

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
