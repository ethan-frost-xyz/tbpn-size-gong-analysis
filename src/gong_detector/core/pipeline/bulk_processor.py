#!/usr/bin/env python3
"""Simple bulk YouTube gong detection script.

Processes multiple YouTube URLs from tbpn_youtube_links.txt file.
Creates no files by default - only when explicitly requested.
"""

import argparse
import sys
import time
from pathlib import Path

from ..data import CSVManager
from ..training.negative_collector import collect_negative_samples
from .detection_pipeline import detect_from_youtube_comprehensive


def format_time(seconds: float) -> str:
    """Convert a second count into a concise human-readable label.

    Parameters
    ----------
    seconds : float
        Duration expressed in seconds.

    Returns
    -------
    str
        Time formatted as either `"Xm Ys"` or `"Xh Ym Zs"` depending on length.
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    else:
        return f"{minutes}m {secs}s"


def read_youtube_links(file_path: Path) -> list[str]:
    """Load a list of YouTube URLs from a manifest file.

    Parameters
    ----------
    file_path : pathlib.Path
        Location of a UTF-8 text file containing one URL per line.

    Returns
    -------
    list[str]
        All valid YouTube URLs discovered in the file.

    Raises
    ------
    FileNotFoundError
        Raised when `file_path` does not exist.
    ValueError
        Raised when no usable URLs are detected in the manifest.
    """
    urls = []

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Check if line looks like a YouTube URL
            if "youtube.com/watch" in line or "youtu.be/" in line:
                urls.append(line)
            elif line and not line.startswith("http"):
                print(
                    f"Warning: Line {line_num} doesn't look like a YouTube URL: {line}"
                )

    if not urls:
        raise ValueError("No valid YouTube URLs found in file")

    return urls


def main() -> None:
    """Run bulk YouTube gong detection from the command line."""
    parser = argparse.ArgumentParser(
        description="Bulk process YouTube URLs for gong detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m gong_detector.core.pipeline.bulk_processor
  python -m gong_detector.core.pipeline.bulk_processor --threshold 0.5
  python -m gong_detector.core.pipeline.bulk_processor --threshold 0.3 --max_threshold 0.8
  python -m gong_detector.core.pipeline.bulk_processor --save_positive_samples
  python -m gong_detector.core.pipeline.bulk_processor --collect_negative_samples --sample_count 10
  python -m gong_detector.core.pipeline.bulk_processor --version_one --csv  # Includes batch LUFS (dual-cache enabled)
  python -m gong_detector.core.pipeline.bulk_processor --csv --test-run 10  # Test with first 10 videos
        """,
    )

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
        "--save_positive_samples",
        action="store_true",
        help="Save detected gong segments to training data folder",
    )
    parser.add_argument(
        "--keep_audio",
        action="store_true",
        help="Keep temporary audio files",
    )
    parser.add_argument(
        "--collect_negative_samples",
        action="store_true",
        help="Collect negative samples instead of detecting gongs",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=5,
        help="Number of negative samples to collect per video (default: 5)",
    )
    parser.add_argument(
        "--version_one",
        action="store_true",
        help="Use the trained classifier for enhanced gong detection",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4000,
        help="Batch size for classifier predictions (larger = faster but more memory, default: 4000)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save all detection results to comprehensive CSV file with batch LUFS + True Peak analysis (EBU R128)",
    )
    parser.add_argument(
        "--no_consolidate",
        action="store_true",
        help="Disable consolidation of overlapping detections (keep all raw detections)",
    )

    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Strict offline mode: require local preprocessed audio; never download",
    )
    parser.add_argument(
        "--test-run",
        type=int,
        default=None,
        help="Test mode: process only the first X videos instead of all videos",
    )

    args = parser.parse_args()

    # Find links file robustly
    links_file = None
    relative_path = "data/tbpn_ytlinks/tbpn_youtube_links.txt"

    # Check current directory first
    if Path(relative_path).exists():
        links_file = Path(relative_path)
    else:
        # Walk up from script location to find project root
        script_dir = Path(__file__).resolve().parent
        for parent in [script_dir] + list(script_dir.parents):
            candidate = parent / relative_path
            if candidate.exists():
                links_file = candidate
                break

    if not links_file:
        print(f"Error: Could not find '{relative_path}'")
        print("Please ensure the file exists in the project data directory")
        sys.exit(1)

    # Read URLs from file
    try:
        urls = read_youtube_links(links_file)
        total_urls = len(urls)

        # Apply test-run limitation if specified
        if args.test_run is not None:
            if args.test_run <= 0:
                print("Error: --test-run must be a positive number")
                sys.exit(1)
            if args.test_run >= total_urls:
                print(
                    f"Warning: --test-run {args.test_run} >= total videos {total_urls}, processing all videos"
                )
            else:
                urls = urls[: args.test_run]
                print(f"TEST MODE: Processing first {len(urls)} of {total_urls} videos")

        print(f"Found {len(urls)} YouTube URLs to process")
    except FileNotFoundError:
        print(f"Error: File '{links_file}' not found")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize CSV manager if requested
    csv_manager = None
    if args.csv:
        csv_manager = CSVManager()
        print(
            "CSV output enabled - batch LUFS + True Peak analysis will be performed across all videos"
        )

    # Store results for CSV processing
    all_results = []  # Store results for CSV processing

    # Start timing
    start_time = time.time()

    # Process each URL
    successful = 0
    failed = 0

    # Memory monitoring for bulk processing
    def check_memory_status() -> tuple[float, bool]:
        """Check current memory usage and return (available_gb, should_continue)."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            used_percent = memory.percent

            # Stop if memory usage is critically high
            if used_percent > 90 or available_gb < 2:
                print(
                    f"[CRITICAL] Memory usage too high: {used_percent:.1f}% used, {available_gb:.1f}GB available"
                )
                print("Stopping bulk processing to prevent system crash")
                return available_gb, False
            elif used_percent > 80 or available_gb < 4:
                print(
                    f"[WARNING] High memory usage: {used_percent:.1f}% used, {available_gb:.1f}GB available"
                )
                return available_gb, True

            return available_gb, True
        except ImportError:
            return 8.0, True  # Assume OK if psutil not available

    def force_memory_cleanup() -> None:
        """Force garbage collection and memory cleanup."""
        import gc
        import os

        # Force garbage collection
        gc.collect()

        # Try to force TensorFlow memory cleanup if available
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except ImportError:
            pass

        # Force OS-level memory cleanup on macOS
        if os.name == "posix":
            try:
                os.system(
                    "purge > /dev/null 2>&1 &"
                )  # macOS memory purge in background
            except Exception:
                pass

    for i, url in enumerate(urls, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing {i}/{len(urls)}: {url}")
        print(f"{'=' * 60}")

        # Add delay between downloads to avoid rate limiting (except for first video)
        if i > 1:
            delay = 5  # 5 second delay between downloads
            print(f"[INFO] Waiting {delay} seconds before next download...")
            time.sleep(delay)

        # Check memory before processing each video
        available_gb, should_continue = check_memory_status()
        if not should_continue:
            print(
                f"[ERROR] Stopping due to insufficient memory after {successful} successful videos"
            )
            break

        # Force cleanup before processing if memory is getting low
        if available_gb < 6:
            print(f"[INFO] Running memory cleanup (available: {available_gb:.1f}GB)")
            force_memory_cleanup()

        if args.collect_negative_samples:
            result = collect_negative_samples(
                youtube_url=url,
                num_samples=args.sample_count,
                threshold=args.threshold,
                max_threshold=args.max_threshold,
                keep_audio=args.keep_audio,
            )
        else:
            result = detect_from_youtube_comprehensive(
                youtube_url=url,
                threshold=args.threshold,
                max_threshold=args.max_threshold,
                should_save_positive_samples=args.save_positive_samples,
                keep_audio=args.keep_audio,
                use_version_one=args.version_one,
                batch_size=args.batch_size,
                consolidate_detections=not args.no_consolidate,
                local_only=args.local_only,
            )

        if result["success"]:
            if args.collect_negative_samples:
                print(f"[OK] Collected {result['sample_count']} negative samples")
            else:
                print(f"[OK] Detected {result['detection_count']} gongs")
                # Store results for CSV processing
                all_results.append(result)

                # Write to CSV incrementally (per-video LUFS already computed in detection_pipeline)
                if csv_manager:
                    try:
                        csv_manager.add_video_detections(
                            video_url=result["video_url"],
                            video_title=result["video_title"],
                            upload_date=result["upload_date"],
                            video_duration=result["video_duration"],
                            max_confidence=result["max_confidence"],
                            threshold=args.threshold,
                            max_threshold=args.max_threshold,
                            detections=result["detections"],
                            video_loudness_metrics=result.get("video_loudness_metrics"),
                            detection_loudness_metrics=result.get(
                                "detection_loudness_metrics"
                            ),
                            detection_lufs_metrics=result.get(
                                "detection_lufs_metrics", []
                            ),
                            detection_dbtp_metrics=result.get(
                                "detection_dbtp_metrics", []
                            ),
                        )
                        print(
                            f"[OK] Added {len(result['detections'])} detections to CSV"
                        )
                    except Exception as e:
                        print(f"[WARNING] CSV write failed for this video: {e}")
            successful += 1
        else:
            print(f"[ERROR] Failed: {result['error_message']}")
            failed += 1

        # Force memory cleanup after each video to prevent accumulation
        if i % 3 == 0 or available_gb < 8:  # Every 3 videos or when memory is low
            print(f"[INFO] Running periodic memory cleanup after video {i}")
            force_memory_cleanup()

    # Final memory cleanup
    print("\n[INFO] Running final memory cleanup...")
    force_memory_cleanup()

    # All videos processed - CSV already written incrementally
    if csv_manager and not args.collect_negative_samples:
        print(
            f"\n[OK] Processed {len(all_results)} videos with incremental CSV writing"
        )

    # Save CSV if requested
    if csv_manager and not args.collect_negative_samples:
        try:
            csv_path = csv_manager.save_comprehensive_csv("bulk_run")
            stats = csv_manager.get_summary_stats()
            print(f"\n[OK] CSV saved to: {csv_path}")
            print(f"  Total detections: {stats.get('total_detections', 0)}")
            print(f"  Unique videos: {stats.get('unique_videos', 0)}")
            print(f"  Average confidence: {stats.get('average_confidence', 0):.3f}")

            # Display loudness statistics if available
            if "avg_detection_peak_dbfs" in stats:
                print("  Audio levels (detection avg):")
                print(
                    f"    Peak dBFS: {stats.get('avg_detection_peak_dbfs', 0):.1f} dB"
                )
                print(f"    RMS dBFS: {stats.get('avg_detection_rms_dbfs', 0):.1f} dB")
                print(
                    f"    Crest factor: {stats.get('avg_detection_crest_factor', 0):.1f}"
                )
        except Exception as e:
            print(f"[ERROR] CSV save failed: {e}")

    # Calculate timing
    end_time = time.time()
    total_time_seconds = end_time - start_time

    # Calculate average time per video (only successful ones)
    avg_time_seconds = total_time_seconds / successful if successful > 0 else 0

    # Print final summary
    print(f"\n{'=' * 60}")
    print("BULK PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total URLs: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {format_time(total_time_seconds)}")
    if successful > 0:
        print(f"Average time per video: {format_time(avg_time_seconds)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
