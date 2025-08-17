#!/usr/bin/env python3
"""Simple bulk YouTube gong detection script.

Processes multiple YouTube URLs from tbpn_youtube_links.txt file.
Creates no files by default - only when explicitly requested.
"""

import argparse
import sys
from pathlib import Path

from ..data import CSVManager
from ..training.negative_collector import collect_negative_samples
from .detection_pipeline import detect_from_youtube_comprehensive


def read_youtube_links(file_path: Path) -> list[str]:
    """Read YouTube URLs from text file.

    Args:
        file_path: Path to text file containing YouTube URLs

    Returns:
        List of YouTube URLs

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If no valid URLs found
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
    """Run bulk YouTube gong detection."""
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
  python -m gong_detector.core.pipeline.bulk_processor --version_one --csv  # Includes batch LUFS
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
        default=2000,
        help="Batch size for classifier predictions (larger = faster but more memory, default: 2000)",
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
        "--use_local_media",
        action="store_true",
        help="Prefer local preprocessed audio cache; download only if missing",
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
                print(f"Warning: --test-run {args.test_run} >= total videos {total_urls}, processing all videos")
            else:
                urls = urls[:args.test_run]
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
        print("CSV output enabled - batch LUFS + True Peak analysis will be performed across all videos")

    # Collect all detection data for batch LUFS computation
    all_detection_data = []  # Store (video_id, timestamps) for batch LUFS
    all_results = []  # Store results for CSV processing

    # Process each URL
    successful = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing {i}/{len(urls)}: {url}")
        print(f"{'=' * 60}")

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
                use_local_media=args.use_local_media,
                local_only=args.local_only,
            )

        if result["success"]:
            if args.collect_negative_samples:
                print(f"✓ Collected {result['sample_count']} negative samples")
            else:
                print(f"✓ Detected {result['detection_count']} gongs")
                # Store results for batch LUFS processing
                all_results.append(result)

                # Collect detection data for batch LUFS computation
                if csv_manager and result["detections"]:
                    from ..utils.youtube_utils import video_id_from_url
                    video_id = video_id_from_url(result["video_url"])
                    if video_id:
                        # Prepare timestamps for LUFS computation with different segment lengths
                        detection_timestamps = {
                            "integrated": [],  # ±2 seconds (4s total) for integrated
                            "short_term": [],  # ±1.5 seconds (3s total) for short-term
                            "momentary": []    # ±0.2 seconds (400ms total) for momentary
                        }

                        for _window_start, _confidence, display_timestamp in result["detections"]:
                            # Integrated LUFS: 4-second window (±2s)
                            integrated_start = max(0, display_timestamp - 2.0)
                            integrated_end = min(result["video_duration"], display_timestamp + 2.0)
                            detection_timestamps["integrated"].append((integrated_start, integrated_end))

                            # Short-term LUFS: 3-second window (±1.5s)
                            shortterm_start = max(0, display_timestamp - 1.5)
                            shortterm_end = min(result["video_duration"], display_timestamp + 1.5)
                            detection_timestamps["short_term"].append((shortterm_start, shortterm_end))

                            # Momentary LUFS: 400ms window (±0.2s)
                            momentary_start = max(0, display_timestamp - 0.2)
                            momentary_end = min(result["video_duration"], display_timestamp + 0.2)
                            detection_timestamps["momentary"].append((momentary_start, momentary_end))

                        all_detection_data.append({
                            "video_id": video_id,
                            "timestamps": detection_timestamps,
                            "result": result
                        })
            successful += 1
        else:
            print(f"✗ Failed: {result['error_message']}")
            failed += 1

    # Compute batch LUFS if we have CSV data and detections
    if csv_manager and all_detection_data and not args.collect_negative_samples:
        print(f"\nComputing batch-weighted LUFS across {len(all_detection_data)} videos with detections...")

        try:
            from ..utils.youtube_utils import (
                compute_batch_weighted_dbtp,
                compute_batch_weighted_lufs,
            )

            # Compute batch-weighted LUFS for all videos together (all three types with different time windows)
            print("  Computing integrated LUFS (4s windows)...")
            integrated_data = []
            for video_data in all_detection_data:
                integrated_data.append({
                    "video_id": video_data["video_id"],
                    "timestamps": video_data["timestamps"]["integrated"],
                    "result": video_data["result"]
                })
            integrated_lufs_results = compute_batch_weighted_lufs(
                all_video_data=integrated_data,
                measurement_type="integrated",
                reference_lufs=-23.0
            )

            print("  Computing short-term LUFS (3s windows)...")
            shortterm_data = []
            for video_data in all_detection_data:
                shortterm_data.append({
                    "video_id": video_data["video_id"],
                    "timestamps": video_data["timestamps"]["short_term"],
                    "result": video_data["result"]
                })
            shortterm_lufs_results = compute_batch_weighted_lufs(
                all_video_data=shortterm_data,
                measurement_type="short_term",
                reference_lufs=-23.0
            )

            print("  Computing momentary LUFS (400ms windows)...")
            momentary_data = []
            for video_data in all_detection_data:
                momentary_data.append({
                    "video_id": video_data["video_id"],
                    "timestamps": video_data["timestamps"]["momentary"],
                    "result": video_data["result"]
                })
            momentary_lufs_results = compute_batch_weighted_lufs(
                all_video_data=momentary_data,
                measurement_type="momentary",
                reference_lufs=-23.0
            )

            # Compute batch-weighted True Peak (dBTP) for all videos together (all three types)
            print("  Computing integrated True Peak (4s windows)...")
            integrated_dbtp_results = compute_batch_weighted_dbtp(
                all_video_data=integrated_data,
                measurement_type="integrated",
                reference_dbtp=-1.0  # EBU R128 True Peak limit
            )

            print("  Computing short-term True Peak (3s windows)...")
            shortterm_dbtp_results = compute_batch_weighted_dbtp(
                all_video_data=shortterm_data,
                measurement_type="short_term",
                reference_dbtp=-1.0
            )

            print("  Computing momentary True Peak (400ms windows)...")
            momentary_dbtp_results = compute_batch_weighted_dbtp(
                all_video_data=momentary_data,
                measurement_type="momentary",
                reference_dbtp=-1.0
            )

            # Process each video's results and add to CSV
            for video_data in all_detection_data:
                video_id = video_data["video_id"]
                result = video_data["result"]

                # Get batch-weighted LUFS results for this video (all three types)
                integrated_results = integrated_lufs_results.get(video_id, [])
                shortterm_results = shortterm_lufs_results.get(video_id, [])
                momentary_results = momentary_lufs_results.get(video_id, [])

                # Get batch-weighted True Peak results for this video (all three types)
                integrated_dbtp_video_results = integrated_dbtp_results.get(video_id, [])
                shortterm_dbtp_video_results = shortterm_dbtp_results.get(video_id, [])
                momentary_dbtp_video_results = momentary_dbtp_results.get(video_id, [])

                # Transform LUFS and True Peak results to expected format
                detection_lufs_metrics = []
                detection_dbtp_metrics = []
                max_results = max(
                    len(integrated_results), len(shortterm_results), len(momentary_results),
                    len(integrated_dbtp_video_results), len(shortterm_dbtp_video_results), len(momentary_dbtp_video_results)
                ) if any([integrated_results, shortterm_results, momentary_results, integrated_dbtp_video_results, shortterm_dbtp_video_results, momentary_dbtp_video_results]) else 0

                for i in range(max_results):
                    # Get LUFS values for each measurement type (with fallbacks)
                    integrated_lufs = integrated_results[i].get("lufs", 0) if i < len(integrated_results) else 0
                    shortterm_lufs = shortterm_results[i].get("lufs", 0) if i < len(shortterm_results) else 0
                    momentary_lufs = momentary_results[i].get("lufs", 0) if i < len(momentary_results) else 0

                    detection_lufs_metrics.append({
                        "integrated_lufs": integrated_lufs,
                        "shortterm_lufs": shortterm_lufs,
                        "momentary_lufs": momentary_lufs,
                    })

                    # Get True Peak values for each measurement type (with fallbacks)
                    integrated_dbtp = integrated_dbtp_video_results[i].get("dbtp", 0) if i < len(integrated_dbtp_video_results) else 0
                    shortterm_dbtp = shortterm_dbtp_video_results[i].get("dbtp", 0) if i < len(shortterm_dbtp_video_results) else 0
                    momentary_dbtp = momentary_dbtp_video_results[i].get("dbtp", 0) if i < len(momentary_dbtp_video_results) else 0

                    detection_dbtp_metrics.append({
                        "integrated_dbtp": integrated_dbtp,
                        "shortterm_dbtp": shortterm_dbtp,
                        "momentary_dbtp": momentary_dbtp,
                    })

                # Add to CSV with batch-weighted LUFS and True Peak
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
                    detection_loudness_metrics=result.get("detection_loudness_metrics"),
                    detection_lufs_metrics=detection_lufs_metrics,
                    detection_dbtp_metrics=detection_dbtp_metrics,
                )

        except Exception as e:
            print(f"Warning: Batch LUFS computation failed: {e}")
            # Fallback: Add results without batch LUFS
            for video_data in all_detection_data:
                result = video_data["result"]
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
                    detection_loudness_metrics=result.get("detection_loudness_metrics"),
                    detection_lufs_metrics=result.get("detection_lufs_metrics", []),
                )

    # Save CSV if requested
    if csv_manager and not args.collect_negative_samples:
        try:
            csv_path = csv_manager.save_comprehensive_csv("bulk_run")
            stats = csv_manager.get_summary_stats()
            print(f"\n✓ CSV saved to: {csv_path}")
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
            print(f"✗ CSV save failed: {e}")

    # Print final summary
    print(f"\n{'=' * 60}")
    print("BULK PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total URLs: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
