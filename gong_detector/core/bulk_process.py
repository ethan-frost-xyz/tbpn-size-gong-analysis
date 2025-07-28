#!/usr/bin/env python3
"""Simple bulk YouTube gong detection script.

Processes multiple YouTube URLs from tbpn_youtube_links.txt file.
Creates no files by default - only when explicitly requested.
"""

import argparse
import sys
from pathlib import Path

from .detect_from_youtube import detect_from_youtube_comprehensive
from .negative_sample_collector import collect_negative_samples


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
                print(f"Warning: Line {line_num} doesn't look like a YouTube URL: {line}")

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
  python -m gong_detector.core.bulk_process
  python -m gong_detector.core.bulk_process --threshold 0.5
  python -m gong_detector.core.bulk_process --threshold 0.3 --max_threshold 0.8
  python -m gong_detector.core.bulk_process --save_positive_samples
  python -m gong_detector.core.bulk_process --collect_negative_samples --sample_count 10
        """,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.925,
        help="Confidence threshold for gong detection (default: 0.925)",
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

    args = parser.parse_args()

    # Find links file in same directory as this script
    script_dir = Path(__file__).parent
    links_file = script_dir / "tbpn_youtube_links.txt"

    # Read URLs from file
    try:
        urls = read_youtube_links(links_file)
        print(f"Found {len(urls)} YouTube URLs to process")
    except FileNotFoundError:
        print(f"Error: File '{links_file}' not found")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Process each URL
    successful = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(urls)}: {url}")
        print(f"{'='*60}")

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
            )

        if result["success"]:
            if args.collect_negative_samples:
                print(f"✓ Collected {result['sample_count']} negative samples")
            else:
                print(f"✓ Detected {result['detection_count']} gongs")
            successful += 1
        else:
            print(f"✗ Failed: {result['error_message']}")
            failed += 1

    # Print final summary
    print(f"\n{'='*60}")
    print("BULK PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total URLs: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
