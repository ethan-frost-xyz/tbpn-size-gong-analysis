#!/usr/bin/env python3
"""Bulk YouTube gong detection script.

Processes multiple YouTube URLs from tbpn_youtube_links.txt using the existing
detect_from_youtube.py functionality. Maintains exact formatting and storage patterns.
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path
from typing import List

# Suppress urllib3 OpenSSL warning
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")


def read_youtube_links(file_path: str) -> List[str]:
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
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Skip lines that are just instructions
            if line.startswith('Here is all the tbpn videos') or line.startswith('python -m'):
                continue
            # Check if line looks like a YouTube URL
            if 'youtube.com/watch' in line or 'youtu.be/' in line:
                urls.append(line)
            elif line and not line.startswith('http'):  # Skip non-URL lines that aren't empty
                print(f"Warning: Line {line_num} doesn't look like a YouTube URL: {line}")
    
    if not urls:
        raise ValueError("No valid YouTube URLs found in file")
    
    return urls


def process_single_url(url: str, threshold: float, save_csv: bool, keep_audio: bool) -> bool:
    """Process a single YouTube URL using existing detect_from_youtube.py.

    Args:
        url: YouTube URL to process
        threshold: Confidence threshold for detection
        save_csv: Whether to save CSV results
        keep_audio: Whether to keep temporary audio files

    Returns:
        True if successful, False if failed
    """
    print(f"\n{'='*60}")
    print(f"Processing: {url}")
    print(f"{'='*60}")
    
    # Build command with same parameters as individual processing
    cmd = [
        sys.executable, "-m", "gong_detector.core.detect_from_youtube",
        url,
        "--threshold", str(threshold),
        "--save_positive_samples"
    ]
    
    if save_csv:
        cmd.extend(["--save_csv", "bulk_results.csv"])
    
    if keep_audio:
        cmd.append("--keep_audio")
    
    try:
        # Run the existing detection script
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Successfully processed: {url}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {url}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error processing {url}: {e}")
        return False


def main() -> None:
    """Run bulk YouTube gong detection."""
    parser = argparse.ArgumentParser(
        description="Bulk process YouTube URLs for gong detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bulk_process.py
  python bulk_process.py --threshold 0.5
  python bulk_process.py --save_csv --keep_audio
        """
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for gong detection (default: 0.4)"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save results to CSV file for each video"
    )
    parser.add_argument(
        "--keep_audio",
        action="store_true",
        help="Keep temporary audio files for training data extraction"
    )
    parser.add_argument(
        "--links_file",
        default=None,
        help="Path to file containing YouTube URLs (default: tbpn_youtube_links.txt in script directory)"
    )
    
    args = parser.parse_args()
    
    # Determine links file path
    if args.links_file is None:
        # Default to tbpn_youtube_links.txt in the same directory as this script
        script_dir = Path(__file__).parent
        links_file = script_dir / "tbpn_youtube_links.txt"
    else:
        links_file = args.links_file
    
    # Read URLs from file
    try:
        urls = read_youtube_links(str(links_file))
        print(f"Found {len(urls)} YouTube URLs to process")
    except FileNotFoundError:
        print(f"Error: File '{links_file}' not found")
        print(f"Expected location: {links_file}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Process each URL
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\nProgress: {i}/{len(urls)}")
        
        if process_single_url(url, args.threshold, args.save_csv, args.keep_audio):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("BULK PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total URLs: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nNote: {failed} URLs failed to process. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll URLs processed successfully!")


if __name__ == "__main__":
    main() 