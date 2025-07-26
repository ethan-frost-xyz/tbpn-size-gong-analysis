#!/usr/bin/env python3
"""Bulk YouTube gong detection script with comprehensive CSV output.

Processes multiple YouTube URLs from tbpn_youtube_links.txt and generates
a comprehensive CSV file containing all detection metadata for analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from gong_detector.core.detect_from_youtube import detect_from_youtube_comprehensive
from gong_detector.core.comprehensive_csv import ComprehensiveCSVManager


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


def process_single_url(
    url: str, 
    threshold: float, 
    save_positive_samples: bool, 
    keep_audio: bool,
    csv_manager: ComprehensiveCSVManager
) -> bool:
    """Process a single YouTube URL and add results to CSV manager.

    Args:
        url: YouTube URL to process
        threshold: Confidence threshold for detection
        save_positive_samples: Whether to save detected segments
        keep_audio: Whether to keep temporary audio files
        csv_manager: CSV manager to collect results

    Returns:
        True if successful, False if failed
    """
    print(f"\n{'='*60}")
    print(f"Processing: {url}")
    print(f"{'='*60}")
    
    try:
        # Use the comprehensive detection function
        result = detect_from_youtube_comprehensive(
            youtube_url=url,
            threshold=threshold,
            save_positive_samples=save_positive_samples,
            keep_audio=keep_audio
        )
        
        if result["success"]:
            print(f"\nStep 1: Downloaded and processed audio")
            print(f"Step 2-4: YAMNet detection complete")
            print(f"\nDetected {result['detection_count']} gongs:")
            
            # Print individual detections
            for i, (window_start, confidence, display_timestamp) in enumerate(result["detections"]):
                from gong_detector.core.results_utils import format_time
                youtube_time = format_time(display_timestamp)
                print(f"  {youtube_time} - Confidence: {confidence:.3f}")
            
            # Add to CSV manager
            csv_manager.add_video_detections(
                video_url=result["video_url"],
                video_title=result["video_title"],
                upload_date=result["upload_date"],
                video_duration=result["video_duration"],
                max_confidence=result["max_confidence"],
                threshold=result["threshold"],
                detections=result["detections"]
            )
            
            print(f"✓ Successfully processed: {url}")
            return True
        else:
            print(f"✗ Failed to process {url}: {result['error_message']}")
            return False
            
    except Exception as e:
        print(f"✗ Unexpected error processing {url}: {e}")
        return False


def main() -> None:
    """Run bulk YouTube gong detection with comprehensive CSV output."""
    parser = argparse.ArgumentParser(
        description="Bulk process YouTube URLs for gong detection with comprehensive CSV output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bulk_process.py
  python bulk_process.py --threshold 0.5
  python bulk_process.py --save_positive_samples --keep_audio
  python bulk_process.py --run_name "tbpn_batch_1"
        """
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for gong detection (default: 0.4)"
    )
    parser.add_argument(
        "--save_positive_samples",
        action="store_true",
        help="Save detected gong segments to training data folder"
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
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional name for this bulk run (used in CSV filename)"
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
    
    # Initialize comprehensive CSV manager
    csv_manager = ComprehensiveCSVManager()
    
    # Process each URL
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\nProgress: {i}/{len(urls)}")
        
        if process_single_url(url, args.threshold, args.save_positive_samples, args.keep_audio, csv_manager):
            successful += 1
        else:
            failed += 1
    
    # Generate comprehensive CSV
    try:
        if csv_manager.detection_records:
            csv_path = csv_manager.save_comprehensive_csv(args.run_name)
            
            # Get and display summary statistics
            stats = csv_manager.get_summary_stats()
            
            print(f"\n{'='*60}")
            print("COMPREHENSIVE CSV GENERATED")
            print(f"{'='*60}")
            print(f"CSV saved to: {csv_path}")
            print(f"Total detections: {stats['total_detections']}")
            print(f"Videos processed: {stats['unique_videos']}")
            print(f"Average confidence: {stats['average_confidence']}")
            print(f"Confidence range: {stats['min_confidence']} - {stats['max_confidence']}")
        else:
            print(f"\n{'='*60}")
            print("NO DETECTIONS FOUND")
            print(f"{'='*60}")
            print("No CSV file generated (no gongs detected)")
    except Exception as e:
        print(f"Error generating comprehensive CSV: {e}")
    
    # Print final summary
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