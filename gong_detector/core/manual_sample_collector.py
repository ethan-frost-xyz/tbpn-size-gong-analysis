#!/usr/bin/env python3
"""Manual sample collector for gong detection training data.

This script downloads YouTube audio and extracts a single audio segment
at a specified timestamp for manual review and labeling. It leverages
existing core utilities and the save_positive_samples function.
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from .results_utils import save_positive_samples
from .youtube_utils import (
    cleanup_old_temp_files,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    setup_directories,
)


def create_manual_detection(timestamp: float, confidence: float = 1.0) -> list[tuple[float, float, float]]:
    """Create a single manual detection tuple for the save_positive_samples function.
    
    Args:
        timestamp: Timestamp in seconds where the gong occurs
        confidence: Confidence value (default 1.0 for manual detections)
        
    Returns:
        List containing a single detection tuple (window_start, confidence, display_timestamp)
    """
    # For manual detections, window_start and display_timestamp are the same
    return [(timestamp, confidence, timestamp)]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Extract a single audio segment from YouTube video for manual review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m gong_detector.core.manual_sample_collector "https://youtube.com/watch?v=VIDEO_ID" 120
  python -m gong_detector.core.manual_sample_collector "https://youtube.com/watch?v=VIDEO_ID" 360 --confidence 0.8
        """,
    )
    
    parser.add_argument("youtube_url", help="YouTube URL to process")
    parser.add_argument("timestamp", type=float, help="Timestamp in seconds where gong occurs")
    parser.add_argument(
        "--confidence",
        type=float,
        default=1.0,
        help="Confidence value for manual detection (default: 1.0)",
    )
    
    return parser


def main() -> None:
    """Run manual sample collection."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup directories
    temp_audio_dir, _ = setup_directories()
    cleanup_old_temp_files(temp_audio_dir)
    
    # Create temporary audio file path
    temp_audio = create_temp_audio_path(temp_audio_dir)
    
    try:
        # Step 1: Download and process audio
        print(f"Step 1: Downloading audio from YouTube...")
        temp_audio, video_title, upload_date = download_and_trim_youtube_audio(
            url=args.youtube_url,
            output_path=temp_audio,
        )
        
        # Step 2: Create manual detection
        print(f"Step 2: Creating manual detection at {args.timestamp}s...")
        manual_detection = create_manual_detection(args.timestamp, args.confidence)
        
        # Step 3: Save sample using existing function
        print(f"Step 3: Extracting and saving audio segment...")
        
        # Determine output directory based on video title
        from .youtube_utils import sanitize_title_for_folder
        safe_title = sanitize_title_for_folder(video_title)
        positive_dir = Path("gong_detector/training/data/raw_samples/positive") / safe_title
        
        # Use existing save_positive_samples function
        save_positive_samples(manual_detection, temp_audio, positive_dir)
        
        print(f"\n✓ Manual sample collected successfully!")
        print(f"  Video: {video_title}")
        print(f"  Timestamp: {args.timestamp}s")
        print(f"  Saved to: {positive_dir}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    finally:
        # Clean up temporary audio file
        try:
            if Path(temp_audio).exists():
                Path(temp_audio).unlink()
                print(f"Cleaned up temporary file: {temp_audio}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary file: {e}")


if __name__ == "__main__":
    main() 