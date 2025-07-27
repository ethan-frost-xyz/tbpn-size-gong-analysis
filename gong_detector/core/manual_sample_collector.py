#!/usr/bin/env python3
"""Manual sample collector for gong detection training data.

This script downloads YouTube audio and extracts a single audio segment
at a specified timestamp for manual review and labeling. It leverages
existing core utilities and the save_positive_samples function.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .results_utils import save_positive_samples
from .youtube_utils import (
    cleanup_old_temp_files,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    setup_directories,
)


def create_manual_detection(
    timestamp: float, confidence: float = 1.0
) -> list[tuple[float, float, float]]:
    """Create a single manual detection tuple for the save_positive_samples function.

    Args:
        timestamp: Timestamp in seconds where the gong occurs
        confidence: Confidence value (default 1.0 for manual detections)

    Returns:
        List containing a single detection tuple (window_start, confidence, display_timestamp)
    """
    # For manual detections, window_start and display_timestamp are the same
    return [(timestamp, confidence, timestamp)]


def process_single_sample(youtube_url: str, timestamp: float, confidence: float = 1.0) -> bool:
    """Process a single YouTube video sample.
    
    Args:
        youtube_url: YouTube URL to process
        timestamp: Timestamp in seconds where gong occurs
        confidence: Confidence value for manual detection
        
    Returns:
        True if successful, False if error occurred
    """
    # Setup directories
    temp_audio_dir, _ = setup_directories()
    cleanup_old_temp_files(temp_audio_dir)

    # Create temporary audio file path
    temp_audio = create_temp_audio_path(temp_audio_dir)

    try:
        # Step 1: Download and process audio
        print("Step 1: Downloading audio from YouTube...")
        temp_audio, video_title, upload_date = download_and_trim_youtube_audio(
            url=youtube_url,
            output_path=temp_audio,
        )

        # Step 2: Create manual detection
        print(f"Step 2: Creating manual detection at {timestamp}s...")
        manual_detection = create_manual_detection(timestamp, confidence)

        # Step 3: Save sample using existing function
        print("Step 3: Extracting and saving audio segment...")

        # Determine output directory based on video title
        from .youtube_utils import sanitize_title_for_folder

        safe_title = sanitize_title_for_folder(video_title)
        positive_dir = (
            Path("gong_detector/training/data/raw_samples/positive") / safe_title
        )

        # Use existing save_positive_samples function
        save_positive_samples(manual_detection, temp_audio, positive_dir)

        print("\n✓ Manual sample collected successfully!")
        print(f"  Video: {video_title}")
        print(f"  Timestamp: {timestamp}s")
        print(f"  Saved to: {positive_dir}")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    finally:
        # Clean up temporary audio file
        try:
            if Path(temp_audio).exists():
                Path(temp_audio).unlink()
                print(f"Cleaned up temporary file: {temp_audio}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary file: {e}")


def main() -> None:
    """Run interactive manual sample collection."""
    print("🎵 Manual Sample Collector")
    print("Enter YouTube link and timestamp (e.g., 'https://youtube.com/watch?v=ABC123 120')")
    print("Type 'n' to exit\n")

    while True:
        try:
            # Get user input
            user_input = input("Enter link and seconds (type 'n' to close): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'n', 'no']:
                print("Goodbye!")
                break
            
            # Parse input
            parts = user_input.split()
            if len(parts) < 2:
                print("❌ Please provide both link and timestamp")
                continue
                
            youtube_url = parts[0]
            try:
                timestamp = float(parts[1])
            except ValueError:
                print("❌ Invalid timestamp. Please enter a number.")
                continue
            
            # Process the sample
            success = process_single_sample(youtube_url, timestamp)
            
            if success:
                print()  # Add spacing for next iteration
            else:
                print()  # Add spacing for next iteration
                        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue


if __name__ == "__main__":
    main()
