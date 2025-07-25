#!/usr/bin/env python3
"""Example usage of the YouTube gong detection with positive sample collection.

This script demonstrates how to use the new --save_positive_samples feature
to automatically collect gong samples from YouTube videos for human review.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Demonstrate the positive sample collection feature."""
    print("YouTube Gong Detection with Positive Sample Collection")
    print("=" * 60)
    
    # Example YouTube URL (replace with your own)
    youtube_url = "https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID"
    
    print(f"Example command:")
    print(f"python -m gong_detector.core.detect_from_youtube '{youtube_url}' --save_positive_samples")
    print()
    
    print("What this does:")
    print("1. Downloads audio from YouTube video")
    print("2. Detects gongs using YAMNet")
    print("3. Extracts 3-second segments around each detection")
    print("4. Saves segments to: gong_detector/training/data/raw_samples/positive/")
    print()
    
    print("Sample output files:")
    print("- gong_15.2s_conf_0.850_1.wav")
    print("- gong_32.1s_conf_0.720_2.wav")
    print("- gong_65.8s_conf_0.680_3.wav")
    print()
    
    print("Human-in-the-loop workflow:")
    print("1. Run detection with --save_positive_samples")
    print("2. Review saved samples in positive/ folder")
    print("3. Keep good samples, delete false positives")
    print("4. Repeat until you have 50 confirmed samples")
    print("5. Then run training pipeline")
    print()
    
    # Check if positive directory exists
    positive_dir = Path("gong_detector/training/data/raw_samples/positive")
    if positive_dir.exists():
        sample_count = len(list(positive_dir.glob("*.wav")))
        print(f"Current positive samples: {sample_count}")
    else:
        print("No positive samples collected yet.")
    
    print()
    print("To get started, replace EXAMPLE_VIDEO_ID with a real YouTube video ID")
    print("and run the command above!")


if __name__ == "__main__":
    main() 