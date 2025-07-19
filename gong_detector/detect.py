#!/usr/bin/env python3
"""Simple command-line gong detection script.

Usage:
    python detect.py <audio_file> [--threshold 0.5]
"""

import argparse
import sys
from pathlib import Path

from yamnet_runner import YAMNetGongDetector


def main() -> None:
    """Main detection script."""
    parser = argparse.ArgumentParser(description="Detect gong sounds in audio files")
    parser.add_argument("audio_file", help="Path to audio file (.wav)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--output", "-o", help="Output CSV file for detections")
    
    args = parser.parse_args()
    
    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"🎵 Detecting gongs in: {audio_path}")
    print(f"🎯 Confidence threshold: {args.threshold}")
    
    try:
        # Initialize detector
        detector = YAMNetGongDetector()
        detector.load_model()
        
        # Process audio
        waveform, sample_rate = detector.load_and_preprocess_audio(str(audio_path))
        scores, _, _ = detector.run_inference(waveform)
        
        # Detect gongs
        detections = detector.detect_gongs(scores, confidence_threshold=args.threshold)
        
        # Print results
        detector.print_detections(detections)
        
        # Save to CSV if requested
        if args.output and detections:
            df = detector.detections_to_dataframe(detections)
            df.to_csv(args.output, index=False)
            print(f"💾 Saved {len(detections)} detections to: {args.output}")
        
        print(f"✅ Found {len(detections)} gong detections")
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
