#!/usr/bin/env python3
"""Batch gong detection tool.

Processes a list of .wav files in a folder, runs YAMNet gong detection,
and stores results in CSV/JSON format. Handles errors gracefully.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from yamnet_runner import YAMNetGongDetector


def find_wav_files(directory: str) -> List[Path]:
    """Find all .wav files in a directory.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of Path objects for .wav files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    wav_files = list(dir_path.glob("*.wav"))
    wav_files.sort()  # Process in consistent order
    
    print(f"Found {len(wav_files)} .wav files in {directory}")
    return wav_files


def process_single_file(
    detector: YAMNetGongDetector, 
    audio_path: Path, 
    confidence_threshold: float = 0.5
) -> Tuple[List[Tuple[float, float]], str]:
    """Process a single audio file for gong detection.
    
    Args:
        detector: Initialized YAMNet detector
        audio_path: Path to audio file
        confidence_threshold: Detection threshold
        
    Returns:
        Tuple of (detections, error_message). Error message is empty if successful.
    """
    try:
        print(f"Processing: {audio_path.name}")
        
        # Load and preprocess audio
        waveform, sample_rate = detector.load_and_preprocess_audio(str(audio_path))
        
        # Run inference
        scores, _, _ = detector.run_inference(waveform)
        
        # Detect gongs
        detections = detector.detect_gongs(scores, confidence_threshold=confidence_threshold)
        
        print(f"  → Found {len(detections)} detections")
        return detections, ""
        
    except Exception as e:
        error_msg = f"Error processing {audio_path.name}: {e}"
        print(f"  ❌ {error_msg}")
        return [], error_msg


def save_detections_csv(detections: List[Tuple[float, float]], output_path: Path) -> None:
    """Save detections to CSV format.
    
    Args:
        detections: List of (timestamp, confidence) tuples
        output_path: Output CSV file path
    """
    with open(output_path, 'w') as f:
        f.write("timestamp_seconds,confidence\n")
        for timestamp, confidence in detections:
            f.write(f"{timestamp:.3f},{confidence:.6f}\n")


def save_detections_json(detections: List[Tuple[float, float]], output_path: Path) -> None:
    """Save detections to JSON format.
    
    Args:
        detections: List of (timestamp, confidence) tuples
        output_path: Output JSON file path
    """
    detection_list = [
        {"timestamp_seconds": timestamp, "confidence": confidence}
        for timestamp, confidence in detections
    ]
    
    with open(output_path, 'w') as f:
        json.dump(detection_list, f, indent=2)


def main() -> None:
    """Main batch detection execution."""
    parser = argparse.ArgumentParser(description="Batch gong detection on WAV files")
    parser.add_argument("directory", help="Directory containing .wav files")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv",
                       help="Output format (default: csv)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.directory)
    output_dir.mkdir(exist_ok=True)
    
    print("🎵 Batch Gong Detection")
    print("=" * 50)
    print(f"Input directory: {args.directory}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Output format: {args.format}")
    print()
    
    try:
        # Find WAV files
        wav_files = find_wav_files(args.directory)
        if not wav_files:
            print("No .wav files found!")
            return
        
        # Initialize detector (once for all files)
        print("Loading YAMNet model...")
        detector = YAMNetGongDetector()
        detector.load_model()
        print()
        
        # Process each file
        results: Dict[str, int] = {}
        errors: List[str] = []
        
        for audio_path in wav_files:
            detections, error_msg = process_single_file(
                detector, audio_path, args.threshold
            )
            
            if error_msg:
                errors.append(error_msg)
                continue
            
            # Save detections
            base_name = audio_path.stem
            if args.format == "csv":
                output_path = output_dir / f"{base_name}_detections.csv"
                save_detections_csv(detections, output_path)
            else:
                output_path = output_dir / f"{base_name}_detections.json"
                save_detections_json(detections, output_path)
            
            results[audio_path.name] = len(detections)
            print(f"  💾 Saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 50)
        
        if results:
            print("Detection counts per file:")
            for filename, count in results.items():
                print(f"  {filename:<30} {count:>3} detections")
            
            total_detections = sum(results.values())
            print(f"\nTotal files processed: {len(results)}")
            print(f"Total detections: {total_detections}")
        
        if errors:
            print(f"\nErrors encountered: {len(errors)}")
            for error in errors:
                print(f"  ❌ {error}")
        
        print("✅ Batch processing complete!")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 