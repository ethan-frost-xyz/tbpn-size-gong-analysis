#!/usr/bin/env python3
"""Simple YouTube gong detector script.

Download YouTube videos, optionally trim audio segments, and detect gongs
using YAMNet. Designed for testing on real podcast episodes.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

import pandas as pd  # type: ignore
import yt_dlp  # type: ignore

from yamnet_runner import YAMNetGongDetector


def download_and_trim_youtube_audio(
    url: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None
) -> str:
    """Download YouTube audio and optionally trim to specified segment.
    
    Args:
        url: YouTube URL to download
        output_path: Path for output WAV file
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
        
    Returns:
        Path to the processed audio file
    """
    print(f"Downloading audio from: {url}")
    
    # Download audio using yt-dlp
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = os.path.join(temp_dir, "temp_audio.%(ext)s")
        
        ydl_opts = {
            "format": "bestaudio/best",
            "extractaudio": True,
            "audioformat": "mp3",
            "outtmpl": temp_audio,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Find the downloaded file
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("temp_audio")]
        if not downloaded_files:
            raise RuntimeError("Failed to download audio from YouTube")
        
        input_file = os.path.join(temp_dir, downloaded_files[0])
        
        # Build ffmpeg command for conversion and optional trimming
        cmd = ["ffmpeg", "-i", input_file]
        
        # Add trimming parameters if specified
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        if duration is not None:
            cmd.extend(["-t", str(duration)])
            
        # Audio conversion settings for YAMNet
        cmd.extend([
            "-ac", "1",        # mono
            "-ar", "16000",    # 16kHz
            "-y",              # overwrite output
            output_path
        ])
        
        print(f"Converting audio to 16kHz mono WAV...")
        if start_time is not None or duration is not None:
            print(f"Trimming: start={start_time}s, duration={duration}s")
            
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Audio saved to: {output_path}")
        
    return output_path


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def print_detection_results(detections: List[Tuple[float, float]]) -> None:
    """Print detection results in a formatted table.
    
    Args:
        detections: List of (timestamp, confidence) tuples
    """
    if not detections:
        print("No gong detections found.")
        return
        
    print("\n" + "="*60)
    print("GONG DETECTIONS")
    print("="*60)
    print(f"{'Timestamp':<12} {'Time':<8} {'Confidence':<12}")
    print("-" * 32)
    
    for timestamp, confidence in detections:
        time_str = format_time(timestamp)
        print(f"{timestamp:<12.2f} {time_str:<8} {confidence:<12.4f}")
        
    print("="*60)


def save_detections_to_csv(
    detections: List[Tuple[float, float]], 
    csv_path: str
) -> None:
    """Save detections to CSV file.
    
    Args:
        detections: List of (timestamp, confidence) tuples
        csv_path: Output CSV file path
    """
    if not detections:
        print("No detections to save.")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp_seconds': [d[0] for d in detections],
        'confidence': [d[1] for d in detections]
    })
    
    # Add formatted time column
    df['time_mmss'] = df['timestamp_seconds'].apply(format_time)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")


def print_summary(
    detections: List[Tuple[float, float]], 
    total_duration: float,
    start_offset: float = 0.0
) -> None:
    """Print detection summary.
    
    Args:
        detections: List of (timestamp, confidence) tuples
        total_duration: Total audio duration in seconds
        start_offset: Start time offset if audio was trimmed
    """
    count = len(detections)
    start_time = format_time(start_offset)
    end_time = format_time(start_offset + total_duration)
    
    print(f"\nSUMMARY: Detected {count} gongs between {start_time} and {end_time}")
    
    if count > 0:
        confidences = [d[1] for d in detections]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Maximum confidence: {max_confidence:.3f}")


def main() -> None:
    """Main function for YouTube gong detection."""
    parser = argparse.ArgumentParser(
        description="Detect gongs in YouTube videos using YAMNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --start_time 5680 --duration 20
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --threshold 0.3 --save_csv results.csv
        """
    )
    
    parser.add_argument(
        "youtube_url",
        help="YouTube URL to process"
    )
    parser.add_argument(
        "--start_time",
        type=int,
        help="Start time in seconds (optional)"
    )
    parser.add_argument(
        "--duration", 
        type=int,
        help="Duration in seconds (optional)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for gong detection (default: 0.4)"
    )
    parser.add_argument(
        "--save_csv",
        help="Save results to CSV file (optional)"
    )
    
    args = parser.parse_args()
    
    # Temporary audio file with unique name
    import uuid
    temp_audio = f"temp_youtube_audio_{uuid.uuid4().hex[:8]}.wav"
    
    try:
        # Download and process audio
        print("Step 1: Downloading and processing audio...")
        download_and_trim_youtube_audio(
            url=args.youtube_url,
            output_path=temp_audio,
            start_time=args.start_time,
            duration=args.duration
        )
        
        # Initialize YAMNet detector
        print("\nStep 2: Loading YAMNet model...")
        detector = YAMNetGongDetector()
        detector.load_model()
        
        # Process audio
        print("\nStep 3: Processing audio...")
        waveform, sample_rate = detector.load_and_preprocess_audio(temp_audio)
        
        # Run inference
        print("\nStep 4: Running gong detection...")
        scores, _, _ = detector.run_inference(waveform)
        
        # Detect gongs
        detections = detector.detect_gongs(scores, args.threshold)
        
        # Print results
        print_detection_results(detections)
        
        # Save to CSV if requested
        if args.save_csv:
            save_detections_to_csv(detections, args.save_csv)
        
        # Print summary
        total_duration = len(waveform) / sample_rate
        start_offset = args.start_time or 0
        print_summary(detections, total_duration, start_offset)
        
        # Print max gong confidence
        max_gong_confidence = scores[:, 172].max()
        print(f"Max gong confidence: {max_gong_confidence:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
        # Clean up temporary file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


if __name__ == "__main__":
    main() 