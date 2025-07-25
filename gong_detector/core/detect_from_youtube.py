#!/usr/bin/env python3
"""Simple YouTube gong detector script.

Download YouTube videos, optionally trim audio segments, and detect gongs
using YAMNet. Designed for testing on real podcast episodes.
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import yt_dlp  # type: ignore

from .yamnet_runner import YAMNetGongDetector
from .audio_utils import extract_audio_slice


def cleanup_old_temp_files(temp_dir: str, max_age_hours: int = 24) -> None:
    """Clean up old temporary audio files.

    Args:
        temp_dir: Directory containing temp files
        max_age_hours: Maximum age in hours before cleanup
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    temp_files = glob.glob(os.path.join(temp_dir, "temp_youtube_audio_*.wav"))

    for temp_file in temp_files:
        file_age = current_time - os.path.getmtime(temp_file)
        if file_age > max_age_seconds:
            try:
                os.remove(temp_file)
                print(f"Cleaned up old temp file: {temp_file}")
            except OSError:
                pass  # File might already be gone


def download_and_trim_youtube_audio(
    url: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
) -> str:
    """Download YouTube audio and optionally trim to specified segment.

    Args:
        url: YouTube URL to download
        output_path: Path for output WAV file
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)

    Returns:
        Path to the processed audio file

    Raises:
        RuntimeError: If download or conversion fails
    """
    print(f"Downloading audio from: {url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = os.path.join(temp_dir, "temp_audio.%(ext)s")

        # Download with yt-dlp
        downloaded_file = _download_youtube_audio(url, temp_audio)

        # Convert and trim with ffmpeg
        _convert_and_trim_audio(downloaded_file, output_path, start_time, duration)

        print(f"Audio saved to: {output_path}")

    return output_path


def _download_youtube_audio(url: str, output_template: str) -> str:
    """Download audio from YouTube using yt-dlp."""
    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": output_template,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file
    temp_dir = os.path.dirname(output_template)
    downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("temp_audio")]

    if not downloaded_files:
        raise RuntimeError("Failed to download audio from YouTube")

    return os.path.join(temp_dir, downloaded_files[0])


def _convert_and_trim_audio(
    input_file: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
) -> None:
    """Convert audio to WAV format and optionally trim it."""
    cmd = ["ffmpeg", "-i", input_file]

    # Add trimming parameters if specified
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    # Audio conversion settings for YAMNet
    cmd.extend(
        [
            "-ac",
            "1",  # mono
            "-ar",
            "16000",  # 16kHz
            "-y",  # overwrite output
            output_path,
        ]
    )

    print("Converting audio to 16kHz mono WAV...")
    if start_time is not None or duration is not None:
        print(f"Trimming: start={start_time}s, duration={duration}s")

    subprocess.run(cmd, check=True, capture_output=True)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_summary(
    detections: list[tuple[float, float]],
    total_duration: float,
    start_offset: float = 0.0,
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


def setup_directories() -> tuple[str, str]:
    """Create necessary directories and return paths.

    Returns:
        Tuple of (temp_audio_dir, csv_results_dir)
    """
    temp_audio_dir = "temp_audio"
    csv_results_dir = "csv_results"

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(csv_results_dir, exist_ok=True)

    return temp_audio_dir, csv_results_dir


def create_temp_audio_path(temp_dir: str) -> str:
    """Create a unique temporary audio file path.

    Args:
        temp_dir: Directory for temporary files

    Returns:
        Path to temporary audio file
    """
    return os.path.join(temp_dir, f"temp_youtube_audio_{uuid.uuid4().hex[:8]}.wav")


def save_results_to_csv(
    detections: list[tuple[float, float]], csv_filename: str, csv_dir: str
) -> None:
    """Save detection results to CSV file.

    Args:
        detections: List of detection tuples
        csv_filename: Name of the CSV file
        csv_dir: Directory to save CSV files
    """
    if not csv_filename.endswith(".csv"):
        csv_filename += ".csv"

    # Ensure directory exists
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, csv_filename)
    detector = YAMNetGongDetector()
    df = detector.detections_to_dataframe(detections)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")


def save_positive_samples(
    detections: list[tuple[float, float]], 
    audio_path: str, 
    positive_dir: Path
) -> None:
    """Save detected gong segments to positive samples folder.

    Args:
        detections: List of (timestamp, confidence) tuples
        audio_path: Path to source audio file
        positive_dir: Directory to save positive samples
    """
    if not detections:
        print("No gong detections to save")
        return

    # Load audio waveform
    detector = YAMNetGongDetector()
    waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)
    
    # Create positive directory if it doesn't exist
    positive_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for i, (timestamp, confidence) in enumerate(detections):
        try:
            # Extract 3-second segment around detection
            segment = extract_audio_slice(
                waveform, 
                timestamp, 
                duration_before=1.0, 
                duration_after=2.0,
                sample_rate=sample_rate
            )
            
            # Save segment with descriptive filename
            filename = f"gong_{timestamp:.1f}s_conf_{confidence:.3f}_{i+1}.wav"
            output_path = positive_dir / filename
            
            # Convert numpy array to WAV file
            import soundfile as sf
            sf.write(str(output_path), segment, sample_rate)
            
            saved_count += 1
            print(f"✓ Saved: {filename}")
            
        except Exception as e:
            print(f"✗ Failed to save segment {i+1}: {e}")
    
    print(f"\nSaved {saved_count} positive samples to: {positive_dir}")


def process_audio_with_yamnet(
    temp_audio: str, threshold: float
) -> tuple[list[tuple[float, float]], float, float]:
    """Process audio file with YAMNet detector.

    Args:
        temp_audio: Path to temporary audio file
        threshold: Confidence threshold for detection

    Returns:
        Tuple of (detections, total_duration, max_gong_confidence)
    """
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

    # Detect gongs with duration validation
    total_duration = len(waveform) / sample_rate
    detections = detector.detect_gongs(
        scores=scores, confidence_threshold=threshold, audio_duration=total_duration
    )

    # Print results using detector's formatted output
    detector.print_detections(detections)

    max_gong_confidence = scores[:, 172].max()

    return detections, total_duration, max_gong_confidence


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Detect gongs in YouTube videos using YAMNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --start_time 5680 --duration 20
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --threshold 0.3 --save_csv results.csv
  python detect_from_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --save_positive_samples
        """,
    )

    parser.add_argument("youtube_url", help="YouTube URL to process")
    parser.add_argument(
        "--start_time", type=int, help="Start time in seconds (optional)"
    )
    parser.add_argument("--duration", type=int, help="Duration in seconds (optional)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for gong detection (default: 0.4)",
    )
    parser.add_argument("--save_csv", help="Save results to CSV file (optional)")
    parser.add_argument(
        "--keep_audio",
        action="store_true",
        help="Keep temporary audio file for training data extraction",
    )
    parser.add_argument(
        "--save_positive_samples",
        action="store_true",
        help="Save detected gong segments to training data folder for human review",
    )

    return parser


def main() -> None:
    """Run YouTube gong detection."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup directories
    temp_audio_dir, csv_results_dir = setup_directories()
    cleanup_old_temp_files(temp_audio_dir)

    # Create temporary audio file path
    temp_audio = create_temp_audio_path(temp_audio_dir)

    try:
        # Step 1: Download and process audio
        print("Step 1: Downloading and processing audio...")
        download_and_trim_youtube_audio(
            url=args.youtube_url,
            output_path=temp_audio,
            start_time=args.start_time,
            duration=args.duration,
        )

        # Step 2-4: Process with YAMNet
        detections, total_duration, max_gong_confidence = process_audio_with_yamnet(
            temp_audio, args.threshold
        )

        # Save to CSV if requested
        if args.save_csv:
            save_results_to_csv(detections, args.save_csv, csv_results_dir)

        # Save positive samples if requested
        if args.save_positive_samples and detections:
            positive_dir = Path("gong_detector/training/data/raw_samples/positive")
            print(f"\nSaving positive samples to: {positive_dir}")
            save_positive_samples(detections, temp_audio, positive_dir)

        # Print summary and max confidence
        start_offset = args.start_time or 0
        print_summary(detections, total_duration, start_offset)
        print(f"Max gong confidence: {max_gong_confidence:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Clean up temporary file
        if not args.keep_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
            print(f"Cleaned up temporary file: {temp_audio}")
        elif args.keep_audio and os.path.exists(temp_audio):
            print(f"Temporary file preserved for training: {temp_audio}")


if __name__ == "__main__":
    main()
