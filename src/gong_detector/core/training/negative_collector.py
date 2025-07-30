#!/usr/bin/env python3
"""Negative sample collector for gong detection training data.

This script collects negative samples (non-gong audio segments) from YouTube videos
for training data. It reuses existing core utilities and integrates with bulk processing.
"""

import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

import numpy as np

from ..detector.yamnet_runner import YAMNetGongDetector
from ..utils.audio_utils import extract_audio_slice
from ..utils.results_utils import format_time_for_filename
from ..utils.youtube_utils import (
    cleanup_old_temp_files,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    setup_directories,
)


def find_negative_timestamps(
    waveform: np.ndarray,
    sample_rate: int,
    num_samples: int,
    gong_detections: list[tuple[float, float, float]],
    min_gap_seconds: float = 30.0,
) -> list[float]:
    """Find random timestamps that are not near gong detections.

    Args:
        waveform: Full audio waveform
        sample_rate: Audio sample rate
        num_samples: Number of negative samples to collect
        gong_detections: List of (window_start, confidence, display_timestamp) tuples
        min_gap_seconds: Minimum gap from gong detections in seconds

    Returns:
        List of timestamps for negative samples
    """
    total_duration = len(waveform) / sample_rate
    negative_timestamps = []

    # Create exclusion zones around gong detections
    exclusion_zones = []
    for _window_start, _, display_timestamp in gong_detections:
        start_zone = max(0, display_timestamp - min_gap_seconds)
        end_zone = min(total_duration, display_timestamp + min_gap_seconds)
        exclusion_zones.append((start_zone, end_zone))

    # Merge overlapping zones
    if exclusion_zones:
        exclusion_zones.sort()
        merged_zones = [exclusion_zones[0]]
        for start, end in exclusion_zones[1:]:
            last_start, last_end = merged_zones[-1]
            if start <= last_end:
                merged_zones[-1] = (last_start, max(last_end, end))
            else:
                merged_zones.append((start, end))
        exclusion_zones = merged_zones

    # Find available regions (simplified logic)
    available_regions = []
    last_end = 0

    for start, end in exclusion_zones:
        if start > last_end + 10:  # Reduced minimum region size
            available_regions.append((last_end, start))
        last_end = end

    # Add final region if there's space
    if total_duration > last_end + 10:
        available_regions.append((last_end, total_duration))

    if not available_regions:
        print("Warning: No suitable regions found for negative samples")
        return []

    # Select random timestamps from available regions
    attempts = 0
    max_attempts = num_samples * 20  # Increased max attempts

    while len(negative_timestamps) < num_samples and attempts < max_attempts:
        attempts += 1

        # Pick random region
        region_start, region_end = random.choice(available_regions)

        # Pick random timestamp within region (with smaller buffer)
        buffer = 10.0  # Reduced buffer
        if region_end - region_start < 2 * buffer:
            continue

        timestamp = random.uniform(region_start + buffer, region_end - buffer)

        # Ensure we're not too close to existing selections
        too_close = False
        for existing in negative_timestamps:
            if abs(timestamp - existing) < 5.0:  # Reduced minimum spacing
                too_close = True
                break

        if not too_close:
            negative_timestamps.append(timestamp)

    return negative_timestamps


def collect_negative_samples(
    youtube_url: str,
    num_samples: int = 5,
    threshold: float = 0.4,
    max_threshold: Optional[float] = None,
    keep_audio: bool = False,
    cookies_from_browser: Optional[str] = None,
) -> dict[str, any]:
    """Collect negative samples from a YouTube video.

    Args:
        youtube_url: YouTube URL to process
        num_samples: Number of negative samples to collect
        threshold: Confidence threshold for gong detection
        max_threshold: Maximum confidence threshold for gong detection
        keep_audio: Whether to keep temporary audio files

    Returns:
        Dictionary with results including success status and sample count
    """
    # Setup directories
    temp_audio_dir, _ = setup_directories()
    cleanup_old_temp_files(temp_audio_dir)

    # Create temporary audio file path
    temp_audio = create_temp_audio_path(temp_audio_dir)

    try:
        # Step 1: Download and process audio
        print("Step 1: Downloading audio from YouTube...")
        # Build yt-dlp options
        yt_dlp_options = {}
        if cookies_from_browser:
            yt_dlp_options["cookiesfrombrowser"] = (cookies_from_browser,)

        temp_audio, video_title, upload_date = download_and_trim_youtube_audio(
            url=youtube_url,
            output_path=temp_audio,
            yt_dlp_options=yt_dlp_options,
        )

        # Step 2: Load YAMNet and detect gongs
        print("Step 2: Detecting gongs to avoid...")
        detector = YAMNetGongDetector()
        detector.load_model()

        waveform, sample_rate = detector.load_and_preprocess_audio(temp_audio)
        scores, _, _ = detector.run_inference(waveform)

        total_duration = len(waveform) / sample_rate
        gong_detections = detector.detect_gongs(
            scores=scores,
            confidence_threshold=threshold,
            max_confidence_threshold=max_threshold,
            audio_duration=total_duration,
        )

        print(f"Found {len(gong_detections)} gong detections to avoid")

        # Step 3: Find negative sample timestamps
        print(f"Step 3: Finding {num_samples} negative sample locations...")
        negative_timestamps = find_negative_timestamps(
            waveform=waveform,
            sample_rate=sample_rate,
            num_samples=num_samples,
            gong_detections=gong_detections,
        )

        if not negative_timestamps:
            return {
                "success": False,
                "error_message": "No suitable regions found for negative samples",
                "sample_count": 0,
            }

        # Step 4: Extract and save negative samples
        print("Step 4: Extracting and saving negative samples...")
        # Get the correct path relative to the project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        negative_base_dir = (
            project_root
            / "src"
            / "gong_detector"
            / "training"
            / "data"
            / "raw_samples"
            / "negative"
        )
        negative_base_dir.mkdir(parents=True, exist_ok=True)

        # Use date-based folder naming if available
        if video_title:
            from ..utils.youtube_utils import create_folder_name_from_title

            folder_name = create_folder_name_from_title(video_title)
            sample_dir = negative_base_dir / folder_name
        elif upload_date:
            from ..utils.youtube_utils import create_folder_name_from_date

            folder_name = create_folder_name_from_date(upload_date)
            sample_dir = negative_base_dir / folder_name
        else:
            sample_dir = negative_base_dir

        sample_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for i, timestamp in enumerate(negative_timestamps, 1):
            try:
                # Extract 3-second segment (matching positive sample format)
                audio_slice = extract_audio_slice(
                    waveform=waveform,
                    timestamp=timestamp,
                    duration_before=0.75,  # 0.75s before (like positive samples)
                    duration_after=2.25,  # 2.25s after (like positive samples)
                    sample_rate=sample_rate,
                )

                # Create filename (matching positive sample format)
                time_str = format_time_for_filename(timestamp)
                filename = f"negative_at_{time_str}_s_{i:02d}.wav"
                output_path = sample_dir / filename

                # Save audio slice
                import soundfile as sf

                sf.write(str(output_path), audio_slice, sample_rate)

                saved_count += 1
                print(f"  Saved: {filename} (at {format_time(timestamp)})")

            except Exception as e:
                print(f"  Error saving sample {i}: {e}")

        print(f"\n✓ Collected {saved_count} negative samples")
        print(f"  Video: {video_title}")
        print(f"  Saved to: {sample_dir}")

        return {
            "success": True,
            "sample_count": saved_count,
            "video_title": video_title,
            "sample_dir": str(sample_dir),
        }

    except Exception as e:
        return {"success": False, "error_message": str(e), "sample_count": 0}

    finally:
        # Clean up temporary audio file unless keep_audio is True
        if not keep_audio:
            try:
                if Path(temp_audio).exists():
                    Path(temp_audio).unlink()
            except Exception:
                pass


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


def main() -> None:
    """Run negative sample collection for a single YouTube URL."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect negative samples from YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m gong_detector.core.negative_sample_collector "https://www.youtube.com/watch?v=VIDEO_ID"
  python -m gong_detector.core.negative_sample_collector "https://www.youtube.com/watch?v=VIDEO_ID" --num_samples 10
  python -m gong_detector.core.negative_sample_collector "https://www.youtube.com/watch?v=VIDEO_ID" --threshold 0.3
        """,
    )

    parser.add_argument("youtube_url", help="YouTube URL to process")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of negative samples to collect (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for gong detection (default: 0.4)",
    )
    parser.add_argument(
        "--max_threshold",
        type=float,
        default=None,
        help="Maximum confidence threshold for gong detection (optional)",
    )
    parser.add_argument(
        "--keep_audio", action="store_true", help="Keep temporary audio files"
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        choices=["chrome", "firefox", "safari", "edge"],
        help="Extract cookies from browser (chrome, firefox, safari, edge)",
    )

    args = parser.parse_args()

    result = collect_negative_samples(
        youtube_url=args.youtube_url,
        num_samples=args.num_samples,
        threshold=args.threshold,
        max_threshold=args.max_threshold,
        keep_audio=args.keep_audio,
        cookies_from_browser=args.cookies_from_browser,
    )

    if not result["success"]:
        print(f"✗ Error: {result['error_message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
