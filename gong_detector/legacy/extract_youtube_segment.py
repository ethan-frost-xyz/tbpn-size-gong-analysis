#!/usr/bin/env python3
"""
Extract YouTube segment and preprocess for YAMNet analysis.

This script downloads a specific segment from a YouTube video
and preprocesses it according to YAMNet requirements.
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import yt_dlp


def download_youtube_segment(
    url: str, start_time: int, duration: int, output_path: str
) -> str:
    """
    Download a specific segment from YouTube video.

    Args:
        url: YouTube URL
        start_time: Start time in seconds
        duration: Duration in seconds
        output_path: Output file path

    Returns:
        Path to downloaded file
    """
    # First download the full video
    temp_path = "temp_full_video"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": temp_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    try:
        print("Downloading full video...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Extract the segment using librosa
        print("Extracting segment...")
        temp_wav_path = temp_path + ".wav"
        waveform, sr = librosa.load(temp_wav_path, sr=16000, mono=True)

        # Calculate segment boundaries
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)

        # Extract segment
        segment = waveform[start_sample:end_sample]

        # Save segment
        import soundfile as sf

        sf.write(output_path, segment, sr)

        # Clean up temp file
        Path(temp_wav_path).unlink(missing_ok=True)

        return output_path

    except Exception as e:
        # Clean up temp file on error
        Path(temp_path + ".wav").unlink(missing_ok=True)
        raise ValueError(f"Failed to download YouTube segment: {e}") from e


def load_for_yamnet(path: str) -> tuple[np.ndarray, int]:
    """
    Load and preprocess audio for YAMNet analysis.

    Args:
        path: Path to audio file

    Returns:
        Tuple of (waveform, sample_rate)
    """
    waveform, sr = librosa.load(path, sr=16000, mono=True)

    # Ensure sample rate is integer
    sr = int(sr)

    # Verify YAMNet requirements
    assert waveform.ndim == 1, "Audio must be mono"
    assert waveform.dtype == "float32", "Librosa should return float32 waveform"
    assert sr == 16000, "Sample rate must be 16,000 Hz"

    print("Waveform range:", waveform.min(), waveform.max())
    print("Sample rate:", sr)
    print("Duration:", len(waveform) / sr, "seconds")
    print("Shape:", waveform.shape)

    return waveform, sr


def save_preprocessed_audio(
    waveform: np.ndarray, output_path: str, sr: int = 16000
) -> None:
    """
    Save preprocessed audio to WAV file.

    Args:
        waveform: Audio waveform
        output_path: Output file path
        sr: Sample rate
    """
    import soundfile as sf

    sf.write(output_path, waveform, sr)
    print(f"Saved preprocessed audio to: {output_path}")


def main() -> None:
    """Main function to extract and preprocess YouTube segment."""
    parser = argparse.ArgumentParser(
        description="Extract YouTube segment and preprocess for YAMNet"
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--start_time", type=int, required=True, help="Start time in seconds"
    )
    parser.add_argument(
        "--duration", type=int, required=True, help="Duration in seconds"
    )
    parser.add_argument(
        "--output", default="yamnet_segment.wav", help="Output file path"
    )

    args = parser.parse_args()

    try:
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading segment from {args.url}")
        print(f"Start time: {args.start_time}s, Duration: {args.duration}s")

        # Download segment
        temp_path = download_youtube_segment(
            args.url, args.start_time, args.duration, str(output_path)
        )

        print(f"Downloaded to: {temp_path}")

        # Preprocess for YAMNet
        print("\nPreprocessing for YAMNet...")
        waveform, sr = load_for_yamnet(temp_path)

        # Save preprocessed audio
        save_preprocessed_audio(waveform, str(output_path), sr)

        print("\n✅ Success! Ready for YAMNet analysis:")
        print(f"   File: {output_path}")
        print(f"   Duration: {len(waveform) / sr:.3f}s")
        print(f"   Sample rate: {sr}Hz")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
