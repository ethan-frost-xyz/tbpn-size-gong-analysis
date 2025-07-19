"""Audio conversion module for downloading and processing audio files.

This module provides functionality to download audio from YouTube URLs or process
local audio files, converting them to the required format for audio analysis.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import ffmpeg  # type: ignore
import yt_dlp  # type: ignore


def convert_youtube_audio(
    url_or_path: str, output_wav_path: str = "audio.wav"
) -> str:
    """Download and convert audio from YouTube URL or local file to WAV format.
    
    Args:
        url_or_path: YouTube URL or local file path to process
        output_wav_path: Output path for the converted WAV file
        
    Returns:
        Path to the converted WAV file
        
    Raises:
        FileNotFoundError: If local file path doesn't exist
        RuntimeError: If download or conversion fails
    """
    output_path = Path(output_wav_path).resolve()
    
    # Check if input is a local file path
    if os.path.exists(url_or_path):
        print(f"Processing local file: {url_or_path}")
        input_path = url_or_path
    else:
        print(f"Downloading audio from YouTube: {url_or_path}")
        input_path = _download_youtube_audio(url_or_path)
    
    try:
        # Convert to WAV with specified format
        _convert_to_wav(input_path, str(output_path))
        
        # Clean up temporary file if it was downloaded
        if not os.path.exists(url_or_path) and os.path.exists(input_path):
            os.remove(input_path)
            
        print(f"Conversion complete! Output saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        # Clean up temporary file on error
        if not os.path.exists(url_or_path) and os.path.exists(input_path):
            os.remove(input_path)
        raise RuntimeError(f"Conversion failed: {e}") from e


def _download_youtube_audio(url: str) -> str:
    """Download audio from YouTube URL to temporary file.
    
    Args:
        url: YouTube URL to download
        
    Returns:
        Path to downloaded audio file
        
    Raises:
        RuntimeError: If download fails
    """
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info to get the filename
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            # Change extension to mp3 since we're extracting audio
            filename = os.path.splitext(filename)[0] + ".mp3"
            
            # Download the audio
            ydl.download([url])
            
            if os.path.exists(filename):
                return filename
            else:
                # Sometimes the filename might be different, find any mp3 in temp dir
                mp3_files = list(Path(temp_dir).glob("*.mp3"))
                if mp3_files:
                    return str(mp3_files[0])
                else:
                    raise RuntimeError("Downloaded file not found")
                    
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {e}") from e


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio file to WAV format with specified parameters.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        
    Raises:
        RuntimeError: If conversion fails
    """
    try:
        # Use ffmpeg to convert to WAV with mono channel and 16kHz sample rate
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                acodec="pcm_s16le",  # 16-bit PCM
                ac=1,                # Mono channel
                ar=16000,           # 16kHz sample rate
                y=None              # Overwrite output file if exists
            )
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e}") from e


def main() -> int:
    """Command-line interface for audio conversion."""
    parser = argparse.ArgumentParser(
        description="Download and convert audio from YouTube or local files"
    )
    parser.add_argument(
        "input",
        help="YouTube URL or local file path to convert"
    )
    parser.add_argument(
        "-o", "--output",
        default="audio.wav",
        help="Output WAV file path (default: audio.wav)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_youtube_audio(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 