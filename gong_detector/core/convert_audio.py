"""Simple audio converter for YouTube URLs and local files."""

import os
import subprocess
from typing import Optional

import yt_dlp  # type: ignore


def convert_youtube_audio(url_or_path: str, output_wav_path: str = "audio.wav") -> str:
    """Convert YouTube URL or local file to WAV format.
    
    Args:
        url_or_path: YouTube URL or local file path
        output_wav_path: Output WAV file path
        
    Returns:
        Path to converted WAV file
    """
    # Handle local file
    if os.path.exists(url_or_path):
        print(f"Converting local file: {url_or_path}")
        input_path = url_or_path
    else:
        # Download from YouTube
        print(f"Downloading from YouTube: {url_or_path}")
        input_path = _download_audio(url_or_path)
    
    # Convert to WAV
    _convert_to_wav(input_path, output_wav_path)
    
    # Clean up downloaded file
    if input_path != url_or_path and os.path.exists(input_path):
        os.remove(input_path)
    
    print(f"Conversion complete! Saved to: {output_wav_path}")
    return output_wav_path


def _download_audio(url: str) -> str:
    """Download audio from YouTube URL."""
    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": "temp_%(title)s.%(ext)s",
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".mp4", ".mp3")
        ydl.download([url])
        return filename


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio to WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz
        "-y",                 # overwrite
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_audio.py <youtube_url_or_file> [output.wav]")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "audio.wav"
    
    try:
        convert_youtube_audio(input_arg, output_arg)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 