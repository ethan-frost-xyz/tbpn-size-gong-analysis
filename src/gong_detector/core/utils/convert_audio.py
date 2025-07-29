"""Simple audio converter for YouTube URLs and local files."""

import os
import subprocess

import yt_dlp  # type: ignore


def convert_youtube_audio(url_or_path: str, output_wav_path: str = "audio.wav") -> str:
    """Convert YouTube URL or local file to WAV format.

    Args:
        url_or_path: YouTube URL or local file path
        output_wav_path: Output WAV file path

    Returns:
        Path to converted WAV file

    Raises:
        FileNotFoundError: If local file doesn't exist
        RuntimeError: If download or conversion fails
        ValueError: If input parameters are invalid
    """
    if not url_or_path.strip():
        raise ValueError("URL or file path cannot be empty")

    if not output_wav_path.strip():
        raise ValueError("Output path cannot be empty")

    # Handle local file
    if os.path.exists(url_or_path):
        print(f"Converting local file: {url_or_path}")
        input_path = url_or_path
    else:
        # Download from YouTube
        print(f"Downloading from YouTube: {url_or_path}")
        input_path = _download_audio(url_or_path)

    try:
        # Convert to WAV
        _convert_to_wav(input_path, output_wav_path)

        print(f"Conversion complete! Saved to: {output_wav_path}")
        return output_wav_path

    finally:
        # Clean up downloaded file
        if input_path != url_or_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass  # File might be in use or already deleted


def _download_audio(url: str) -> str:
    """Download audio from YouTube URL.

    Args:
        url: YouTube URL to download

    Returns:
        Path to downloaded audio file

    Raises:
        RuntimeError: If download fails
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": "temp_%(title)s.%(ext)s",
        "quiet": True,  # Reduce yt-dlp output noise
    }

    # Add cookies if available
    from .youtube_utils import get_cookies_path

    cookies_path = get_cookies_path()
    if cookies_path:
        print(f"Using cookies from: {cookies_path}")
        ydl_opts["cookiefile"] = cookies_path
    else:
        print(
            "No cookies file found. If you encounter bot detection, create a cookies.txt file."
        )
        print(
            "See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
        )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise RuntimeError("Failed to extract video information")

            filename = ydl.prepare_filename(info)
            # Handle different possible extensions
            for ext in [".webm", ".mp4", ".m4a"]:
                filename = filename.replace(ext, ".mp3")

            ydl.download([url])

            if not os.path.exists(filename):
                raise RuntimeError(f"Downloaded file not found: {filename}")

            return filename

    except Exception as e:
        if "Sign in to confirm you're not a bot" in str(e):
            print("\nBot detection detected! To fix this:")
            print("1. Create a cookies.txt file with your YouTube cookies")
            print("2. Place it in the project root or your home directory")
            print(
                "3. See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
            )
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"YouTube download failed: {e}") from e


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio to WAV using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If ffmpeg conversion fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16kHz sample rate
        "-y",  # overwrite output
        "-loglevel",
        "error",  # Reduce ffmpeg output noise
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg completed but output file was not created")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown ffmpeg error"
        raise RuntimeError(f"Audio conversion failed: {error_msg}") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "FFmpeg not found. Please install ffmpeg and ensure it's in your PATH."
        ) from e


def validate_audio_file(file_path: str) -> bool:
    """Validate that a file exists and appears to be an audio file.

    Args:
        file_path: Path to the file to validate

    Returns:
        True if file exists and has an audio extension
    """
    if not os.path.exists(file_path):
        return False

    audio_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".aac",
        ".webm",
        ".mp4",
    }
    _, ext = os.path.splitext(file_path.lower())

    return ext in audio_extensions


def get_audio_info(file_path: str) -> dict:
    """Get basic information about an audio file using ffprobe.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary with audio file information

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If ffprobe fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        import json

        probe_data = json.loads(result.stdout)

        # Extract basic info
        format_info = probe_data.get("format", {})
        streams = probe_data.get("streams", [])

        audio_stream = None
        for stream in streams:
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        return {
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "format_name": format_info.get("format_name", "unknown"),
            "sample_rate": (
                int(audio_stream.get("sample_rate", 0)) if audio_stream else 0
            ),
            "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
            "codec_name": (
                audio_stream.get("codec_name", "unknown") if audio_stream else "unknown"
            ),
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get audio info: {e}") from e
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Failed to parse audio info: {e}") from e


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_audio.py <youtube_url_or_file> [output.wav]")
        sys.exit(1)

    input_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "audio.wav"

    try:
        result_path = convert_youtube_audio(input_arg, output_arg)

        # Show basic info about the converted file
        if validate_audio_file(result_path):
            try:
                info = get_audio_info(result_path)
                print(f"Duration: {info['duration']:.2f} seconds")
                print(f"Sample rate: {info['sample_rate']} Hz")
                print(f"Channels: {info['channels']}")
            except RuntimeError:
                pass  # Info gathering failed, but conversion succeeded

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
