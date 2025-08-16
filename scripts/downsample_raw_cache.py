#!/usr/bin/env python3
"""Downsample existing high-quality WAV files to 16kHz for storage optimization.

This script converts existing 48kHz WAV files to 16kHz while preserving
adequate quality for LUFS analysis and dramatically reducing storage usage.
"""

import logging
import os
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def get_audio_info(file_path: str) -> dict:
    """Get audio file information using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        import json
        probe_data = json.loads(result.stdout)
        
        # Extract audio stream info
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return {
                    "sample_rate": int(stream.get("sample_rate", 0)),
                    "channels": int(stream.get("channels", 0)),
                    "duration": float(probe_data.get("format", {}).get("duration", 0)),
                    "size": int(probe_data.get("format", {}).get("size", 0)),
                }
        return {}
    except Exception as e:
        logger.error(f"Failed to get audio info for {file_path}: {e}")
        return {}


def downsample_wav_files() -> None:
    """Downsample all high-quality WAV files in raw cache to 16kHz."""
    project_root = find_project_root()
    raw_cache_dir = project_root / "data/local_media/raw"
    
    if not raw_cache_dir.exists():
        logger.info("Raw cache directory does not exist")
        return
    
    # Find all WAV files
    wav_files = list(raw_cache_dir.glob("*.wav"))
    
    if not wav_files:
        logger.info("No WAV files found in raw cache")
        return
    
    logger.info(f"Found {len(wav_files)} WAV files to check for downsampling")
    
    downsampled_count = 0
    skipped_count = 0
    failed_count = 0
    total_space_saved = 0
    
    for wav_file in wav_files:
        video_id = wav_file.stem
        
        # Get current audio info
        audio_info = get_audio_info(str(wav_file))
        if not audio_info:
            logger.warning(f"Could not get audio info for {video_id}, skipping")
            failed_count += 1
            continue
        
        current_sample_rate = audio_info.get("sample_rate", 0)
        current_size = audio_info.get("size", 0)
        
        # Skip if already 16kHz or lower
        if current_sample_rate <= 16000:
            logger.info(f"✓ {video_id} already at {current_sample_rate}Hz, skipping")
            skipped_count += 1
            continue
        
        logger.info(f"Downsampling {video_id} from {current_sample_rate}Hz to 16kHz...")
        logger.info(f"  Current size: {current_size / (1024**3):.2f} GB")
        
        # Create temporary downsampled file
        wav_tmp = raw_cache_dir / f"{video_id}.tmp.wav"
        
        try:
            # Downsample to 16kHz with high-quality resampling
            # Use 16-bit PCM to balance quality and storage
            cmd = [
                "ffmpeg",
                "-i", str(wav_file),
                "-map", "a:0",  # Use first audio stream
                "-acodec", "pcm_s16le",  # 16-bit PCM (sufficient for LUFS analysis)
                "-ar", "16000",  # 16kHz sample rate (adequate for loudness analysis)
                "-af", "aresample=resampler=soxr:precision=28:cheby=1",  # High-quality resampling
                "-vn",  # no video
                "-sn",  # no subtitles
                "-dn",  # no data
                "-y",  # overwrite
                "-nostdin",  # non-interactive
                "-loglevel", "error",  # minimal output
                str(wav_tmp),
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Verify the downsampled file
            new_audio_info = get_audio_info(str(wav_tmp))
            if not new_audio_info or new_audio_info.get("sample_rate") != 16000:
                raise RuntimeError("Downsampled file verification failed")
            
            new_size = new_audio_info.get("size", 0)
            space_saved = current_size - new_size
            total_space_saved += space_saved
            
            # Atomically replace the original file
            os.replace(wav_tmp, wav_file)
            
            logger.info(f"✓ Successfully downsampled {video_id}")
            logger.info(f"  New size: {new_size / (1024**3):.2f} GB")
            logger.info(f"  Space saved: {space_saved / (1024**3):.2f} GB")
            
            downsampled_count += 1
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ FFmpeg downsampling failed for {video_id}: {e.stderr}")
            failed_count += 1
            
            # Clean up temp file if it exists
            if wav_tmp.exists():
                try:
                    wav_tmp.unlink()
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"✗ Unexpected error downsampling {video_id}: {e}")
            failed_count += 1
            
            # Clean up temp file if it exists
            if wav_tmp.exists():
                try:
                    wav_tmp.unlink()
                except OSError:
                    pass
    
    logger.info("=" * 60)
    logger.info(f"Downsampling complete:")
    logger.info(f"  Downsampled: {downsampled_count} files")
    logger.info(f"  Skipped (already 16kHz): {skipped_count} files")
    logger.info(f"  Failed: {failed_count} files")
    logger.info(f"  Total space saved: {total_space_saved / (1024**3):.2f} GB")
    
    if downsampled_count > 0:
        logger.info("✓ Raw cache downsampling successful!")
        logger.info("Files are now optimized for LUFS analysis with minimal storage usage.")


if __name__ == "__main__":
    try:
        downsample_wav_files()
    except KeyboardInterrupt:
        logger.info("Downsampling cancelled by user")
    except Exception as e:
        logger.error(f"Downsampling failed: {e}")
        exit(1)
