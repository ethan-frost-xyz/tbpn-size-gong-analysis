#!/usr/bin/env python3
"""Migration script to convert existing WebM raw cache files to WAV format.

This script converts all existing WebM files in the raw cache to high-quality WAV
format for optimal compatibility with loudness analysis tools.
"""

import logging
import os
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def migrate_webm_to_wav() -> None:
    """Convert all WebM files in raw cache to WAV format."""
    project_root = find_project_root()
    raw_cache_dir = project_root / "data/local_media/raw"

    if not raw_cache_dir.exists():
        logger.info("Raw cache directory does not exist, nothing to migrate")
        return

    # Find all WebM files
    webm_files = list(raw_cache_dir.glob("*.webm"))

    if not webm_files:
        logger.info("No WebM files found in raw cache")
        return

    logger.info(f"Found {len(webm_files)} WebM files to convert")

    converted_count = 0
    failed_count = 0

    for webm_file in webm_files:
        video_id = webm_file.stem
        wav_file = raw_cache_dir / f"{video_id}.wav"
        wav_tmp = raw_cache_dir / f"{video_id}.tmp.wav"

        # Skip if WAV already exists
        if wav_file.exists():
            logger.info(f"WAV already exists for {video_id}, skipping")
            continue

        try:
            logger.info(f"Converting {webm_file.name} to WAV...")

            # Convert to storage-optimized WAV for LUFS analysis
            # 16kHz 16-bit PCM provides sufficient quality for loudness measurements while minimizing storage
            cmd = [
                "ffmpeg",
                "-i",
                str(webm_file),
                "-map",
                "a:0",  # Use first audio stream
                "-acodec",
                "pcm_s16le",  # 16-bit PCM (sufficient dynamic range for LUFS)
                "-ar",
                "16000",  # 16kHz sample rate (adequate for loudness analysis, saves storage)
                "-af",
                "aresample=resampler=soxr:precision=28:cheby=1",  # High-quality resampling
                "-vn",  # no video
                "-sn",  # no subtitles
                "-dn",  # no data
                "-y",  # overwrite
                "-nostdin",  # non-interactive
                "-loglevel",
                "error",  # minimal output
                str(wav_tmp),
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Atomically move to final location
            os.replace(wav_tmp, wav_file)

            # Verify the WAV file was created successfully
            if wav_file.exists() and wav_file.stat().st_size > 0:
                logger.info(f"✓ Successfully converted {video_id}")

                # Optionally remove the original WebM file to save space
                # Uncomment the next line if you want to delete WebM files after conversion
                # webm_file.unlink()

                converted_count += 1
            else:
                logger.error(
                    f"✗ Conversion failed for {video_id}: output file is empty or missing"
                )
                failed_count += 1

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ FFmpeg conversion failed for {video_id}: {e.stderr}")
            failed_count += 1

            # Clean up temp file if it exists
            if wav_tmp.exists():
                try:
                    wav_tmp.unlink()
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"✗ Unexpected error converting {video_id}: {e}")
            failed_count += 1

            # Clean up temp file if it exists
            if wav_tmp.exists():
                try:
                    wav_tmp.unlink()
                except OSError:
                    pass

    logger.info(
        f"Migration complete: {converted_count} converted, {failed_count} failed"
    )

    if converted_count > 0:
        logger.info("✓ Raw cache migration successful!")
        logger.info(
            "Note: Original WebM files are preserved. You can delete them manually if desired."
        )

    if failed_count > 0:
        logger.warning(
            f"⚠ {failed_count} files failed to convert. Check the logs above for details."
        )


if __name__ == "__main__":
    try:
        migrate_webm_to_wav()
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        exit(1)
