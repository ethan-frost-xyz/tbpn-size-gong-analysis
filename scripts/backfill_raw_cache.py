#!/usr/bin/env python3
"""Backfill script to populate raw cache for existing preprocessed files.

This script finds existing preprocessed audio files that don't have corresponding
raw files and downloads the raw audio to complete the dual-cache setup.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gong_detector.core.utils.local_media import LocalMediaIndex, LocalMediaEntry
from gong_detector.core.utils.youtube_utils import (
    download_and_trim_youtube_audio,
    video_id_from_url,
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def find_missing_raw_files() -> List[Tuple[str, str, str]]:
    """Find preprocessed files that don't have corresponding raw files.

    Returns:
        List of tuples: (video_id, preprocessed_path, source_url)
    """
    logger = logging.getLogger(__name__)
    
    # Find project root
    current = Path(__file__).resolve().parent
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            project_root = parent
            break

    if not project_root:
        raise RuntimeError("Could not find project root")

    local_media_base = project_root / "data/local_media"
    preprocessed_dir = local_media_base / "preprocessed"
    raw_dir = local_media_base / "raw"
    index_path = local_media_base / "index.json"

    if not preprocessed_dir.exists():
        logger.info("No preprocessed directory found")
        return []

    # Load existing index
    index_data = {}
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")

    missing_raw = []
    
    # Find all preprocessed files
    for preprocessed_file in preprocessed_dir.glob("*_16k_mono.wav"):
        video_id = preprocessed_file.stem.replace("_16k_mono", "")
        
        # Check if raw file exists
        raw_files = list(raw_dir.glob(f"{video_id}.*"))
        if not raw_files:
            # Get source URL from index or construct a placeholder
            source_url = ""
            if video_id in index_data:
                source_url = index_data[video_id].get("source_url", "")
            
            if not source_url:
                # Try to construct a URL (this is a fallback)
                source_url = f"https://www.youtube.com/watch?v={video_id}"
            
            missing_raw.append((video_id, str(preprocessed_file), source_url))
            logger.info(f"Missing raw for: {video_id}")

    return missing_raw


def backfill_raw_file(video_id: str, source_url: str, dry_run: bool = False) -> bool:
    """Backfill raw file for a single video.

    Args:
        video_id: YouTube video ID
        source_url: Source YouTube URL
        dry_run: If True, don't actually download

    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        if dry_run:
            logger.info(f"[DRY RUN] Would backfill raw for: {video_id}")
            return True

        logger.info(f"Backfilling raw for: {video_id}")
        
        # Download and cache (this will create both raw and preprocessed)
        # We use a temporary output path since we only want to trigger the cache
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_output = temp_file.name

        try:
            # This will download raw and create preprocessed, updating the index
            audio_path, video_title, upload_date = download_and_trim_youtube_audio(
                url=source_url,
                output_path=temp_output,
                start_time=None,
                duration=None,
            )
            
            # Clean up the temporary file
            os.unlink(temp_output)
            
            logger.info(f"✓ Successfully backfilled raw for: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backfill raw for {video_id}: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_output):
                os.unlink(temp_output)
            return False

    except Exception as e:
        logger.error(f"Error processing {video_id}: {e}")
        return False


def main() -> int:
    """Main backfill function."""
    parser = argparse.ArgumentParser(
        description="Backfill raw cache for existing preprocessed files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually downloading",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting raw cache backfill...")

    try:
        # Find missing raw files
        missing_raw = find_missing_raw_files()
        
        if not missing_raw:
            logger.info("No missing raw files found!")
            return 0

        logger.info(f"Found {len(missing_raw)} files missing raw cache")

        # Apply limit if specified
        if args.limit:
            missing_raw = missing_raw[:args.limit]
            logger.info(f"Limited to {len(missing_raw)} files for testing")

        # Process each missing raw file
        successful = 0
        failed = 0

        for i, (video_id, preprocessed_path, source_url) in enumerate(missing_raw, 1):
            logger.info(f"Processing {i}/{len(missing_raw)}: {video_id}")
            
            if backfill_raw_file(video_id, source_url, args.dry_run):
                successful += 1
            else:
                failed += 1

        # Summary
        logger.info(f"\nBackfill complete:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {len(missing_raw)}")

        if args.dry_run:
            logger.info("This was a dry run - no files were actually downloaded")

        return 0 if failed == 0 else 1

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
