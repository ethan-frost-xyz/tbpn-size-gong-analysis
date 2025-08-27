#!/usr/bin/env python3
"""TBPN YouTube Channel Link Extractor.

Extracts full-episode video links from the TBPN YouTube channel using yt-dlp.
Filters for long-form content (live streams) and excludes short clips.

Usage:
    python scripts/extract_tbpn_links.py [--min-duration MINUTES] [--output FILE]

Features:
- Uses existing yt-dlp dependency
- Leverages existing cookie system for bot detection bypass
- Filters by duration and title patterns to identify full episodes
- Outputs in same format as existing tbpn_youtube_links.txt files
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yt_dlp  # type: ignore

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gong_detector.core.utils.youtube.downloader import get_cookies_path


# Custom exceptions
class TBPNExtractionError(Exception):
    """Base exception for TBPN video extraction errors."""

    pass


class PlaylistExtractionError(TBPNExtractionError):
    """Exception raised when playlist extraction fails."""

    pass


class LocalMediaIndexError(TBPNExtractionError):
    """Exception raised when local media index operations fail."""

    pass


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TBPN Playlist URL - official curated list of episodes
TBPN_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLBV_0ax_G8bowGFK97Nrv0DBmxaDObKgA"

# No filtering needed since playlist is already curated


def extract_playlist_videos(playlist_url: str = TBPN_PLAYLIST_URL) -> list[tuple[str, float, str, str]]:
    """Extract all video metadata from TBPN YouTube playlist.

    Args:
        playlist_url: YouTube playlist URL to extract from.

    Returns:
        List of tuples containing (title, duration_seconds, video_id, upload_date).

    Raises:
        PlaylistExtractionError: If playlist extraction fails.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # Don't download, just get metadata
    }

    # Add cookies if available
    cookies_path = get_cookies_path()
    if cookies_path:
        logger.info(f"Using cookies from: {cookies_path}")
        ydl_opts["cookiefile"] = cookies_path
    else:
        logger.warning(
            "No cookies file found. If you encounter bot detection, create a cookies.txt file."
        )

    logger.info(f"Extracting videos from playlist: {playlist_url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract playlist info
            info = ydl.extract_info(playlist_url, download=False)

            if not info or "entries" not in info:
                raise PlaylistExtractionError("Failed to extract playlist videos")

            videos = []
            for entry in info["entries"]:
                if entry:  # Skip None entries
                    title = entry.get("title", "")
                    duration = entry.get("duration", 0)
                    video_id = entry.get("id", "")
                    upload_date = entry.get("upload_date", "")

                    if title and duration and video_id:
                        videos.append((title, float(duration), video_id, upload_date))

            logger.info(f"Extracted {len(videos)} videos from playlist")
            return videos

    except Exception as e:
        if "Sign in to confirm you're not a bot" in str(e):
            logger.error("\nBot detection detected! To fix this:")
            logger.error("1. Create a cookies.txt file with your YouTube cookies")
            logger.error("2. Place it in the project root or your home directory")
            logger.error(
                "3. See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
            )
        raise PlaylistExtractionError(f"Failed to extract playlist videos: {e}") from e


def get_newest_existing_date(existing_files: list[str]) -> str:
    """Get the newest upload date from existing video files.

    Args:
        existing_files: List of file paths containing YouTube URLs

    Returns:
        Newest upload date in YYYYMMDD format, or empty string if none found
    """
    logger.info("Finding newest date from existing video collections...")

    newest_date = ""
    total_videos = 0

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }

    # Add cookies if available
    cookies_path = get_cookies_path()
    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

    for file_path in existing_files:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        logger.info(f"Checking dates in: {file_path}")

        with open(file_path) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not urls:
            continue

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                for url in urls:
                    try:
                        info = ydl.extract_info(url, download=False)
                        upload_date = info.get("upload_date", "")
                        if upload_date and upload_date > newest_date:
                            newest_date = upload_date
                            total_videos += 1
                    except Exception as e:
                        logger.debug(f"Could not get date for {url}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue

    if newest_date:
        logger.info(f"Newest existing episode: {newest_date} (checked {total_videos} videos)")
    else:
        logger.warning("No existing dates found")

    return newest_date


def get_downloaded_video_info(local_media_index_path: str = "data/local_media/index.json") -> tuple[set[str], Optional[str]]:
    """Get downloaded video IDs and the oldest download date.

    Args:
        local_media_index_path: Path to the local media index JSON file.

    Returns:
        Tuple of (downloaded_video_ids, oldest_upload_date).
        oldest_upload_date is None if no downloads found.
    """
    if not os.path.exists(local_media_index_path):
        logger.info(f"Local media index not found at: {local_media_index_path}")
        return set(), None

    try:
        with open(local_media_index_path) as f:
            index_data = json.load(f)

        downloaded_ids = set(index_data.keys())
        oldest_date = None

        # Find the oldest upload date among downloaded videos
        for video_info in index_data.values():
            upload_date = video_info.get("upload_date")
            if upload_date:
                if oldest_date is None or upload_date < oldest_date:
                    oldest_date = upload_date

        logger.info(f"Found {len(downloaded_ids)} already downloaded videos in local media")
        if oldest_date:
            logger.info(f"Oldest downloaded episode: {oldest_date}")

        return downloaded_ids, oldest_date

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to read local media index: {e}")
        return set(), None


def filter_episodes(
    videos: list[tuple[str, float, str, str]],
    exclude_downloaded: bool = True,
    downloaded_video_ids: Optional[set[str]] = None,
    after_date: Optional[str] = None,
) -> list[tuple[str, float, str, str]]:
    """Filter playlist videos to exclude already downloaded ones and old episodes.

    Args:
        videos: List of (title, duration_seconds, video_id, upload_date).
        exclude_downloaded: Whether to exclude already downloaded videos.
        downloaded_video_ids: Set of video IDs that are already downloaded.
        after_date: Only include videos uploaded after this date (YYYYMMDD format).

    Returns:
        Filtered list of episodes.
    """
    filtered_episodes = []

    if exclude_downloaded and downloaded_video_ids:
        logger.info(f"Excluding {len(downloaded_video_ids)} already downloaded videos")

    if after_date:
        logger.info(f"Only including episodes uploaded after {after_date}")

    for title, duration, video_id, upload_date in videos:
        # Check if already downloaded
        if exclude_downloaded and downloaded_video_ids and video_id in downloaded_video_ids:
            logger.debug(f"Skipping already downloaded video: {title} ({video_id})")
            continue

        # Check date filter if enabled
        if after_date and upload_date:
            try:
                if upload_date <= after_date:
                    logger.debug(f"Skipping old video: {title} (uploaded {upload_date}, need after {after_date})")
                    continue
            except (ValueError, TypeError):
                # If we can't parse the date, include it anyway
                logger.debug(f"Could not parse upload date for: {title}")
        elif after_date and not upload_date:
            # If we have a date filter but no upload date, skip to be safe
            logger.debug(f"Skipping video with no upload date: {title}")
            continue

        # Include this episode
        filtered_episodes.append((title, duration, video_id, upload_date))
        logger.debug(f"Found episode: {title} ({duration/60:.1f} min)")

    logger.info(f"Found {len(filtered_episodes)} episodes")
    return filtered_episodes


def save_video_links(
    videos: list[tuple[str, float, str, str]],
    output_file: str,
    include_metadata: bool = False,
) -> None:
    """Save video links to file, overwriting existing content.

    This creates a clean "todo list" of episodes that need to be downloaded.

    Args:
        videos: List of (title, duration_seconds, video_id, upload_date).
        output_file: Output file path.
        include_metadata: Whether to include metadata comments.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        if include_metadata and videos:
            f.write(f"# TBPN Episodes to Download - Generated {datetime.now().isoformat()}\n")
            f.write(f"# Total new episodes: {len(videos)}\n")
            f.write(f"# Duration range: {min(v[1]/60 for v in videos):.1f} - {max(v[1]/60 for v in videos):.1f} minutes\n")
            f.write("#\n")

        for title, duration, video_id, upload_date in videos:
            url = f"https://www.youtube.com/watch?v={video_id}"

            if include_metadata:
                f.write(f"# {title} ({duration/60:.1f} min) - {upload_date}\n")

            f.write(f"{url}\n")

    logger.info(f"Saved {len(videos)} new episode links to: {output_file}")


def main():
    """Extract TBPN video links from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract full-episode video links from TBPN YouTube channel"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/tbpn_ytlinks/tbpn_youtube_links.txt",
        help="Output file path (default: data/tbpn_ytlinks/tbpn_youtube_links.txt)"
    )

    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata comments in output file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without saving"
    )

    parser.add_argument(
        "--exclude-downloaded",
        action="store_true",
        default=True,
        help="Exclude videos that are already downloaded in local media (default: True)"
    )
    parser.add_argument(
        "--include-downloaded",
        action="store_true",
        help="Include videos even if they're already downloaded (overrides --exclude-downloaded)"
    )
    parser.add_argument(
        "--local-media-index",
        type=str,
        default="data/local_media/index.json",
        help="Path to local media index file (default: data/local_media/index.json)"
    )

    args = parser.parse_args()

    # Handle downloaded video filtering logic
    if args.include_downloaded:
        args.exclude_downloaded = False

    # Get downloaded video info (IDs and oldest date)
    downloaded_video_ids = set()
    oldest_downloaded_date = None
    if args.exclude_downloaded:
        downloaded_video_ids, oldest_downloaded_date = get_downloaded_video_info(args.local_media_index)

    try:
        # Extract all videos from TBPN playlist
        logger.info("Starting TBPN playlist extraction...")
        all_videos = extract_playlist_videos()

        # Filter episodes
        episodes = filter_episodes(
            all_videos,
            args.exclude_downloaded,
            downloaded_video_ids,
            oldest_downloaded_date,
        )

        if not episodes:
            # Create empty file to indicate no episodes to download
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("# No new episodes to download - collection is up to date\n")
            logger.info("No new episodes found. Your collection is up to date!")
            return

        # Sort by duration (longest first) to prioritize main episodes
        episodes.sort(key=lambda x: x[1], reverse=True)

        if args.dry_run:
            logger.info(f"\nWould extract {len(episodes)} episodes:")
            for title, duration, _video_id, _upload_date in episodes[:10]:  # Show first 10
                logger.info(f"  {title} ({duration/60:.1f} min)")
            if len(episodes) > 10:
                logger.info(f"  ... and {len(episodes) - 10} more")
        else:
            # Save to file (overwriting existing content)
            save_video_links(episodes, args.output, args.include_metadata)

            # Show summary
            logger.info("\nExtraction complete!")
            logger.info(f"New episodes to download: {len(episodes)}")
            logger.info(f"Duration range: {min(v[1]/60 for v in episodes):.1f} - {max(v[1]/60 for v in episodes):.1f} minutes")
            logger.info(f"Download list saved to: {args.output}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
