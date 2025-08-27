#!/usr/bin/env python3
"""TBPN YouTube Channel Link Extractor.

Extracts full-episode video links from the TBPN YouTube channel using yt-dlp.
Filters for long-form content (live streams) and excludes short clips.

Usage:
    # Simple update (recommended for daily use)
    python scripts/extract_tbpn_links.py --update

    # Advanced CLI with options
    python scripts/extract_tbpn_links.py [--options...]

    # Dry run to see what would be extracted
    python scripts/extract_tbpn_links.py --dry-run

Features:
- Uses existing yt-dlp dependency
- Leverages existing cookie system for bot detection bypass
- Parses dates from episode titles (yt-dlp doesn't provide them)
- Filters episodes newer than your latest downloaded episode
- Outputs in same format as existing tbpn_youtube_links.txt files
- Simple --update mode for daily use
"""

import argparse
import json
import logging
import os
import re
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
TBPN_PLAYLIST_URL = (
    "https://www.youtube.com/playlist?list=PLBV_0ax_G8bowGFK97Nrv0DBmxaDObKgA"
)

# No filtering needed since playlist is already curated


def parse_date_from_title(title: str) -> str:
    """Parse upload date from TBPN episode title.

    Args:
        title: Episode title containing date information

    Returns:
        Date in YYYYMMDD format, or empty string if no date found
    """
    # Month name to number mapping
    months = {
        "January": "01",
        "February": "02",
        "March": "03",
        "April": "04",
        "May": "05",
        "June": "06",
        "July": "07",
        "August": "08",
        "September": "09",
        "October": "10",
        "November": "11",
        "December": "12",
    }

    # Pattern to match: "Day, Month Day(th/st/nd/rd)" or "Day, Month Day(th/st/nd/rd) | ..."
    # Examples: "Tuesday, August 26th", "David Senra LIVE in The Ultradome | Thursday, August 21st"
    pattern = r"(\w+),\s+(\w+)\s+(\d+)(?:st|nd|rd|th)?"
    match = re.search(pattern, title)

    if match:
        day_name, month_name, day = match.groups()
        if month_name in months:
            day_padded = day.zfill(2)
            # Assume 2025 for recent episodes
            return f"2025{months[month_name]}{day_padded}"

    return ""


def extract_playlist_videos(
    playlist_url: str = TBPN_PLAYLIST_URL,
) -> list[tuple[str, float, str, str]]:
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
        "extract_flat": True,  # Get flat metadata for faster extraction
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
                    try:
                        title = entry.get("title", "")
                        duration = entry.get("duration", 0)
                        video_id = entry.get("id", "")
                        upload_date = entry.get("upload_date", "")

                        # If no upload_date, try to parse from title
                        if not upload_date:
                            upload_date = parse_date_from_title(title)

                        if title and duration and video_id:
                            videos.append(
                                (title, float(duration), video_id, upload_date)
                            )
                    except Exception as video_error:
                        # Skip problematic videos but continue with others
                        logger.debug(f"Skipping video due to error: {video_error}")
                        continue

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
            urls = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

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
        logger.info(
            f"Newest existing episode: {newest_date} (checked {total_videos} videos)"
        )
    else:
        logger.warning("No existing dates found")

    return newest_date


def get_downloaded_video_info(
    local_media_index_path: str = "data/local_media/index.json",
) -> tuple[set[str], Optional[str]]:
    """Get downloaded video IDs and the newest download date.

    Args:
        local_media_index_path: Path to the local media index JSON file.

    Returns:
        Tuple of (downloaded_video_ids, newest_upload_date).
        newest_upload_date is None if no downloads found.
    """
    if not os.path.exists(local_media_index_path):
        logger.info(f"Local media index not found at: {local_media_index_path}")
        return set(), None

    try:
        with open(local_media_index_path) as f:
            index_data = json.load(f)

        downloaded_ids = set(index_data.keys())
        newest_date = None

        # Find the newest upload date among downloaded videos
        for video_info in index_data.values():
            upload_date = video_info.get("upload_date")
            if upload_date:
                if newest_date is None or upload_date > newest_date:
                    newest_date = upload_date

        logger.info(
            f"Found {len(downloaded_ids)} already downloaded videos in local media"
        )
        if newest_date:
            logger.info(f"Newest downloaded episode: {newest_date}")

        return downloaded_ids, newest_date

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to read local media index: {e}")
        return set(), None


def filter_episodes(
    videos: list[tuple[str, float, str, str]],
    exclude_downloaded: bool = True,
    downloaded_video_ids: Optional[set[str]] = None,
    after_date: Optional[str] = None,
) -> list[tuple[str, float, str, str]]:
    """Filter playlist videos to exclude already downloaded ones and episodes before cutoff date.

    Args:
        videos: List of (title, duration_seconds, video_id, upload_date).
        exclude_downloaded: Whether to exclude already downloaded videos.
        downloaded_video_ids: Set of video IDs that are already downloaded.
        after_date: Only include videos uploaded after this date (YYYYMMDD format).
                   This is typically the newest downloaded episode date.

    Returns:
        Filtered list of episodes.
    """
    filtered_episodes = []

    if exclude_downloaded and downloaded_video_ids:
        logger.info(f"Excluding {len(downloaded_video_ids)} already downloaded videos")

    if after_date:
        logger.info(
            f"Only including episodes uploaded after {after_date} (newer than newest downloaded)"
        )

    for title, duration, video_id, upload_date in videos:
        # Check if already downloaded
        if (
            exclude_downloaded
            and downloaded_video_ids
            and video_id in downloaded_video_ids
        ):
            logger.debug(f"Skipping already downloaded video: {title} ({video_id})")
            continue

        # Check date filter if enabled
        if after_date and upload_date:
            try:
                if upload_date <= after_date:
                    logger.debug(
                        f"Skipping old video: {title} (uploaded {upload_date}, need after {after_date})"
                    )
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
        logger.debug(f"Found episode: {title} ({duration / 60:.1f} min)")

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
            f.write(
                f"# TBPN Episodes to Download - Generated {datetime.now().isoformat()}\n"
            )
            f.write(f"# Total new episodes: {len(videos)}\n")
            f.write(
                f"# Duration range: {min(v[1] / 60 for v in videos):.1f} - {max(v[1] / 60 for v in videos):.1f} minutes\n"
            )
            f.write("#\n")

        for title, duration, video_id, upload_date in videos:
            url = f"https://www.youtube.com/watch?v={video_id}"

            if include_metadata:
                f.write(f"# {title} ({duration / 60:.1f} min) - {upload_date}\n")

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
        help="Output file path (default: data/tbpn_ytlinks/tbpn_youtube_links.txt)",
    )

    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata comments in output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without saving",
    )

    parser.add_argument(
        "--exclude-downloaded",
        action="store_true",
        default=True,
        help="Exclude videos that are already downloaded in local media (default: True)",
    )
    parser.add_argument(
        "--include-downloaded",
        action="store_true",
        help="Include videos even if they're already downloaded (overrides --exclude-downloaded)",
    )
    parser.add_argument(
        "--local-media-index",
        type=str,
        default="data/local_media/index.json",
        help="Path to local media index file (default: data/local_media/index.json)",
    )

    args = parser.parse_args()

    # Handle downloaded video filtering logic
    if args.include_downloaded:
        args.exclude_downloaded = False

    # Get downloaded video info (IDs and newest date)
    downloaded_video_ids = set()
    newest_downloaded_date = None
    if args.exclude_downloaded:
        downloaded_video_ids, newest_downloaded_date = get_downloaded_video_info(
            args.local_media_index
        )

    try:
        # Extract all videos from TBPN playlist
        logger.info("Starting TBPN playlist extraction...")
        all_videos = extract_playlist_videos()

        # Filter episodes
        episodes = filter_episodes(
            all_videos,
            args.exclude_downloaded,
            downloaded_video_ids,
            newest_downloaded_date,
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
            for title, duration, _video_id, _upload_date in episodes[
                :10
            ]:  # Show first 10
                logger.info(f"  {title} ({duration / 60:.1f} min)")
            if len(episodes) > 10:
                logger.info(f"  ... and {len(episodes) - 10} more")
        else:
            # Save to file (overwriting existing content)
            save_video_links(episodes, args.output, args.include_metadata)

            # Show summary
            logger.info("\nExtraction complete!")
            logger.info(f"New episodes to download: {len(episodes)}")
            logger.info(
                f"Duration range: {min(v[1] / 60 for v in episodes):.1f} - {max(v[1] / 60 for v in episodes):.1f} minutes"
            )
            logger.info(f"Download list saved to: {args.output}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


def update_download_list():
    """Convenience function to update the download list with user-friendly output."""
    import os

    # Run the extraction with default settings
    args = argparse.Namespace(
        output="data/tbpn_ytlinks/tbpn_youtube_links.txt",
        include_metadata=False,
        dry_run=False,
        exclude_downloaded=True,
        include_downloaded=False,
        local_media_index="data/local_media/index.json"
    )

    try:
        # Extract all videos from TBPN playlist
        logger.info("Starting TBPN playlist extraction...")
        all_videos = extract_playlist_videos()

        # Get downloaded video info
        downloaded_video_ids, newest_downloaded_date = get_downloaded_video_info(
            args.local_media_index
        )

        # Filter episodes
        episodes = filter_episodes(
            all_videos,
            args.exclude_downloaded,
            downloaded_video_ids,
            newest_downloaded_date,
        )

        if not episodes:
            # Create empty file to indicate no episodes to download
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("# No new episodes to download - collection is up to date\n")
        else:
            # Sort by duration (longest first) to prioritize main episodes
            episodes.sort(key=lambda x: x[1], reverse=True)
            # Save to file (overwriting existing content)
            save_video_links(episodes, args.output, args.include_metadata)

        # Count non-comment lines (actual URLs)
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                lines = f.readlines()
            episode_count = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))

            if episode_count > 0:
                print("\n✓ Found {} new episode(s) ready for download!".format(episode_count))
                print("Download list updated: {}".format(args.output))
                print("")
                print("Next steps:")
                print("1. Run your download script on the updated file")
                print("2. Run this script again after downloading to refresh the list")
            else:
                print("")
                print("✓ Your collection is up to date - no new episodes found.")
                print("The download list is empty.")
        else:
            logger.error("Output file not found: {}".format(args.output))
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to update download list: {e}")
        return False


def main():
    """Main entry point for the script."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--update":
        # Run the convenience update function
        success = update_download_list()
        sys.exit(0 if success else 1)
    else:
        # Run the normal extraction function
        _main()


def _main():
    """Extract TBPN video links from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract full-episode video links from TBPN YouTube channel",
        epilog="""
Examples:
  python scripts/extract_tbpn_links.py --update              # Simple update (recommended)
  python scripts/extract_tbpn_links.py --dry-run             # Preview what would be extracted
  python scripts/extract_tbpn_links.py --include-metadata    # Include episode details
        """
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tbpn_ytlinks/tbpn_youtube_links.txt",
        help="Output file path (default: data/tbpn_ytlinks/tbpn_youtube_links.txt)",
    )

    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata comments in output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without saving",
    )

    parser.add_argument(
        "--exclude-downloaded",
        action="store_true",
        default=True,
        help="Exclude videos that are already downloaded in local media (default: True)",
    )
    parser.add_argument(
        "--include-downloaded",
        action="store_true",
        help="Include videos even if they're already downloaded (overrides --exclude-downloaded)",
    )
    parser.add_argument(
        "--local-media-index",
        type=str,
        default="data/local_media/index.json",
        help="Path to local media index file (default: data/local_media/index.json)",
    )

    args = parser.parse_args()

    # Handle downloaded video filtering logic
    if args.include_downloaded:
        args.exclude_downloaded = False

    # Get downloaded video info (IDs and newest date)
    downloaded_video_ids = set()
    newest_downloaded_date = None
    if args.exclude_downloaded:
        downloaded_video_ids, newest_downloaded_date = get_downloaded_video_info(
            args.local_media_index
        )

    try:
        # Extract all videos from TBPN playlist
        logger.info("Starting TBPN playlist extraction...")
        all_videos = extract_playlist_videos()

        # Filter episodes
        episodes = filter_episodes(
            all_videos,
            args.exclude_downloaded,
            downloaded_video_ids,
            newest_downloaded_date,
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
            for title, duration, _video_id, _upload_date in episodes[
                :10
            ]:  # Show first 10
                logger.info(f"  {title} ({duration / 60:.1f} min)")
            if len(episodes) > 10:
                logger.info(f"  ... and {len(episodes) - 10} more")
        else:
            # Save to file (overwriting existing content)
            save_video_links(episodes, args.output, args.include_metadata)

            # Show summary
            logger.info("\nExtraction complete!")
            logger.info(f"New episodes to download: {len(episodes)}")
            logger.info(
                f"Duration range: {min(v[1] / 60 for v in episodes):.1f} - {max(v[1] / 60 for v in episodes):.1f} minutes"
            )
            logger.info(f"Download list saved to: {args.output}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
