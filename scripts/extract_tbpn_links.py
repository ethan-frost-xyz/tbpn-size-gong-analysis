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
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yt_dlp  # type: ignore

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gong_detector.core.utils.youtube.downloader import get_cookies_path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TBPN Channel URLs - try multiple sources for live episodes
TBPN_CHANNEL_URLS = [
    "https://www.youtube.com/channel/UC-DRzaGnL_vtBUpCFH5M0tg/videos",  # All videos
    "https://www.youtube.com/channel/UC-DRzaGnL_vtBUpCFH5M0tg/streams", # Live streams
    "https://www.youtube.com/@TBPNLive/streams",  # Alternative live streams URL
]

# Default filtering criteria for live episodes
DEFAULT_MIN_DURATION = 60  # minutes
FULL_EPISODE_KEYWORDS = [
    "weekly recap",
    "weekend special",
    "full analysis",
    "we interviewed",
    "special episode",
]

# Days of the week for live episodes (Monday = 0, Sunday = 6)
WEEKDAY_NUMBERS = [0, 1, 2, 3, 4]  # Monday through Friday


def extract_channel_videos(channel_urls: list[str] | None = None) -> list[tuple[str, float, str, str]]:
    """Extract all video metadata from TBPN YouTube channel sources.

    Args:
        channel_urls: List of YouTube channel URLs to try (defaults to TBPN_CHANNEL_URLS)

    Returns:
        List of tuples: (title, duration_seconds, video_id, upload_date)

    Raises:
        RuntimeError: If extraction fails from all sources
    """
    if channel_urls is None:
        channel_urls = TBPN_CHANNEL_URLS

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

    all_videos = []
    seen_video_ids = set()

    for channel_url in channel_urls:
        logger.info(f"Extracting videos from: {channel_url}")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract playlist info (channel videos)
                info = ydl.extract_info(channel_url, download=False)

                if not info or "entries" not in info:
                    logger.warning(f"No videos found in: {channel_url}")
                    continue

                videos_from_source = 0
                for entry in info["entries"]:
                    if entry:  # Skip None entries
                        title = entry.get("title", "")
                        duration = entry.get("duration", 0)
                        video_id = entry.get("id", "")
                        upload_date = entry.get("upload_date", "")

                        if title and duration and video_id and video_id not in seen_video_ids:
                            all_videos.append((title, float(duration), video_id, upload_date))
                            seen_video_ids.add(video_id)
                            videos_from_source += 1

                logger.info(f"Found {videos_from_source} unique videos from this source")

        except Exception as e:
            logger.warning(f"Failed to extract from {channel_url}: {e}")
            continue

    if not all_videos:
        raise RuntimeError("Failed to extract videos from any source")

    logger.info(f"Total unique videos extracted: {len(all_videos)}")
    return all_videos


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


def filter_full_episodes(
    videos: list[tuple[str, float, str, str]],
    min_duration_minutes: int = DEFAULT_MIN_DURATION,
    weekdays_only: bool = True,
    after_date: str | None = None
) -> list[tuple[str, float, str, str]]:
    """Filter videos to identify full live episodes.

    Args:
        videos: List of (title, duration_seconds, video_id, upload_date)
        min_duration_minutes: Minimum duration in minutes for full episodes
        weekdays_only: Only include videos that were likely live on weekdays
        after_date: Only include videos uploaded after this date (YYYYMMDD format)

    Returns:
        Filtered list of full episode videos
    """
    from datetime import datetime

    min_duration_seconds = min_duration_minutes * 60
    full_episodes = []

    logger.info(f"Filtering for episodes longer than {min_duration_minutes} minutes")
    if weekdays_only:
        logger.info("Filtering for weekday live episodes only")
    if after_date:
        logger.info(f"Filtering for episodes uploaded after {after_date}")

    for title, duration, video_id, upload_date in videos:
        # Check duration threshold
        if duration < min_duration_seconds:
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

        # Check weekday filter if enabled
        if weekdays_only and upload_date:
            try:
                # Parse upload date (format: YYYYMMDD)
                upload_dt = datetime.strptime(upload_date, "%Y%m%d")
                weekday = upload_dt.weekday()  # Monday = 0, Sunday = 6

                # Skip weekend uploads (Saturday=5, Sunday=6)
                if weekday not in WEEKDAY_NUMBERS:
                    logger.debug(f"Skipping weekend video: {title} (uploaded on {upload_dt.strftime('%A')})")
                    continue
            except (ValueError, TypeError):
                # If we can't parse the date, include it anyway
                logger.debug(f"Could not parse upload date for: {title}")

        # Check for full episode keywords in title
        title_lower = title.lower()
        is_full_episode = any(keyword in title_lower for keyword in FULL_EPISODE_KEYWORDS)

        # For live episodes, be more selective - require either keywords OR very long duration
        if is_full_episode or duration >= (min_duration_minutes * 1.5 * 60):  # 1.5x threshold for non-keyword matches
            full_episodes.append((title, duration, video_id, upload_date))
            weekday_str = ""
            if upload_date:
                try:
                    upload_dt = datetime.strptime(upload_date, "%Y%m%d")
                    weekday_str = f" ({upload_dt.strftime('%A')})"
                except (ValueError, TypeError):
                    pass
            logger.debug(f"Found full episode: {title} ({duration/60:.1f} min){weekday_str}")

    logger.info(f"Found {len(full_episodes)} full episodes")
    return full_episodes


def save_video_links(
    videos: list[tuple[str, float, str, str]],
    output_file: str,
    include_metadata: bool = False
) -> None:
    """Save video links to file.

    Args:
        videos: List of (title, duration_seconds, video_id, upload_date)
        output_file: Output file path
        include_metadata: Whether to include metadata comments
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        if include_metadata:
            f.write(f"# TBPN Full Episodes - Generated {datetime.now().isoformat()}\n")
            f.write(f"# Total episodes: {len(videos)}\n")
            f.write(f"# Duration range: {min(v[1]/60 for v in videos):.1f} - {max(v[1]/60 for v in videos):.1f} minutes\n")
            f.write("#\n")

        for title, duration, video_id, upload_date in videos:
            url = f"https://www.youtube.com/watch?v={video_id}"

            if include_metadata:
                f.write(f"# {title} ({duration/60:.1f} min) - {upload_date}\n")

            f.write(f"{url}\n")

    logger.info(f"Saved {len(videos)} video links to: {output_file}")


def main():
    """Extract TBPN video links from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract full-episode video links from TBPN YouTube channel"
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=DEFAULT_MIN_DURATION,
        help=f"Minimum duration in minutes for full episodes (default: {DEFAULT_MIN_DURATION})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tbpn_ytlinks/tbpn_full_episodes_auto.txt",
        help="Output file path (default: data/tbpn_ytlinks/tbpn_full_episodes_auto.txt)"
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
        "--weekdays-only",
        action="store_true",
        default=True,
        help="Only include videos uploaded on weekdays (default: True)"
    )
    parser.add_argument(
        "--include-weekends",
        action="store_true",
        help="Include weekend videos (overrides --weekdays-only)"
    )
    parser.add_argument(
        "--after-date",
        type=str,
        help="Only extract videos uploaded after this date (YYYYMMDD format)"
    )
    parser.add_argument(
        "--newer-than-existing",
        action="store_true",
        help="Automatically find newest date from existing files and only extract newer videos"
    )
    parser.add_argument(
        "--existing-files",
        nargs="*",
        default=["data/tbpn_ytlinks/tbpn_youtube_links.txt"],
        help="Existing video files to check for newest date (default: tbpn_youtube_links.txt)"
    )

    args = parser.parse_args()

    # Handle weekday filtering logic
    if args.include_weekends:
        args.weekdays_only = False

    # Handle date filtering logic
    after_date = None
    if args.newer_than_existing:
        after_date = get_newest_existing_date(args.existing_files)
        if not after_date:
            logger.warning("Could not determine newest existing date, extracting all videos")
    elif args.after_date:
        after_date = args.after_date
        logger.info(f"Using manually specified date filter: {after_date}")

    try:
        # Extract all videos from TBPN channel
        logger.info("Starting TBPN channel extraction...")
        all_videos = extract_channel_videos()

        # Filter for full episodes
        full_episodes = filter_full_episodes(all_videos, args.min_duration, args.weekdays_only, after_date)

        if not full_episodes:
            if after_date:
                logger.info(f"No new episodes found after {after_date}. Your collection is up to date!")
            else:
                logger.warning("No full episodes found matching criteria")
            return

        # Sort by duration (longest first) to prioritize main episodes
        full_episodes.sort(key=lambda x: x[1], reverse=True)

        if args.dry_run:
            logger.info(f"\nWould extract {len(full_episodes)} full episodes:")
            for title, duration, _video_id, _upload_date in full_episodes[:10]:  # Show first 10
                logger.info(f"  {title} ({duration/60:.1f} min)")
            if len(full_episodes) > 10:
                logger.info(f"  ... and {len(full_episodes) - 10} more")
        else:
            # Save to file
            save_video_links(full_episodes, args.output, args.include_metadata)

            # Show summary
            logger.info("\nExtraction complete!")
            logger.info(f"Total full episodes found: {len(full_episodes)}")
            logger.info(f"Duration range: {min(v[1]/60 for v in full_episodes):.1f} - {max(v[1]/60 for v in full_episodes):.1f} minutes")
            logger.info(f"Output saved to: {args.output}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
