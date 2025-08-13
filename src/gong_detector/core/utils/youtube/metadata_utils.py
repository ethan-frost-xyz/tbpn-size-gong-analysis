"""YouTube metadata and URL parsing utilities.

This module provides functions for extracting video IDs from URLs,
formatting video titles for filesystem use, and creating folder names.
"""

import re
from datetime import datetime


def video_id_from_url(url: str) -> str:
    """Extract YouTube video ID from a URL.

    Args:
        url: YouTube URL to extract video ID from

    Returns:
        YouTube video ID or empty string if not found
    """
    # youtu.be/<id>
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    # youtube.com/watch?v=<id>
    match = re.search(r"v=([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    # youtube.com/embed/<id>
    match = re.search(r"/embed/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    return ""


def create_folder_name_from_title(video_title: str) -> str:
    """Create folder name from video title date information.

    Args:
        video_title: Video title from YouTube (e.g., "TBPN | Monday, July 7th")

    Returns:
        Folder name in format tbpn_monday_july_7th
    """
    if not video_title:
        return "tbpn_unknown_date"

    try:
        # Look for patterns like "TBPN | Monday, July 7th" or "Monday, July 7th"
        # Pattern to match day, month, and day number
        # Matches: "Monday, July 7th", "Tuesday, August 15th", etc.
        pattern = r"(\w+),\s+(\w+)\s+(\d+)(?:st|nd|rd|th)?"
        match = re.search(pattern, video_title)

        if match:
            day_name = match.group(1).lower()  # monday, tuesday, etc.
            month_name = match.group(2).lower()  # july, august, etc.
            day_num = int(match.group(3))

            # Get ordinal suffix for day
            if 10 <= day_num <= 20:  # Special case for 11th, 12th, 13th
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_num % 10, "th")

            return f"tbpn_{day_name}_{month_name}_{day_num}{suffix}"

        return "tbpn_unknown_date"

    except (ValueError, IndexError):
        return "tbpn_unknown_date"


def create_folder_name_from_date(upload_date: str) -> str:
    """Create folder name from YouTube upload date in format tbpn_dayname_month_dayordinal.

    Args:
        upload_date: Upload date from YouTube (format: YYYYMMDD)

    Returns:
        Folder name in format tbpn_monday_june_23rd
    """
    if not upload_date or len(upload_date) != 8:
        return "tbpn_unknown_date"

    try:
        year = int(upload_date[:4])
        month = int(upload_date[4:6])
        day = int(upload_date[6:8])

        # Create datetime object
        date_obj = datetime(year, month, day)

        # Get day name and month name
        day_name = date_obj.strftime("%A").lower()  # monday, tuesday, etc.
        month_name = date_obj.strftime("%B").lower()  # january, february, etc.

        # Get ordinal suffix for day
        if 10 <= day <= 20:  # Special case for 11th, 12th, 13th
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

        return f"tbpn_{day_name}_{month_name}_{day}{suffix}"

    except (ValueError, IndexError):
        return "tbpn_unknown_date"


def sanitize_title_for_folder(title: str) -> str:
    """Convert video title to safe folder name.

    Args:
        title: Video title from YouTube

    Returns:
        Sanitized folder name safe for filesystem
    """
    # Convert to lowercase
    sanitized = title.lower()
    # Remove commas
    sanitized = sanitized.replace(",", "")
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", sanitized)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized
