#!/usr/bin/env python3
"""Debug script to isolate bulk download issues.

This script helps identify what's causing bulk downloads to fail after the first successful download.
"""

import sys
import time
import logging
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gong_detector.core.utils.youtube.downloader import download_youtube_audio
from gong_detector.core.utils.youtube.metadata_utils import video_id_from_url

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_download(url: str, index: int) -> bool:
    """Test downloading a single YouTube URL."""
    logger.info(f"=== Testing Download {index} ===")
    logger.info(f"URL: {url}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = f"{temp_dir}/temp_audio.%(ext)s"

            start_time = time.time()
            downloaded_file, video_title, upload_date = download_youtube_audio(
                url=url,
                output_template=output_template
            )
            end_time = time.time()

            logger.info("✓ Download successful!")
            logger.info(f"  Title: {video_title}")
            logger.info(f"  Upload Date: {upload_date}")
            logger.info(f"  File: {downloaded_file}")
            logger.info(f"  Download time: {end_time - start_time:.2f} seconds")
            logger.info(f"  File size: {Path(downloaded_file).stat().st_size / (1024*1024):.2f} MB")

            return True

    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False

def test_memory_usage():
    """Test current memory usage."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info("=== Memory Status ===")
        logger.info(f"  Total: {memory.total / (1024**3):.1f} GB")
        logger.info(f"  Available: {memory.available / (1024**3):.1f} GB")
        logger.info(f"  Used: {memory.percent:.1f}%")
        logger.info(f"  Used (GB): {memory.used / (1024**3):.1f} GB")
        return memory.available / (1024**3) > 2.0  # At least 2GB available
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return True

def test_cookie_access():
    """Test if cookies are accessible."""
    try:
        from gong_detector.core.utils.youtube.downloader import get_cookies_path
        cookie_path = get_cookies_path()
        if cookie_path:
            logger.info(f"✓ Cookies found at: {cookie_path}")
            cookie_size = Path(cookie_path).stat().st_size
            logger.info(f"  Cookie file size: {cookie_size} bytes")
            return True
        else:
            logger.warning("No cookies file found")
            return False
    except Exception as e:
        logger.error(f"Error checking cookies: {e}")
        return False

def main():
    """Run bulk download diagnostics."""
    logger.info("=== Bulk Download Diagnostics ===")

    # Check memory
    memory_ok = test_memory_usage()
    if not memory_ok:
        logger.error("Insufficient memory - aborting")
        return

    # Check cookies
    cookies_ok = test_cookie_access()

    # Get URLs to test (first few from the links file)
    try:
        links_file = Path("data/tbpn_ytlinks/tbpn_youtube_links.txt")
        if not links_file.exists():
            logger.error(f"Links file not found: {links_file}")
            return

        urls = []
        with open(links_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and ('youtube.com' in line or 'youtu.be' in line):
                    urls.append(line)
                    if len(urls) >= 5:  # Test first 5 URLs
                        break

        logger.info(f"Found {len(urls)} URLs to test")

        # Test each URL with increasing delay
        results = []
        for i, url in enumerate(urls, 1):
            # Test memory before each download
            memory_ok = test_memory_usage()
            if not memory_ok:
                logger.error(f"Insufficient memory before download {i} - stopping")
                break

            success = test_single_download(url, i)
            results.append(success)

            if i < len(urls):  # Don't sleep after last download
                delay = 5 + (i * 2)  # Increasing delay: 7s, 9s, 11s...
                logger.info(f"Waiting {delay} seconds before next download...")
                time.sleep(delay)

        # Summary
        logger.info("\n=== Results Summary ===")
        successful = sum(results)
        total = len(results)
        logger.info(f"Successful downloads: {successful}/{total}")

        if successful == 0:
            logger.error("All downloads failed!")
        elif successful < total:
            logger.warning(f"Some downloads failed after {successful} successful ones")
            logger.info("This suggests an issue with subsequent downloads")
        else:
            logger.info("All downloads successful!")

        # Check for patterns
        if results and not results[0]:
            logger.error("First download failed - check URL or network")
        elif results and results[0] and any(not r for r in results[1:]):
            logger.info("Pattern: First download OK, subsequent downloads failed")
            logger.info("Possible causes:")
            logger.info("  - YouTube rate limiting")
            logger.info("  - Cookie/session issues")
            logger.info("  - Memory accumulation")
            logger.info("  - Temporary file conflicts")

    except Exception as e:
        logger.error(f"Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
