#!/bin/bash
# Quick script to check for new TBPN episodes and update your collection
# Usage: ./scripts/update_tbpn_links.sh

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Update the download list with new episodes only
echo "Updating TBPN download list..."

python scripts/extract_tbpn_links.py --exclude-downloaded

# Count non-comment lines (actual URLs)
episode_count=$(grep -v '^#' data/tbpn_ytlinks/tbpn_youtube_links.txt 2>/dev/null | wc -l | tr -d ' ')

if [ "$episode_count" -gt 0 ]; then
    echo ""
    echo "✓ Found $episode_count new episode(s) ready for download!"
    echo "Download list updated: data/tbpn_ytlinks/tbpn_youtube_links.txt"
    echo ""
    echo "Next steps:"
    echo "1. Run your download script on the updated file"
    echo "2. Run this script again after downloading to refresh the list"
else
    echo ""
    echo "✓ Your collection is up to date - no new episodes found."
    echo "The download list is empty."
fi
