#!/bin/bash
# Quick script to check for new TBPN episodes and update your collection
# Usage: ./scripts/update_tbpn_links.sh

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Check for new episodes
echo "Checking for new TBPN episodes..."
python scripts/extract_tbpn_links.py --newer-than-existing --output data/tbpn_ytlinks/tbpn_new_episodes.txt

# Check if any new episodes were found
if [ -f "data/tbpn_ytlinks/tbpn_new_episodes.txt" ] && [ -s "data/tbpn_ytlinks/tbpn_new_episodes.txt" ]; then
    echo ""
    echo "New episodes found! Saved to: data/tbpn_ytlinks/tbpn_new_episodes.txt"
    echo "Episode count: $(wc -l < data/tbpn_ytlinks/tbpn_new_episodes.txt)"
    echo ""
    echo "To add them to your main collection:"
    echo "  cat data/tbpn_ytlinks/tbpn_new_episodes.txt >> data/tbpn_ytlinks/tbpn_youtube_links.txt"
else
    echo "Your collection is up to date - no new episodes found."
fi
