#!/bin/bash
# Quick script to check for new TBPN episodes and update your collection
# Usage: ./scripts/update_tbpn_links.sh

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Check for new episodes and append directly to main file
echo "Checking for new TBPN episodes..."
initial_count=$(wc -l < data/tbpn_ytlinks/tbpn_youtube_links.txt 2>/dev/null || echo "0")

python scripts/extract_tbpn_links.py --exclude-downloaded --append

final_count=$(wc -l < data/tbpn_ytlinks/tbpn_youtube_links.txt 2>/dev/null || echo "0")
new_episodes=$((final_count - initial_count))

if [ $new_episodes -gt 0 ]; then
    echo ""
    echo "✓ Found and added $new_episodes new episode(s) to your collection!"
    echo "Total episodes: $final_count"
else
    echo "Your collection is up to date - no new episodes found."
fi
