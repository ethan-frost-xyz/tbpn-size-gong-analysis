#!/usr/bin/env python3
"""Update local media index to point to WAV files instead of WebM files.

This script updates the index.json file to reference the new WAV files
instead of the old WebM files after migration.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def update_index_for_wav() -> None:
    """Update local media index to point to WAV files."""
    try:
        # Import here to avoid path issues
        import sys
        sys.path.insert(0, 'src')
        from gong_detector.core.utils.local_media import LocalMediaIndex
        
        # Load the index
        index = LocalMediaIndex()
        
        if not hasattr(index, '_index') or not index._index:
            logger.info("No index found or index is empty")
            return
        
        updated_count = 0
        
        # Update each entry
        for video_id, entry in index._index.items():
            raw_path = entry.get('raw_path', '')
            
            if raw_path and raw_path.endswith('.webm'):
                # Convert WebM path to WAV path
                wav_path = raw_path.replace('.webm', '.wav')
                
                # Check if WAV file exists
                if Path(wav_path).exists():
                    logger.info(f"Updating {video_id}: {Path(raw_path).name} -> {Path(wav_path).name}")
                    
                    # Update the entry
                    entry['raw_path'] = wav_path
                    updated_count += 1
                else:
                    logger.warning(f"WAV file not found for {video_id}: {wav_path}")
            else:
                logger.debug(f"Skipping {video_id}: already WAV or no raw path")
        
        if updated_count > 0:
            # Save the updated index
            index.save()
            logger.info(f"✓ Updated {updated_count} entries in index")
            logger.info("✓ Index saved successfully")
        else:
            logger.info("No updates needed - all entries already point to WAV files")
            
    except Exception as e:
        logger.error(f"Failed to update index: {e}")
        raise


if __name__ == "__main__":
    try:
        update_index_for_wav()
    except Exception as e:
        logger.error(f"Index update failed: {e}")
        exit(1)
