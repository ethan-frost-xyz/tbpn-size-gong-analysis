# Folder Structure

## Organized Output Folders

### `temp_audio/`
- **Purpose**: Temporary WAV files during processing
- **Contents**: Downloaded and converted audio files
- **Cleanup**: Files are automatically removed after processing
- **Auto-cleanup**: Old files (>24 hours) are cleaned up automatically

### `csv_results/`
- **Purpose**: All detection results and analysis outputs
- **Contents**: CSV files with gong detection timestamps and confidence scores
- **Naming**: Files are saved with descriptive names
- **Format**: `timestamp_seconds,confidence` columns

## Usage

```bash
# Run detection - files automatically go to organized folders
python core/detect_from_youtube.py "YOUR_URL" --save_csv my_results

# Results will be saved to:
# - temp_audio/temp_youtube_audio_*.wav (temporary, auto-cleaned)
# - csv_results/my_results.csv (permanent results)
```

## Benefits

- ✅ **Clean workspace**: No more scattered temp files
- ✅ **Organized results**: All CSVs in one place
- ✅ **Auto-cleanup**: Prevents disk space issues
- ✅ **Simple workflow**: Just specify CSV name, folders handle the rest 