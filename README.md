# TBPN Size Gong Analysis

YAMNet-based audio event detection system for identifying gong sounds in audio files.

## Quick Start

```bash
# Create and activate a local virtual environment (Python 3.12 recommended)
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies inside the venv
pip install -r requirements.txt

# Run detection pipeline
python -m src.gong_detector.core.pipeline.detection_pipeline
```

### Local media cache (optional)

- Cache directory: `data/local_media/`
  - `raw/` — Original audio files for LUFS analysis (preserves source format)
  - `preprocessed/` — 16kHz mono WAVs named `VIDEOID_16k_mono.wav`
  - `index.json` — metadata (title, upload_date, raw_path, preprocessed_path, timestamps)

- Recommended bulk usage:
  - Prefer local, fallback to download:
    ```bash
    PYTHONPATH=src python -m gong_detector.core.pipeline.bulk_processor --use_local_media --version_one --csv
    ```
  - Strict offline (error if missing locally):
    ```bash
    PYTHONPATH=src python -m gong_detector.core.pipeline.bulk_processor --local_only
    ```

The interactive menu (`src/gong_detector/run_gong_detector.py`) includes toggles for “Use local media” and “Local only” in both Single Video and Bulk modes.

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/
ruff check src/gong_detector/ tests/
```

## Project Structure

```
src/gong_detector/     # Main package
├── core/             # Core functionality
├── training/         # Training data
tests/                # Test suite
data/                 # Data files
  └── local_media/    # Optional dual-cache for raw and preprocessed audio
      ├── raw/        # Original audio for LUFS analysis
      ├── preprocessed/
      └── index.json
config/               # Configuration
```
