# TBPN Size Gong Analysis

YAMNet-based audio event detection system for identifying gong sounds in audio files.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run detection pipeline
python -m src.gong_detector.core.pipeline.detection_pipeline
```

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
config/               # Configuration
```

## License

MIT License
