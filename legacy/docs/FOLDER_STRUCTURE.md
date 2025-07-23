# Legacy Code Organization

This directory contains archived code, examples, and documentation from the
TBPN gong detection project that are no longer part of the core functionality
but may be useful for reference.

## Directory Structure

```text
legacy/
├── analysis/
│   └── compare_spectrograms.py   - Spectrogram comparison tool for audio analysis
├── examples/
│   ├── create_test_audio.py      - Test audio file generation
│   ├── example_usage.py          - Example usage of core functionality
│   └── extract_youtube_segment.py - YouTube audio extraction utility
├── docs/
│   ├── FOLDER_STRUCTURE.md       - This file
│   └── README_spectrogram_comparison.md - Documentation for spectrogram comparison
└── __init__.py                   - Package initialization
```

## Component Descriptions

### Analysis Tools

- `compare_spectrograms.py`: Visual comparison tool for analyzing gong audio
  characteristics against reference samples

### Example Scripts

- `create_test_audio.py`: Generates test audio files for development and testing
- `example_usage.py`: Demonstrates usage of core gong detection functionality
- `extract_youtube_segment.py`: Utility for extracting audio segments from YouTube
  videos

### Documentation

- `FOLDER_STRUCTURE.md`: Explains the organization of legacy code
- `README_spectrogram_comparison.md`: Guide for using the spectrogram comparison
  tool

## Note

These components were used during development but are not required for the core
gong detection functionality. They are preserved for reference and potential future
use in similar audio analysis tasks.
