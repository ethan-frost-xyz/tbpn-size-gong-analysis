# Spectrogram Comparison Script

This script compares the spectrograms of TBPN gong audio against reference gong clips to help analyze frequency characteristics, duration, and harmonic structure.

## Usage

```bash
python compare_spectrograms.py <tbpn_gong_path> <reference_gong_path>
```

### Example
```bash
python compare_spectrograms.py samples/tbpn_gong.wav samples/reference_gong.wav
```

## Features

- **Side-by-side spectrogram visualization** using Mel-scale frequency analysis
- **Audio statistics** including duration, peak amplitude, and RMS levels
- **Consistent scaling** for meaningful comparison
- **16kHz sample rate** for YAMNet compatibility

## Output

The script displays:
1. Two side-by-side Mel spectrograms (dB scale)
2. Audio statistics for both files
3. Visual comparison of:
   - Frequency concentration
   - Time duration
   - Harmonic structure
   - Reverb characteristics

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## File Organization

- `samples/` - Place your audio files here
- `test_results/` - Test outputs and analysis results
- `compare_spectrograms.py` - Main comparison script 