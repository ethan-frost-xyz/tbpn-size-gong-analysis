# Quick Start: Spectrogram Comparison

## ✅ **Script is Ready!**

The `compare_spectrograms.py` script has been created and tested successfully.

## 🚀 **How to Use:**

### 1. **With Test Audio (Ready Now):**

```bash
# Compare the generated test files
python compare_spectrograms.py samples/tbpn_gong.wav samples/reference_gong.wav
```

### 2. **With Your Real Audio Files:**

```bash
# Replace with your actual TBPN gong and reference files
python compare_spectrograms.py path/to/your_tbpn_gong.wav path/to/reference_gong.wav
```

## 📊 **What You Get:**

1. **Side-by-side Mel spectrograms** (dB scale)
2. **Audio statistics** (duration, peak amplitude)
3. **Saved plot** in `test_results/spectrogram_comparison.png`

## 🔍 **What to Look For:**

- **Frequency concentration**: Low vs high frequency energy
- **Time duration**: Short burst vs long decay
- **Harmonic structure**: Complex vs simple frequency patterns
- **Reverb tail**: Length of sound decay

## 📁 **Files Created:**

- `compare_spectrograms.py` - Main comparison script
- `create_test_audio.py` - Generate test audio files
- `samples/` - Test audio files
- `test_results/` - Output plots and results

## 🎯 **Example Output:**

The script will show:

```text
Loading TBPN gong: samples/tbpn_gong.wav
Loading reference gong: samples/reference_gong.wav
Generating spectrogram comparison...

Spectrogram comparison saved to: test_results/spectrogram_comparison.png

=== Audio Statistics ===
TBPN Gong: 2.500s, Peak: 0.800
Reference Gong: 2.000s, Peak: 0.700
```

## 💡 **Next Steps:**

1. Run the script with your real TBPN gong audio
2. Compare with a reference gong from YAMNet/AudioSet
3. Analyze the visual differences in frequency patterns
4. Use insights to improve gong detection accuracy
