# Quick Start: YouTube Gong Detection with Positive Sample Collection

## **New Feature: Automatic Positive Sample Collection**

You can now automatically collect gong samples from YouTube videos for human-in-the-loop review!

## **New Feature: Bulk Processing**

You can now process multiple YouTube URLs at once using the bulk processing script!

## **How to Use:**

### **Basic Detection:**
```bash
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID"
```

### **With Positive Sample Collection:**
```bash
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --save_positive_samples
```

### **With Custom Threshold:**
```bash
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --threshold 0.5 --save_positive_samples
```

### **Bulk Processing Multiple URLs:**
```bash
# Process all URLs in tbpn_youtube_links.txt
python gong_detector/bulk_process.py

# With custom threshold
python gong_detector/bulk_process.py --threshold 0.5

# Save CSV results for each video
python gong_detector/bulk_process.py --save_csv

# Keep temporary audio files
python gong_detector/bulk_process.py --keep_audio
```

## **What Happens:**

### **Single Video Processing:**
1. **Downloads YouTube audio** (using yt-dlp)
2. **Detects gongs** (using YAMNet)
3. **Extracts 3-second segments** around each detection
4. **Saves to training folder** for human review

### **Bulk Processing:**
1. **Reads URLs** from `tbpn_youtube_links.txt`
2. **Processes each URL** using the same detection pipeline
3. **Maintains exact formatting** and storage patterns
4. **Provides progress updates** and error handling
5. **Continues processing** even if some URLs fail

## **Output Files:**

- **CSV results**: `csv_results/` (if using `--save_csv`)
- **Positive samples**: `gong_detector/training/data/raw_samples/positive/`
- **Sample filenames**: `gong_1.5s_conf_0.850_1.wav`

## **Human-in-the-Loop Workflow:**

### **Single Video:**
1. Run detection with `--save_positive_samples`
2. Review saved samples in `positive/` folder
3. Keep good samples, delete false positives
4. Repeat until you have 50 confirmed samples
5. Then run training pipeline

### **Bulk Processing:**
1. Run bulk processing: `python gong_detector/bulk_process.py`
2. Review all saved samples across multiple dated folders
3. Keep good samples, delete false positives
4. Much faster collection of training data
5. Then run training pipeline

## **Example Output:**

```text
Step 1: Downloading and processing audio...
Step 2: Loading YAMNet model...
Step 3: Processing audio...
Step 4: Running gong detection...

Detected 3 gongs:
  00:00:15.2 - Confidence: 0.850
  00:00:32.1 - Confidence: 0.720
  00:01:05.8 - Confidence: 0.680

Saving positive samples to: gong_detector/training/data/raw_samples/positive
✓ Saved: gong_15.2s_conf_0.850_1.wav
✓ Saved: gong_32.1s_conf_0.720_2.wav
✓ Saved: gong_65.8s_conf_0.680_3.wav

Saved 3 positive samples to: gong_detector/training/data/raw_samples/positive

## **Bulk Processing Example:**

```text
Found 8 YouTube URLs to process

Progress: 1/8
============================================================
Processing: https://www.youtube.com/watch?v=oo9qWcWRQvw
============================================================
Step 1: Downloading and processing audio...
Step 2: Loading YAMNet model...
Step 3: Processing audio...
Step 4: Running gong detection...

Detected 2 gongs:
  00:00:15.2 - Confidence: 0.850
  00:00:32.1 - Confidence: 0.720

Saving positive samples to: gong_detector/training/data/raw_samples/positive/tbpn_07_11_2025
✓ Saved: s_conf_0.850_at_00_15_12_1.wav
✓ Saved: s_conf_0.720_at_00_32_01_2.wav

✓ Successfully processed: https://www.youtube.com/watch?v=oo9qWcWRQvw

Progress: 2/8
...

============================================================
BULK PROCESSING COMPLETE
============================================================
Total URLs: 8
Successful: 8
Failed: 0

All URLs processed successfully!
```


# Quick Start: Spectrogram Comparison

## **Script is Ready!**

The `compare_spectrograms.py` script has been created and tested successfully.

## **How to Use:**

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

## **What You Get:**

1. **Side-by-side Mel spectrograms** (dB scale)
2. **Audio statistics** (duration, peak amplitude)
3. **Saved plot** in `test_results/spectrogram_comparison.png`

## **What to Look For:**

- **Frequency concentration**: Low vs high frequency energy
- **Time duration**: Short burst vs long decay
- **Harmonic structure**: Complex vs simple frequency patterns
- **Reverb tail**: Length of sound decay

## **Files Created:**

- `compare_spectrograms.py` - Main comparison script
- `create_test_audio.py` - Generate test audio files
- `samples/` - Test audio files
- `test_results/` - Output plots and results

## **Example Output:**

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

## **Next Steps:**

1. Run the script with your real TBPN gong audio
2. Compare with a reference gong from YAMNet/AudioSet
3. Analyze the visual differences in frequency patterns
4. Use insights to improve gong detection accuracy

---