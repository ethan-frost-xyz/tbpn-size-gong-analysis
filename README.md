# TBPN Gong Abuse Detection System
i made this because sometimes the boys absolutely rip the size gong, especially when there is a big raise


## Quick Start

```bash
# Setup environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Interactive menu (recommended)
cd src/gong_detector && python run_gong_detector.py

# Or direct command
python -m src.gong_detector.core.pipeline.detection_pipeline "YOUTUBE_URL"
```

## Dependencies

**Core ML:**
- `tensorflow==2.19.0` - YAMNet model (pinned for stability)
- `librosa==0.11.0` - Audio processing (pinned to avoid v1.0 breaking changes)
- `pyloudnorm==0.1.1` - EBU R128 analysis

**Audio/Video:**
- `yt-dlp>=2023.7.6` - YouTube downloads
- `ffmpeg` - Audio conversion (system dependency)

**Utilities:**
- `numpy>=1.24.0,<2.0.0` - Data processing
- `pandas>=2.0.0,<3.0.0` - Data handling
- `scikit-learn==1.6.1` - Trained classifier
- `setuptools<81` - Prevents pkg_resources deprecation warnings

### Known Issues & Warnings

**Non-breaking warnings you may see:**
- `pkg_resources is deprecated` - From TensorFlow Hub, will be fixed in future releases
- `PySoundFile failed. Trying audioread instead` - Normal fallback behavior for some audio formats
- `__audioread_load Deprecated` - Librosa deprecation, non-breaking until v1.0
- `Short-term/Momentary LUFS approximated` - Normal when precise measurements aren't available

**Version pinning rationale:**
- Dependencies are pinned to tested, working versions to prevent compatibility issues
- Use `pip install -r requirements.txt` for exact reproducible environment


