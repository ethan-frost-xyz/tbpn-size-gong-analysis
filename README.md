# TBPN Gong Analysis

TBPN Gong Analysis is a reproducible ML pipeline with TensorFlow for detecting gong hits in TBPN episodes. The system downloads or reuses cached video and audio, runs a YAMNet-based classifier augmented with a trained Random Forest, verifies detections through saved artifacts, and exports structured datasets and charts for downstream analysis.

## Workflow
1. Acquire source media from curated YouTube playlists using `yt-dlp`, storing raw downloads plus preprocessed 16 kHz mono WAV files in the local cache.
2. Preprocess and segment audio, then run YAMNet embeddings through the trained gong classifier to generate timestamped events and confidence scores.
3. Filter, consolidate, and enrich detections with LUFS, True Peak, PLR, and crest-factor metrics to support manual verification.
4. Persist reviewed detections, loudness statistics, and contextual metadata to reproducible CSV inventories and optional waveform snippets.
5. Feed the aggregated dataset into notebooks and chart scripts that publish the canonical gong frequency, loudness, and funding visualizations.

## Quickstart
1. Create a Python environment (3.11 or newer) and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. (Optional) Place authenticated cookies at `config/cookies.txt` so `yt-dlp` can access private or age-restricted videos.
3. Launch the interactive menu:
   ```bash
   python src/gong_detector/run_gong_detector.py
   ```
   Select **Bulk Processing** to cache every episode listed in `data/tbpn_ytlinks/tbpn_youtube_links.txt`, run detections with the trained classifier, and export a CSV summary with LUFS and True Peak metrics.
4. Command-line example without the menu:
   ```bash
   source .venv/bin/activate
   PYTHONPATH=src python -m gong_detector.core.pipeline.detection_pipeline \
       "https://youtube.com/watch?v=<VIDEO_ID>" \
       --use_version_one --save_csv data/csv_results/demo.csv --save_positive_samples
   ```
5. Results include:
   - Cached audio in `data/local_media/raw/` and `data/local_media/preprocessed/` (git-ignored local cache)
   - Detection CSV files under `data/csv_results/` (generated outputs, not version controlled)
   - Optional positive sample WAV files for audit trails

## Project Structure
```
.
├── config/                     Settings plus optional cookies for yt-dlp
├── data/
│   ├── csv_results/            Exported detection tables
│   └── tbpn_ytlinks/           Episode link lists for bulk runs
├── docs/                       Working notes and references
├── notebooks/
│   └── charts/                 Notebooks and scripts that build the visuals
├── scripts/ (local)            Personal helper scripts (git-ignored)
├── src/gong_detector/
│   ├── core/                   Detection engine, pipelines, utils, and training tools
│   └── run_gong_detector.py    Menu runner for the whole pipeline
├── tests/                      Unit, integration, and functional checks
└── requirements.txt            Locked dependency list
```

## Usage Notes
- **Interactive pipeline**: `src/gong_detector/run_gong_detector.py` wraps single-video, bulk, sample collection, and audio conversion workflows behind a keyboard-driven menu.
- **Batch runners**: `python -m gong_detector.core.pipeline.bulk_processor` processes the canonical episode list, performs LUFS and True Peak analysis, and can operate strictly offline when the local cache is populated.
- **Training utilities**: `gong_detector/core/training` provides manual and negative sample collectors plus scripts for embedding extraction and classifier retraining.
- **Cache management**: `gong_detector/core/utils/local_media.py` maintains the dual-cache index. Migration helpers such as `scripts/migrate_raw_cache_to_wav.py` and `scripts/update_index_for_wav.py` keep cached assets consistent.
- **Git hygiene**: Generated artifacts (`data/local_media/`, `data/temp_audio/`, exported CSV/LUFS reports) and personal helpers in `scripts/` stay out of version control; rerun the pipeline or rebuild scripts locally as needed.
- **Dataset generation and charts**: The `CSVManager` assembles PLR, LUFS, and confidence metrics. Notebooks in `notebooks/charts` turn the exports into reproducible visualizations.
- **Configuration**: Tune thresholds, batch sizes, and memory guards in `config/settings.py`. Place YouTube playlists or ad-hoc link lists in `data/tbpn_ytlinks/` to drive bulk runs.

## Caveats and Limitations
- Microphone placement and studio acoustics vary across episodes, so PLR and LUFS values should be compared within the same production era.
- Reverb-heavy rooms and overlapping speech can mask gong strikes, reducing classifier confidence despite audible events.
- YouTube normalization tamps down True Peak values on re-encoded audio, so raw dBTP readings may underestimate the original broadcast level.
- Loudness perception remains subjective; retain human review of saved samples when curating canonical datasets.
- Offline runs require a fully populated cache; missing assets trigger download attempts unless `--local_only` is set.

## References
- YAMNet: https://tfhub.dev/google/yamnet/1
- EBU R128 Loudness (via `pyloudnorm`): https://tech.ebu.ch/publications/r128
- yt-dlp project: https://github.com/yt-dlp/yt-dlp
