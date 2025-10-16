# TBPN Gong Analysis

TBPN Gong Analysis is a reproducible ML pipeline with TensorFlow for detecting gong hits in TBPN episodes. The system downloads or reuses cached video and audio, runs a YAMNet-based classifier augmented with a trained Random Forest, verifies detections through saved artifacts, and exports structured datasets and charts for downstream analysis.

update: project was featured on the show!

## Workflow
1. Acquire source media from curated YouTube playlists using `yt-dlp`, storing raw downloads plus preprocessed 16 kHz mono WAV files in the local cache.
2. Preprocess and segment audio, then run YAMNet embeddings through the trained gong classifier to generate timestamped events and confidence scores.
3. Filter, consolidate, and enrich detections with LUFS, True Peak, PLR, and crest-factor metrics to support manual verification.
4. Persist reviewed detections, loudness statistics, and contextual metadata to reproducible CSV inventories and optional waveform snippets.
5. Feed the aggregated dataset into notebooks and chart scripts that publish the canonical gong frequency, loudness, and funding visualizations.

## Quickstart
... Launch the interactive menu:
   
   python src/gong_detector/run_gong_detector.py
   
   Select **Bulk Processing** to cache every episode listed in `data/tbpn_ytlinks/tbpn_youtube_links.txt`, run detections with the trained classifier, and export a CSV summary with LUFS and True Peak metrics.
   
5. Results include:
   - Cached audio in `data/local_media/raw/` and `data/local_media/preprocessed/` (git-ignored local cache)
   - Detection CSV files under `data/csv_results/` (generated outputs, not version controlled)
   - Optional positive sample WAV files for audit trails

## Project Structure

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



## References
- YAMNet: https://tfhub.dev/google/yamnet/1
- EBU R128 Loudness (via `pyloudnorm`): https://tech.ebu.ch/publications/r128
- yt-dlp project: https://github.com/yt-dlp/yt-dlp
