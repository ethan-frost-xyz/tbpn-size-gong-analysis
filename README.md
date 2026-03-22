# TBPN Gong Analysis

TBPN Gong Analysis is a reproducible ML pipeline with TensorFlow for detecting gong hits in TBPN episodes. The system downloads or reuses cached video and audio, runs a YAMNet-based classifier augmented with a trained Random Forest, verifies detections through saved artifacts, and exports structured datasets and charts for downstream analysis.

[update: project was featured on the show!](https://www.ethanfrost.org/tbpn-gong-analysis)

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

## Project structure

| Path | Description |
| --- | --- |
| `config/` | Runtime settings, including dual-cache defaults and optional `cookies.txt` for yt-dlp. |
| `data/csv_results/` | Timestamped CSV exports with detection metadata, LUFS, True Peak, and PLR columns. |
| `data/local_media/` | Git-ignored cache storing raw downloads, preprocessed WAV files, and `index.json` metadata for offline reruns. |
| `data/tbpn_ytlinks/` | Curated episode manifests that drive bulk detection jobs. |
| `docs/` | Internal notes kept for future engineering work. |
| `notebooks/charts/` | Python notebooks and scripts that build publication-ready charts from detection exports. |
| `scripts/` (local) | Personal helper scripts kept git-ignored; recreate locally as needed. |
| `src/gong_detector/core/` | Detection engine, pipelines, loudness utilities, CSV manager, and sample collection tools. |
| `src/gong_detector/run_gong_detector.py` | Interactive menu entry point that orchestrates the detection pipeline. |
| `tests/` | Unit, integration, and functional tests for detector, pipeline, cache, and utility layers. |
| `requirements.txt` | Locked dependency list at the repo root. |



## References
- YAMNet: https://tfhub.dev/google/yamnet/1
- EBU R128 Loudness (via `pyloudnorm`): https://tech.ebu.ch/publications/r128
- yt-dlp project: https://github.com/yt-dlp/yt-dlp
