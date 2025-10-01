# Repository Inventory

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
