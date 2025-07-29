# TBPN Size Gong Analysis

A machine learning project for detecting gong sounds in audio files using YAMNet and custom classifiers.

## Project Structure

This project follows modern Python best practices with a src-layout structure:

```
tbpn-size-gong-analysis/
├── src/                          # Source code
│   └── gong_detector/           # Main package
│       ├── core/                # Core functionality
│       │   ├── data/           # Data management
│       │   ├── detector/       # Audio detection
│       │   ├── models/         # ML models
│       │   ├── pipeline/       # Processing pipelines
│       │   ├── training/       # Training utilities
│       │   └── utils/          # Utility functions
│       └── training/           # Training data and scripts
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── functional/             # Functional tests
├── config/                      # Configuration files
├── data/                        # Data files
│   ├── csv_dir/               # CSV data
│   ├── csv_results/           # Results data
│   └── tbpn_ytlinks/         # YouTube links
├── docs/                        # Documentation
├── logs/                        # Application logs
├── static/                      # Static files
├── templates/                   # Jinja2 templates
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up the environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Run the detection pipeline:**
   ```bash
   python -m src.gong_detector.core.pipeline.detection_pipeline
   ```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Type Checking
```bash
mypy src/
```

## Configuration

Configuration is managed through environment variables and the `config/settings.py` file. Key settings include:

- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `AUDIO_FORMAT`: Audio format (default: "wav")
- `LOG_LEVEL`: Logging level (default: "INFO")

## Data

- **Training Data**: Located in `src/gong_detector/training/data/`
- **Results**: Stored in `data/csv_results/`
- **YouTube Links**: Stored in `data/tbpn_ytlinks/`

## Models

Trained models are stored in `src/gong_detector/core/models/` and include:
- YAMNet embeddings
- Custom classifiers
- Configuration files

## Contributing

1. Follow the project structure guidelines
2. Write tests for new functionality
3. Use type hints throughout
4. Follow PEP 8 style guidelines
5. Update documentation as needed

## License

[Add your license information here]
