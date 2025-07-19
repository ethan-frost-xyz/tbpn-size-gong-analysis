# Gong Detector

A Python project for detecting gong sounds using YAMNet (Yet Another Mobile Network) for audio classification.

## 🚀 Features

- Audio classification using TensorFlow Hub's YAMNet model
- Support for various audio formats (WAV, MP3, etc.)
- Comprehensive test suite with pytest
- Automated CI/CD with GitHub Actions
- Code quality enforcement with Ruff

## 📋 Requirements

- Python 3.9 or higher
- TensorFlow 2.13.0 or higher
- TensorFlow Hub 0.14.0 or higher
- NumPy 1.24.0 or higher
- Pandas 2.0.0 or higher

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ethan-frost-xyz/tbpn-size-gong-analysis.git
   cd tbpn-size-gong-analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r gong_detector/requirements.txt
   ```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install pytest pytest-cov ruff

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=gong_detector --cov-report=html
```

## 🔍 Code Quality

This project uses Ruff for code quality enforcement:

```bash
# Install Ruff
pip install ruff

# Check code quality
ruff check gong_detector/ tests/

# Auto-fix issues
ruff check gong_detector/ tests/ --fix
```

## 🚀 CI/CD

This project includes automated CI/CD with GitHub Actions that:

- ✅ Runs tests on Python 3.9, 3.10, and 3.11
- ✅ Checks code quality with Ruff
- ✅ Generates test coverage reports
- ✅ Uploads coverage to Codecov

## 📁 Project Structure

```
tbpn-size-gong-analysis/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD configuration
├── gong_detector/
│   ├── __init__.py
│   ├── detect.py               # Main detection logic
│   ├── test_yamnet.py          # YAMNet test script
│   └── requirements.txt        # Dependencies
├── tests/
│   ├── __init__.py
│   └── test_basic.py           # Basic tests
├── .gitignore                  # Git ignore rules
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Ethan Frost** - [ethan-frost-xyz](https://github.com/ethan-frost-xyz)

## 🙏 Acknowledgments

- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) for audio classification
- [TensorFlow Hub](https://tfhub.dev/) for model hosting
- [Ruff](https://github.com/astral-sh/ruff) for fast Python linting 