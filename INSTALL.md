# 🚀 Installation Guide for macOS

This guide will help you set up the gong-detector project on your new Mac from scratch.

## 📋 Prerequisites

- macOS (tested on macOS 13+)
- Terminal access
- Internet connection

## 🛠️ Quick Setup (Recommended)

Run the automated setup script:

```bash
bash setup.sh
```

This script will:
- Install Homebrew (if not present)
- Install system dependencies (Python, ffmpeg, etc.)
- Create a virtual environment
- Install all Python dependencies
- Set up the project for development

## 🔧 Manual Setup

If you prefer to set up manually or the automated script fails:

### 1. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install System Dependencies

```bash
# Install Python 3.12
brew install python@3.12

# Install ffmpeg for audio processing
brew install ffmpeg

# Install additional tools
brew install git sox
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## ✅ Verification

Test that everything is working:

```bash
# Activate virtual environment
source venv/bin/activate

# Test TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Test ffmpeg
ffmpeg -version

# Run tests
pytest
```

## 🎯 Common Issues & Solutions

### TensorFlow Installation Issues
If TensorFlow fails to install:
```bash
# Try installing with specific version
pip install tensorflow==2.15.0

# Or use conda if pip fails
conda install tensorflow
```

### ffmpeg Not Found
If ffmpeg is not found after installation:
```bash
# Add to PATH
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Permission Issues
If you get permission errors:
```bash
# Fix Homebrew permissions
sudo chown -R $(whoami) /opt/homebrew
```

## 🚀 Next Steps

After successful installation:

1. **Read the documentation**: Check `README.md` and `QUICK_START.md`
2. **Test the system**: Try the example in `QUICK_START.md`
3. **Explore the code**: Look at the modules in `gong_detector/core/`

## 📚 Project Structure

```
tbpn-size-gong-analysis/
├── gong_detector/          # Main project code
│   ├── core/              # Core modules
│   ├── samples/           # Sample audio files
│   └── test_results/      # Test outputs
├── requirements.txt       # Python dependencies
├── Brewfile              # System dependencies
├── setup.sh              # Automated setup script
└── INSTALL.md            # This file
```

## 🆘 Getting Help

If you encounter issues:

1. Check the error messages carefully
2. Ensure all dependencies are installed
3. Verify your Python version (3.9+ required)
4. Make sure ffmpeg is in your PATH
5. Try running the setup script again

For project-specific help, check the main `README.md` file. 