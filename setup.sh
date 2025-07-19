#!/bin/bash

# Setup script for gong-detector project on macOS
# Run with: bash setup.sh

set -e  # Exit on any error

echo "🎵 Setting up gong-detector project..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew bundle

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install project in development mode
echo "🔨 Installing project in development mode..."
pip install -e .

echo "✅ Setup complete!"
echo ""
echo "🚀 To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Test the installation: python -c 'import tensorflow; print(\"TensorFlow:\", tensorflow.__version__)'"
echo "3. Run tests: pytest"
echo "4. Check the QUICK_START.md file for usage examples" 