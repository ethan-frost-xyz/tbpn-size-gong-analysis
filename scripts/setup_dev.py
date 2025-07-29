#!/usr/bin/env python3
"""Development setup script for gong_detector project."""

import os
import sys
from pathlib import Path


def setup_development_environment():
    """Set up the development environment."""
    print("Setting up development environment...")
    
    # Add src to Python path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"Added {src_path} to Python path")
    
    # Create necessary directories
    dirs_to_create = [
        "logs",
        "data/processed",
        "data/raw",
        "tests/data",
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    print("Development environment setup complete!")


if __name__ == "__main__":
    setup_development_environment() 