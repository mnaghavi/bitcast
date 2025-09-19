#!/usr/bin/env python3
"""
Setup script for Bitcoin Price Predictor
Installs required dependencies and sets up the environment.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    print("🚀 Setting up Bitcoin Price Predictor...")
    print("=" * 50)

    setup_directories()
    install_requirements()

    print("\n✅ Setup complete!")
    print("\nTo get started, run:")
    print("  python predict.py")
    print("\nFor first time use (downloads data and trains model):")
    print("  python predict.py --update-data --retrain")

if __name__ == "__main__":
    main()