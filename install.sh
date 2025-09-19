#!/bin/bash
# BitCast Installation Script

echo "🚀 Setting up BitCast - Bitcoin Forecast Experiments..."
echo "======================================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data models

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use BitCast:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run predictions: python predict.py --update-data --retrain"
echo ""
echo "For quick predictions (after first setup):"
echo "   python predict.py"