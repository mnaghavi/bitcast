#!/bin/bash
# Git repository setup script for BitCast

echo "🚀 Setting up Git repository for BitCast"
echo "========================================"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
else
    echo "📦 Git repository already exists"
fi

# Add all files
echo "📁 Adding files to Git..."
git add .

# Create initial commit
echo "📝 Creating initial commit..."
git commit -m "🎉 Initial commit: BitCast - Bitcoin Forecast Experiments

✨ Features:
- Ensemble of LSTM, GRU, and Transformer models
- 40+ technical indicators and market data
- User-selectable prediction periods
- Robust scaling and bias correction
- Comprehensive documentation

🤖 Generated with Claude Code"

echo ""
echo "✅ Git repository setup complete!"
echo ""
echo "🌐 To upload to GitHub:"
echo "   1. Create a new repository on GitHub"
echo "   2. Run: git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo "   3. Run: git branch -M main"
echo "   4. Run: git push -u origin main"
echo ""
echo "💡 Repository is ready for GitHub with:"
echo "   • Comprehensive .gitignore"
echo "   • MIT License"
echo "   • Professional README with examples"
echo "   • Clean project structure"