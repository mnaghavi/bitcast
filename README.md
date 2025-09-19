# BitCast 🚀
### *Bitcoin Forecast Experiments*

An advanced machine learning research project that experiments with Bitcoin price forecasting using ensemble models (LSTM, GRU, and Transformer). Features sophisticated technical analysis with 40+ indicators, market data integration, and user-selectable prediction periods.

> **BitCast** = **Bit**coin Fore**cast** - A comprehensive toolkit for experimenting with cryptocurrency price prediction models and techniques.

## 🔬 What is BitCast?

**BitCast** stands for **Bitcoin Forecast Experiments** - a comprehensive research toolkit for experimenting with cryptocurrency price prediction models. The project explores various machine learning approaches, from simple LSTM networks to sophisticated ensemble models combining multiple architectures.

### 🎯 Project Goals
- **Research**: Experiment with different ML approaches for crypto forecasting
- **Education**: Learn about time series prediction and technical analysis
- **Innovation**: Explore ensemble methods and advanced feature engineering
- **Accessibility**: Make advanced ML models easy to use for anyone

### 🏗️ Architecture Philosophy
BitCast uses an **experimental approach** - we implement multiple models and let you choose the best one for your use case:
- **Legacy Models**: Simple LSTM for basic experimentation
- **Advanced Ensemble**: Combined LSTM + GRU + Transformer for maximum accuracy
- **Flexible Framework**: Easy to add new models and features

## 🌟 Features

### 🧠 Advanced AI Models
- **Ensemble Architecture**: Combines LSTM, GRU, and Transformer models
- **40+ Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands, momentum, volatility
- **Market Context**: Integrates S&P 500 and Gold correlations
- **Robust Scaling**: Handles outliers better than standard normalization

### 📊 Comprehensive Analysis
- **6 Years Historical Data**: More training data for better accuracy
- **Multiple Timeframes**: 5min to yearly analysis
- **Advanced Features**: Support/resistance, volatility clustering, market correlations
- **Time-based Features**: Day of week, month, quarter effects

### 🎯 Flexible Predictions
- **User-Selectable Periods**: Any timeframe from 1 day to 2 years
- **Multiple Horizons**: Standard periods (day, week, month, quarter, year)
- **Confidence Levels**: Based on prediction timeframe
- **Model Selection**: Choose between ensemble or individual models

### 🔧 Enhanced Reliability
- **Bias Correction**: Addresses bearish prediction bias
- **Huber Loss**: More stable training with outliers
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive optimization

## Quick Start

### 1. Setup Environment

```bash
# Run the installation script (creates virtual environment and installs dependencies)
./install.sh

# OR manually setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. First Run (Download Data & Train Model)

```bash
# Activate virtual environment first
source venv/bin/activate

# Download data and train advanced ensemble model
python predict_advanced.py --update-data --retrain
```

This will:
- Download 6 years of Bitcoin + market data
- Train ensemble of LSTM, GRU, and Transformer models (10-15 minutes)
- Generate comprehensive predictions

### 3. Make Predictions

```bash
# Activate virtual environment
source venv/bin/activate

# Quick multi-horizon predictions
python predict_advanced.py --all-horizons

# Specific time period
python predict_advanced.py --period "2 weeks"
python predict_advanced.py --period 30
python predict_advanced.py --period "3 months"

# Choose specific model
python predict_advanced.py --period week --model lstm
python predict_advanced.py --period month --model ensemble

# List all available periods
python predict_advanced.py --list-periods
```

### 4. Legacy Simple Model (Original)

```bash
# Still available - basic LSTM model
python predict.py --update-data --retrain
python predict.py
```

## 📊 Sample Output

### Multi-Horizon Predictions
```
🚀 BitCast - Bitcoin Forecast Experiments
==================================================
🤖 Using ENSEMBLE model

📈 Current Bitcoin Price: $43,250.00

🎯 Multi-Horizon Price Predictions:
----------------------------------------
   🟢 Next Day     : $44,120.30 (+2.0%)
   🔵 Next Week    : $45,890.75 (+6.1%)
   🟢 Next Month   : $48,650.20 (+12.5%)
   🔵 Next Quarter : $52,180.40 (+20.7%)
   🟢 Next Year    : $67,240.85 (+55.5%)

⏰ Generated at: 2024-01-15 14:30:25

📊 Model Features:
   • Ensemble of LSTM, GRU, and Transformer
   • 40+ technical indicators and market data
   • Robust scaling for outlier resistance
   • Multiple timeframe analysis
```

### Specific Period Prediction
```
🚀 BitCast - Bitcoin Forecast Experiments
🤖 Using ENSEMBLE model

📈 Current Bitcoin Price: $43,250.00

🎯 2 Weeks Prediction:
------------------------------
   Predicted Price: $45,320.15 (+4.8%)
   Confidence Level: High
   Model Used: ENSEMBLE
```

## 🏗️ Model Architecture

### Ensemble Components
1. **LSTM Branch**: 128→64→32 units with L2 regularization
2. **GRU Branch**: 128→64→32 units with dropout layers
3. **Transformer Branch**: Multi-head attention with 8 heads
4. **Combined Output**: Weighted ensemble of all branches

### Advanced Features (40+)
- **Price Features**: Returns, log returns, price ranges, body sizes
- **Moving Averages**: Multiple timeframes (7, 14, 21, 50, 100, 200 days)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, momentum
- **Market Data**: S&P 500 and Gold correlations
- **Time Features**: Day of week, month, quarter effects
- **Support/Resistance**: Dynamic levels and distances

### Training Improvements
- **Lookback Period**: 90 days (vs 60 in basic model)
- **Data**: 6 years of historical data
- **Loss Function**: Huber loss (robust to outliers)
- **Scaling**: RobustScaler (handles extreme values better)
- **Regularization**: L2 penalty, dropout, early stopping

## 📁 Project Structure

```
bitcast/
├── predict_advanced.py       # 🚀 Advanced prediction script
├── advanced_predictor.py     # 🧠 Ensemble model implementation
├── predict.py               # 📈 Legacy simple predictor
├── bitcoin_predictor.py     # 🔧 Basic LSTM model
├── install.sh              # ⚙️  Auto-installation script
├── requirements.txt        # 📋 Python dependencies
├── README.md              # 📖 Documentation
├── LICENSE               # ⚖️  MIT License
├── .gitignore           # 🚫 Git ignore rules
├── data/                # 💾 Downloaded data (auto-created)
│   └── bitcoin_data.csv
└── models/             # 🤖 Trained models (auto-created)
    ├── ensemble_model.keras
    ├── lstm_model.keras
    ├── gru_model.keras
    ├── transformer_model.keras
    └── advanced_scaler.joblib
```

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- pandas, numpy, scikit-learn
- yfinance for data download

## Disclaimer

⚠️ **Important**: This model is for educational and research purposes only. Cryptocurrency markets are highly volatile and unpredictable. Never make investment decisions based solely on these predictions. Always do your own research and consider consulting with financial advisors.

## Technical Details

The model uses several technical indicators:
- **Moving Averages**: 7, 21, and 50-day periods
- **RSI**: Relative Strength Index for momentum
- **Bollinger Bands**: Price volatility indicators
- **Volume Analysis**: Trading volume patterns
- **Price Ratios**: High/low and price change patterns

## 🚀 Getting Started - Complete Guide

### For GitHub Users
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bitcast.git
cd bitcast

# Quick setup and first run
./install.sh
source venv/bin/activate
python predict_advanced.py --update-data --retrain
```

### Available Commands
```bash
# Quick predictions (most common use case)
python predict_advanced.py --all-horizons

# Specific predictions
python predict_advanced.py --period "2 weeks"
python predict_advanced.py --period 90
python predict_advanced.py --period "6 months"

# Model comparison
python predict_advanced.py --period month --model ensemble
python predict_advanced.py --period month --model lstm
python predict_advanced.py --period month --model transformer

# Data management
python predict_advanced.py --update-data      # Refresh data
python predict_advanced.py --retrain          # Retrain models
```

## 🔧 Troubleshooting

### Common Issues
1. **Installation Problems**: Use the automated installer: `./install.sh`
2. **Memory Issues**: Reduce lookback period in `advanced_predictor.py`
3. **Network Issues**: Check internet connection for Yahoo Finance data
4. **Model Training Fails**: Ensure sufficient disk space (>2GB)

### Performance Tips
- **First Run**: Expect 10-15 minutes for full model training
- **Subsequent Runs**: ~30 seconds for predictions with existing models
- **Memory Usage**: ~2-4GB RAM during training, <1GB for predictions
- **Storage**: ~500MB for models and data

## 📈 Understanding Results

### Confidence Levels
- **High** (1-7 days): Most reliable, based on short-term patterns
- **Medium** (1-4 weeks): Good accuracy, considers market trends
- **Low** (1-3 months): Moderate reliability, affected by market volatility
- **Very Low** (6+ months): Educational only, highly speculative

### Model Performance
- **Ensemble**: Best overall accuracy, combines all approaches
- **LSTM**: Good for trend following, handles sequences well
- **GRU**: Faster training, good for short-term predictions
- **Transformer**: Best for pattern recognition, attention mechanisms

## 🤝 Contributing

BitCast is designed as an **experimental platform** - we welcome researchers, students, and enthusiasts to contribute new ideas and improvements!

### 🔬 Research Areas
- **New Models**: Implement CNN, GAN, or Attention-based architectures
- **Feature Engineering**: Add sentiment analysis, on-chain metrics, macro indicators
- **Ensemble Methods**: Experiment with voting, stacking, or boosting approaches
- **Data Sources**: Integrate news, social media, or alternative data feeds

### 🛠️ Technical Improvements
- **Performance**: Model optimization, GPU acceleration, distributed training
- **Infrastructure**: Docker containerization, cloud deployment, APIs
- **Interface**: Web dashboard, mobile app, real-time streaming
- **Testing**: Backtesting framework, A/B testing, model validation

### 📚 Documentation & Education
- **Tutorials**: Step-by-step guides for different use cases
- **Research Papers**: Analysis of model performance and market insights
- **Code Examples**: Jupyter notebooks with detailed explanations
- **Video Content**: Walkthroughs and educational materials

## ⚠️ Important Disclaimers

**This software is for educational and research purposes only.**

- Cryptocurrency markets are extremely volatile and unpredictable
- Past performance does not guarantee future results
- Never invest more than you can afford to lose
- Always do your own research (DYOR)
- Consider consulting with financial advisors
- The models may have biases and limitations
- Predictions become less reliable over longer time horizons

**No Investment Advice**: Nothing in this software constitutes investment advice, financial advice, trading advice, or any other sort of advice.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

---

## 🌟 Support BitCast

**⭐ If BitCast helps with your research or learning, please give it a star on GitHub!**

**🔬 Have ideas for new experiments? We'd love to hear them - open an issue!**

**🤝 Want to contribute? Check out our contributing guidelines above!**

### 📞 Connect with the Community
- **GitHub Discussions**: Share results, ask questions, propose experiments
- **Issues**: Report bugs, request features, suggest improvements
- **Pull Requests**: Contribute code, documentation, or research

---

*BitCast - Where Bitcoin meets Machine Learning Experiments 🚀*