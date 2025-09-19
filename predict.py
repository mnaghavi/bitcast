#!/usr/bin/env python3
"""
BitCast - Legacy Bitcoin Prediction Script
Simple LSTM-based Bitcoin price prediction (legacy version).
"""

import argparse
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from bitcoin_predictor import BitcoinPredictor

def main():
    parser = argparse.ArgumentParser(description='Predict Bitcoin prices')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model with latest data')
    parser.add_argument('--update-data', action='store_true', help='Update training data')
    args = parser.parse_args()

    print("🚀 BitCast - Legacy Predictor")
    print("=" * 35)

    try:
        predictor = BitcoinPredictor()

        if args.update_data or not os.path.exists('data/bitcoin_data.csv'):
            print("📊 Downloading/updating Bitcoin data...")
            predictor.download_data()

        if args.retrain or not os.path.exists('models/bitcoin_model.keras'):
            print("🤖 Training prediction model...")
            predictor.train_model()

        print("🔮 Generating predictions...")
        predictions = predictor.predict()

        current_price = predictor.get_current_price()

        print(f"\n📈 Current Bitcoin Price: ${current_price:,.2f}")
        print("\n🎯 Price Predictions:")
        print("-" * 30)

        for period, price in predictions.items():
            change = ((price - current_price) / current_price) * 100
            change_str = f"({change:+.1f}%)"
            print(f"{period:12}: ${price:8,.2f} {change_str}")

        print(f"\n⏰ Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n⚠️  Disclaimer: This is for educational purposes only.")
        print("   Never invest based solely on predictions!")

    except Exception as prediction_error:
        print(f"❌ Error: {prediction_error}")
        sys.exit(1)

if __name__ == "__main__":
    main()