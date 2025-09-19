#!/usr/bin/env python3
"""
BitCast - Bitcoin Forecast Experiments
Advanced prediction script with ensemble models and user-selectable periods.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from advanced_predictor import AdvancedBitcoinPredictor

def parse_prediction_period(period_str):
    """Parse user input for prediction period."""
    period_str = period_str.lower().strip()

    # Handle numeric input (assume days)
    try:
        days = int(period_str)
        if days <= 0:
            raise ValueError("Days must be positive")
        return days, f"{days} day{'s' if days != 1 else ''}"
    except ValueError:
        pass

    # Handle text input
    period_mapping = {
        'day': 1,
        'tomorrow': 1,
        '1d': 1, '1day': 1,
        'week': 7, '1w': 7, '1week': 7,
        '2w': 14, '2weeks': 14,
        'month': 30, '1m': 30, '1month': 30,
        '2m': 60, '2months': 60,
        '3m': 90, '3months': 90, 'quarter': 90,
        '6m': 180, '6months': 180,
        'year': 365, '1y': 365, '1year': 365,
        '2y': 730, '2years': 730
    }

    for key, days in period_mapping.items():
        if key in period_str:
            return days, key.replace('1', '').replace('s', '') if key.startswith('1') else key

    raise ValueError(f"Unable to parse prediction period: {period_str}")

def print_prediction_options():
    """Print available prediction options."""
    print("\n📅 Available prediction periods:")
    print("   • Numbers: 1, 7, 30, 365 (interpreted as days)")
    print("   • Text: day, week, month, quarter, year")
    print("   • Specific: 2w, 3m, 6m, 2y")
    print("   • Examples: 'tomorrow', '2 weeks', '3 months'")

def main():
    parser = argparse.ArgumentParser(
        description='BitCast - Bitcoin Forecast Experiments (Advanced Ensemble Models)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python predict_advanced.py --period week
  python predict_advanced.py --period 30
  python predict_advanced.py --period "3 months"
  python predict_advanced.py --all-horizons
  python predict_advanced.py --retrain --period month
        '''
    )

    parser.add_argument('--period', '-p', type=str,
                       help='Prediction period (e.g., "day", "week", "month", "30", "3m")')
    parser.add_argument('--all-horizons', action='store_true',
                       help='Show predictions for all standard time horizons')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain all models with latest data')
    parser.add_argument('--update-data', action='store_true',
                       help='Download/update training data')
    parser.add_argument('--model', choices=['ensemble', 'lstm', 'gru', 'transformer'],
                       default='ensemble', help='Model to use for prediction')
    parser.add_argument('--list-periods', action='store_true',
                       help='List available prediction periods')

    args = parser.parse_args()

    if args.list_periods:
        print_prediction_options()
        return

    print("🚀 BitCast - Bitcoin Forecast Experiments")
    print("=" * 55)
    print(f"🤖 Using {args.model.upper()} model")

    try:
        predictor = AdvancedBitcoinPredictor()

        # Update data if requested
        if args.update_data or not os.path.exists('data/bitcoin_data.csv'):
            print("📊 Downloading/updating Bitcoin data with market indicators...")
            predictor.download_data()

        # Retrain models if requested
        if args.retrain or not os.path.exists('models/ensemble_model.keras'):
            print("🤖 Training ensemble of advanced models...")
            print("   • LSTM + GRU + Transformer ensemble")
            print("   • 90-day lookback with 40+ features")
            print("   • RobustScaler for outlier handling")
            print("   • Huber loss for stability")
            predictor.train_models()

        current_price = predictor.get_current_price()
        print(f"\n📈 Current Bitcoin Price: ${current_price:,.2f}")

        if args.all_horizons:
            print("\n🎯 Multi-Horizon Price Predictions:")
            print("-" * 40)

            predictions = predictor.predict_multiple_horizons()

            for period, price in predictions.items():
                change = ((price - current_price) / current_price) * 100
                change_str = f"({change:+.1f}%)"

                # Color coding for terminal
                if change > 5:
                    icon = "🟢"
                elif change > 0:
                    icon = "🔵"
                elif change > -5:
                    icon = "🟡"
                else:
                    icon = "🔴"

                print(f"   {icon} {period:12}: ${price:8,.2f} {change_str}")

        elif args.period:
            try:
                days, period_name = parse_prediction_period(args.period)
                print(f"\n🎯 {period_name.title()} Prediction:")
                print("-" * 30)

                use_ensemble = (args.model == 'ensemble')
                predicted_price = predictor.predict_price(days, use_ensemble)

                change = ((predicted_price - current_price) / current_price) * 100
                change_str = f"({change:+.1f}%)"

                # Prediction confidence based on time horizon
                if days <= 7:
                    confidence = "High"
                elif days <= 30:
                    confidence = "Medium"
                elif days <= 90:
                    confidence = "Low"
                else:
                    confidence = "Very Low"

                print(f"   Predicted Price: ${predicted_price:,.2f} {change_str}")
                print(f"   Confidence Level: {confidence}")
                print(f"   Model Used: {args.model.upper()}")

            except ValueError as e:
                print(f"❌ Error parsing period: {e}")
                print_prediction_options()
                sys.exit(1)

        else:
            print("\n🎯 Quick Predictions (Ensemble Model):")
            print("-" * 35)

            quick_periods = {'Tomorrow': 1, 'Next Week': 7, 'Next Month': 30}

            for period_name, days in quick_periods.items():
                predicted_price = predictor.predict_price(days, use_ensemble=True)
                change = ((predicted_price - current_price) / current_price) * 100
                change_str = f"({change:+.1f}%)"
                print(f"   {period_name:12}: ${predicted_price:8,.2f} {change_str}")

        print(f"\n⏰ Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📊 Model Features:")
        print("   • Ensemble of LSTM, GRU, and Transformer")
        print("   • 40+ technical indicators and market data")
        print("   • Robust scaling for outlier resistance")
        print("   • Multiple timeframe analysis")

        print("\n⚠️  Investment Disclaimer:")
        print("   This is for educational and research purposes only.")
        print("   Cryptocurrency markets are highly volatile and unpredictable.")
        print("   Never invest based solely on predictions!")
        print("   Always do your own research and consult financial advisors.")

        print(f"\n💡 Tip: Use '--all-horizons' for complete analysis")
        print(f"   or '--period <timeframe>' for specific predictions")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Try: python predict_advanced.py --update-data --retrain")
        print(f"   • Check internet connection for data download")
        print(f"   • Ensure sufficient disk space for model training")
        sys.exit(1)

if __name__ == "__main__":
    main()