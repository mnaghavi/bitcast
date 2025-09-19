#!/usr/bin/env python3
"""
BitCast - Bitcoin Forecast Experiments
Advanced prediction script with ensemble models and user-selectable periods.
"""

import argparse
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from advanced_predictor import AdvancedBitcoinPredictor

def parse_prediction_period(period_str):
    """
    Parse user input for prediction period with flexible formats.

    Supports:
    - Single numbers: "7", "30", "365"
    - Single periods: "week", "month", "year", "tomorrow"
    - Multiple periods: "1,2,3 days", "2,4,6 weeks", "1,3,6 months"
    - Ranges: "1-5 days", "2-8 weeks", "3-12 months"
    - Mixed: "1,3,5-7 days"
    """
    import re
    period_str = period_str.lower().strip()

    # Handle numeric input (assume days)
    try:
        days = int(period_str)
        if days <= 0:
            raise ValueError("Days must be positive")
        return [days], f"{days} day{'s' if days != 1 else ''}"
    except ValueError:
        pass

    # Check for multiple periods or ranges
    if any(char in period_str for char in [',', '-']) and any(unit in period_str for unit in ['day', 'week', 'month', 'year']):
        return parse_multiple_periods(period_str)

    # Handle text input for single periods
    period_mapping = {
        'day': 1, 'tomorrow': 1,
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
            clean_key = key.replace('1', '').replace('s', '') if key.startswith('1') else key
            return [days], clean_key

    raise ValueError(f"Unable to parse prediction period: {period_str}")


def parse_multiple_periods(period_str):
    """Parse multiple periods like '1,2,3 days' or '2-5 weeks' or '1,3,5-7 months'."""
    import re

    # Extract unit (days, weeks, months, years)
    unit_mapping = {
        'day': 1, 'days': 1,
        'week': 7, 'weeks': 7,
        'month': 30, 'months': 30,
        'year': 365, 'years': 365
    }

    unit = None
    multiplier = 1

    for unit_name, mult in unit_mapping.items():
        if unit_name in period_str:
            unit = unit_name
            multiplier = mult
            break

    if not unit:
        raise ValueError("Must specify unit: days, weeks, months, or years")

    # Extract numbers part (everything before the unit)
    numbers_part = period_str.split(unit)[0].strip()

    # Parse the numbers part for ranges and lists
    periods = []

    # Split by comma first
    for part in numbers_part.split(','):
        part = part.strip()

        # Check if it's a range (contains -)
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start <= 0 or end <= 0 or start > end:
                    raise ValueError("Invalid range")
                periods.extend(range(start, end + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range format: {part}") from e
        else:
            # Single number
            try:
                num = int(part)
                if num <= 0:
                    raise ValueError("Numbers must be positive")
                periods.append(num)
            except ValueError as e:
                raise ValueError(f"Invalid number: {part}") from e

    if not periods:
        raise ValueError("No valid periods found")

    # Remove duplicates and sort
    periods = sorted(list(set(periods)))

    # Convert to days
    days_list = [p * multiplier for p in periods]

    # Create description
    if len(periods) == 1:
        desc = f"{periods[0]} {unit}"
    else:
        desc = f"{','.join(map(str, periods))} {unit}"

    return days_list, desc

def print_prediction_options():
    """Print available prediction options."""
    print("\n📅 Available prediction periods:")
    print("   • Single numbers: 1, 7, 30, 365 (interpreted as days)")
    print("   • Text: day, week, month, quarter, year, tomorrow")
    print("   • Specific: 2w, 3m, 6m, 2y")
    print("   • Multiple periods: '1,2,3 days', '2,4,6 weeks'")
    print("   • Ranges: '1-5 days', '2-8 weeks', '3-12 months'")
    print("   • Mixed: '1,3,5-7 days', '1,2,6-12 months'")
    print("\n📝 Examples:")
    print("   • '1,2,3,4,5 days' - Next 5 days")
    print("   • '2,4,6 weeks' - 2nd, 4th, and 6th week")
    print("   • '1-6 months' - Next 6 months")
    print("   • '1,3,6-12 months' - 1st, 3rd, and 6th-12th months")

def main():
    parser = argparse.ArgumentParser(
        description='BitCast - Bitcoin Forecast Experiments (Advanced Ensemble Models)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python predict_advanced.py --period week
  python predict_advanced.py --period "1,2,3 days"
  python predict_advanced.py --period "2,4,6 weeks"
  python predict_advanced.py --period "1-5 days"
  python predict_advanced.py --period "1,3,6-12 months"
  python predict_advanced.py --all-horizons
  python predict_advanced.py --retrain --period month
        '''
    )

    parser.add_argument('--period', '-p', type=str,
                       help='Prediction period(s) - supports single, multiple, and ranges (e.g., "week", "1,2,3 days", "1-5 months")')
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
                days_list, period_name = parse_prediction_period(args.period)

                if len(days_list) == 1:
                    # Single prediction
                    days = days_list[0]
                    print(f"\n🎯 {period_name.title()} Prediction:")
                    print("-" * 40)

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

                else:
                    # Multiple predictions
                    print(f"\n🎯 {period_name.title()} Predictions:")
                    print("-" * 50)

                    use_ensemble = (args.model == 'ensemble')
                    predictions = predictor.predict_multiple_days(days_list, use_ensemble)

                    for days in sorted(days_list):
                        predicted_price = predictions[days]
                        change = ((predicted_price - current_price) / current_price) * 100
                        change_str = f"({change:+.1f}%)"

                        # Color coding for terminal
                        if change > 10:
                            icon = "🟢"
                        elif change > 0:
                            icon = "🔵"
                        elif change > -10:
                            icon = "🟡"
                        else:
                            icon = "🔴"

                        # Determine unit for display
                        if 'day' in period_name:
                            unit = 'day' if days == 1 else 'days'
                            display_period = f"{days} {unit}"
                        elif 'week' in period_name:
                            weeks = days // 7
                            unit = 'week' if weeks == 1 else 'weeks'
                            display_period = f"{weeks} {unit}"
                        elif 'month' in period_name:
                            months = days // 30
                            unit = 'month' if months == 1 else 'months'
                            display_period = f"{months} {unit}"
                        elif 'year' in period_name:
                            years = days // 365
                            unit = 'year' if years == 1 else 'years'
                            display_period = f"{years} {unit}"
                        else:
                            display_period = f"{days} days"

                        print(f"   {icon} {display_period:15}: ${predicted_price:8,.2f} {change_str}")

                    print(f"\n   Total Predictions: {len(days_list)}")
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

        print("\n💡 Tip: Use '--all-horizons' for complete analysis")
        print("   or '--period <timeframe>' for specific predictions")

    except Exception as prediction_error:
        print(f"❌ Error: {prediction_error}")
        print("\n🔧 Troubleshooting:")
        print("   • Try: python predict_advanced.py --update-data --retrain")
        print("   • Check internet connection for data download")
        print("   • Ensure sufficient disk space for model training")
        sys.exit(1)

if __name__ == "__main__":
    main()