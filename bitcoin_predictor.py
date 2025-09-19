"""
BitCast - Legacy Bitcoin Price Prediction Model
Basic LSTM neural network for Bitcoin price prediction (legacy version).
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')

class BitcoinPredictor:
    def __init__(self, lookback_days=60):
        """
        Initialize Bitcoin predictor with configurable lookback period.

        Args:
            lookback_days (int): Number of days to look back for predictions
        """
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

        # Create directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

    def download_data(self):
        """Download Bitcoin historical data using Yahoo Finance."""
        print("Downloading Bitcoin data...")

        # Download 5 years of data to have enough training samples
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)

        # Download Bitcoin data
        btc_data = yf.download('BTC-USD',
                              start=start_date.strftime('%Y-%m-%d'),
                              end=end_date.strftime('%Y-%m-%d'),
                              auto_adjust=False)

        if btc_data.empty:
            raise Exception("Failed to download Bitcoin data")

        # Flatten MultiIndex columns if present
        if isinstance(btc_data.columns, pd.MultiIndex):
            btc_data.columns = [col[0] for col in btc_data.columns]

        # Add technical indicators
        btc_data = self._add_technical_indicators(btc_data)

        # Save the data
        btc_data.to_csv('data/bitcoin_data.csv')
        print(f"Downloaded {len(btc_data)} days of Bitcoin data")

        return btc_data

    def _add_technical_indicators(self, df):
        """Add technical indicators to the dataset."""
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = rolling_mean + (rolling_std * 2)
        df['BB_lower'] = rolling_mean - (rolling_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']

        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = (df['Volume'] / df['Volume_MA']).fillna(1.0)

        # Price change indicators
        df['Price_change'] = df['Close'].pct_change().fillna(0.0)
        df['High_Low_ratio'] = (df['High'] / df['Low']).fillna(1.0)

        # Fill remaining NaN values before dropping rows
        df = df.bfill().ffill()

        # Drop any remaining NaN values created by rolling calculations
        df = df.dropna()

        return df

    def prepare_data(self, data):
        """Prepare data for LSTM model training."""
        # Select features for training
        feature_columns = ['Close', 'Volume', 'MA_7', 'MA_21', 'MA_50',
                          'RSI', 'BB_width', 'Volume_ratio', 'Price_change', 'High_Low_ratio']

        # Ensure we have all columns
        available_columns = [col for col in feature_columns if col in data.columns]
        features = data[available_columns].values

        # Scale the features
        scaled_features = self.scaler.fit_transform(features)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_features)):
            X.append(scaled_features[i-self.lookback_days:i])
            y.append(scaled_features[i, 0])  # Predict 'Close' price (first column)

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train_model(self):
        """Train the LSTM model on Bitcoin data."""
        # Load or download data
        if os.path.exists('data/bitcoin_data.csv'):
            data = pd.read_csv('data/bitcoin_data.csv', index_col=0, parse_dates=True)
        else:
            data = self.download_data()

        print("Preparing training data...")
        X, y = self.prepare_data(data)

        # Split data (80% training, 20% testing)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Build and train model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        # Evaluate model
        train_predictions = self.model.predict(X_train, verbose=0)
        test_predictions = self.model.predict(X_test, verbose=0)

        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)

        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")

        # Save model and scaler
        self.model.save('models/bitcoin_model.keras')
        joblib.dump(self.scaler, 'models/scaler.joblib')

        print("Model training completed and saved!")

        return history

    def load_model(self):
        """Load trained model and scaler."""
        if not os.path.exists('models/bitcoin_model.keras'):
            raise Exception("No trained model found. Please run with --retrain flag.")

        self.model = load_model('models/bitcoin_model.keras')
        self.scaler = joblib.load('models/scaler.joblib')

    def get_current_price(self):
        """Get current Bitcoin price."""
        ticker = yf.Ticker("BTC-USD")
        current_data = ticker.history(period="1d")
        return float(current_data['Close'].iloc[-1])

    def predict(self):
        """Generate predictions for next day, month, and year."""
        if self.model is None:
            self.load_model()

        # Load recent data
        if os.path.exists('data/bitcoin_data.csv'):
            data = pd.read_csv('data/bitcoin_data.csv', index_col=0, parse_dates=True)
        else:
            data = self.download_data()

        # Get recent data for prediction
        feature_columns = ['Close', 'Volume', 'MA_7', 'MA_21', 'MA_50',
                          'RSI', 'BB_width', 'Volume_ratio', 'Price_change', 'High_Low_ratio']
        available_columns = [col for col in feature_columns if col in data.columns]

        recent_data = data[available_columns].tail(self.lookback_days).values
        scaled_recent = self.scaler.transform(recent_data)

        # Reshape for model input
        X_pred = scaled_recent.reshape(1, self.lookback_days, len(available_columns))

        predictions = {}
        current_sequence = X_pred.copy()

        # Predict for different time horizons
        time_horizons = {
            'Next Day': 1,
            'Next Week': 7,
            'Next Month': 30,
            'Next Year': 365
        }

        for period, days in time_horizons.items():
            sequence = current_sequence.copy()

            # For longer predictions, we simulate day by day
            for _ in range(days):
                pred_scaled = self.model.predict(sequence, verbose=0)[0][0]

                # Create next day's features (simplified approach)
                next_features = sequence[0, -1, :].copy()
                next_features[0] = pred_scaled  # Update close price

                # Update sequence for next prediction
                sequence = np.roll(sequence, -1, axis=1)
                sequence[0, -1, :] = next_features

            # Convert back to original scale
            dummy_features = np.zeros((1, len(available_columns)))
            dummy_features[0, 0] = pred_scaled
            predicted_price = self.scaler.inverse_transform(dummy_features)[0, 0]

            predictions[period] = max(predicted_price, 0)  # Ensure positive price

        return predictions

if __name__ == "__main__":
    # Test the predictor
    predictor = BitcoinPredictor()
    predictor.download_data()
    predictor.train_model()
    predictions = predictor.predict()

    print("Bitcoin Price Predictions:")
    for period, price in predictions.items():
        print(f"{period}: ${price:,.2f}")