"""
BitCast - Bitcoin Forecast Experiments
Advanced ensemble prediction model using LSTM, GRU, and Transformer architectures.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input,
                                   MultiHeadAttention, LayerNormalization,
                                   GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedBitcoinPredictor:
    def __init__(self, lookback_days=90):
        """
        Initialize advanced Bitcoin predictor with ensemble models.

        Args:
            lookback_days (int): Number of days to look back for predictions
        """
        self.lookback_days = lookback_days
        self.scaler = RobustScaler()  # Better for outliers
        self.models = {}
        self.feature_scaler = RobustScaler()

        # Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

    def download_data(self):
        """Download Bitcoin historical data with additional market data."""
        print("Downloading Bitcoin and market data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=6*365)  # 6 years for more data

        # Download Bitcoin data
        btc_data = yf.download('BTC-USD',
                              start=start_date.strftime('%Y-%m-%d'),
                              end=end_date.strftime('%Y-%m-%d'),
                              auto_adjust=False)

        # Download additional market data for context
        try:
            spy_data = yf.download('SPY', start=start_date.strftime('%Y-%m-%d'),
                                 end=end_date.strftime('%Y-%m-%d'), auto_adjust=False)
            gold_data = yf.download('GLD', start=start_date.strftime('%Y-%m-%d'),
                                  end=end_date.strftime('%Y-%m-%d'), auto_adjust=False)

            # Flatten MultiIndex if present
            for data in [btc_data, spy_data, gold_data]:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

            # Add market context
            btc_data['SPY_Close'] = spy_data['Close'].reindex(btc_data.index).fillna(method='ffill')
            btc_data['Gold_Close'] = gold_data['Close'].reindex(btc_data.index).fillna(method='ffill')

        except Exception as market_data_error:
            print(f"Warning: Could not download market data: {market_data_error}")

        # Flatten MultiIndex columns if present
        if isinstance(btc_data.columns, pd.MultiIndex):
            btc_data.columns = [col[0] for col in btc_data.columns]

        if btc_data.empty:
            raise ValueError("Failed to download Bitcoin data")

        # Add advanced technical indicators
        btc_data = self._add_advanced_indicators(btc_data)

        # Save the data
        btc_data.to_csv('data/bitcoin_data.csv')
        print(f"Downloaded {len(btc_data)} days of Bitcoin data with advanced indicators")

        return btc_data

    def _add_advanced_indicators(self, df):
        """Add comprehensive technical indicators."""
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']

        # Multiple timeframe moving averages
        for window in [7, 14, 21, 50, 100, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']

        # Exponential moving averages
        for span in [12, 26, 50]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # RSI with multiple periods
        for rsi_period in [14, 30]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))

        # Bollinger Bands with multiple periods
        for window in [20, 50]:
            rolling_mean = df['Close'].rolling(window=window).mean()
            rolling_std = df['Close'].rolling(window=window).std()
            df[f'BB_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'BB_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'BB_width_{window}'] = df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']
            df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / df[f'BB_width_{window}']

        # Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_price_trend'] = df['Volume'] * df['Returns']

        # Price momentum
        for mom_period in [5, 10, 20]:
            df[f'Momentum_{mom_period}'] = df['Close'] / df['Close'].shift(mom_period) - 1

        # Volatility measures
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()

        # Support and resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support_distance'] = (df['Close'] - df['Support']) / df['Close']
        df['Resistance_distance'] = (df['Resistance'] - df['Close']) / df['Close']

        # Market context features (if available)
        if 'SPY_Close' in df.columns:
            df['SPY_Returns'] = df['SPY_Close'].pct_change()
            df['BTC_SPY_correlation'] = df['Returns'].rolling(window=30).corr(df['SPY_Returns'])

        if 'Gold_Close' in df.columns:
            df['Gold_Returns'] = df['Gold_Close'].pct_change()
            df['BTC_Gold_correlation'] = df['Returns'].rolling(window=30).corr(df['Gold_Returns'])

        # Time-based features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter

        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.dropna()

        return df

    def prepare_data(self, data):
        """Prepare data with better feature engineering."""
        # Select comprehensive features
        feature_columns = [
            'Close', 'Volume', 'Returns', 'Log_Returns', 'Price_Range', 'Body_Size',
            'MA_7', 'MA_14', 'MA_21', 'MA_50', 'MA_100', 'MA_200',
            'MA_7_ratio', 'MA_21_ratio', 'MA_50_ratio', 'MA_200_ratio',
            'EMA_12', 'EMA_26', 'EMA_50',
            'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI_14', 'RSI_30',
            'BB_width_20', 'BB_position_20', 'BB_width_50', 'BB_position_50',
            'Volume_ratio', 'Volume_price_trend',
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            'Volatility_10', 'Volatility_30',
            'Support_distance', 'Resistance_distance',
            'DayOfWeek', 'Month', 'Quarter'
        ]

        # Add market context if available
        if 'BTC_SPY_correlation' in data.columns:
            feature_columns.extend(['SPY_Returns', 'BTC_SPY_correlation'])
        if 'BTC_Gold_correlation' in data.columns:
            feature_columns.extend(['Gold_Returns', 'BTC_Gold_correlation'])

        # Filter available columns
        available_columns = [col for col in feature_columns if col in data.columns]
        features = data[available_columns].values

        # Use RobustScaler for better outlier handling
        scaled_features = self.scaler.fit_transform(features)

        # Create sequences
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_features)):
            X.append(scaled_features[i-self.lookback_days:i])
            y.append(scaled_features[i, 0])  # Close price is first column

        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape):
        """Build LSTM model."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            LSTM(32, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        return model

    def build_gru_model(self, input_shape):
        """Build GRU model."""
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape,
                kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            GRU(64, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            GRU(32, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        return model

    def build_transformer_model(self, input_shape):
        """Build Transformer-based model."""
        inputs = Input(shape=input_shape)

        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=8, key_dim=input_shape[-1]//8
        )(inputs, inputs)
        attn_output = LayerNormalization()(attn_output + inputs)

        # Second attention layer
        attn_output2 = MultiHeadAttention(
            num_heads=4, key_dim=input_shape[-1]//4
        )(attn_output, attn_output)
        attn_output2 = LayerNormalization()(attn_output2 + attn_output)

        # Global pooling and dense layers
        x = GlobalAveragePooling1D()(attn_output2)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)

        return Model(inputs, outputs)

    def build_ensemble_model(self, input_shape):
        """Build ensemble of all models."""
        inputs = Input(shape=input_shape)

        # LSTM branch
        lstm_branch = LSTM(64, return_sequences=False)(inputs)
        lstm_branch = Dropout(0.2)(lstm_branch)
        lstm_out = Dense(32, activation='relu')(lstm_branch)

        # GRU branch
        gru_branch = GRU(64, return_sequences=False)(inputs)
        gru_branch = Dropout(0.2)(gru_branch)
        gru_out = Dense(32, activation='relu')(gru_branch)

        # Attention branch
        attn_branch = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
        attn_branch = GlobalAveragePooling1D()(attn_branch)
        attn_branch = Dropout(0.2)(attn_branch)
        attn_out = Dense(32, activation='relu')(attn_branch)

        # Combine all branches
        combined = Concatenate()([lstm_out, gru_out, attn_out])
        combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined)
        combined = Dropout(0.2)(combined)
        outputs = Dense(1)(combined)

        return Model(inputs, outputs)

    def train_models(self):
        """Train ensemble of models."""
        # Load data
        if os.path.exists('data/bitcoin_data.csv'):
            data = pd.read_csv('data/bitcoin_data.csv', index_col=0, parse_dates=True)
        else:
            data = self.download_data()

        print("Preparing training data...")
        X, y = self.prepare_data(data)

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape}")

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)

        # Train ensemble model (main model)
        print("Training ensemble model...")
        ensemble_model = self.build_ensemble_model((X_train.shape[1], X_train.shape[2]))
        ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Less sensitive to outliers
            metrics=['mae']
        )

        ensemble_history = ensemble_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate ensemble
        train_pred = ensemble_model.predict(X_train, verbose=0)
        test_pred = ensemble_model.predict(X_test, verbose=0)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        print(f"Ensemble - Training MAE: {train_mae:.4f}, Testing MAE: {test_mae:.4f}")

        # Save ensemble model
        self.models['ensemble'] = ensemble_model
        ensemble_model.save('models/ensemble_model.keras')

        # Train individual models for comparison
        model_configs = {
            'lstm': self.build_lstm_model,
            'gru': self.build_gru_model,
            'transformer': self.build_transformer_model
        }

        for name, build_func in model_configs.items():
            print(f"Training {name.upper()} model...")
            model = build_func((X_train.shape[1], X_train.shape[2]))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='huber',
                metrics=['mae']
            )

            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )

            self.models[name] = model
            model.save(f'models/{name}_model.keras')

            # Quick evaluation
            test_pred = model.predict(X_test, verbose=0)
            test_mae = mean_absolute_error(y_test, test_pred)
            print(f"{name.upper()} - Testing MAE: {test_mae:.4f}")

        # Save scaler
        joblib.dump(self.scaler, 'models/advanced_scaler.joblib')
        print("All models trained and saved!")

        return ensemble_history

    def load_models(self):
        """Load trained models."""
        if not os.path.exists('models/ensemble_model.keras'):
            raise FileNotFoundError("No trained models found. Please run with --retrain flag.")

        self.models['ensemble'] = load_model('models/ensemble_model.keras')
        self.scaler = joblib.load('models/advanced_scaler.joblib')

        # Load individual models if available
        for model_name in ['lstm', 'gru', 'transformer']:
            model_path = f'models/{model_name}_model.keras'
            if os.path.exists(model_path):
                self.models[model_name] = load_model(model_path)

    def get_current_price(self):
        """Get current Bitcoin price."""
        ticker = yf.Ticker("BTC-USD")
        current_data = ticker.history(period="1d")
        if current_data.empty:
            # Fallback to last known price from data
            if os.path.exists('data/bitcoin_data.csv'):
                data = pd.read_csv('data/bitcoin_data.csv', index_col=0, parse_dates=True)
                return float(data['Close'].iloc[-1])
            return 100000.0  # Fallback value
        return float(current_data['Close'].iloc[-1])

    def predict_price(self, days_ahead=1, use_ensemble=True):
        """
        Predict Bitcoin price for specific days ahead.

        Args:
            days_ahead (int): Number of days to predict ahead
            use_ensemble (bool): Whether to use ensemble model
        """
        if not self.models:
            self.load_models()

        # Load recent data
        if os.path.exists('data/bitcoin_data.csv'):
            data = pd.read_csv('data/bitcoin_data.csv', index_col=0, parse_dates=True)
        else:
            data = self.download_data()

        # Prepare features same way as training
        feature_columns = [col for col in data.columns if col in [
            'Close', 'Volume', 'Returns', 'Log_Returns', 'Price_Range', 'Body_Size',
            'MA_7', 'MA_14', 'MA_21', 'MA_50', 'MA_100', 'MA_200',
            'MA_7_ratio', 'MA_21_ratio', 'MA_50_ratio', 'MA_200_ratio',
            'EMA_12', 'EMA_26', 'EMA_50',
            'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI_14', 'RSI_30',
            'BB_width_20', 'BB_position_20', 'BB_width_50', 'BB_position_50',
            'Volume_ratio', 'Volume_price_trend',
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            'Volatility_10', 'Volatility_30',
            'Support_distance', 'Resistance_distance',
            'DayOfWeek', 'Month', 'Quarter',
            'SPY_Returns', 'BTC_SPY_correlation',
            'Gold_Returns', 'BTC_Gold_correlation'
        ]]

        recent_features = data[feature_columns].tail(self.lookback_days).values
        scaled_recent = self.scaler.transform(recent_features)

        # Predict iteratively
        current_sequence = scaled_recent.copy()
        pred_list = []

        model_key = 'ensemble' if use_ensemble else 'lstm'
        model = self.models.get(model_key, self.models['ensemble'])

        for day in range(days_ahead):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.lookback_days, len(feature_columns))

            # Get prediction
            pred_scaled = model.predict(X_pred, verbose=0)[0][0]
            pred_list.append(pred_scaled)

            # Update sequence for next prediction
            next_features = current_sequence[-1].copy()
            next_features[0] = pred_scaled  # Update close price

            # Update some technical indicators based on new price
            # This is simplified - in reality, we'd need to recalculate all indicators
            if len(feature_columns) > 1:
                # Update returns
                if day > 0:
                    next_features[2] = (pred_scaled - current_sequence[-1, 0]) / current_sequence[-1, 0]

            # Shift sequence and add new features
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_features

        # Convert back to original price scale using a more stable approach
        final_pred_scaled = pred_list[-1]
        current_scaled = scaled_recent[-1, 0]

        # Get recent actual prices for scaling reference
        recent_prices = data['Close'].tail(10).values
        current_actual_price = recent_prices[-1]

        # Calculate the predicted change in scaled space
        scaled_change = final_pred_scaled - current_scaled

        # Apply a conservative scaling factor to prevent extreme predictions
        # Scale down the change based on time horizon to make it more realistic
        if days_ahead <= 7:
            max_change = 0.15  # Maximum 15% change for short term
        elif days_ahead <= 30:
            max_change = 0.30  # Maximum 30% change for medium term
        else:
            max_change = 0.50  # Maximum 50% change for long term

        scaled_change = np.clip(scaled_change, -max_change, max_change)

        # For longer horizons, apply a dampening factor
        if days_ahead > 1:
            # Reduce volatility for longer predictions
            dampening = min(1.0, 7.0 / days_ahead)
            scaled_change *= dampening

        # Apply change to current price
        predicted_price = current_actual_price * (1.0 + scaled_change)

        # Additional safety bounds
        min_price = current_actual_price * 0.5   # No less than 50% of current price
        max_price = current_actual_price * 2.0   # No more than 2x current price

        predicted_price = max(min_price, min(max_price, abs(predicted_price)))

        return float(predicted_price)

    def predict_multiple_days(self, days_list, use_ensemble=True):
        """
        Predict Bitcoin prices for multiple specific days.

        Args:
            days_list (list): List of days ahead to predict
            use_ensemble (bool): Whether to use ensemble model

        Returns:
            dict: Dictionary mapping days to predicted prices
        """
        predictions = {}

        for days in days_list:
            predicted_price = self.predict_price(days, use_ensemble)
            predictions[days] = predicted_price

        return predictions

    def predict_multiple_horizons(self):
        """Predict for multiple time horizons with ensemble voting."""
        horizons = {
            'Next Day': 1,
            'Next Week': 7,
            'Next Month': 30,
            'Next Quarter': 90,
            'Next Year': 365
        }

        horizon_predictions = {}

        # Get ensemble predictions
        for time_period, days in horizons.items():
            ensemble_pred = self.predict_price(days, use_ensemble=True)
            horizon_predictions[time_period] = ensemble_pred

        return horizon_predictions

if __name__ == "__main__":
    predictor = AdvancedBitcoinPredictor()
    predictor.download_data()
    predictor.train_models()
    horizon_predictions = predictor.predict_multiple_horizons()

    current_price = predictor.get_current_price()
    print(f"\nCurrent Bitcoin Price: ${current_price:,.2f}")

    for time_period, price in horizon_predictions.items():
        change = ((price - current_price) / current_price) * 100
        print(f"{time_period}: ${price:,.2f} ({change:+.1f}%)")