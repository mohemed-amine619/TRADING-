import os
import json
import logging
import traceback
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# ======================================================
# SETUP & CONFIGURATION
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - (TRAINER) %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_config():
    """Loads the central configuration file."""
    with open("config.json", "r") as f:
        return json.load(f)


# ======================================================
# FEATURE ENGINEERING (UPGRADED)
# ======================================================
def add_technical_features(df):
    """Calculates and adds a comprehensive set of technical indicator features."""
    df = df.copy()

    # Correctly parse time from CSV string format, which is crucial
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Basic Features
    df["ret"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()

    # Momentum: RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(window=14).mean()
    down = -delta.clip(upper=0).rolling(window=14).mean()
    rs = up / (down + 1e-10)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # Momentum: MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # Volatility: ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14).mean()

    # Volatility: Bollinger Bands Percentage
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Momentum: Stochastic Oscillator (%K)
    low14 = df["low"].rolling(window=14).min()
    high14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14)

    # Volume: On-Balance Volume (OBV)
    if "tick_volume" in df.columns:
        df["obv"] = (np.sign(df["close"].diff()) * df["tick_volume"]).fillna(0).cumsum()
        df["obv"] = MinMaxScaler().fit_transform(df[["obv"]])

    # Time-Based Features
    df["day_of_week"] = df["time"].dt.dayofweek
    df["hour_of_day"] = df["time"].dt.hour

    df.dropna(inplace=True)
    return df


# ======================================================
# DATA PREPARATION
# ======================================================
def prepare_data(config):
    logging.info("Starting data preparation...")
    paths, model_cfg = config["paths"], config["model"]
    df = pd.read_csv(paths["live_data_file"])
    logging.info(f"Loaded {len(df)} rows from {paths['live_data_file']}")
    df = add_technical_features(df)

    # --- SAFEGUARD ---
    # Add the crucial check here to prevent crashes
    if df.empty:
        raise ValueError(
            "DataFrame is empty after feature calculation. Ensure historical data file is large enough for all indicator lookback periods."
        )

    future_price = df["close"].shift(-model_cfg["horizon"])
    df["target"] = ((future_price - df["close"]) / df["close"]) > model_cfg[
        "label_threshold"
    ]
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(
            "DataFrame is empty after calculating the target variable. Check horizon and data size."
        )

    feature_cols = model_cfg.get("feature_columns")
    if not feature_cols:
        raise KeyError("FATAL: 'feature_columns' array not found in config.")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"FATAL: Feature columns missing from data: {missing_cols}")

    features, target = df[feature_cols].values, df["target"].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    scaler_path = paths["scaler"]
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    seq_len = model_cfg["seq_len"]
    X, y = [
        scaled_features[i : i + seq_len] for i in range(len(scaled_features) - seq_len)
    ], [target[i + seq_len] for i in range(len(scaled_features) - seq_len)]
    X, y = np.array(X), np.array(y)
    logging.info(f"Created {len(X)} sequences of length {seq_len}.")

    return train_test_split(X, y, test_size=model_cfg["test_split"], random_state=42)


# ======================================================
# MODEL TRAINING & MAIN EXECUTION
# ======================================================
def train_model(config, X_train, y_train, X_val, y_val):
    model_path, model_cfg = config["paths"]["model"], config["model"]

    if os.path.exists(model_path):
        logging.info(f"Found existing model at {model_path}. Loading for retraining.")
        model = load_model(model_path)
    else:
        logging.info("No existing model found. Creating a new one.")
        model = Sequential(
            [
                LSTM(
                    100,
                    return_sequences=True,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                ),
                Dropout(0.2),
                LSTM(100, return_sequences=False),
                Dropout(0.2),
                Dense(50, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    logging.info("Starting model training...")
    model.fit(
        X_train,
        y_train,
        epochs=model_cfg["epochs"],
        batch_size=model_cfg["batch"],
        validation_data=(X_val, y_val),
        verbose=1,
    )
    model.save(model_path)
    logging.info(f"✅ Model training complete. Saved to {model_path}")


if __name__ == "__main__":
    logging.info("===== STARTING MODEL TRAINING PIPELINE =====")
    config = load_config()
    lock_file = config["paths"]["model_lock_file"]

    # This import was missing for the error logging
    import traceback

    with open(lock_file, "w") as f:
        f.write(datetime.now().isoformat())

    try:
        X_train, X_val, y_train, y_val = prepare_data(config)
        train_model(config, X_train, y_train, X_val, y_val)
    except Exception as e:
        logging.error(f"A critical error occurred during the training pipeline: {e}")
        logging.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
    finally:
        last_trained_file = config["paths"]["last_trained_file"]
        with open(last_trained_file, "w") as f:
            # Corrected deprecated function call to be version-agnostic
            f.write(datetime.now(timezone.utc).isoformat())
        logging.info(f"Updated last trained timestamp in {last_trained_file}")

        if os.path.exists(lock_file):
            os.remove(lock_file)
        logging.info("Removed model lock file. Training complete.")
