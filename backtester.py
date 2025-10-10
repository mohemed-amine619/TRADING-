import os
import json
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# ======================================================
# SETUP & CONFIGURATION
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - (BACKTESTER) %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_config():
    """Loads the central configuration file."""
    with open("config.json", "r") as f:
        return json.load(f)


# ======================================================
# FEATURE ENGINEERING (MUST BE IDENTICAL TO TRAINER)
# ======================================================
def add_technical_features(df):
    """Calculates and adds a comprehensive set of technical indicator features."""
    df = df.copy()

    # Correctly parse time from CSV string format
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
# BACKTESTING CORE
# ======================================================
class Backtester:
    def __init__(self, config):
        self.config = config
        self.paths = config["paths"]
        self.model_cfg = config["model"]
        self.exec_cfg = config["execution"]
        self.backtest_cfg = config.get(
            "backtest", {}
        )  # NEW: Load backtest specific config

        self.model = tf.keras.models.load_model(self.paths["model"])
        self.scaler = joblib.load(self.paths["scaler"])
        logging.info("✅ Model and scaler loaded successfully.")

    def run(self):
        logging.info("Starting backtest...")

        # 1. Load and prepare data
        df = pd.read_csv(self.paths["live_data_file"])

        # --- NEW: Filter data by start date ---
        start_date_str = self.backtest_cfg.get("start_date")
        if start_date_str:
            try:
                start_date = pd.to_datetime(start_date_str)
                df["time"] = pd.to_datetime(
                    df["time"]
                )  # Ensure time column is datetime
                original_rows = len(df)
                df = df[df["time"] >= start_date].copy().reset_index(drop=True)
                logging.info(
                    f"Filtering data for backtest start date >= {start_date_str}. Kept {len(df)} of {original_rows} rows."
                )
            except Exception as e:
                logging.warning(
                    f"Could not parse 'start_date' from config: {e}. Running on full dataset."
                )

        df = add_technical_features(df)

        if df.empty:
            logging.error(
                "FATAL: DataFrame is empty after feature calculation and date filtering."
            )
            return

        feature_cols = self.model_cfg["feature_columns"]
        features = df[feature_cols].values
        scaled_features = self.scaler.transform(features)

        # 2. Simulation loop (logic remains the same)
        trades = []
        position = None
        initial_balance = 10000.0
        balance = initial_balance
        equity_curve = [initial_balance]
        seq_len = self.model_cfg["seq_len"]

        for i in range(seq_len, len(scaled_features)):
            sequence = np.expand_dims(scaled_features[i - seq_len : i], axis=0)
            prediction = self.model.predict(sequence, verbose=0)[0][0]

            current_price = df["close"].iloc[i]
            current_time = df["time"].iloc[i]
            atr = df["atr14"].iloc[i]

            if position:
                pnl = 0
                close_reason = None
                if position == "long":
                    sl_hit, tp_hit = (
                        current_price <= entry_sl,
                        current_price >= entry_tp,
                    )
                    pnl = (
                        (current_price - entry_price)
                        * (balance / entry_price)
                        * self.exec_cfg.get("lot", 0.01)
                    )
                else:
                    sl_hit, tp_hit = (
                        current_price >= entry_sl,
                        current_price <= entry_tp,
                    )
                    pnl = (
                        (entry_price - current_price)
                        * (balance / entry_price)
                        * self.exec_cfg.get("lot", 0.01)
                    )

                if sl_hit:
                    close_reason = "Stop Loss"
                elif tp_hit:
                    close_reason = "Take Profit"
                elif (
                    position == "long"
                    and prediction <= self.model_cfg["sell_threshold"]
                ):
                    close_reason = "Opposing Signal"
                elif (
                    position == "short"
                    and prediction >= self.model_cfg["buy_threshold"]
                ):
                    close_reason = "Opposing Signal"

                if close_reason:
                    balance += pnl
                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "pnl": pnl,
                            "side": position,
                            "reason": close_reason,
                        }
                    )
                    position = None
                    equity_curve.append(balance)

            if not position:
                if prediction >= self.model_cfg["buy_threshold"]:
                    position, entry_price, entry_time = (
                        "long",
                        current_price,
                        current_time,
                    )
                    entry_sl, entry_tp = entry_price - (atr * 1.5), entry_price + (
                        atr * 2.0
                    )
                elif prediction <= self.model_cfg["sell_threshold"]:
                    position, entry_price, entry_time = (
                        "short",
                        current_price,
                        current_time,
                    )
                    entry_sl, entry_tp = entry_price + (atr * 1.5), entry_price - (
                        atr * 2.0
                    )

        self.report(trades, equity_curve, df)

    def report(self, trades, equity_curve, df):
        # ... (reporting logic remains the same)
        if not trades:
            logging.warning("No trades were executed during the backtest.")
            return
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        total_trades = len(trades_df)
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = trades_df["pnl"].sum()
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        equity = pd.Series(equity_curve)
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min() * 100
        print(
            "\n" + "=" * 50 + "\n" + " " * 15 + "BACKTESTING REPORT" + "\n" + "=" * 50
        )
        print(f" Period Analyzed:       {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        print(f" Initial Balance:       ${10000:,.2f}")
        print("-" * 50)
        print(f" Net Profit:              ${total_pnl:,.2f}")
        print(f" Total Trades:            {total_trades}")
        print(f" Win Rate:                {win_rate:.2f}%")
        print(f" Profit Factor:           {profit_factor:.2f}")
        print(f" Max Drawdown:            {max_drawdown:.2f}%")
        print("=" * 50 + "\n")
        plt.figure(figsize=(12, 6))
        plt.plot(equity)
        plt.title("Equity Curve")
        plt.xlabel("Trade Number")
        plt.ylabel("Balance ($)")
        plt.grid(True)
        plot_path = os.path.join("reports", "equity_curve.png")
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        logging.info(f"✅ Equity curve plot saved to {plot_path}")


if __name__ == "__main__":
    config = load_config()
    backtester = Backtester(config)
    backtester.run()
