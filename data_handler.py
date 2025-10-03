# data_handler.py
import pandas as pd
import MetaTrader5 as mt5
import ta
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import config

def mt5_connect():
    # Credentials are now passed directly to initialize
    if not mt5.initialize(login=config.MT5_LOGIN, password=config.MT5_PASS, server=config.MT5_SERVER):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    print("MT5 connected successfully.")

def fetch_data():
    mt5_connect()
    rates = mt5.copy_rates_from_pos(config.SYMBOL, config.TIMEFRAME, 0, config.N_BARS_FETCH)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"Failed to fetch any data for symbol {config.SYMBOL}. "
            f"Please ensure the symbol is in your MT5 Market Watch and the name is correct."
        )

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.dropna()

def add_features(df):
    df_feat = df.copy()
    df_feat['ret'] = df_feat['close'].pct_change()
    df_feat['rsi'] = ta.momentum.RSIIndicator(df_feat['close']).rsi()
    df_feat['macd'] = ta.trend.MACD(df_feat['close']).macd_diff()
    df_feat['atr'] = ta.volatility.AverageTrueRange(df_feat['high'], df_feat['low'], df_feat['close']).average_true_range()
    bb = ta.volatility.BollingerBands(df_feat['close'])
    df_feat['bb_width'] = bb.bollinger_wband()
    df_feat.replace([float('inf'), -float('inf')], None, inplace=True)
    df_feat.bfill(inplace=True)
    df_feat.ffill(inplace=True)
    return df_feat

def create_labels(df):
    future_price = df['close'].shift(-config.LABEL_HORIZON)
    future_ret = (future_price - df['close']) / df['close']
    labels = pd.Series(0, index=df.index)
    labels.loc[future_ret > config.UP_THRESHOLD_REL] = 1 # BUY
    labels.loc[future_ret < config.DOWN_THRESHOLD_REL] = 2 # SELL
    return labels[:-config.LABEL_HORIZON]

def prepare_data(df):
    features_df = add_features(df)
    labels = create_labels(df)
    features_df = features_df.iloc[:len(labels)]
    
    split_idx = int(len(features_df) * (1 - config.TEST_SPLIT))
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]
    y_train = labels.iloc[:split_idx]
    y_test = labels.iloc[split_idx:]
    
    feature_cols = [col for col in train_df.columns if col not in ['time', 'open', 'high', 'low', 'tick_volume', 'spread', 'real_volume']]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])
    
    joblib.dump(scaler, config.SCALER_FILE)
    
    X_train, y_train = build_sequences(X_train_scaled, y_train.values)
    X_test, y_test = build_sequences(X_test_scaled, y_test.values)
    
    return X_train, y_train, X_test, y_test, scaler, feature_cols, split_idx

def build_sequences(data, labels):
    X, y = [], []
    for i in range(len(data) - config.SEQ_LEN):
        X.append(data[i:i + config.SEQ_LEN])
        y.append(labels[i + config.SEQ_LEN])
    return np.array(X), np.array(y)