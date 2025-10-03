# backtester.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import MetaTrader5 as mt5
import config

def get_point_value():
    if not mt5.initialize():
        print("Could not initialize MT5 for point value check.")
        return 0.01 # Fallback for XAUUSD
    symbol_info = mt5.symbol_info(config.SYMBOL)
    mt5.shutdown()
    return symbol_info.point if symbol_info else 0.01

POINT_VALUE = get_point_value()

def run_backtest(model, X_test, y_test, df_full, split_idx):
    predictions_prob = model.predict(X_test)
    predictions = np.argmax(predictions_prob, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, target_names=['HOLD', 'BUY', 'SELL'], digits=4))

    backtest_df = df_full.iloc[split_idx:].copy()
    test_signals = pd.Series(predictions, index=backtest_df.index[config.SEQ_LEN:len(predictions)+config.SEQ_LEN])
    backtest_df['signal'] = test_signals

    pnl = []
    position = 0
    entry_price = 0
    spread_cost = 20 * POINT_VALUE # Example spread cost

    for i in range(len(backtest_df) - 1):
        current_signal = backtest_df['signal'].iloc[i]
        
        if position == 1 and current_signal != 1:
            profit = (backtest_df['open'].iloc[i+1] - entry_price) - spread_cost
            pnl.append(profit)
            position = 0
        elif position == -1 and current_signal != 2:
            profit = (entry_price - backtest_df['open'].iloc[i+1]) - spread_cost
            pnl.append(profit)
            position = 0
        
        if position == 0:
            if current_signal == 1:
                entry_price = backtest_df['open'].iloc[i+1]
                position = 1
            elif current_signal == 2:
                entry_price = backtest_df['open'].iloc[i+1]
                position = -1

    pnl_series = pd.Series(pnl)
    cum_pnl = pnl_series.cumsum()
    
    metrics = {
        "Total PnL ($ per Lot)": cum_pnl.iloc[-1] if not cum_pnl.empty else 0,
        "Sharpe Ratio": (pnl_series.mean() / pnl_series.std()) * np.sqrt(252*24*12) if pnl_series.std() > 0 else 0,
        "Max Drawdown": (cum_pnl - cum_pnl.expanding().max()).min(),
        "Total Trades": len(pnl_series),
    }

    print("\n--- Backtest Metrics ---")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")
        
    pnl_df = pd.DataFrame({'pnl': cum_pnl})
    pnl_df.to_excel(config.BACKTEST_FILE)
    print(f"\nBacktest PnL saved to {config.BACKTEST_FILE}")