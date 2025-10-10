# data_collector.py
"""
Continuously fetches latest bars from MT5 and appends to data/btcusd_live.csv
Run this script in its own terminal/session.
"""

import os
import time
import json
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

CFG = "config.json"
if not os.path.exists(CFG):
    raise FileNotFoundError("config.json not found")

with open(CFG, "r") as f:
    cfg = json.load(f)

ACCOUNT = cfg["account"]
SYMBOL = cfg["symbol"]
TF_MIN = cfg.get("timeframe_minutes", 5)
N_BARS = int(cfg.get("n_bars", 6000))
SLEEP = int(cfg["execution"].get("sleep_seconds", 300))
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "btcusd_live.csv")

def tf_const(mins):
    m = {1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15,
         30: mt5.TIMEFRAME_M30, 60: mt5.TIMEFRAME_H1, 240: mt5.TIMEFRAME_H4, 1440: mt5.TIMEFRAME_D1}
    return m.get(mins, mt5.TIMEFRAME_M5)

def connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    if ACCOUNT.get("login"):
        ok = mt5.login(ACCOUNT["login"], password=ACCOUNT["password"], server=ACCOUNT["server"])
        if not ok:
            acct = mt5.account_info()
            if acct is None or acct.login != ACCOUNT["login"]:
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    mt5.symbol_select(SYMBOL, True)
    print("✅ MT5 connected for data collection")

def fetch_bars(n=500):
    tf = tf_const(TF_MIN)
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def append_unique(df_new, csv_path=CSV_PATH):
    # convert time to ISO for dedupe
    df_new = df_new.copy()
    if "time" in df_new.columns:
        df_new["time_iso"] = df_new["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError("No 'time' column returned from MT5")

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        if "time_iso" in df_old.columns:
            existing = set(df_old["time_iso"].astype(str).tolist())
        else:
            existing = set()
        df_to_append = df_new[~df_new["time_iso"].isin(existing)].copy()
        if not df_to_append.empty:
            df_to_append.to_csv(csv_path, mode="a", index=False, header=False)
            print(f"{datetime.now()} - Appended {len(df_to_append)} rows to {csv_path}")
        else:
            print(f"{datetime.now()} - No new rows to append.")
    else:
        # write header
        df_new.to_csv(csv_path, index=False)
        print(f"{datetime.now()} - Created {csv_path} with {len(df_new)} rows")

def main():
    connect()
    while True:
        try:
            df = fetch_bars(n=1000 if N_BARS > 1000 else N_BARS)
            if df is None:
                print(f"{datetime.now()} - No bars returned, retry after {SLEEP}s")
                time.sleep(SLEEP)
                continue
            append_unique(df)
            time.sleep(SLEEP)
        except KeyboardInterrupt:
            print("Interrupted by user. Shutting down data collector.")
            break
        except Exception as e:
            print("Data collector error:", e)
            time.sleep(SLEEP)

if __name__ == "__main__":
    main()
