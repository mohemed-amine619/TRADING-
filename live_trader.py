# live_trader.py
import os
import json
import time
import traceback
from datetime import datetime, timedelta, timezone
import sys
import subprocess
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import tensorflow as tf
import joblib
import requests
import logging

# -----------------------
# Logging setup
# -----------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "bot.log")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger("live_trader")

# -----------------------
# Load config
# -----------------------
CFG = "config.json"
if not os.path.exists(CFG):
    log.critical("config.json not found. Exit.")
    raise FileNotFoundError("config.json not found")

with open(CFG, "r") as f:
    cfg = json.load(f)

# read commonly used config values with defaults
ACCOUNT = cfg.get("account", {})
SYMBOL = cfg.get("symbol", "BTCUSDm")
TF_MIN = int(cfg.get("timeframe_minutes", 5))
SEQ_LEN = int(cfg.get("model", {}).get("seq_len", 32))
SLEEP = int(cfg.get("execution", {}).get("sleep_seconds", 300))
LOT = float(cfg.get("execution", {}).get("lot", 0.01))
DEVIATION = int(cfg.get("execution", {}).get("deviation", 50))
MAX_OPEN = int(cfg.get("execution", {}).get("max_open_trades", 1))
# stop_loss and take_profit in config are interpreted as *pips/points* (integers)
SL_PIPS = float(cfg.get("execution", {}).get("stop_loss", 0))
TP_USD = float(cfg.get("execution", {}).get("take_profit_usd", 30.0))


MODEL_PATH = cfg.get("paths", {}).get("model", "models/lstm_btc_model.h5")
SCALER_PATH = cfg.get("paths", {}).get("scaler", "models/scaler_btc.save")
TRADES_FILE = cfg.get("paths", {}).get("trades_file", "data/trades_history.csv")
LAST_TRAINED_FILE = cfg.get("paths", {}).get(
    "last_trained_file", "models/last_trained.txt"
)
LOCK_FILE = cfg.get("paths", {}).get("model_lock_file", "models/training.lock")

TG_TOKEN = cfg.get("telegram", {}).get("bot_token")
TG_CHAT_ID = cfg.get("telegram", {}).get("chat_id")
NOTIFY_ON_HOLD = cfg.get("telegram", {}).get("notify_on_hold", False)

BUY_THRESH = float(cfg.get("model", {}).get("buy_threshold", 0.55))
SELL_THRESH = float(cfg.get("model", {}).get("sell_threshold", 0.45))

# parameter that controls how we scale raw prediction -> 0..1
# e.g. pred_scale = 1000 will amplify returns like 0.0006 -> 0.6-ish after sigmoid/tanh mapping
PRED_SCALE = float(cfg.get("model", {}).get("pred_scale", 1000.0))

FEATURE_COLS = cfg.get("model", {}).get(
    "feature_columns",
    ["close", "ret", "ma5", "ma10", "ma20", "rsi14", "macd", "atr14"],
)

# predictions excel
PRED_DIR = cfg.get("paths", {}).get("outputs", "data")
os.makedirs(PRED_DIR, exist_ok=True)
PRED_FILE = os.path.join(PRED_DIR, "predictions.xlsx")
PRED_BATCH = int(cfg.get("model", {}).get("pred_batch", 10))


# -----------------------
# Helpers: Telegram
# -----------------------
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        log.warning("Telegram not configured (token/chat_id missing).")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(
            url, data={"chat_id": TG_CHAT_ID, "text": text}, timeout=10
        )
        if resp.status_code != 200:
            log.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
        return True
    except Exception as e:
        log.exception("Telegram send exception:")
        return False


def send_telegram_file(path, filename=None):
    if not TG_TOKEN or not TG_CHAT_ID:
        log.warning("Telegram file send skipped: not configured.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument"
        files = {"document": open(path, "rb")}
        data = {"chat_id": TG_CHAT_ID}
        r = requests.post(url, files=files, data=data, timeout=30)
        if r.status_code != 200:
            log.warning(f"Telegram file send failed: {r.status_code} {r.text}")
            return False
        return True
    except Exception:
        log.exception("Failed to send file to telegram")
        return False


# -----------------------
# Feature engineering (copied/improved)
# -----------------------
def add_technical_features(df):
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    # basic
    df["ret"] = df["close"].pct_change().fillna(0)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-10)
    df["rsi14"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# -----------------------
# Prediction recording
# -----------------------
_pred_buffer = []


def record_prediction_to_excel(symbol, pred, action, price):
    global _pred_buffer
    _pred_buffer.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "prediction": float(pred),
            "action": action,
            "price": float(price),
        }
    )
    if len(_pred_buffer) >= PRED_BATCH:
        df = pd.DataFrame(_pred_buffer)
        if os.path.exists(PRED_FILE):
            try:
                existing = pd.read_excel(PRED_FILE)
                df = pd.concat([existing, df], ignore_index=True)
            except Exception:
                pass
        df.to_excel(PRED_FILE, index=False)
        log.info(f"Saved {len(_pred_buffer)} predictions to {PRED_FILE}")
        # send file by telegram
        send_telegram_file(PRED_FILE)
        _pred_buffer = []


# -----------------------
# Prediction normalization
# -----------------------
def map_raw_pred_to_score(raw_pred, scale=PRED_SCALE):
    """
    Map raw prediction (e.g. expected return like 0.0006) to [0,1].
    Uses a scaled tanh mapping:
        score = 0.5 + 0.5 * tanh(scale * raw_pred)
    If you prefer a sigmoid, change here.
    scale controls sensitivity (tune via config.json -> model.pred_scale).
    """
    try:
        x = float(raw_pred)
    except Exception:
        x = 0.0
    s = np.tanh(scale * x)
    score = 0.5 + 0.5 * s
    return float(score)


# -----------------------
# MT5 helpers
# -----------------------
def mt5_connect_or_die():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    if ACCOUNT.get("login"):
        ok = mt5.login(
            ACCOUNT.get("login"),
            password=ACCOUNT.get("password"),
            server=ACCOUNT.get("server"),
        )
        if not ok:
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    mt5.symbol_select(SYMBOL, True)
    acc = mt5.account_info()
    log.info(
        f"Connected to MT5 account: {getattr(acc,'login','?')} | Balance: {getattr(acc,'balance','?')}"
    )
    send_telegram(
        f"✅ Connected to MT5: {ACCOUNT.get('server','?')} | Balance: {getattr(acc,'balance','n/a')}"
    )


def fetch_bars(n):
    tf_map = {
        1: mt5.TIMEFRAME_M1,
        5: mt5.TIMEFRAME_M5,
        15: mt5.TIMEFRAME_M15,
        30: mt5.TIMEFRAME_M30,
        60: mt5.TIMEFRAME_H1,
        240: mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
    }
    tf = tf_map.get(TF_MIN, mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def current_open_positions_count():
    pos = mt5.positions_get(symbol=SYMBOL)
    return 0 if pos is None else len(pos)


# -----------------------
# SL/TP price calculation (pips/points -> price)
# -----------------------


def calculate_sl_tp_from_usd(price, side, tp_usd, lot, symbol_info):
    """
    Calculates SL/TP in price terms based on desired USD profit.
    tp_usd: target profit in USD (e.g. 30)
    lot: trade size
    symbol_info: MT5 symbol info (to get tick value and point size)
    """
    if tp_usd <= 0:
        return None, None

    tick_value = getattr(symbol_info, "trade_tick_value", 1.0)
    tick_size = getattr(symbol_info, "trade_tick_size", 1e-5)
    point = getattr(symbol_info, "point", 1e-5)

    # Calculate price distance per tick in USD
    # Profit = (price_delta / tick_size) * tick_value * lot
    # => price_delta = (tp_usd / (tick_value * lot)) * tick_size
    price_delta = (tp_usd / (tick_value * lot)) * tick_size

    if side == "buy":
        tp_price = price + price_delta
        sl_price = price - price_delta / 2  # optional SL = half TP distance
    else:
        tp_price = price - price_delta
        sl_price = price + price_delta / 2

    return sl_price, tp_price


# -----------------------
# Trading order
# -----------------------
def place_order(side, prediction_score):
    # enforce max open trades
    open_cnt = current_open_positions_count()
    if open_cnt >= MAX_OPEN:
        log.info(
            f"Max open trades {MAX_OPEN} reached ({open_cnt}). Skipping new order."
        )
        return None

    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None:
        log.error("No tick/info for symbol; cannot place order.")
        return None

    price = float(tick.ask) if side == "buy" else float(tick.bid)
    typ = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

    # compute SL/TP prices from config pips
    sl_price, tp_price = calculate_sl_tp_from_usd(price, side, TP_USD, LOT, info)

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": float(LOT),
        "type": typ,
        "price": price,
        "deviation": int(DEVIATION),
        "magic": 999999,
        "comment": f"AI Trader | score={prediction_score:.4f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # include sl/tp only if computed
    #if sl_price is not None:
        #req["sl"] = float(sl_price) 
    if tp_price is not None:
        req["tp"] = float(tp_price)

    result = mt5.order_send(req)
    if result is None:
        log.error("MT5 returned None for order_send (no result).")
        return None

    # log result details
    try:
        log.info(
            f"OrderSendResult: retcode={result.retcode}, comment={getattr(result,'comment', '')}, order={getattr(result,'order', None)}, deal={getattr(result,'deal', None)}"
        )
    except Exception:
        log.exception("While logging order result")

    return result


# -----------------------
# Model loading
# -----------------------
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler file not found in paths.")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# -----------------------
# Main
# -----------------------
def main():
    # connect
    try:
        mt5_connect_or_die()
    except Exception as e:
        log.exception("MT5 connect failed")
        return

    # load model/scaler
    try:
        model, scaler = load_model_and_scaler()
        log.info("Model and scaler loaded.")
    except Exception as e:
        log.exception("Failed to load model/scaler")
        send_telegram(f"❌ Failed to load model/scaler: {e}")
        return

    prediction_buffer = []  # for optional local usage

    while True:
        try:
            # optional: retrain trigger based on last_trained file age
            if os.path.exists(LAST_TRAINED_FILE):
                try:
                    with open(LAST_TRAINED_FILE, "r") as f:
                        ts = f.read().strip()
                    last = datetime.fromisoformat(ts)
                    if (datetime.now(timezone.utc) - last) > timedelta(hours=24):
                        # non-blocking retrain trigger
                        log.info(
                            "Last trained older than 24h -> trigger retrain (non-blocking)."
                        )
                        send_telegram("🔁 Triggering daily retrain (background).")
                        try:
                            subprocess.Popen(
                                [sys.executable, "run_pipeline.py"], shell=False
                            )
                        except Exception:
                            log.exception("Failed to start retrain subprocess")
                except Exception:
                    pass

            # get market data
            bars = fetch_bars(n=max(SEQ_LEN + 50, 200))
            if bars is None or len(bars) < SEQ_LEN:
                log.info("Not enough bars, sleeping...")
                time.sleep(SLEEP)
                continue

            df = add_technical_features(bars)
            # ensure feature columns are present
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                log.error(f"Missing feature columns: {missing}. Sleep and retry.")
                time.sleep(SLEEP)
                continue

            features = df[FEATURE_COLS].values
            # scale
            try:
                scaled = scaler.transform(features)
            except Exception:
                log.exception("Scaler.transform failed")
                time.sleep(SLEEP)
                continue

            if scaled.shape[0] < SEQ_LEN:
                log.info("Not enough rows after scaling for seq_len.")
                time.sleep(SLEEP)
                continue

            X = np.expand_dims(scaled[-SEQ_LEN:], axis=0)
            raw_pred = model.predict(X, verbose=0)
            # raw_pred can be shape (1,1) or (1,) etc.
            try:
                raw_val = float(np.array(raw_pred).ravel()[0])
            except Exception:
                raw_val = float(raw_pred)

            # normalize to score [0,1]
            score = map_raw_pred_to_score(raw_val, scale=PRED_SCALE)

            action = "HOLD"
            if score >= BUY_THRESH:
                action = "BUY"
            elif score <= SELL_THRESH:
                action = "SELL"

            # log & notif (hold notifications optional)
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            msg = f"📊 {SYMBOL} | raw={raw_val:.6f} | score={score:.4f} | action={action} | time={now}"
            log.info(msg)
            if NOTIFY_ON_HOLD or action != "HOLD":
                send_telegram(msg)

            # record prediction & send excel every PRED_BATCH
            last_price = float(df["close"].iloc[-1])
            record_prediction_to_excel(SYMBOL, raw_val, action, last_price)

            # execute trade if action
            if action == "BUY":
                res = place_order("buy", score)
                if res and getattr(res, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                    send_telegram(
                        f"✅ BUY executed for {SYMBOL} @ {getattr(res,'price',last_price)}"
                    )
                else:
                    send_telegram(f"❌ BUY failed: {getattr(res,'comment',str(res))}")
            elif action == "SELL":
                res = place_order("sell", score)
                if res and getattr(res, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                    send_telegram(
                        f"✅ SELL executed for {SYMBOL} @ {getattr(res,'price',last_price)}"
                    )
                else:
                    send_telegram(f"❌ SELL failed: {getattr(res,'comment',str(res))}")

            # update trade history CSV (non-blocking)
            try:
                from_date = datetime.now() - timedelta(days=90)
                deals = mt5.history_deals_get(from_date, datetime.now())
                if deals:
                    deals_df = pd.DataFrame(list(deals))
                    # try to select common columns
                    cols = [
                        c
                        for c in [
                            "time",
                            "symbol",
                            "volume",
                            "price",
                            "profit",
                            "comment",
                        ]
                        if c in deals_df.columns
                    ]
                    if len(cols) > 0:
                        deals_df.to_csv(TRADES_FILE, index=False)
            except Exception:
                log.debug("update trade history failed (non-fatal)")

            # sleep
            time.sleep(SLEEP)

        except KeyboardInterrupt:
            log.info("User interrupted - exiting.")
            break
        except Exception as e:
            log.exception(f"Error in main loop: {e}")
            send_telegram(f"🚨 Live trader error: {e}")
            time.sleep(SLEEP)

    mt5.shutdown()


if __name__ == "__main__":
    main()
