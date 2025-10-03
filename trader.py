# trader.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import joblib
import tensorflow as tf
import logging
import threading
from telegram.ext import Updater, CommandHandler
import config
import data_handler
from notifications import TelegramNotifier

trading_active = True
notifier = TelegramNotifier()

def start_command(update, context):
    global trading_active
    trading_active = True
    welcome_message = (
        "👋 أهلاً بك!\n"
        "بوت التداول متصل وجاهز للعمل.\n\n"
        "✅ تم تفعيل التداول الحي.\n\n"
        "أرسل الأوامر التالية للتحكم:\n"
        "🔹 /status - للتحقق من حالة الحساب.\n"
        "🔹 /stop - لإيقاف التداول مؤقتاً."
    )
    update.message.reply_text(welcome_message)
    notifier.send_message("▶️ *Live Trading Resumed via Telegram Command*")

def stop_command(update, context):
    global trading_active
    trading_active = False
    update.message.reply_text("⏹️ Live trading has been PAUSED.")
    notifier.send_message("⏹️ *Live Trading Paused*")

def status_command(update, context):
    try:
        data_handler.mt5_connect()
        acc_info = mt5.account_info()
        mt5.shutdown()
        status_msg = (f"📊 *Bot Status*\n"
                      f"*Trading Active:* {'Yes' if trading_active else 'No'}\n"
                      f"*Balance:* {acc_info.balance:.2f}\n*Equity:* {acc_info.equity:.2f}")
    except Exception as e:
        status_msg = f"Error fetching status: {e}"
    update.message.reply_text(status_msg, parse_mode='Markdown')

def run_telegram_bot():
    token = getattr(config, 'TELEGRAM_TOKEN', None)
    if not token:
        logging.warning("Telegram token not found, remote control is disabled.")
        return
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("stop", stop_command))
    dp.add_handler(CommandHandler("status", status_command))
    logging.info("Telegram command listener started.")
    updater.start_polling()
    updater.idle()

def run_live_trading():
    global trading_active
    logging.info("--- Starting Live Trading Loop ---")
    telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    telegram_thread.start()
    
    model = tf.keras.models.load_model(config.MODEL_FILE)
    scaler = joblib.load(config.SCALER_FILE)
    
    try:
        while True:
            if not trading_active:
                time.sleep(10)
                continue
            
            data_handler.mt5_connect()
            rates_df = data_handler.fetch_data()
            features_df = data_handler.add_features(rates_df)
            
            feature_cols = [c for c in features_df.columns if c not in ['time', 'open', 'high', 'low', 'tick_volume', 'spread', 'real_volume']]
            last_sequence_unscaled = features_df.iloc[-config.SEQ_LEN:][feature_cols]
            last_sequence_scaled = scaler.transform(last_sequence_unscaled.values)
            X_live = last_sequence_scaled.reshape(1, config.SEQ_LEN, last_sequence_scaled.shape[1])
            
            pred_prob = model.predict(X_live)[0]
            signal = np.argmax(pred_prob) # 0=HOLD, 1=BUY, 2=SELL
            
            if signal in [1, 2]: # If BUY or SELL
                logging.info(f"{'BUY' if signal==1 else 'SELL'} signal triggered with prob {pred_prob[signal]:.2f}")
                notifier.send_message(f"{'✅ BUY' if signal==1 else '🔻 SELL'} Signal\nProb: {pred_prob[signal]:.2f}")
                # Add trade execution logic here
            else:
                logging.info(f"HOLD signal. Prob: {pred_prob[0]:.2f}")

            mt5.shutdown()
            logging.info(f"Cycle finished. Sleeping for {config.SLEEP_SECONDS} seconds.")
            time.sleep(config.SLEEP_SECONDS)
            
    except KeyboardInterrupt:
        logging.warning("Live trading stopped by user.")
    finally:
        trading_active = False
        notifier.send_message("🛑 *Live Trading Terminated*")