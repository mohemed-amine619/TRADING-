# main.py
import os
import logging
from notifications import TelegramNotifier
import config
import data_handler
import model_builder
import backtester
import trader

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("logs/bot.log", encoding='utf-8'), logging.StreamHandler()])

notifier = TelegramNotifier()

def main():
    try:
        logging.info("==============================================")
        logging.info("🚀 Bot starting up...")
        notifier.send_message("🚀 *Bot Starting Up*")

        logging.info("Step 1: Fetching and preparing data...")
        raw_df = data_handler.fetch_data()
        X_train, y_train, X_test, y_test, scaler, feature_cols, split_idx = data_handler.prepare_data(raw_df)
        logging.info(f"Data prepared. Train shape: {X_train.shape}")

        if getattr(config, 'TUNE_HYPERPARAMS', True):
            logging.info("Step 2: Finding best model hyperparameters...")
            best_params = model_builder.find_best_model(X_train, y_train)
        else:
            logging.warning("Step 2: Hyperparameter tuning disabled.")
            best_params = {'num_lstm_layers': 1, 'lstm_units_0': 128, 'dropout_0': 0.3, 'dense_units': 64, 'learning_rate': 0.001}

        logging.info("Step 3: Training final model...")
        model = model_builder.train_final_model(best_params, X_train, y_train, X_test, y_test)
        
        logging.info("Step 4: Running backtest...")
        backtester.run_backtest(model, X_test, y_test, raw_df, split_idx)
        
        if getattr(config, 'RUN_LIVE', False):
            logging.info("Step 5: Handing over to live trading...")
            notifier.send_message("✅ *Setup Complete*. Starting live trading.")
            trader.run_live_trading()
        else:
            notifier.send_message("✅ *Process Finished*. Live trading disabled.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        notifier.send_message(f"🚨 *FATAL ERROR*: {e}")
    finally:
        logging.info("🛑 Bot shutting down.")
        logging.info("==============================================\n")

if __name__ == "__main__":
    main()