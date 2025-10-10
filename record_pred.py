import os
import json
import pandas as pd
from datetime import datetime
import requests

# -----------------------
# Load config.json
# -----------------------
CFG_FILE = "config.json"
if not os.path.exists(CFG_FILE):
    # Create a dummy config file if it doesn't exist for testing purposes
    with open(CFG_FILE, "w") as f:
        json.dump({
            "telegram": {
                "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
                "chat_id": "YOUR_TELEGRAM_CHAT_ID"
            }
        }, f, indent=4)
    print(f"{CFG_FILE} not found. A dummy file has been created. Please edit it with your credentials.")

with open(CFG_FILE, "r") as f:
    cfg = json.load(f)

TG_TOKEN = cfg.get("telegram", {}).get("bot_token")
TG_CHAT_ID = cfg.get("telegram", {}).get("chat_id")

if not TG_TOKEN or not TG_CHAT_ID or TG_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
    print("Warning: Telegram bot token or chat ID not configured in config.json. File sending will be disabled.")
    TG_TOKEN = None # Disable telegram functionality

# -----------------------
# Logs / prediction file
# -----------------------
LOGS_DIR = "logs"
DATA_DIR = "data"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


LOG_FILE = os.path.join(LOGS_DIR, "bot.log")
PRED_FILE = os.path.join(DATA_DIR, "predictions.xlsx")

# Counter
prediction_counter = 0

# -----------------------
# Functions
# -----------------------
def send_file_telegram(file_path, caption=""):
    """Send a file to Telegram."""
    if not TG_TOKEN:
        print("Telegram not configured. Skipping file send.")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument"
        with open(file_path, "rb") as f:
            response = requests.post(url, files={"document": f}, data={"chat_id": TG_CHAT_ID, "caption": caption})
            if response.status_code == 200:
                print(f"Successfully sent {os.path.basename(file_path)} to Telegram.")
            else:
                print(f"Error sending {os.path.basename(file_path)} to Telegram: {response.text}")
    except Exception as e:
        print(f"Error sending file to Telegram: {e}")

def record_prediction(symbol, prediction, action):
    """Record prediction to Excel and log file."""
    global prediction_counter

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Append to log file
    try:
        with open(LOG_FILE, "a") as log_f:
            log_f.write(f"{now} | {symbol} | {prediction:.4f} | {action}\n")
        print(f"Recorded prediction for {symbol} to {LOG_FILE}")
    except IOError as e:
        print(f"Error writing to log file {LOG_FILE}: {e}")
        return

    # Append to Excel
    try:
        if os.path.exists(PRED_FILE):
            df = pd.read_excel(PRED_FILE)
        else:
            df = pd.DataFrame(columns=["time", "symbol", "prediction", "action"])

        new_record = pd.DataFrame([{
            "time": now,
            "symbol": symbol,
            "prediction": prediction,
            "action": action
        }])
        
        df = pd.concat([df, new_record], ignore_index=True)

        df.to_excel(PRED_FILE, index=False)
        print(f"Saved predictions to {PRED_FILE}")

    except Exception as e:
        print(f"Error writing to Excel file {PRED_FILE}: {e}")
        return

    # Increment counter and send files every 10 predictions
    prediction_counter += 1
    if prediction_counter >= 10:
        print("10 predictions recorded. Exporting and sending files to Telegram...")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_pred_file = os.path.join(DATA_DIR, f"predictions_{timestamp}.xlsx")

        # Rename the current PRED_FILE to the unique name before sending
        if os.path.exists(PRED_FILE):
            try:
                os.rename(PRED_FILE, unique_pred_file)
                print(f"Renamed {PRED_FILE} to {unique_pred_file}")
                send_file_telegram(unique_pred_file, caption=f"📈 Predictions batch ({timestamp})")
            except OSError as e:
                print(f"Error renaming and sending prediction file: {e}")
        else:
            print(f"Prediction file {PRED_FILE} not found. Cannot send.")

        send_file_telegram(LOG_FILE, caption="📝 Bot log")
        prediction_counter = 0 # Reset counter

def clear_predictions():
    """Deletes the main prediction and log files if they exist."""
    print("Attempting to clear prediction files...")
    if os.path.exists(PRED_FILE):
        try:
            os.remove(PRED_FILE)
            print(f"Successfully deleted {PRED_FILE}")
        except OSError as e:
            print(f"Error deleting file {PRED_FILE}: {e}")
    else:
        print(f"{PRED_FILE} not found. Nothing to delete.")

    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            print(f"Successfully deleted {LOG_FILE}")
        except OSError as e:
            print(f"Error deleting file {LOG_FILE}: {e}")
    else:
        print(f"{LOG_FILE} not found. Nothing to delete.")

# Example usage:
if __name__ == '__main__':
    print("--- Running example usage ---")

    # Record 10 predictions to trigger the file sending
    print("\n1. Recording 10 sample predictions to trigger export...")
    for i in range(10):
        record_prediction("BTC/USD", 65000.5678 + i, "BUY")
        
    print("\n2. Recording another prediction to start a new file...")
    record_prediction("ETH/USD", 3500.1234, "SELL")
    
    # Clear the files
    print("\n3. Clearing the current (not archived) prediction files...")
    clear_predictions()

    print("\n--- Example usage finished ---")
