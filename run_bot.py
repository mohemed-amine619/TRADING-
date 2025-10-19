import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import time
import os
import json
import requests  # Importez la bibliothèque pour les requêtes HTTP

# --- 1. CHARGER LA CONFIGURATION ---
print("Chargement du fichier config.json...")
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERREUR : Le fichier config.json est introuvable.")
    quit()

# Configuration MT5
MT5_CONFIG = CONFIG["mt5_config"]
SYMBOL = MT5_CONFIG["symbol"]
TIMEFRAME_STRING = MT5_CONFIG["timeframe_string"]
TIMEFRAME_SECONDS = MT5_CONFIG["timeframe_seconds"]
DATA_FETCH_COUNT = MT5_CONFIG["data_fetch_count"]

# Mapping des timeframes
TIMEFRAME_MAP = {
    "TIMEFRAME_H1": mt5.TIMEFRAME_H1,
    "TIMEFRAME_M5": mt5.TIMEFRAME_M5,
    "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
    "TIMEFRAME_M30": mt5.TIMEFRAME_M30,
    "TIMEFRAME_H4": mt5.TIMEFRAME_H4,
    "TIMEFRAME_D1": mt5.TIMEFRAME_D1,
}
TIMEFRAME = TIMEFRAME_MAP.get(TIMEFRAME_STRING)

# Configuration Modèle
MODEL_CONFIG = CONFIG["model_config"]
SEQUENCE_LENGTH = MODEL_CONFIG["sequence_length"]
MODEL_PATH = MODEL_CONFIG["model_path"]
SCALER_PATH = MODEL_CONFIG["scaler_path"]
FEATURES_PATH = MODEL_CONFIG["features_path"]

# Configuration Bot
BOT_CONFIG = CONFIG["bot_config"]
CONFIDENCE_THRESHOLD = BOT_CONFIG["confidence_threshold"]

# NOUVELLE Configuration Telegram
TELEGRAM_CONFIG = CONFIG["telegram_config"]
BOT_TOKEN = TELEGRAM_CONFIG["bot_token"]
CHAT_ID = TELEGRAM_CONFIG["chat_id"]

# --- 2. CHARGER LE MODÈLE PROFESSIONNEL ET LE SCALER ---
print(f"Chargement du modèle : {MODEL_PATH}")
model = load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(FEATURES_PATH, "r") as f:
    feature_cols = [line.strip() for line in f]
num_features = len(feature_cols)
print(f"Modèle, scaler et {num_features} features chargés.")


# --- 3. NOUVELLE FONCTION D'ENVOI TELEGRAM ---
def send_telegram_signal(symbol, timeframe, signal, confidence, current_price):
    """
    Envoie un message formaté à votre bot Telegram.
    """
    # Emojis pour les signaux
    signal_emoji = "📈" if signal == "BUY" else "📉"

    # Formater le message en Markdown
    message_text = (
        f"🚨 *Nouveau Signal de Trading* 🚨\n\n"
        f"{signal_emoji} *Symbole:* `{symbol}`\n"
        f"🕒 *Timeframe:* `{timeframe}`\n\n"
        f"➡️ *Signal:* **{signal}**\n"
        f"📊 *Confiance:* `{confidence:.2f}%`\n"
        f"💲 *Prix Actuel:* `{current_price:.2f}`"
    )

    # Construire l'URL de l'API Telegram
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # Définir le payload
    payload = {"chat_id": CHAT_ID, "text": message_text, "parse_mode": "Markdown"}

    # Envoyer la requête
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print(f"Signal {signal} envoyé à Telegram avec succès.")
        else:
            print(f"Erreur lors de l'envoi à Telegram : {response.text}")
    except Exception as e:
        print(f"Exception lors de la connexion à Telegram : {e}")


# --- 4. BOUCLE PRINCIPALE DU BOT ---
print("Connexion à MT5...")
if not mt5.initialize():
    print("initialize() a échoué, code d'erreur =", mt5.last_error())
    quit()
print("Bot connecté à MT5. En attente de signaux...")

while True:
    try:
        # --- 5. Obtenir les dernières données ---
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, DATA_FETCH_COUNT)

        if rates is None or len(rates) < DATA_FETCH_COUNT:
            print("Impossible d'obtenir assez de données. Nouvel essai...")
            time.sleep(60)
            continue

        # --- 6. PRÉTRAITEMENT DES DONNÉES LIVE ---
        df = pd.DataFrame(rates)
        df = df[["open", "high", "low", "close", "tick_volume"]]
        df = df.astype(float)

        # Calcul des indicateurs
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, append=True)

        df = df.dropna()

        if len(df) < SEQUENCE_LENGTH:
            print("Pas assez de données après calcul des indicateurs. En attente...")
            time.sleep(TIMEFRAME_SECONDS)
            continue

        last_bars = df.tail(SEQUENCE_LENGTH)

        # Obtenir le prix actuel pour le message
        current_price = last_bars["close"].iloc[-1]

        X_live_df = last_bars[feature_cols]

        # --- 7. Mise à l'échelle et remodelage ---
        scaled_data = scaler.transform(X_live_df.values)
        X_live = np.array([scaled_data])
        X_live = np.reshape(X_live, (1, SEQUENCE_LENGTH, num_features))

        # --- 8. Faire la prédiction ---
        prediction_probs = model.predict(X_live)[0]
        predicted_class_index = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_class_index]

        # --- 9. Générer le signal ---
        signal = "HOLD"  # Par défaut
        if predicted_class_index == 1:  # Index 1 = BUY
            signal = "BUY"
        elif predicted_class_index == 2:  # Index 2 = SELL
            signal = "SELL"

        print(f"Prédiction : {signal} (Confiance : {confidence:.2f})")

        # --- 10. ENVOYER LE SIGNAL À TELEGRAM ---
        # Nous n'envoyons que les signaux d'action (pas HOLD)
        # et seulement si la confiance dépasse notre seuil

        if signal != "HOLD" and confidence >= CONFIDENCE_THRESHOLD:
            send_telegram_signal(
                SYMBOL, TIMEFRAME_STRING, signal, confidence * 100, current_price
            )
        elif signal == "HOLD":
            print("Signal: HOLD. Pas de message envoyé.")
        else:
            print(
                f"Signal: {signal} mais confiance ({confidence:.2f}) < Seuil. Pas de message envoyé."
            )

        # --- 11. Attendre la prochaine bougie ---
        print(f"En attente de {TIMEFRAME_SECONDS} secondes pour la prochaine bougie...")
        time.sleep(TIMEFRAME_SECONDS)

    except Exception as e:
        print(f"Une erreur est survenue dans la boucle principale : {e}")
        mt5.shutdown()
        time.sleep(60)
        if not mt5.initialize():
            print("Échec de la ré-initialisation de MT5. Arrêt.")
            break
