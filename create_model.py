import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Conv1D,
    MaxPooling1D,
    Flatten,
    TimeDistributed,
)
from tensorflow.keras.utils import to_categorical
import pickle
import json  # Importer json pour lire la config

# --- 1. CHARGER LA CONFIGURATION ---
print("Chargement du fichier config.json...")
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERREUR : Le fichier config.json est introuvable.")
    quit()

# Lire les sections de la config
MT5_CONFIG = CONFIG["mt5_config"]
MODEL_CONFIG = CONFIG["model_config"]

# Assigner les variables
SYMBOL = MT5_CONFIG["symbol"]
TIMEFRAME_STRING = MT5_CONFIG["timeframe_string"]
DATA_FETCH_COUNT = 10000  # Nous avons besoin de beaucoup de données pour l'entraînement

MODEL_PATH = MODEL_CONFIG["model_path"]
SCALER_PATH = MODEL_CONFIG["scaler_path"]
FEATURES_PATH = MODEL_CONFIG["features_path"]
SEQUENCE_LENGTH = MODEL_CONFIG["sequence_length"]

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

# --- 2. CONNEXION À MT5 ET RÉCUPÉRATION DES DONNÉES ---
print(f"Connexion à MT5 pour télécharger {DATA_FETCH_COUNT} barres...")
if not mt5.initialize():
    print("initialize() a échoué, code d'erreur =", mt5.last_error())
    quit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, DATA_FETCH_COUNT)
mt5.shutdown()
print("Téléchargement des données terminé. Déconnexion de MT5.")

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.set_index("time")
df = df[["open", "high", "low", "close", "tick_volume"]]
df = df.astype(float)

# --- 3. FEATURE ENGINEERING (INDICATEURS TECHNIQUES) ---
print("Calcul des indicateurs techniques (RSI, MACD, BBands)...")
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, append=True)  # Bollinger Bands

# Nettoyer les données : supprimer les lignes avec NaN (créées par les indicateurs)
df = df.dropna()

# --- 4. CRÉATION DE LA CIBLE DE CLASSIFICATION (y) ---
# 0 = HOLD, 1 = BUY, 2 = SELL
future_shift = -5  # Regarder 5 barres dans le futur
pct_threshold = 0.005  # Seuil de 0.5% de changement

# Calculer le prix futur et le % de changement
df["future_price"] = df["close"].shift(future_shift)
df["pct_change"] = (df["future_price"] - df["close"]) / df["close"]

# Créer le signal (cible y)
df["target"] = 0  # Défaut: HOLD
df.loc[df["pct_change"] > pct_threshold, "target"] = 1  # BUY
df.loc[df["pct_change"] < -pct_threshold, "target"] = 2  # SELL

# Supprimer les lignes NaN créées par le décalage futur
df = df.dropna()

# --- 5. PRÉPARATION DES DONNÉES POUR L'IA ---
print("Préparation des données pour l'entraînement...")

# Nos features (X) sont toutes les colonnes SAUF celles cibles
feature_cols = [
    col for col in df.columns if col not in ["target", "future_price", "pct_change"]
]
X_data = df[feature_cols].values
y_data = df["target"].values

# Mettre à l'échelle les features (X)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data)

# Encoder la cible (y) en "one-hot"
# 0 -> [1, 0, 0] (HOLD)
# 1 -> [0, 1, 0] (BUY)
# 2 -> [0, 0, 1] (SELL)
y_categorical = to_categorical(y_data, num_classes=3)

# --- 6. CRÉATION DES SÉQUENCES ---
num_features = len(feature_cols)

X_seq, y_seq = [], []
for i in range(SEQUENCE_LENGTH, len(X_scaled)):
    X_seq.append(X_scaled[i - SEQUENCE_LENGTH : i, :])
    y_seq.append(y_categorical[i, :])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

print(f"Données prêtes : {X_seq.shape[0]} séquences créées.")

# --- 7. CONSTRUCTION DU MODÈLE CNN-LSTM ---
# --- 7. CONSTRUCTION DU MODÈLE CNN-LSTM ---
print(f"Construction du modèle CNN-LSTM... Shape d'entrée : {X_seq.shape[1:]}")

model = Sequential()

# Couche CNN pour trouver les motifs (SANS TimeDistributed)
# Elle prend directement en entrée la forme (60, 14)
# La UserWarning est normale, nous la corrigeons en définissant input_shape ici
model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(SEQUENCE_LENGTH, num_features),
    )
)

model.add(MaxPooling1D(pool_size=2))

# Nous n'utilisons PAS Flatten ici, car l'LSTM a besoin d'une séquence

# Couche LSTM pour comprendre la séquence de motifs
# (elle reçoit la séquence de la couche MaxPooling)
model.add(LSTM(units=100, return_sequences=False))

# Une fois que l'LSTM a traité la séquence, la sortie est "plate"
# Nous pouvons donc ajouter les couches Dense
model.add(Dense(units=50, activation="relu"))

# Couche de sortie finale
model.add(Dense(units=3, activation="softmax"))  # 3 neurones (HOLD, BUY, SELL)

# Compilation pour la classification
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- 8. ENTRAÎNEMENT DU MODÈLE ---
print("Début de l'entraînement... (Cela peut prendre plusieurs minutes)")
model.fit(X_seq, y_seq, batch_size=32, epochs=50, validation_split=0.2)
print("Entraînement terminé !")

# --- 9. SAUVEGARDE DU MODÈLE, DU SCALER ET DES FEATURES ---
model.save(MODEL_PATH)
print(f"Modèle sauvegardé sous : {MODEL_PATH}")

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler sauvegardé sous : {SCALER_PATH}")

# Sauvegarder la liste des features, dans l'ordre
with open(FEATURES_PATH, "w") as f:
    for col in feature_cols:
        f.write(f"{col}\n")
print(f"Liste des features sauvegardée sous : {FEATURES_PATH}")

print("\n--- Processus d'entraînement terminé avec succès ! ---")
