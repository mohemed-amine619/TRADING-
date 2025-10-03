# model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import optuna
from sklearn.utils import class_weight
import numpy as np
import config

def build_lstm_model(hp, input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # --- هذا هو التعديل ---
    # أولاً، نقترح عدد الطبقات ونحفظه في متغير
    num_layers = hp.suggest_int('num_lstm_layers', 1, 2)
    
    # ثانياً، نستخدم المتغير الذي حفظناه
    for i in range(num_layers):
        model.add(layers.LSTM(
            units=hp.suggest_int(f'lstm_units_{i}', 64, 256, step=32),
            # ثالثاً، نستخدم نفس المتغير هنا أيضاً
            return_sequences=(i < num_layers - 1)
        ))
        model.add(layers.Dropout(hp.suggest_float(f'dropout_{i}', 0.2, 0.5)))
    # --- نهاية التعديل ---
        
    model.add(layers.Dense(hp.suggest_int('dense_units', 32, 128, step=32), activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    
    learning_rate = hp.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def optuna_objective(trial, X_train, y_train):
    tf.keras.backend.clear_session()
    val_split = 0.15
    split_idx = int(len(X_train) * (1 - val_split))
    X_t, X_v = X_train[:split_idx], X_train[split_idx:]
    y_t, y_v = y_train[:split_idx], y_train[split_idx:]
    
    model = build_lstm_model(trial, input_shape=(X_train.shape[1], X_train.shape[2]))
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_t), y=y_t)
    cw_dict = {i: w for i, w in enumerate(cw)}
    
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=15, batch_size=config.BATCH_SIZE, class_weight=cw_dict,
                        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=0)
    return history.history['val_accuracy'][-1]

def find_best_model(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train), n_trials=config.OPTUNA_TRIALS)
    print("Best trial found:")
    print(f"  Value: {study.best_value}")
    print("  Params: ", study.best_params)
    return study.best_params

def train_final_model(best_params, X_train, y_train, X_test, y_test):
    final_model = build_lstm_model(optuna.trial.FixedTrial(best_params), input_shape=(X_train.shape[1], X_train.shape[2]))
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {i: w for i, w in enumerate(cw)}
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
          callbacks.ModelCheckpoint(config.MODEL_FILE, save_best_only=True, monitor='val_loss')]
    final_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, class_weight=cw_dict, callbacks=cb, verbose=2)
    return final_model