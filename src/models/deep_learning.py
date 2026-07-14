import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class MultivariateLSTMForecaster:
    """
    membungkus model Multivariate LSTM untuk peramalan Intensitas Karbon.
    """
    
    def __init__(self, sequence_length, n_features, model_dir="saved_models/dl_weights/"):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_dir = model_dir
        self.model = None

        # Memastikan direktori penyimpanan model sudah ada
        os.makedirs(self.model_dir, exist_ok=True)

    def build_model(self, units=64, dropout_rate=0.2, learning_rate=0.001):
        """
        Membangun arsitektur LSTM. 
        Nilai default bisa di-override dengan hasil terbaik dari Optuna.
        """
        self.model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dense(1) # Prediksi 1 langkah ke depan (Carbon Intensity atau Residualnya)
        ])

        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("Arsitektur model berhasil di-build.")
        self.model.summary()
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_name="optuna_best_lstm.keras"):
        """
        Training loop dengan EarlyStopping dan ModelCheckpoint.
        """
        if self.model is None:
            raise ValueError("Model belum di-build, Panggil build_model() dulu.")

        filepath = os.path.join(self.model_dir, model_name)

        # Callback 1: Stop training kalau val_loss ngga turun-turun (hindari overfitting)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        # Callback 2: Simpan HANYA bobot model yang punya val_loss paling kecil
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False, # Simpan full architecture & weights
            verbose=1
        )

        print(f"\nTraining... Bobot terbaik bakal disimpan di: {filepath}")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        return history

    def predict(self, X_test):
        """
        Fase inference/forecasting pakai data testing.
        """
        if self.model is None:
            raise ValueError("model tidak ditemukan! Build atau load_pretrained_model() dulu.")
        
        print("Melakukan prediksi...")
        return self.model.predict(X_test)

    def load_pretrained_model(self, model_name="optuna_best_lstm.keras"):
        """
        Fungsi buat manggil model yang udah mateng (buat dipake di production/API).
        """
        filepath = os.path.join(self.model_dir, model_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} tidak ditemukan!")
            
        self.model = tf.keras.models.load_model(filepath)
        print(f"Sukses nge-load model dari {filepath}")