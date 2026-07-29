import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


class LSTMForecaster:
    """
    Generalized LSTM Forecaster yang support:
    - Univariate (1 output): untuk prediksi CI saja atau RE saja
    - Multivariate (2+ outputs): untuk prediksi CI dan RE secara bersamaan
    - Configurable layers & hyperparameters (untuk Optuna tuning)
    """

    def __init__(self, sequence_length, n_features, n_outputs=1,
                 model_dir="saved_models/dl_weights/"):
        """
        Parameters
        ----------
        sequence_length : int
            Jumlah timestep input (look_back). Default 24.
        n_features : int
            Jumlah fitur input per timestep.
        n_outputs : int
            Jumlah output (1 untuk univariate, 2 untuk multivariate CI+RE).
        model_dir : str
            Direktori untuk menyimpan model weights.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model_dir = model_dir
        self.model = None

        os.makedirs(self.model_dir, exist_ok=True)

    def build_model(self, units=64, n_layers=1, dropout_rate=0.2,
                    learning_rate=0.001, dense_units=32):
        """
        Membangun arsitektur LSTM.

        Parameters
        ----------
        units : int
            Jumlah unit LSTM per layer. Default 64.
        n_layers : int
            Jumlah LSTM layer. Default 1.
        dropout_rate : float
            Dropout rate setelah tiap LSTM layer. Default 0.2.
        learning_rate : float
            Learning rate untuk Adam optimizer. Default 0.001.
        dense_units : int
            Jumlah unit Dense layer sebelum output. Default 32.

        Returns
        -------
        tf.keras.Model
        """
        layers = [Input(shape=(self.sequence_length, self.n_features))]

        # LSTM layers
        for i in range(n_layers):
            # return_sequences=True untuk semua layer kecuali yang terakhir
            return_seq = (i < n_layers - 1)
            layers.append(LSTM(units, return_sequences=return_seq))
            layers.append(Dropout(dropout_rate))

        # Dense layers
        layers.append(Dense(dense_units, activation='relu'))
        layers.append(Dense(self.n_outputs))

        self.model = Sequential(layers)

        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        print(f"\n{'='*60}")
        print(f"Model built: {n_layers} LSTM layer(s), {units} units, "
              f"{self.n_outputs} output(s)")
        print(f"{'='*60}")
        self.model.summary()
        return self.model

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, patience=15,
              model_name="lstm_best.keras", verbose=1):
        """
        Training loop dengan EarlyStopping dan ModelCheckpoint.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Data training.
        X_val, y_val : np.ndarray
            Data validasi (test set).
        epochs : int
            Maksimum epoch. Default 100.
        batch_size : int
            Batch size. Default 32.
        patience : int
            EarlyStopping patience. Default 15.
        model_name : str
            Nama file untuk menyimpan model terbaik.
        verbose : int
            Verbosity level. Default 1.

        Returns
        -------
        tf.keras.callbacks.History
        """
        if self.model is None:
            raise ValueError("Model belum di-build! Panggil build_model() dulu.")

        filepath = os.path.join(self.model_dir, model_name)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        print(f"\nTraining... Model terbaik akan disimpan di: {filepath}")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=verbose
        )
        return history

    def predict(self, X_test):
        """Inference/forecasting."""
        if self.model is None:
            raise ValueError("Model tidak ditemukan! Build atau load dulu.")

        return self.model.predict(X_test)

    def load_pretrained_model(self, model_name="lstm_best.keras"):
        """Load model yang sudah di-train."""
        filepath = os.path.join(self.model_dir, model_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} tidak ditemukan!")

        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded dari {filepath}")