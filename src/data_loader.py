import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import MSTL


class TimeSeriesDataLoader:
    """
    Data loader untuk pipeline LSTM forecasting Carbon Intensity & Renewable Energy.:
    - Support Univariate (CI saja / RE saja) dan Multivariate (CI + RE)
    - MSTL Decomposition (daily period=24, weekly period=168)
    - PACF-based lag features
    - Multiseasonality dummies (H1-H24, B1-B12, M1H1-M7H24)
    """

    # Column names sesuai dataset cleaned
    CI_COL = 'carbon_intensity'
    RE_COL = 'renewable_percentage'

    def __init__(self):
       self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

    # DATA LOADING
    def load_processed_data(self, filepath):
        """
        Load dataset hasil preprocessing.
        
        Dataset diasumsikan sudah:
        - digabungkan 2024–2025
        - timestamp sudah valid
        - duplicate timestamp sudah ditangani
        - invalid values sudah ditangani
        - anomaly/outlier sudah dikoreksi
        - missing values sudah diimputasi
        """
        print(f"Loading processed data from {filepath}...")

        df = pd.read_csv(filepath)

        df.columns = df.columns.str.strip()

        required_cols = [
            'datetime',
            self.CI_COL,
            self.RE_COL
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        df['datetime'] = pd.to_datetime(
            df['datetime'],
            utc=True
        )

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        df = df[
            [self.CI_COL, self.RE_COL]
        ].copy()

        df[self.CI_COL] = pd.to_numeric(
            df[self.CI_COL],
            errors='coerce'
        )

        df[self.RE_COL] = pd.to_numeric(
            df[self.RE_COL],
            errors='coerce'
        )

        print(f"Data loaded. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Range: {df.index.min()} → {df.index.max()}")
        print(f"Missing values:\n{df.isna().sum()}")

        return df

    # =========================================================================
    # 2. FEATURE ENGINEERING
    # =========================================================================

    def add_mstl_decomposition(self, df, periods=(24, 168)):
        """
        Menguraikan Yt = Trend + Seasonal_24 + Seasonal_168 + Residual
        menggunakan MSTL (Multi-Seasonal-Trend decomposition using LOESS).
        
        Sesuai arahan Bu Regita:
        - Daily: period=24
        - Weekly: period=168 (7×24)
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame dengan CI dan/atau RE columns.
        periods : tuple
            Periods untuk MSTL decomposition. Default (24, 168).
            
        Returns
        -------
        pd.DataFrame
            DataFrame dengan kolom tambahan: 
            {col}_trend, {col}_seasonal_24, {col}_seasonal_168, {col}_resid
        """
        df_mstl = df.copy()

        for col, prefix in [(self.CI_COL, 'CI'), (self.RE_COL, 'RE')]:
            if col not in df_mstl.columns:
                continue

            print(f"  MSTL decomposition untuk {col} (periods={periods})...")
            mstl_result = MSTL(df_mstl[col], periods=periods).fit()

            df_mstl[f'{prefix}_trend'] = mstl_result.trend
            df_mstl[f'{prefix}_resid'] = mstl_result.resid

            # Seasonal components — satu per period
            seasonal_df = mstl_result.seasonal
            for i, period in enumerate(periods):
                # MSTL returns seasonal columns named 'seasonal_{period}'
                seasonal_col_name = f'seasonal_{period}'
                if seasonal_col_name in seasonal_df.columns:
                    df_mstl[f'{prefix}_seasonal_{period}'] = seasonal_df[seasonal_col_name]
                else:
                    # Fallback: ambil by index
                    df_mstl[f'{prefix}_seasonal_{period}'] = seasonal_df.iloc[:, i]

        print(f"  MSTL selesai. Shape: {df_mstl.shape}")
        return df_mstl

    def add_pacf_lags(self, df, target_col, lags, prefix):
        """
        Menambahkan input lag berdasarkan hasil PACF.
        
        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
            Nama kolom target (e.g., 'carbon_intensity')
        lags : list of int
            Daftar lag yang signifikan dari PACF analysis.
        prefix : str
            Prefix untuk nama kolom lag (e.g., 'CI' atau 'RE')
            
        Returns
        -------
        pd.DataFrame
            DataFrame dengan kolom lag tambahan.
        """
        df_lags = df.copy()
        for lag in lags:
            df_lags[f'{prefix}_lag_{lag}'] = df_lags[target_col].shift(lag)
        # JANGAN dropna di sini — biar dihandle setelah semua fitur ditambahkan
        return df_lags

    def add_time_features(self, df):
        """
        Menambahkan fitur waktu berbasis cyclical encoding (sin/cos).
        Ini lebih efisien untuk LSTM daripada one-hot dummy yang banyak.
        
        Features: hour, dayofweek, month — masing-masing di-encode jadi sin & cos.
        
        Returns
        -------
        pd.DataFrame
            DataFrame dengan 6 kolom tambahan.
        """
        df_time = df.copy()
        
        # Hour of day (0-23) → sin/cos
        hour = df_time.index.hour
        df_time['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6) → sin/cos
        dow = df_time.index.dayofweek
        df_time['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df_time['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Month (1-12) → sin/cos
        month = df_time.index.month
        df_time['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        df_time['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        
        return df_time

    def add_seasonal_dummies(self, df, daily=True, monthly=True, 
                              weekly_hour=True, drop_first=True):
        """
        Membuat Multiseasonality Dummies sesuai instruksi Bu Regita.
        
        Parameters
        ----------
        daily : bool
            Dummy H1-H24 (hour of day)
        monthly : bool
            Dummy B1-B12 (month)
        weekly_hour : bool
            Dummy M1H1-M7H24 (weekday × hour interaction)
        drop_first : bool
            Drop kolom pertama tiap grup dummy untuk menghindari 
            multikolinearitas sempurna.
        """
        df_dummy = df.copy()

        if daily:
            hour_of_day = df_dummy.index.hour + 1  # H1-H24
            dummies_h = pd.get_dummies(hour_of_day, prefix="H", drop_first=drop_first).astype(int)
            dummies_h.index = df_dummy.index
            df_dummy = pd.concat([df_dummy, dummies_h], axis=1)

        if monthly:
            month = df_dummy.index.month  # B1-B12
            dummies_b = pd.get_dummies(month, prefix="B", drop_first=drop_first).astype(int)
            dummies_b.index = df_dummy.index
            df_dummy = pd.concat([df_dummy, dummies_b], axis=1)

        if weekly_hour:
            weekday = df_dummy.index.weekday + 1  # 1-7
            hour_of_day = df_dummy.index.hour + 1  # 1-24
            interaction = "M" + weekday.astype(str) + "H" + hour_of_day.astype(str)
            dummies_w = pd.get_dummies(interaction, drop_first=drop_first).astype(int)
            dummies_w.index = df_dummy.index
            df_dummy = pd.concat([df_dummy, dummies_w], axis=1)

        print(f"  Seasonal dummies added. Total columns: {len(df_dummy.columns)}")
        return df_dummy

    # SPLIT, SCALE, WINDOWING
    def split_and_scale(
        self,
        df,
        feature_cols,
        target_cols,
        train_ratio=0.8
    ):
        """
        Temporal split dan Min-Max normalization.

        Scaler hanya di-fit menggunakan training data
        untuk mencegah data leakage.
        """

        train_size = int(len(df) * train_ratio)

        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # ---------------------------------------------------------
        # Fit scaler ONLY on training data
        # ---------------------------------------------------------
        self.feature_scaler.fit(
            train_df[feature_cols]
        )

        # ---------------------------------------------------------
        # Transform train & test using same scaler
        # ---------------------------------------------------------
        train_scaled = self.feature_scaler.transform(
            train_df[feature_cols]
        )

        test_scaled = self.feature_scaler.transform(
            test_df[feature_cols]
        )

        scaled_train_df = pd.DataFrame(
            train_scaled,
            columns=feature_cols,
            index=train_df.index
        )

        scaled_test_df = pd.DataFrame(
            test_scaled,
            columns=feature_cols,
            index=test_df.index
        )

        print(
            f"Split: "
            f"Train={len(scaled_train_df)}, "
            f"Test={len(scaled_test_df)}"
        )

        print(
            f"Features scaled: {len(feature_cols)}"
        )

        return (
            scaled_train_df,
            scaled_test_df
        )

    def create_sliding_window(self, scaled_df, target_cols, look_back=24, forecast_horizon=1):
        """
        Membuat sliding window untuk LSTM.
        
        Parameters
        ----------
        scaled_df : pd.DataFrame
            Data yang sudah di-scale.
        target_cols : list of str
            Target columns. Len=1 untuk univariate, len=2 untuk multivariate.
        look_back : int
            Jumlah timestep ke belakang. Default 24 (1 hari).
        forecast_horizon : int
            Jumlah timestep ke depan untuk prediksi. Default 1 (one-step ahead).
            
        Returns
        -------
        tuple: (X: np.ndarray, y: np.ndarray)
            X shape: (samples, look_back, n_features)
            y shape: (samples, forecast_horizon) untuk univariate
                     (samples, forecast_horizon, n_targets) untuk multivariate
        """
        X, y = [], []
        data = scaled_df.values
        target_indices = [scaled_df.columns.get_loc(col) for col in target_cols]

        for i in range(len(data) - look_back - forecast_horizon + 1):
            X.append(data[i:(i + look_back), :])

            if len(target_indices) == 1:
                # Univariate: y shape = (forecast_horizon,)
                y.append(data[(i + look_back):(i + look_back + forecast_horizon), target_indices[0]])
            else:
                # Multivariate: y shape = (forecast_horizon, n_targets)
                y.append(data[(i + look_back):(i + look_back + forecast_horizon)][:, target_indices])

        X = np.array(X)
        y = np.array(y)

        # Squeeze forecast_horizon dimension jika = 1
        if forecast_horizon == 1:
            y = y.squeeze(axis=1)

        print(f"  Sliding window: X={X.shape}, y={y.shape}")
        return X, y

    # INVERSE TRANSFORM (untuk evaluasi)
    def inverse_transform_predictions(
        self,
        y_pred,
        target_cols,
        feature_cols
    ):
        """
        Mengembalikan prediksi dari skala [0, 1]
        ke skala asli.

        Parameters
        ----------
        y_pred : np.ndarray
            Prediksi scaled.

        target_cols : list of str
            Kolom target.

        feature_cols : list of str
            Seluruh feature yang digunakan saat scaling.

        Returns
        -------
        np.ndarray
            Prediksi pada skala asli.
        """

        y_pred = np.asarray(y_pred)

        # ---------------------------------------------------------
        # UNIVARIATE / MULTIVARIATE ONE-STEP
        # ---------------------------------------------------------
        if y_pred.ndim == 1:
            if len(target_cols) > 1:
                y_pred = y_pred.reshape(1, -1) 
            else:
                y_pred = y_pred.reshape(-1, 1) 
        # ---------------------------------------------------------
        # Pastikan target index
        # ---------------------------------------------------------
        target_indices = [
            feature_cols.index(col)
            for col in target_cols
        ]

        # ---------------------------------------------------------
        # Dummy array untuk inverse transform
        # ---------------------------------------------------------
        dummy = np.zeros(
            (y_pred.shape[0], len(feature_cols))
        )

        dummy[:, target_indices] = y_pred

        inversed = self.feature_scaler.inverse_transform(
            dummy
        )

        return inversed[:, target_indices]