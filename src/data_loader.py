import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import MSTL


class TimeSeriesDataLoader:
    """
    Data loader untuk pipeline LSTM forecasting Carbon Intensity & Renewable Energy.
    
    Dirancang sesuai pedoman Bu Regita:
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
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================

    def load_and_clean(self, filepath):
        """
        Membaca CSV dataset yang sudah di-preprocessing.
        Expects columns: datetime, carbon_intensity, renewable_percentage
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        # Parse datetime & set as index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Pastikan hanya kolom yang kita butuhkan
        cols_to_keep = [c for c in [self.CI_COL, self.RE_COL] if c in df.columns]
        df = df[cols_to_keep]

        # Resample ke hourly & interpolasi jika ada gap
        df = df.resample('h').interpolate(method='linear')
        df.dropna(inplace=True)

        print(f"Data cleaned. Shape: {df.shape}")
        print(f"Range: {df.index.min()} → {df.index.max()}")
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

    # =========================================================================
    # 3. SPLIT, SCALE, WINDOWING
    # =========================================================================

    def split_and_scale(self, df, feature_cols, target_cols, train_ratio=0.8):
        """
        Split data secara temporal (tanpa shuffle) lalu scale.
        
        Parameters
        ----------
        df : pd.DataFrame
        feature_cols : list of str
            Semua kolom yang masuk ke sliding window (termasuk target).
        target_cols : list of str
            Kolom yang jadi target prediksi (1 untuk univariate, 2 untuk multivariate).
        train_ratio : float
            Proporsi data training. Default 0.8.
            
        Returns
        -------
        tuple: (scaled_train_df, scaled_test_df)
        """
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # Fit scaler hanya pada data training
        self.target_scaler.fit(train_df[target_cols])
        self.feature_scaler.fit(train_df[feature_cols])

        # Transform
        scaled_train = self.feature_scaler.transform(train_df[feature_cols])
        scaled_test = self.feature_scaler.transform(test_df[feature_cols])

        scaled_train_df = pd.DataFrame(scaled_train, columns=feature_cols, index=train_df.index)
        scaled_test_df = pd.DataFrame(scaled_test, columns=feature_cols, index=test_df.index)

        print(f"  Split: Train={len(scaled_train_df)}, Test={len(scaled_test_df)}")
        return scaled_train_df, scaled_test_df

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

    # =========================================================================
    # 4. INVERSE TRANSFORM (untuk evaluasi)
    # =========================================================================

    def inverse_transform_predictions(self, y_pred, target_cols, feature_cols):
        """
        Inverse transform prediksi kembali ke skala asli.
        
        Karena kita pakai feature_scaler untuk semua fitur, kita perlu 
        membuat dummy array full-size lalu extract target columns saja.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Prediksi yang masih dalam skala [0,1]. Shape: (n_samples, n_targets)
        target_cols : list of str
            Nama kolom target.
        feature_cols : list of str
            Nama semua feature columns (untuk reconstruct full array).
            
        Returns
        -------
        np.ndarray
            Prediksi dalam skala asli.
        """
        n_features = len(feature_cols)
        n_samples = len(y_pred)

        # Pastikan y_pred 2D
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Buat array dummy dengan zeros
        dummy = np.zeros((n_samples, n_features))

        # Isi posisi target columns dengan nilai prediksi
        for i, col in enumerate(target_cols):
            col_idx = feature_cols.index(col)
            dummy[:, col_idx] = y_pred[:, i]

        # Inverse transform seluruh array
        inversed = self.feature_scaler.inverse_transform(dummy)

        # Extract hanya target columns
        result = np.column_stack([
            inversed[:, feature_cols.index(col)] for col in target_cols
        ])

        return result