import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataLoader:
    def __init__(self):
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_col = 'Carbon intensity gCO₂eq/kWh (direct)'

    def load_and_clean(self, filepath):
        """Membaca CSV dan merapikan data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
        df.set_index('Datetime (UTC)', inplace=True)
        
        cols_to_keep = [
            self.target_col, 
            'Carbon-free energy percentage (CFE%)', 
            'Renewable energy percentage (RE%)'
        ]
        df = df[cols_to_keep]
        
        df = df.resample('h').interpolate(method='linear')
        print(f"Data cleaned. Total baris: {len(df)}")
        return df
    
    def split_and_scale(self, df, feature_cols, train_ratio=0.8):
            """Split data (tanpa ngacak urutan waktu) lalu scale."""
            print("Splitting and Scaling data...")
            
            train_size = int(len(df) * train_ratio)
            train_df = df.iloc[:train_size].copy()
            test_df = df.iloc[train_size:].copy()
            self.target_scaler.fit(train_df[[self.target_col]])
            self.feature_scaler.fit(train_df[feature_cols])
            
            scaled_train_array = self.feature_scaler.transform(train_df[feature_cols])
            scaled_test_array = self.feature_scaler.transform(test_df[feature_cols])
            
            scaled_train_df = pd.DataFrame(scaled_train_array, columns=feature_cols, index=train_df.index)
            scaled_test_df = pd.DataFrame(scaled_test_array, columns=feature_cols, index=test_df.index)
            
            return scaled_train_df, scaled_test_df
        
    def scale_data(self, df, feature_cols):
        print("Scaling data...")
        self.target_scaler.fit(df[[self.target_col]])
        
        scaled_array = self.feature_scaler.fit_transform(df[feature_cols])
        
        scaled_df = pd.DataFrame(scaled_array, columns=feature_cols, index=df.index)
        return scaled_df

    def create_sliding_window(self, scaled_df, look_back=24, forecast_horizon=1):
        """
        Ngebelah time series jadi format Machine Learning (X dan Y).
        - look_back: Seberapa jauh ke belakang LSTM ngeliat (misal 24 jam)
        - forecast_horizon: Seberapa jauh ke depan LSTM nebak (1 = One step ahead)
        """
        X, y = [], []
        data = scaled_df.values
        
        target_idx = scaled_df.columns.get_loc(self.target_col)

        for i in range(len(data) - look_back - forecast_horizon + 1):
            X.append(data[i : (i + look_back), :])
            y.append(data[(i + look_back) : (i + look_back + forecast_horizon), target_idx])
            
        return np.array(X), np.array(y)