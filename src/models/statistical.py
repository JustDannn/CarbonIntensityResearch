import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.deterministic import Fourier, DeterministicProcess
from prophet import Prophet

class StatisticalForecaster:
    def __init__(self):
        pass

    def run_ar_model(self, train_data, p=24):
        """
        AutoRegressive (AR) Model.
        p = 24 artinya melihat 24 jam ke belakang.
        """
        print(f"Running AR({p}) model...")
        # Order (p, 0, 0) merepresentasikan AR murni
        model = ARIMA(train_data, order=(p, 0, 0))
        results = model.fit()
        return results

    def run_ma_model(self, train_data, q=24):
        """
        Moving Average (MA) Model.
        q = 24 artinya melihat error 24 jam ke belakang.
        """
        print(f"Running MA({q}) model...")
        # Order (0, 0, q) merepresentasikan MA murni
        model = ARIMA(train_data, order=(0, 0, q))
        results = model.fit()
        return results

    def run_sarimax_fourier(self, train_data):
        """
        SARIMAX dengan Deret Fourier untuk Multiple Seasonality.
        Menggunakan DeterministicProcess untuk menggabungkan pola 24 jam & 168 jam.
        """
        print("Running SARIMAX with Fourier features...")
        
        # Bikin Fourier harian dan mingguan secara terpisah
        fourier_24 = Fourier(period=24, order=2)
        fourier_168 = Fourier(period=168, order=2)
        
        # Gabungin pakai DeterministicProcess
        dp = DeterministicProcess(
            index=train_data.index,
            additional_terms=[fourier_24, fourier_168],
            drop=True
        )
        
        # Generate fitur X untuk training
        exog_train = dp.in_sample()
        
        # Fit SARIMAX
        model = SARIMAX(train_data, exog=exog_train, order=(1, 1, 1))
        results = model.fit(disp=False)
        
        return results, dp

    def run_prophet_decomposition(self, train_data_df):
        """
        Prophet Model untuk membedah Trend dan Multiple Seasonality.
        Dataframe input harus punya 2 kolom: 'ds' (datetime) dan 'y' (target).
        """
        print("Running Prophet Decomposition...")
        # Inisialisasi Prophet
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
        model.fit(train_data_df)
        
        forecast = model.predict(train_data_df)

        # Kembalikan forecast lengkap supaya plot_components punya kolom uncertainty yang dibutuhkan.
        return model, forecast

    def run_mstl_decomposition(self, train_data):
        """
        Multiple Seasonal-Trend decomposition using Loess (MSTL).
        Periode musiman di-set 24 (harian) dan 168 (mingguan).
        """
        print("Running MSTL Decomposition...")
        res = MSTL(train_data, periods=(24, 168)).fit()
        
        # Ekstrak komponen
        trend = res.trend
        seasonal_24 = res.seasonal['seasonal_24']
        seasonal_168 = res.seasonal['seasonal_168']
        residual = res.resid
        
        return res, trend, seasonal_24, seasonal_168, residual