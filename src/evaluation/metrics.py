import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred):
        """
        Menghitung 4 metrik standar untuk peramalan (Forecasting).
        Pastikan y_true dan y_pred bentuknya array 1D.
        """
        # Konversi ke numpy array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # RMSE (Root Mean Squared Error) - Sensitif terhadap outlier/error besar
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE (Mean Absolute Error) - Rata-rata melesetnya berapa gCO2/kWh
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error) - Melesetnya berapa persen (%)
        # Ditambah epsilon kecil (1e-10) biar ga "error divide by zero"
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # MSE (Mean Squared Error)
        mse = mean_squared_error(y_true, y_pred)

        return {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'MAPE (%)': round(mape, 4),
            'MSE': round(mse, 4)
        }