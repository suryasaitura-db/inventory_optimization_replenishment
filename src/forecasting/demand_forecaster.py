"""
Demand Forecasting Models for Pharmaceutical Inventory
Prophet and ARIMA implementations with MLflow integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")


class DemandForecaster:
    """
    Demand forecasting using Prophet and ARIMA models
    Includes MLflow integration for model tracking and versioning
    """

    def __init__(self, model_type='prophet', mlflow_tracking=False):
        """
        Initialize forecaster

        Args:
            model_type: 'prophet' or 'arima'
            mlflow_tracking: Enable MLflow experiment tracking
        """
        self.model_type = model_type.lower()
        self.mlflow_tracking = mlflow_tracking and MLFLOW_AVAILABLE
        self.model = None
        self.is_fitted = False

        if self.model_type == 'prophet' and not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        if self.model_type == 'arima' and not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is not installed. Install with: pip install statsmodels")

    def prepare_data_prophet(self, df, date_col='date', value_col='demand'):
        """
        Prepare data for Prophet model

        Args:
            df: DataFrame with date and demand columns
            date_col: Name of date column
            value_col: Name of value column

        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[value_col]
        })
        return prophet_df.sort_values('ds').reset_index(drop=True)

    def fit_prophet(self, df, **prophet_params):
        """
        Fit Prophet model

        Args:
            df: DataFrame with 'ds' and 'y' columns
            **prophet_params: Additional Prophet parameters

        Returns:
            Fitted Prophet model
        """
        if self.mlflow_tracking:
            mlflow.start_run(run_name=f"prophet_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params(prophet_params)

        # Default parameters optimized for pharmaceutical inventory
        default_params = {
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        default_params.update(prophet_params)

        self.model = Prophet(**default_params)
        self.model.fit(df)
        self.is_fitted = True

        if self.mlflow_tracking:
            mlflow.sklearn.log_model(self.model, "prophet_model")
            mlflow.end_run()

        return self.model

    def fit_arima(self, series, order=(1, 1, 1)):
        """
        Fit ARIMA model

        Args:
            series: Time series data
            order: (p, d, q) order for ARIMA

        Returns:
            Fitted ARIMA model
        """
        if self.mlflow_tracking:
            mlflow.start_run(run_name=f"arima_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({'p': order[0], 'd': order[1], 'q': order[2]})

        self.model = ARIMA(series, order=order)
        self.model_fit = self.model.fit()
        self.is_fitted = True

        if self.mlflow_tracking:
            mlflow.log_metric('aic', self.model_fit.aic)
            mlflow.log_metric('bic', self.model_fit.bic)
            mlflow.end_run()

        return self.model_fit

    def predict_prophet(self, periods=30, freq='D'):
        """
        Generate predictions using Prophet

        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, etc.)

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

    def predict_arima(self, periods=30):
        """
        Generate predictions using ARIMA

        Args:
            periods: Number of periods to forecast

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.model_fit.forecast(steps=periods)
        return forecast

    def calculate_mape(self, actual, predicted):
        """
        Calculate Mean Absolute Percentage Error

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            MAPE value
        """
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Avoid division by zero
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

        return round(mape, 2)

    def calculate_rmse(self, actual, predicted):
        """
        Calculate Root Mean Squared Error

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            RMSE value
        """
        actual = np.array(actual)
        predicted = np.array(predicted)

        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        return round(rmse, 2)

    def evaluate_model(self, actual, predicted):
        """
        Evaluate model performance

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'mape': self.calculate_mape(actual, predicted),
            'rmse': self.calculate_rmse(actual, predicted),
            'mae': round(np.mean(np.abs(actual - predicted)), 2)
        }

        if self.mlflow_tracking:
            mlflow.start_run()
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow.end_run()

        return metrics


def example_usage():
    """Example usage of DemandForecaster"""
    print("=" * 80)
    print("DEMAND FORECASTING EXAMPLE")
    print("=" * 80 + "\n")

    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)

    # Trend + Seasonality + Random
    trend = np.linspace(100, 150, len(dates))
    seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    random_noise = np.random.normal(0, 10, len(dates))
    demand = trend + seasonality + random_noise
    demand = np.maximum(demand, 0)  # Ensure non-negative

    df = pd.DataFrame({
        'date': dates,
        'demand': demand
    })

    print(f"Generated {len(df)} days of synthetic demand data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Demand stats: min={df['demand'].min():.1f}, max={df['demand'].max():.1f}, mean={df['demand'].mean():.1f}\n")

    if PROPHET_AVAILABLE:
        print("Training Prophet model...")
        forecaster = DemandForecaster(model_type='prophet', mlflow_tracking=False)

        # Prepare data
        prophet_df = forecaster.prepare_data_prophet(df)

        # Fit model
        forecaster.fit_prophet(prophet_df)

        # Make predictions
        forecast = forecaster.predict_prophet(periods=30)

        print(f"✓ Prophet model trained successfully")
        print(f"  Forecasted next 30 days")
        print(f"  Mean forecast: {forecast['yhat'].mean():.1f}")
        print(f"  Forecast range: {forecast['yhat'].min():.1f} to {forecast['yhat'].max():.1f}\n")

        # Evaluate on training data (last 30 days)
        train_tail = prophet_df.tail(30)
        pred_tail = forecaster.model.predict(train_tail)

        metrics = forecaster.evaluate_model(
            train_tail['y'].values,
            pred_tail['yhat'].values
        )

        print(f"Model Performance Metrics:")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE:  {metrics['mae']:.2f}")

    else:
        print("⚠️  Prophet not available. Skipping Prophet example.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    example_usage()
