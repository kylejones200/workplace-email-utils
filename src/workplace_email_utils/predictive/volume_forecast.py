"""
Email volume forecasting.

Forecasts future email volumes using time series analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VolumeForecastModel:
    """Container for volume forecasting model."""
    method: str = 'moving_average'  # 'moving_average', 'exponential_smoothing', 'ml'
    model: Optional[any] = None
    avg_volume: float = 0.0
    trend: float = 0.0  # Volume trend (positive = increasing)


def prepare_volume_data(
    df: pd.DataFrame,
    date_col: str = 'date_parsed',
    time_period: str = 'day'
) -> pd.Series:
    """
    Prepare time series data for volume forecasting.
    
    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        time_period: Time period ('hour', 'day', 'week', 'month')
        
    Returns:
        Series with time-indexed email counts
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    if len(df) == 0:
        return pd.Series(dtype=float)
    
    # Group by time period
    if time_period == 'hour':
        df['period'] = df[date_col].dt.floor('H')
    elif time_period == 'day':
        df['period'] = df[date_col].dt.date
    elif time_period == 'week':
        df['period'] = df[date_col].dt.to_period('W')
    elif time_period == 'month':
        df['period'] = df[date_col].dt.to_period('M')
    else:
        raise ValueError(f"Unknown time period: {time_period}")
    
    # Count emails per period
    volume_series = df.groupby('period').size()
    volume_series.index = pd.to_datetime(volume_series.index)
    volume_series = volume_series.sort_index()
    
    return volume_series


def forecast_moving_average(
    volume_series: pd.Series,
    periods: int = 7,
    forecast_horizon: int = 7
) -> Tuple[np.ndarray, Dict]:
    """
    Forecast using moving average.
    
    Args:
        volume_series: Time series of email volumes
        periods: Number of periods for moving average
        forecast_horizon: Number of periods to forecast
        
    Returns:
        Tuple of (forecast_values, metrics_dict)
    """
    if len(volume_series) < periods:
        # Not enough data, use simple average
        avg = volume_series.mean() if len(volume_series) > 0 else 0.0
        forecast = np.array([avg] * forecast_horizon)
        return forecast, {'method': 'simple_average', 'avg_volume': avg}
    
    # Calculate moving average
    ma = volume_series.rolling(window=periods).mean()
    last_ma = ma.iloc[-1] if not ma.empty else volume_series.mean()
    
    # Simple forecast: use last moving average
    forecast = np.array([last_ma] * forecast_horizon)
    
    # Calculate trend
    if len(volume_series) >= 2:
        recent = volume_series.tail(periods)
        trend = (recent.iloc[-1] - recent.iloc[0]) / len(recent) if len(recent) > 1 else 0.0
    else:
        trend = 0.0
    
    metrics = {
        'method': 'moving_average',
        'avg_volume': last_ma,
        'trend': trend,
        'window': periods
    }
    
    return forecast, metrics


def forecast_exponential_smoothing(
    volume_series: pd.Series,
    alpha: float = 0.3,
    forecast_horizon: int = 7
) -> Tuple[np.ndarray, Dict]:
    """
    Forecast using exponential smoothing.
    
    Args:
        volume_series: Time series of email volumes
        alpha: Smoothing parameter (0-1)
        forecast_horizon: Number of periods to forecast
        
    Returns:
        Tuple of (forecast_values, metrics_dict)
    """
    if len(volume_series) == 0:
        return np.array([0.0] * forecast_horizon), {'method': 'exponential_smoothing', 'avg_volume': 0.0}
    
    # Simple exponential smoothing
    smoothed = [volume_series.iloc[0]]
    for value in volume_series.iloc[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    
    last_smoothed = smoothed[-1]
    forecast = np.array([last_smoothed] * forecast_horizon)
    
    metrics = {
        'method': 'exponential_smoothing',
        'avg_volume': last_smoothed,
        'alpha': alpha
    }
    
    return forecast, metrics


def forecast_email_volume(
    df: pd.DataFrame,
    model: Optional[VolumeForecastModel] = None,
    date_col: str = 'date_parsed',
    time_period: str = 'day',
    forecast_horizon: int = 7
) -> Tuple[np.ndarray, Dict]:
    """
    Forecast future email volumes.
    
    Args:
        df: DataFrame with email data
        model: Optional VolumeForecastModel
        date_col: Column name for dates
        time_period: Time period for aggregation
        forecast_horizon: Number of periods to forecast
        
    Returns:
        Tuple of (forecast_values, metrics_dict)
    """
    if model is None:
        model = VolumeForecastModel()
    
    # Prepare data
    volume_series = prepare_volume_data(df, date_col=date_col, time_period=time_period)
    
    if len(volume_series) == 0:
        logger.warning("No volume data available for forecasting")
        return np.array([0.0] * forecast_horizon), {'method': 'none', 'avg_volume': 0.0}
    
    # Forecast based on method
    if model.method == 'moving_average':
        forecast, metrics = forecast_moving_average(volume_series, forecast_horizon=forecast_horizon)
    elif model.method == 'exponential_smoothing':
        forecast, metrics = forecast_exponential_smoothing(volume_series, forecast_horizon=forecast_horizon)
    elif model.method == 'ml' and model.model:
        # ML-based forecasting (would use trained model)
        logger.warning("ML-based volume forecasting not yet implemented. Using moving average.")
        forecast, metrics = forecast_moving_average(volume_series, forecast_horizon=forecast_horizon)
    else:
        forecast, metrics = forecast_moving_average(volume_series, forecast_horizon=forecast_horizon)
    
    return forecast, metrics


def train_volume_forecast_model(
    df: pd.DataFrame,
    date_col: str = 'date_parsed',
    time_period: str = 'day',
    method: str = 'moving_average'
) -> VolumeForecastModel:
    """
    Train volume forecasting model.
    
    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        time_period: Time period for aggregation
        method: Forecasting method
        
    Returns:
        Trained VolumeForecastModel
    """
    logger.info(f"Training volume forecast model (method: {method})...")
    
    volume_series = prepare_volume_data(df, date_col=date_col, time_period=time_period)
    
    if len(volume_series) == 0:
        return VolumeForecastModel()
    
    avg_volume = volume_series.mean()
    
    # Calculate trend
    if len(volume_series) >= 2:
        recent = volume_series.tail(min(7, len(volume_series)))
        trend = (recent.iloc[-1] - recent.iloc[0]) / len(recent) if len(recent) > 1 else 0.0
    else:
        trend = 0.0
    
    return VolumeForecastModel(
        method=method,
        avg_volume=avg_volume,
        trend=trend
    )


if __name__ == "__main__":
    # Test volume forecasting
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing volume forecasting...")
    df = load_emails('maildir', data_format='maildir', max_rows=1000)
    df = extract_temporal_features(df)
    
    forecast, metrics = forecast_email_volume(df, time_period='day', forecast_horizon=7)
    print(f"\nVolume forecast (next 7 days):")
    print(f"  Forecast: {forecast}")
    print(f"  Metrics: {metrics}")

