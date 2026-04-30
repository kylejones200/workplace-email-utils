"""
Temporal anomaly detection.

Detects anomalies in temporal patterns: off-hours spikes, unusual frequency, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TemporalAnomalyResult:
    """Container for temporal anomaly detection results."""
    anomalous_periods: List[str]
    anomaly_scores: Dict[str, float]
    anomaly_types: Dict[str, List[str]]
    metrics: Dict


def detect_volume_spikes(
    df: pd.DataFrame,
    date_col: str = 'date_parsed',
    time_period: str = 'hour',
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect unusual spikes in email volume.
    
    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        time_period: Time period to analyze ('hour', 'day', 'week')
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with volume spike flags
    """
    logger.info(f"Detecting volume spikes (period: {time_period})...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    if len(df) == 0:
        return df
    
    # Group by time period
    if time_period == 'hour':
        df['period'] = df[date_col].dt.floor('H')
    elif time_period == 'day':
        df['period'] = df[date_col].dt.date
    elif time_period == 'week':
        df['period'] = df[date_col].dt.to_period('W')
    else:
        raise ValueError(f"Unknown time period: {time_period}")
    
    # Count emails per period
    period_counts = df.groupby('period').size()
    
    if len(period_counts) < 2:
        df['is_volume_spike'] = 0
        df['volume_spike_score'] = 0.0
        return df
    
    # Calculate z-scores
    mean_count = period_counts.mean()
    std_count = period_counts.std()
    
    if std_count > 0:
        z_scores = (period_counts - mean_count) / std_count
    else:
        z_scores = pd.Series(0.0, index=period_counts.index)
    
    # Flag spikes
    spike_periods = z_scores[z_scores > threshold].index
    
    df['is_volume_spike'] = df['period'].isin(spike_periods).astype(int)
    df['volume_spike_score'] = df['period'].map(z_scores).fillna(0.0)
    
    logger.info(f"Detected {len(spike_periods)} volume spike periods")
    return df


def detect_off_hours_anomalies(
    df: pd.DataFrame,
    date_col: str = 'date_parsed',
    business_hours: tuple = (9, 17),
    weekend_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Detect anomalies in off-hours communication patterns.
    
    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        business_hours: Tuple of (start_hour, end_hour)
        weekend_threshold: Minimum ratio of weekend emails to flag as anomaly
        
    Returns:
        DataFrame with off-hours anomaly flags
    """
    logger.info("Detecting off-hours anomalies...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    if len(df) == 0:
        return df
    
    # Extract temporal features
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = (
        (df['hour'] >= business_hours[0]) & 
        (df['hour'] < business_hours[1]) &
        (df['is_weekend'] == 0)
    ).astype(int)
    df['is_off_hours'] = (df['is_business_hours'] == 0).astype(int)
    
    # Calculate off-hours ratio per sender
    sender_stats = df.groupby('sender').agg({
        'is_off_hours': 'mean',
        'is_weekend': 'mean'
    })
    
    # Flag senders with high off-hours ratio
    high_off_hours = sender_stats[sender_stats['is_off_hours'] > 0.3].index
    high_weekend = sender_stats[sender_stats['is_weekend'] > weekend_threshold].index
    
    df['is_off_hours_anomaly'] = (
        df['sender'].isin(high_off_hours) | 
        df['sender'].isin(high_weekend)
    ).astype(int)
    
    logger.info(f"Detected {df['is_off_hours_anomaly'].sum()} off-hours anomaly emails")
    return df


def detect_response_time_anomalies(
    df: pd.DataFrame,
    date_col: str = 'date_parsed',
    thread_col: str = 'thread_id',
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalous response times in email threads.
    
    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        thread_col: Column name for thread IDs
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with response time anomaly flags
    """
    logger.info("Detecting response time anomalies...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, thread_col])
    
    if len(df) == 0 or thread_col not in df.columns:
        df['is_response_time_anomaly'] = 0
        return df
    
    # Calculate response times within threads
    response_times = []
    
    for thread_id in df[thread_col].unique():
        thread_df = df[df[thread_col] == thread_id].sort_values(date_col)
        
        if len(thread_df) < 2:
            continue
        
        for i in range(1, len(thread_df)):
            time_diff = (thread_df.iloc[i][date_col] - thread_df.iloc[i-1][date_col]).total_seconds() / 3600
            if time_diff > 0:
                response_times.append(time_diff)
    
    if not response_times:
        df['is_response_time_anomaly'] = 0
        return df
    
    # Calculate z-scores
    mean_rt = np.mean(response_times)
    std_rt = np.std(response_times)
    
    # Flag threads with anomalous response times
    df['is_response_time_anomaly'] = 0
    
    for thread_id in df[thread_col].unique():
        thread_df = df[df[thread_col] == thread_id].sort_values(date_col)
        
        if len(thread_df) < 2:
            continue
        
        for i in range(1, len(thread_df)):
            time_diff = (thread_df.iloc[i][date_col] - thread_df.iloc[i-1][date_col]).total_seconds() / 3600
            
            if time_diff > 0 and std_rt > 0:
                z_score = abs((time_diff - mean_rt) / std_rt)
                if z_score > threshold:
                    idx = thread_df.iloc[i].name
                    df.loc[idx, 'is_response_time_anomaly'] = 1
    
    logger.info(f"Detected {df['is_response_time_anomaly'].sum()} response time anomalies")
    return df


def detect_temporal_anomalies(
    df: pd.DataFrame,
    date_col: str = 'date_parsed',
    volume_threshold: float = 2.0,
    response_time_threshold: float = 2.0
) -> TemporalAnomalyResult:
    """
    Comprehensive temporal anomaly detection.
    
    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        volume_threshold: Threshold for volume spikes
        response_time_threshold: Threshold for response time anomalies
        
    Returns:
        TemporalAnomalyResult with all temporal anomalies
    """
    logger.info("Detecting temporal anomalies...")
    
    df = df.copy()
    
    # Detect volume spikes
    df = detect_volume_spikes(df, date_col=date_col, threshold=volume_threshold)
    
    # Detect off-hours anomalies
    df = detect_off_hours_anomalies(df, date_col=date_col)
    
    # Detect response time anomalies
    df = detect_response_time_anomalies(df, date_col=date_col, threshold=response_time_threshold)
    
    # Combined anomaly score
    df['temporal_anomaly_score'] = (
        df.get('volume_spike_score', 0) * 0.4 +
        df.get('is_off_hours_anomaly', 0) * 0.3 +
        df.get('is_response_time_anomaly', 0) * 0.3
    )
    df['is_temporal_anomaly'] = (df['temporal_anomaly_score'] > 0.3).astype(int)
    
    # Get anomalous periods
    if 'period' in df.columns:
        anomalous_periods = df[df['is_volume_spike'] == 1]['period'].unique().tolist()
    else:
        anomalous_periods = []
    
    anomaly_scores = dict(zip(df.index, df['temporal_anomaly_score']))
    
    anomaly_types = {
        'volume_spike': df[df['is_volume_spike'] == 1].index.tolist(),
        'off_hours': df[df['is_off_hours_anomaly'] == 1].index.tolist(),
        'response_time': df[df['is_response_time_anomaly'] == 1].index.tolist()
    }
    
    metrics = {
        'total_anomalies': df['is_temporal_anomaly'].sum(),
        'volume_spikes': df['is_volume_spike'].sum(),
        'off_hours_anomalies': df['is_off_hours_anomaly'].sum(),
        'response_time_anomalies': df['is_response_time_anomaly'].sum(),
        'anomalous_periods': len(anomalous_periods)
    }
    
    logger.info(f"Temporal anomalies: {metrics}")
    
    return TemporalAnomalyResult(
        anomalous_periods=anomalous_periods,
        anomaly_scores=anomaly_scores,
        anomaly_types=anomaly_types,
        metrics=metrics
    )


if __name__ == "__main__":
    # Test temporal anomaly detection
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing temporal anomaly detection...")
    df = load_emails('maildir', data_format='maildir', max_rows=1000)
    df = extract_temporal_features(df)
    
    result = detect_temporal_anomalies(df)
    print(f"\nTemporal anomalies: {result.metrics}")

