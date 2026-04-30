"""
Temporal analysis functions.

Analyzes time-based patterns, trends, and anomalies in email data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_email_volume_trends(df: pd.DataFrame, 
                                 freq: str = 'D',
                                 date_col: str = 'date_parsed') -> pd.DataFrame:
    """
    Compute email volume trends over time.
    
    Args:
        df: DataFrame with temporal features
        freq: Frequency for aggregation ('D', 'W', 'M', 'H')
        date_col: Name of date column
        
    Returns:
        DataFrame with volume trends
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return pd.DataFrame()
    
    # Remove rows without dates
    df_valid = df[df[date_col].notna()].copy()
    
    if len(df_valid) == 0:
        logger.warning("No valid dates found for volume analysis")
        return pd.DataFrame()
    
    # Set date as index
    df_valid = df_valid.set_index(date_col)
    
    # Resample by frequency
    volume = df_valid.resample(freq).size()
    volume_df = pd.DataFrame({
        'count': volume,
        'date': volume.index
    })
    
    # Compute rolling statistics
    volume_df['rolling_mean'] = volume_df['count'].rolling(window=7, center=True).mean()
    volume_df['rolling_std'] = volume_df['count'].rolling(window=7, center=True).std()
    
    return volume_df.reset_index(drop=True)


def analyze_response_times(df: pd.DataFrame,
                          sender_col: str = 'sender',
                          recipient_col: str = 'recipients',
                          date_col: str = 'date_parsed',
                          subject_col: str = 'subject') -> pd.DataFrame:
    """
    Analyze response times between emails.
    
    Attempts to match reply emails with original emails and compute response times.
    
    Args:
        df: DataFrame with email data
        sender_col: Name of sender column
        recipient_col: Name of recipients column
        date_col: Name of date column
        subject_col: Name of subject column
        
    Returns:
        DataFrame with response time analysis
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return pd.DataFrame()
    
    logger.info("Analyzing response times")
    
    df_valid = df[df[date_col].notna()].copy()
    df_valid = df_valid.sort_values(date_col).reset_index(drop=True)
    
    response_times = []
    
    # Simple heuristic: match emails by subject (RE:, Fwd:, etc.) and sender/recipient swap
    for idx, email in df_valid.iterrows():
        subject = str(email.get(subject_col, '')).upper()
        sender = str(email.get(sender_col, '')).lower()
        recipients = email.get(recipient_col, [])
        
        if not isinstance(recipients, list):
            recipients = []
        
        recipients = [str(r).lower() for r in recipients if r]
        
        # Look for reply indicators
        if any(keyword in subject for keyword in ['RE:', 'RE :', 'FWD:', 'FWD :']):
            # Try to find original email
            # Check if any recipient is the current sender
            for recipient in recipients:
                # Find emails from this recipient to the current sender with similar subject
                clean_subject = subject.replace('RE:', '').replace('RE :', '').replace('FWD:', '').replace('FWD :', '').strip()
                
                matching = df_valid[
                    (df_valid[sender_col].str.lower() == recipient) &
                    (df_valid[sender_col].str.lower() != sender) &
                    (df_valid[date_col] < email[date_col]) &
                    (df_valid[subject_col].str.upper().str.contains(clean_subject[:20], regex=False, na=False))
                ]
                
                if len(matching) > 0:
                    # Get most recent matching email
                    original = matching.iloc[-1]
                    response_time = (email[date_col] - original[date_col]).total_seconds() / 3600  # hours
                    
                    response_times.append({
                        'original_email_id': original.get('doc_id', ''),
                        'reply_email_id': email.get('doc_id', ''),
                        'original_sender': original.get(sender_col, ''),
                        'reply_sender': sender,
                        'subject': clean_subject[:50],
                        'response_time_hours': response_time,
                        'response_time_days': response_time / 24,
                    })
                    break
    
    if response_times:
        response_df = pd.DataFrame(response_times)
        logger.info(f"Found {len(response_df)} email response pairs")
        logger.info(f"Average response time: {response_df['response_time_hours'].mean():.2f} hours")
        return response_df
    else:
        logger.warning("No response time pairs found")
        return pd.DataFrame()


def compute_communication_velocity(df: pd.DataFrame,
                                   date_col: str = 'date_parsed',
                                   sender_col: str = 'sender',
                                   window_hours: int = 24) -> pd.DataFrame:
    """
    Compute communication velocity (rate of email exchange).
    
    Measures how quickly emails are exchanged within conversations.
    
    Args:
        df: DataFrame with email data
        date_col: Name of date column
        sender_col: Name of sender column
        window_hours: Time window in hours for velocity calculation
        
    Returns:
        DataFrame with velocity metrics
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return pd.DataFrame()
    
    logger.info(f"Computing communication velocity (window: {window_hours} hours)")
    
    df_valid = df[df[date_col].notna()].copy()
    df_valid = df_valid.sort_values(date_col).reset_index(drop=True)
    
    velocities = []
    
    for idx, email in df_valid.iterrows():
        email_time = email[date_col]
        window_start = email_time - pd.Timedelta(hours=window_hours)
        
        # Count emails in the time window
        window_emails = df_valid[
            (df_valid[date_col] >= window_start) &
            (df_valid[date_col] <= email_time)
        ]
        
        # Count unique participants
        participants = set()
        participants.add(str(email.get(sender_col, '')).lower())
        
        for _, e in window_emails.iterrows():
            participants.add(str(e.get(sender_col, '')).lower())
            recipients = e.get('recipients', [])
            if isinstance(recipients, list):
                for r in recipients:
                    participants.add(str(r).lower())
        
        # Compute velocity metrics
        email_count = len(window_emails)
        participant_count = len(participants)
        
        velocities.append({
            'doc_id': email.get('doc_id', ''),
            'date': email_time,
            'emails_in_window': email_count,
            'participants_in_window': participant_count,
            'velocity': email_count / window_hours,  # emails per hour
            'diversity': participant_count,  # number of unique participants
        })
    
    velocity_df = pd.DataFrame(velocities)
    
    if len(velocity_df) > 0:
        logger.info(f"Average velocity: {velocity_df['velocity'].mean():.2f} emails/hour")
        logger.info(f"Max velocity: {velocity_df['velocity'].max():.2f} emails/hour")
    
    return velocity_df


def detect_temporal_anomalies(df: pd.DataFrame,
                              date_col: str = 'date_parsed',
                              method: str = 'zscore',
                              threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect temporal anomalies in email patterns.
    
    Identifies unusual patterns in email timing, volume, or frequency.
    
    Args:
        df: DataFrame with temporal features
        date_col: Name of date column
        method: Anomaly detection method ('zscore', 'iqr')
        threshold: Threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly flags and scores
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return df.copy()
    
    logger.info(f"Detecting temporal anomalies (method: {method})")
    
    df_valid = df[df[date_col].notna()].copy()
    
    if len(df_valid) == 0:
        return df.copy()
    
    anomalies = []
    
    # Compute daily volume
    daily_volume = compute_email_volume_trends(df_valid, freq='D', date_col=date_col)
    
    if len(daily_volume) > 0:
        if method == 'zscore':
            mean_volume = daily_volume['count'].mean()
            std_volume = daily_volume['count'].std()
            daily_volume['zscore'] = (daily_volume['count'] - mean_volume) / std_volume
            daily_volume['is_anomaly'] = (daily_volume['zscore'].abs() > threshold).astype(int)
        elif method == 'iqr':
            Q1 = daily_volume['count'].quantile(0.25)
            Q3 = daily_volume['count'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            daily_volume['is_anomaly'] = (
                (daily_volume['count'] < lower_bound) | 
                (daily_volume['count'] > upper_bound)
            ).astype(int)
    
    # Merge anomaly flags back to original dataframe
    df_result = df.copy()
    df_result['is_temporal_anomaly'] = 0
    
    if len(daily_volume) > 0:
        daily_volume['date_only'] = pd.to_datetime(daily_volume['date']).dt.date
        df_result['date_only'] = pd.to_datetime(df_result[date_col]).dt.date
        
        anomaly_dates = set(daily_volume[daily_volume['is_anomaly'] == 1]['date_only'])
        df_result['is_temporal_anomaly'] = df_result['date_only'].isin(anomaly_dates).astype(int)
        
        df_result = df_result.drop('date_only', axis=1)
    
    n_anomalies = df_result['is_temporal_anomaly'].sum()
    logger.info(f"Detected {n_anomalies} temporal anomalies")
    
    return df_result


if __name__ == "__main__":
    # Test analysis functions
    from workplace_email_utils.ingest.email_parser import load_emails
    from extractors import extract_temporal_features
    
    print("Testing temporal analysis...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    df = extract_temporal_features(df)
    
    print("\n1. Email Volume Trends:")
    volume = compute_email_volume_trends(df)
    print(volume.head(10))
    
    print("\n2. Response Times:")
    responses = analyze_response_times(df)
    if len(responses) > 0:
        print(responses.head(10))
    
    print("\n3. Communication Velocity:")
    velocity = compute_communication_velocity(df)
    print(velocity.head(10))
    
    print("\n4. Temporal Anomalies:")
    df_with_anomalies = detect_temporal_anomalies(df)
    print(f"Anomalies detected: {df_with_anomalies['is_temporal_anomaly'].sum()}")

