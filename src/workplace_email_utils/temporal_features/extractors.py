"""
Temporal feature extraction module.

Extracts time-based features from email data including:
- Hour, day, week, month patterns
- Business hours detection
- Time intervals
- Temporal patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TemporalFeatures:
    """Container for temporal features."""
    feature_matrix: np.ndarray      # (n_docs, n_temporal_features)
    feature_names: List[str]
    date_series: pd.Series          # Parsed dates


def parse_email_date(date_str: str) -> Optional[datetime]:
    """
    Parse email date string to datetime object.
    
    Handles various date formats commonly found in email headers.
    
    Args:
        date_str: Date string from email header
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    if pd.isna(date_str) or not date_str:
        return None
    
    # Common date formats in email headers
    date_formats = [
        '%a, %d %b %Y %H:%M:%S %z',      # RFC 2822 format
        '%a, %d %b %Y %H:%M:%S %Z',      # With timezone name
        '%d %b %Y %H:%M:%S %z',          # Without day name
        '%Y-%m-%d %H:%M:%S',             # ISO format
        '%Y-%m-%d',                      # Date only
        '%d/%m/%Y %H:%M:%S',             # Alternative format
        '%m/%d/%Y %H:%M:%S',             # US format
    ]
    
    date_str = str(date_str).strip()
    
    # Try parsing with each format
    for fmt in date_formats:
        try:
            # Handle timezone offsets like -0800 (PDT)
            if '(%PST)' in date_str or '(PDT)' in date_str:
                date_str = date_str.replace('(PST)', '').replace('(PDT)', '').strip()
            
            # Parse date
            dt = datetime.strptime(date_str, fmt)
            return dt
        except (ValueError, AttributeError):
            continue
    
    # Try pandas parsing as fallback
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return None


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from email DataFrame.
    
    Adds temporal features to the DataFrame including:
    - Parsed datetime
    - Hour, day of week, month, year
    - Business hours indicators
    - Time intervals
    - Weekend indicators
    
    Args:
        df: DataFrame with email data (must have 'date' column)
        
    Returns:
        DataFrame with added temporal features
    """
    logger.info("Extracting temporal features")
    df = df.copy()
    
    # Parse dates
    logger.info("Parsing email dates")
    df['date_parsed'] = df['date'].apply(parse_email_date)
    
    # Convert to pandas datetime Series (handles None values properly)
    df['date_parsed'] = pd.to_datetime(df['date_parsed'], errors='coerce')
    
    # Normalize timezones - convert all to UTC, then remove timezone info for consistency
    if df['date_parsed'].dt.tz is not None:
        df['date_parsed'] = df['date_parsed'].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # Some dates might be tz-aware, some not - handle both
        df.loc[df['date_parsed'].notna(), 'date_parsed'] = df.loc[df['date_parsed'].notna(), 'date_parsed'].apply(
            lambda x: x.tz_localize(None) if hasattr(x, 'tz_localize') and x.tz is None 
                     else x.tz_convert('UTC').tz_localize(None) if hasattr(x, 'tz_convert')
                     else x
        )
    
    # Count successfully parsed dates
    n_parsed = df['date_parsed'].notna().sum()
    logger.info(f"Successfully parsed {n_parsed}/{len(df)} dates")
    
    if n_parsed == 0:
        logger.warning("No dates could be parsed. Temporal features will be limited.")
        return df
    
    # Extract basic temporal features (only for valid dates)
    df['hour'] = df['date_parsed'].dt.hour
    df['day_of_week'] = df['date_parsed'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['date_parsed'].dt.day
    df['week'] = df['date_parsed'].dt.isocalendar().week
    df['month'] = df['date_parsed'].dt.month
    df['year'] = df['date_parsed'].dt.year
    df['quarter'] = df['date_parsed'].dt.quarter
    
    # Day name for easier interpretation
    df['day_name'] = df['date_parsed'].dt.day_name()
    
    # Time-based indicators
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_after_hours'] = ((df['hour'] < 9) | (df['hour'] > 17)).astype(int)
    df['is_morning'] = df['hour'].between(6, 12).astype(int)
    df['is_afternoon'] = df['hour'].between(12, 18).astype(int)
    df['is_evening'] = df['hour'].between(18, 22).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    
    # Temporal intervals (time since previous email)
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Time since last email from same sender
    df['time_since_last_sender'] = df.groupby('sender')['date_parsed'].diff()
    df['hours_since_last_sender'] = df['time_since_last_sender'].dt.total_seconds() / 3600
    
    # Time since last email to same recipient
    # For multiple recipients, we'll calculate for the first recipient
    df['first_recipient'] = df['recipients'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    df['time_since_last_recipient'] = df.groupby('first_recipient')['date_parsed'].diff()
    df['hours_since_last_recipient'] = df['time_since_last_recipient'].dt.total_seconds() / 3600
    
    # Time since last email overall
    df['time_since_last_email'] = df['date_parsed'].diff()
    df['hours_since_last_email'] = df['time_since_last_email'].dt.total_seconds() / 3600
    
    # Days since epoch (for time series analysis)
    epoch = pd.Timestamp('1970-01-01', tz=None)
    df['days_since_epoch'] = (df['date_parsed'] - epoch).dt.total_seconds() / 86400  # Convert to days
    
    # Cyclical encoding for time features (useful for ML models)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Fill NaN values in time intervals
    df['hours_since_last_sender'] = df['hours_since_last_sender'].fillna(0)
    df['hours_since_last_recipient'] = df['hours_since_last_recipient'].fillna(0)
    df['hours_since_last_email'] = df['hours_since_last_email'].fillna(0)
    
    logger.info(f"Extracted {len([c for c in df.columns if c.startswith(('hour', 'day', 'month', 'is_', 'time_', 'days_'))])} temporal features")
    
    return df


def get_temporal_feature_matrix(df: pd.DataFrame) -> TemporalFeatures:
    """
    Extract temporal features as a numerical matrix.
    
    Args:
        df: DataFrame with temporal features (after extract_temporal_features)
        
    Returns:
        TemporalFeatures object with feature matrix
    """
    # Select numerical temporal features
    temporal_cols = [
        'hour', 'day_of_week', 'day_of_month', 'week', 'month', 'year', 'quarter',
        'is_weekend', 'is_business_hours', 'is_after_hours',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'hours_since_last_sender', 'hours_since_last_recipient', 'hours_since_last_email',
        'days_since_epoch',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    
    # Filter to columns that exist in DataFrame
    available_cols = [col for col in temporal_cols if col in df.columns]
    
    # Extract feature matrix
    feature_matrix = df[available_cols].fillna(0).values.astype(np.float32)
    
    return TemporalFeatures(
        feature_matrix=feature_matrix,
        feature_names=available_cols,
        date_series=df.get('date_parsed', pd.Series())
    )


if __name__ == "__main__":
    # Test the extractor
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing temporal feature extraction...")
    df = load_emails('maildir', data_format='maildir', max_rows=100)
    df = extract_temporal_features(df)
    
    print(f"\nExtracted features: {len([c for c in df.columns if 'date' in c.lower() or 'hour' in c.lower() or 'day' in c.lower()])}")
    print(f"\nSample temporal features:")
    print(df[['date_parsed', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours']].head(10))

