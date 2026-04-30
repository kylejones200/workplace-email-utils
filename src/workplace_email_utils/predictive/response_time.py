"""
Response time prediction.

Predicts how long it will take for an email to receive a response.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResponseTimePredictor:
    """Container for response time prediction model."""
    model: Optional[any] = None
    method: str = 'heuristic'  # 'heuristic' or 'ml'
    avg_response_time: float = 24.0  # Average response time in hours


def extract_response_time_features(
    df: pd.DataFrame,
    text_col: str = 'text',
    sender_col: str = 'sender',
    date_col: str = 'date_parsed'
) -> pd.DataFrame:
    """
    Extract features for response time prediction.
    
    Features:
    - Urgency indicators
    - Sender importance
    - Time of day
    - Day of week
    - Email length
    - Question count
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        sender_col: Column name for sender
        date_col: Column name for dates
        
    Returns:
        DataFrame with feature columns
    """
    df = df.copy()
    
    # Urgency keywords
    urgency_keywords = ['urgent', 'asap', 'immediately', 'critical', 'deadline']
    text_lower = df[text_col].fillna('').astype(str).str.lower()
    df['urgency_count'] = text_lower.apply(
        lambda x: sum(1 for kw in urgency_keywords if kw in x)
    )
    
    # Question marks (may indicate need for response)
    df['question_count'] = df[text_col].fillna('').astype(str).str.count('\?')
    
    # Email length
    df['text_length'] = df[text_col].fillna('').astype(str).str.len()
    
    # Temporal features
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['hour'] = df[date_col].dt.hour
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17) & (df['is_weekend'] == 0)).astype(int)
    else:
        df['hour'] = 12
        df['day_of_week'] = 2
        df['is_weekend'] = 0
        df['is_business_hours'] = 1
    
    # Sender importance (if available)
    if 'sender_degree' in df.columns:
        df['sender_importance'] = df['sender_degree'].fillna(0)
    else:
        df['sender_importance'] = 0
    
    return df


def predict_response_time_heuristic(
    df: pd.DataFrame,
    text_col: str = 'text'
) -> np.ndarray:
    """
    Predict response time using heuristic rules.
    
    Rules:
    - Urgent emails: 1-2 hours
    - Business hours: 4-8 hours
    - Off-hours: 12-24 hours
    - Weekend: 24-48 hours
    - High sender importance: faster response
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        
    Returns:
        Array of predicted response times (hours)
    """
    predictions = []
    
    for _, row in df.iterrows():
        base_time = 24.0  # Default: 24 hours
        
        # Urgency adjustment
        urgency_count = row.get('urgency_count', 0)
        if urgency_count > 0:
            base_time = 2.0  # Urgent: 2 hours
        elif urgency_count == 0:
            base_time = 12.0  # Normal: 12 hours
        
        # Temporal adjustment
        is_business_hours = row.get('is_business_hours', 1)
        is_weekend = row.get('is_weekend', 0)
        
        if is_weekend:
            base_time *= 2.0  # Weekend: double time
        elif not is_business_hours:
            base_time *= 1.5  # Off-hours: 1.5x time
        
        # Sender importance adjustment
        sender_importance = row.get('sender_importance', 0)
        if sender_importance > 100:
            base_time *= 0.7  # Important sender: 30% faster
        elif sender_importance > 50:
            base_time *= 0.85  # Moderately important: 15% faster
        
        # Question count adjustment (more questions = faster response expected)
        question_count = row.get('question_count', 0)
        if question_count > 0:
            base_time *= max(0.8, 1.0 - question_count * 0.1)
        
        predictions.append(max(0.5, base_time))  # Minimum 0.5 hours
    
    return np.array(predictions)


def predict_response_time(
    df: pd.DataFrame,
    predictor: Optional[ResponseTimePredictor] = None,
    text_col: str = 'text'
) -> np.ndarray:
    """
    Predict response times for emails.
    
    Args:
        df: DataFrame with email data
        predictor: Optional ResponseTimePredictor
        text_col: Column name for email text
        
    Returns:
        Array of predicted response times (hours)
    """
    if predictor is None:
        predictor = ResponseTimePredictor()
    
    # Extract features
    df_features = extract_response_time_features(df, text_col=text_col)
    
    if predictor.method == 'heuristic':
        predictions = predict_response_time_heuristic(df_features, text_col=text_col)
    elif predictor.method == 'ml' and predictor.model:
        # ML-based prediction (would use trained model)
        logger.warning("ML-based response time prediction not yet implemented. Using heuristic.")
        predictions = predict_response_time_heuristic(df_features, text_col=text_col)
    else:
        predictions = predict_response_time_heuristic(df_features, text_col=text_col)
    
    return predictions


def train_response_time_model(
    df: pd.DataFrame,
    actual_response_times: Optional[pd.Series] = None,
    text_col: str = 'text'
) -> ResponseTimePredictor:
    """
    Train ML model for response time prediction.
    
    Args:
        df: DataFrame with email data
        actual_response_times: Optional Series with actual response times
        text_col: Column name for email text
        
    Returns:
        Trained ResponseTimePredictor
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.warning("scikit-learn not available. Using heuristic predictor.")
        return ResponseTimePredictor()
    
    if actual_response_times is None:
        logger.warning("No actual response times provided. Using heuristic predictor.")
        return ResponseTimePredictor()
    
    logger.info("Training response time prediction model...")
    
    # Extract features
    df_features = extract_response_time_features(df, text_col=text_col)
    
    # Prepare data
    feature_cols = ['urgency_count', 'question_count', 'text_length', 
                    'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'sender_importance']
    X = df_features[[c for c in feature_cols if c in df_features.columns]].fillna(0)
    y = actual_response_times.values
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f"Response time model R² score: {score:.3f}")
    
    return ResponseTimePredictor(model=model, method='ml', avg_response_time=np.mean(y))


if __name__ == "__main__":
    # Test response time prediction
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing response time prediction...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    df = extract_temporal_features(df)
    
    predictions = predict_response_time(df)
    print(f"\nPredicted response times:")
    print(f"  Mean: {np.mean(predictions):.1f} hours")
    print(f"  Median: {np.median(predictions):.1f} hours")
    print(f"  Range: {np.min(predictions):.1f} - {np.max(predictions):.1f} hours")

