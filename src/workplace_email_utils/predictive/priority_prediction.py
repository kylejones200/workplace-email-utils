"""
Priority prediction.

Predicts email priority based on content, sender, and context.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PriorityPredictor:
    """Container for priority prediction model."""
    model: Optional[any] = None
    method: str = 'heuristic'  # 'heuristic' or 'ml'
    priority_weights: Dict[str, float] = None


def predict_priority_score_heuristic(
    df: pd.DataFrame,
    text_col: str = 'text'
) -> np.ndarray:
    """
    Predict priority scores using heuristic rules.
    
    Priority factors:
    - Urgency keywords
    - Sender importance
    - Time of day
    - Question count
    - Action required indicators
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        
    Returns:
        Array of priority scores (0-1, higher = more important)
    """
    scores = []
    
    for _, row in df.iterrows():
        score = 0.5  # Base priority
        
        # Urgency keywords
        urgency_count = row.get('urgency_count', 0)
        if urgency_count > 0:
            score += min(urgency_count * 0.15, 0.3)
        
        # Sender importance
        sender_importance = row.get('sender_importance', 0)
        if sender_importance > 100:
            score += 0.2
        elif sender_importance > 50:
            score += 0.1
        
        # Off-hours/weekend (may indicate urgency)
        is_off_hours = row.get('is_off_hours', 0)
        is_weekend = row.get('is_weekend', 0)
        if is_off_hours or is_weekend:
            score += 0.1
        
        # Questions (action needed)
        question_count = row.get('question_count', 0)
        if question_count > 0:
            score += min(question_count * 0.05, 0.15)
        
        # Action required (if available)
        if 'action_required' in row and row.get('action_required', 0) == 1:
            score += 0.2
        
        # Normalize to 0-1
        score = max(0.0, min(1.0, score))
        scores.append(score)
    
    return np.array(scores)


def predict_priority_score(
    df: pd.DataFrame,
    predictor: Optional[PriorityPredictor] = None,
    text_col: str = 'text'
) -> np.ndarray:
    """
    Predict priority scores for emails.
    
    Args:
        df: DataFrame with email data
        predictor: Optional PriorityPredictor
        text_col: Column name for email text
        
    Returns:
        Array of priority scores (0-1)
    """
    if predictor is None:
        predictor = PriorityPredictor()
    
    # Extract features if needed
    if 'urgency_count' not in df.columns:
        from workplace_email_utils.classification.priority import extract_priority_features
        df = extract_priority_features(df, text_col=text_col)
    
    if predictor.method == 'heuristic':
        predictions = predict_priority_score_heuristic(df, text_col=text_col)
    elif predictor.method == 'ml' and predictor.model:
        # ML-based prediction
        logger.warning("ML-based priority prediction not yet implemented. Using heuristic.")
        predictions = predict_priority_score_heuristic(df, text_col=text_col)
    else:
        predictions = predict_priority_score_heuristic(df, text_col=text_col)
    
    return predictions


def train_priority_predictor(
    df: pd.DataFrame,
    priority_labels: Optional[pd.Series] = None,
    text_col: str = 'text'
) -> PriorityPredictor:
    """
    Train ML model for priority prediction.
    
    Args:
        df: DataFrame with email data
        priority_labels: Optional Series with priority labels (0-1 scores)
        text_col: Column name for email text
        
    Returns:
        Trained PriorityPredictor
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.warning("scikit-learn not available. Using heuristic predictor.")
        return PriorityPredictor()
    
    if priority_labels is None:
        logger.warning("No priority labels provided. Using heuristic predictor.")
        return PriorityPredictor()
    
    logger.info("Training priority prediction model...")
    
    # Extract features
    from workplace_email_utils.classification.priority import extract_priority_features
    df_features = extract_priority_features(df, text_col=text_col)
    
    # Prepare data
    feature_cols = ['urgency_count', 'question_count', 'sender_importance',
                    'is_off_hours', 'is_weekend', 'is_action_required']
    X = df_features[[c for c in feature_cols if c in df_features.columns]].fillna(0)
    y = priority_labels.values
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f"Priority prediction model R² score: {score:.3f}")
    
    return PriorityPredictor(model=model, method='ml')


if __name__ == "__main__":
    # Test priority prediction
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing priority prediction...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    df = extract_temporal_features(df)
    
    predictions = predict_priority_score(df)
    print(f"\nPriority predictions:")
    print(f"  Mean: {np.mean(predictions):.2f}")
    print(f"  High priority (>0.7): {(predictions > 0.7).sum()}")
    print(f"  Low priority (<0.3): {(predictions < 0.3).sum()}")

