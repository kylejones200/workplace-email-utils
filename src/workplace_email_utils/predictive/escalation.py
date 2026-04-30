"""
Escalation prediction.

Predicts likelihood of email escalation (forwarding, CC to management, etc.).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EscalationPredictor:
    """Container for escalation prediction model."""
    model: Optional[any] = None
    method: str = 'heuristic'  # 'heuristic' or 'ml'
    escalation_threshold: float = 0.5


def extract_escalation_features(
    df: pd.DataFrame,
    text_col: str = 'text',
    subject_col: str = 'subject'
) -> pd.DataFrame:
    """
    Extract features for escalation prediction.
    
    Features:
    - Negative sentiment
    - Urgency indicators
    - Complaint keywords
    - Time since first email
    - Response time delays
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        subject_col: Column name for subject
        
    Returns:
        DataFrame with feature columns
    """
    df = df.copy()
    
    # Complaint keywords
    complaint_keywords = ['complaint', 'dissatisfied', 'unhappy', 'unacceptable', 
                         'poor service', 'bad experience', 'disappointed', 'frustrated']
    text_lower = df[text_col].fillna('').astype(str).str.lower()
    df['complaint_count'] = text_lower.apply(
        lambda x: sum(1 for kw in complaint_keywords if kw in x)
    )
    
    # Urgency (escalation risk)
    urgency_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
    df['urgency_count'] = text_lower.apply(
        lambda x: sum(1 for kw in urgency_keywords if kw in x)
    )
    
    # Negative sentiment indicators
    negative_words = ['problem', 'issue', 'error', 'wrong', 'failed', 'broken', 
                     'unable', 'cannot', 'won\'t', 'didn\'t']
    df['negative_sentiment_count'] = text_lower.apply(
        lambda x: sum(1 for word in negative_words if word in x)
    )
    
    # Email length (long emails may indicate detailed complaints)
    df['text_length'] = df[text_col].fillna('').astype(str).str.len()
    
    # Question count (many questions may indicate confusion/issues)
    df['question_count'] = df[text_col].fillna('').astype(str).str.count('\?')
    
    return df


def predict_escalation_risk_heuristic(
    df: pd.DataFrame,
    text_col: str = 'text'
) -> np.ndarray:
    """
    Predict escalation risk using heuristic rules.
    
    Risk factors:
    - Complaints
    - Negative sentiment
    - Urgency
    - Long emails (detailed issues)
    - Many questions
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        
    Returns:
        Array of escalation risk scores (0-1)
    """
    risks = []
    
    for _, row in df.iterrows():
        risk = 0.3  # Base risk
        
        # Complaint indicators
        complaint_count = row.get('complaint_count', 0)
        if complaint_count > 0:
            risk += min(complaint_count * 0.2, 0.4)
        
        # Negative sentiment
        negative_count = row.get('negative_sentiment_count', 0)
        if negative_count > 0:
            risk += min(negative_count * 0.1, 0.3)
        
        # Urgency
        urgency_count = row.get('urgency_count', 0)
        if urgency_count > 0:
            risk += min(urgency_count * 0.15, 0.2)
        
        # Long emails (detailed complaints)
        text_length = row.get('text_length', 0)
        if text_length > 1000:
            risk += 0.1
        
        # Many questions (confusion/issues)
        question_count = row.get('question_count', 0)
        if question_count > 3:
            risk += 0.1
        
        # Normalize to 0-1
        risk = max(0.0, min(1.0, risk))
        risks.append(risk)
    
    return np.array(risks)


def predict_escalation_risk(
    df: pd.DataFrame,
    predictor: Optional[EscalationPredictor] = None,
    text_col: str = 'text'
) -> np.ndarray:
    """
    Predict escalation risk for emails.
    
    Args:
        df: DataFrame with email data
        predictor: Optional EscalationPredictor
        text_col: Column name for email text
        
    Returns:
        Array of escalation risk scores (0-1)
    """
    if predictor is None:
        predictor = EscalationPredictor()
    
    # Extract features
    df_features = extract_escalation_features(df, text_col=text_col)
    
    if predictor.method == 'heuristic':
        predictions = predict_escalation_risk_heuristic(df_features, text_col=text_col)
    elif predictor.method == 'ml' and predictor.model:
        # ML-based prediction
        logger.warning("ML-based escalation prediction not yet implemented. Using heuristic.")
        predictions = predict_escalation_risk_heuristic(df_features, text_col=text_col)
    else:
        predictions = predict_escalation_risk_heuristic(df_features, text_col=text_col)
    
    return predictions


def train_escalation_model(
    df: pd.DataFrame,
    escalation_labels: Optional[pd.Series] = None,
    text_col: str = 'text'
) -> EscalationPredictor:
    """
    Train ML model for escalation prediction.
    
    Args:
        df: DataFrame with email data
        escalation_labels: Optional Series with escalation labels (0-1 scores)
        text_col: Column name for email text
        
    Returns:
        Trained EscalationPredictor
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.warning("scikit-learn not available. Using heuristic predictor.")
        return EscalationPredictor()
    
    if escalation_labels is None:
        logger.warning("No escalation labels provided. Using heuristic predictor.")
        return EscalationPredictor()
    
    logger.info("Training escalation prediction model...")
    
    # Extract features
    df_features = extract_escalation_features(df, text_col=text_col)
    
    # Prepare data
    feature_cols = ['complaint_count', 'urgency_count', 'negative_sentiment_count',
                    'text_length', 'question_count']
    X = df_features[[c for c in feature_cols if c in df_features.columns]].fillna(0)
    y = (escalation_labels > 0.5).astype(int).values  # Binary classification
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f"Escalation prediction model accuracy: {score:.3f}")
    
    return EscalationPredictor(model=model, method='ml', escalation_threshold=0.5)


if __name__ == "__main__":
    # Test escalation prediction
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing escalation prediction...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    
    predictions = predict_escalation_risk(df)
    print(f"\nEscalation risk predictions:")
    print(f"  Mean risk: {np.mean(predictions):.2f}")
    print(f"  High risk (>0.7): {(predictions > 0.7).sum()}")
    print(f"  Low risk (<0.3): {(predictions < 0.3).sum()}")

