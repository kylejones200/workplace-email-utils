"""
Priority classification models.

Classifies emails into priority levels: high, medium, low.
Uses multiple features including content, sender, temporal patterns, and action detection.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PriorityClassifier:
    """Container for priority classification model."""
    model: any
    vectorizer: any
    scaler: any
    feature_columns: List[str]
    priority_levels: List[str] = None  # ['low', 'medium', 'high']
    accuracy: float = 0.0


def extract_priority_features(
    df: pd.DataFrame,
    text_col: str = 'text',
    sender_col: str = 'sender',
    date_col: str = 'date_parsed'
) -> pd.DataFrame:
    """
    Extract features for priority classification.
    
    Features:
    - TF-IDF text features
    - Sender importance (if available)
    - Temporal features (hour, day of week)
    - Action-required indicators
    - Email length
    - Subject urgency indicators
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        sender_col: Column name for sender
        date_col: Column name for parsed date
        
    Returns:
        DataFrame with feature columns
    """
    logger.info("Extracting priority features")
    
    features_df = df.copy()
    
    # Text length
    features_df['text_length'] = features_df[text_col].fillna('').astype(str).str.len()
    features_df['word_count'] = features_df[text_col].fillna('').astype(str).str.split().str.len()
    
    # Urgency keywords in subject/body
    urgency_keywords = ['urgent', 'asap', 'immediately', 'critical', 'important', 'deadline', 
                       'as soon as possible', 'time sensitive', 'emergency']
    
    text_lower = features_df[text_col].fillna('').astype(str).str.lower()
    features_df['urgency_keyword_count'] = text_lower.apply(
        lambda x: sum(1 for kw in urgency_keywords if kw in x)
    )
    features_df['has_urgency'] = (features_df['urgency_keyword_count'] > 0).astype(int)
    
    # Question marks (may indicate action needed)
    features_df['question_count'] = features_df[text_col].fillna('').astype(str).str.count('\?')
    features_df['has_questions'] = (features_df['question_count'] > 0).astype(int)
    
    # Exclamation marks (urgency/importance)
    features_df['exclamation_count'] = features_df[text_col].fillna('').astype(str).str.count('!')
    features_df['has_exclamations'] = (features_df['exclamation_count'] > 0).astype(int)
    
    # Temporal features (if available)
    if date_col in features_df.columns and features_df[date_col].notna().any():
        try:
            features_df['hour'] = pd.to_datetime(features_df[date_col], errors='coerce').dt.hour
            features_df['day_of_week'] = pd.to_datetime(features_df[date_col], errors='coerce').dt.dayofweek
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            features_df['is_business_hours'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 17)).astype(int)
            features_df['is_off_hours'] = ((features_df['hour'] < 9) | (features_df['hour'] > 17)).astype(int)
        except:
            pass
    
    # Sender features (if graph features available)
    if 'sender_degree' in features_df.columns:
        features_df['sender_importance'] = features_df['sender_degree'].fillna(0)
    else:
        features_df['sender_importance'] = 0
    
    # Action required (if available)
    if 'action_required' in features_df.columns:
        features_df['is_action_required'] = features_df['action_required'].fillna(0).astype(int)
    else:
        features_df['is_action_required'] = 0
    
    return features_df


def train_priority_classifier(
    df: pd.DataFrame,
    priority_col: Optional[str] = None,
    auto_generate_labels: bool = True,
    text_col: str = 'text',
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42
) -> PriorityClassifier:
    """
    Train priority classification model.
    
    If priority_col is not provided, auto-generates labels based on features:
    - High: Urgency keywords, off-hours, questions, high sender importance
    - Medium: Action required, some urgency indicators
    - Low: Regular emails without urgency indicators
    
    Args:
        df: DataFrame with email data
        priority_col: Column name with priority labels ('high', 'medium', 'low')
        auto_generate_labels: Auto-generate labels if priority_col not provided
        text_col: Column name for email text
        model_type: Type of classifier
        test_size: Proportion for testing
        random_state: Random seed
        
    Returns:
        Trained PriorityClassifier
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for priority classification")
    
    logger.info("Training priority classifier")
    
    # Extract features
    features_df = extract_priority_features(df, text_col=text_col)
    
    # Generate or use existing labels
    if priority_col and priority_col in df.columns:
        labels = df[priority_col].values
    elif auto_generate_labels:
        logger.info("Auto-generating priority labels based on features...")
        labels = _generate_priority_labels(features_df)
    else:
        raise ValueError("Must provide priority_col or set auto_generate_labels=True")
    
    # Map labels to numbers
    priority_levels = ['low', 'medium', 'high']
    label_map = {level: i for i, level in enumerate(priority_levels)}
    y = np.array([label_map.get(str(label).lower(), 0) for label in labels])
    
    # Prepare text features
    texts = features_df[text_col].fillna('').astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    X_text = vectorizer.fit_transform(texts).toarray()
    
    # Prepare structured features
    structured_features = [
        'text_length', 'word_count', 'urgency_keyword_count', 'has_urgency',
        'question_count', 'has_questions', 'exclamation_count', 'has_exclamations',
        'is_action_required', 'sender_importance'
    ]
    
    # Add temporal features if available
    if 'hour' in features_df.columns:
        structured_features.extend(['hour', 'day_of_week', 'is_weekend', 
                                   'is_business_hours', 'is_off_hours'])
    
    X_struct = features_df[[f for f in structured_features if f in features_df.columns]].fillna(0).values
    
    # Combine features
    X = np.hstack([X_text, X_struct])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model
    logger.info(f"Training {model_type} classifier...")
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.3f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=priority_levels)}")
    
    return PriorityClassifier(
        model=model,
        vectorizer=vectorizer,
        scaler=scaler,
        feature_columns=structured_features,
        priority_levels=priority_levels,
        accuracy=accuracy
    )


def _generate_priority_labels(df: pd.DataFrame) -> List[str]:
    """
    Auto-generate priority labels based on features.
    
    Rules:
    - High: Has urgency keywords AND (off-hours OR questions OR high sender importance)
    - Medium: Has action required OR some urgency indicators
    - Low: Default (regular emails)
    """
    labels = []
    
    for _, row in df.iterrows():
        urgency_count = row.get('urgency_keyword_count', 0)
        has_urgency = row.get('has_urgency', 0)
        is_off_hours = row.get('is_off_hours', 0)
        has_questions = row.get('has_questions', 0)
        sender_importance = row.get('sender_importance', 0)
        is_action_required = row.get('is_action_required', 0)
        
        # High priority conditions
        if (has_urgency and urgency_count >= 2) or \
           (has_urgency and (is_off_hours or has_questions)) or \
           (sender_importance > 100 and has_urgency):
            labels.append('high')
        # Medium priority conditions
        elif is_action_required or has_urgency or has_questions or sender_importance > 50:
            labels.append('medium')
        # Low priority (default)
        else:
            labels.append('low')
    
    return labels


def predict_priority(
    classifier: PriorityClassifier,
    df: pd.DataFrame,
    text_col: str = 'text',
    return_proba: bool = False
) -> np.ndarray:
    """
    Predict priority levels for emails.
    
    Args:
        classifier: Trained PriorityClassifier
        df: DataFrame with email data
        text_col: Column name for email text
        return_proba: If True, return probability distribution
        
    Returns:
        Array of priority predictions or probability distributions
    """
    # Extract features
    features_df = extract_priority_features(df, text_col=text_col)
    
    # Prepare text features
    texts = features_df[text_col].fillna('').astype(str).tolist()
    X_text = classifier.vectorizer.transform(texts).toarray()
    
    # Prepare structured features
    available_features = [f for f in classifier.feature_columns if f in features_df.columns]
    X_struct = features_df[available_features].fillna(0).values
    
    # Pad if features missing
    if len(available_features) < len(classifier.feature_columns):
        padding = np.zeros((len(features_df), len(classifier.feature_columns) - len(available_features)))
        X_struct = np.hstack([X_struct, padding])
    
    # Combine and scale
    X = np.hstack([X_text, X_struct])
    X_scaled = classifier.scaler.transform(X)
    
    # Predict
    if return_proba:
        predictions = classifier.model.predict_proba(X_scaled)
    else:
        predictions = classifier.model.predict(X_scaled)
        # Map back to labels
        predictions = np.array([classifier.priority_levels[int(p)] for p in predictions])
    
    return predictions


if __name__ == "__main__":
    # Test priority classifier
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing priority classification...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    df = extract_temporal_features(df)
    
    classifier = train_priority_classifier(df, auto_generate_labels=True)
    
    # Test prediction
    test_df = df.head(10)
    predictions = predict_priority(classifier, test_df)
    
    print("\nPriority predictions:")
    for i, (_, row) in enumerate(test_df.iterrows()):
        print(f"  {row.get('subject', '')[:50]}: {predictions[i]}")

