"""
Unified email classification pipeline.

Combines all classification models into a single unified pipeline:
- Action required detection
- Priority classification (high/medium/low)
- Category classification (sales, support, HR, etc.)
- Folder prediction (inbox, sent, etc.)
- Urgency detection
- Spam detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import logging

from .action_detection import ActionClassifier, predict_action_required
from .priority import PriorityClassifier, predict_priority
from .category import CategoryClassifier, predict_category

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmailClassifications:
    """Container for all email classifications."""
    action_required: np.ndarray  # Binary (1=action, 0=no action)
    action_probability: np.ndarray  # Probability of action required
    priority: np.ndarray  # ['low', 'medium', 'high']
    priority_probability: np.ndarray  # Probability distribution
    category: np.ndarray  # Category labels
    category_probability: np.ndarray  # Probability distribution
    urgency_score: np.ndarray  # Urgency score (0-1)
    is_spam: np.ndarray  # Binary (1=spam, 0=not spam)
    spam_probability: np.ndarray  # Probability of spam
    predicted_folder: np.ndarray  # Predicted folder type


@dataclass
class UnifiedClassifier:
    """Unified classification model combining all classifiers."""
    action_classifier: Optional[ActionClassifier] = None
    priority_classifier: Optional[PriorityClassifier] = None
    category_classifier: Optional[CategoryClassifier] = None
    folder_predictor: Optional[any] = None
    spam_classifier: Optional[any] = None


def detect_urgency(
    df: pd.DataFrame,
    text_col: str = 'text',
    subject_col: str = 'subject'
) -> np.ndarray:
    """
    Detect urgency score for emails (0-1 scale).
    
    Uses heuristics:
    - Urgency keywords in subject/body
    - Off-hours sending
    - Question marks (action needed)
    - Exclamation marks (importance)
    - Response time requests
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        subject_col: Column name for subject
        
    Returns:
        Array of urgency scores (0.0 to 1.0)
    """
    urgency_keywords = {
        'high': ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'time sensitive'],
        'medium': ['important', 'please', 'soon', 'deadline', 'need', 'required'],
        'low': ['whenever', 'if possible', 'at your convenience']
    }
    
    urgency_scores = []
    
    for _, row in df.iterrows():
        score = 0.0
        max_score = 0.0
        
        # Combine subject and body
        text = (str(row.get(subject_col, '')) + ' ' + str(row.get(text_col, ''))).lower()
        
        # High urgency keywords
        high_count = sum(1 for kw in urgency_keywords['high'] if kw in text)
        score += high_count * 0.3
        max_score += 0.3
        
        # Medium urgency keywords
        medium_count = sum(1 for kw in urgency_keywords['medium'] if kw in text)
        score += medium_count * 0.15
        max_score += 0.15
        
        # Low urgency keywords (decrease score)
        low_count = sum(1 for kw in urgency_keywords['low'] if kw in text)
        score -= low_count * 0.1
        
        # Off-hours sending (increases urgency)
        if 'is_off_hours' in row and row.get('is_off_hours') == 1:
            score += 0.1
            max_score += 0.1
        
        # Weekend sending (increases urgency)
        if 'is_weekend' in row and row.get('is_weekend') == 1:
            score += 0.1
            max_score += 0.1
        
        # Question marks (action needed)
        question_count = text.count('?')
        score += min(question_count * 0.05, 0.2)
        max_score += 0.2
        
        # Exclamation marks (importance)
        exclamation_count = text.count('!')
        score += min(exclamation_count * 0.03, 0.15)
        max_score += 0.15
        
        # Normalize to 0-1
        if max_score > 0:
            normalized_score = min(score / max_score, 1.0)
        else:
            normalized_score = 0.0
        
        urgency_scores.append(max(0.0, normalized_score))
    
    return np.array(urgency_scores)


def detect_spam(
    df: pd.DataFrame,
    text_col: str = 'text',
    sender_col: str = 'sender'
) -> tuple:
    """
    Detect spam/junk emails using heuristics.
    
    Indicators:
    - Suspicious keywords
    - Unusual sender patterns
    - High exclamation/question ratio
    - Suspicious links
    - All caps
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        sender_col: Column name for sender
        
    Returns:
        Tuple of (spam_predictions, spam_probabilities)
    """
    spam_keywords = [
        'free', 'click here', 'limited time', 'act now', 'winner', 'prize',
        'guarantee', 'risk free', 'opt out', 'unsubscribe', 'viagra',
        'casino', 'lottery', 'congratulations', 'urgent action required'
    ]
    
    spam_scores = []
    
    for _, row in df.iterrows():
        score = 0.0
        
        text = str(row.get(text_col, '')).lower()
        subject = str(row.get('subject', '')).lower()
        full_text = subject + ' ' + text
        
        # Suspicious keywords
        keyword_count = sum(1 for kw in spam_keywords if kw in full_text)
        score += keyword_count * 0.15
        
        # All caps in subject
        if subject:
            caps_ratio = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
            if caps_ratio > 0.5:
                score += 0.2
        
        # Excessive exclamation/question marks
        exclamation_count = full_text.count('!')
        question_count = full_text.count('?')
        if exclamation_count + question_count > 5:
            score += 0.2
        
        # Suspicious links (basic check)
        if 'http://' in full_text or 'https://' in full_text:
            link_count = full_text.count('http')
            if link_count > 3:
                score += 0.15
        
        # Sender patterns (basic)
        sender = str(row.get(sender_col, '')).lower()
        if any(sus in sender for sus in ['noreply', 'no-reply', 'mailer', 'notification']):
            score += 0.1
        
        # Normalize to probability (0-1)
        spam_prob = min(score, 1.0)
        spam_scores.append(spam_prob)
    
    spam_probabilities = np.array(spam_scores)
    spam_predictions = (spam_probabilities > 0.5).astype(int)
    
    return spam_predictions, spam_probabilities


def predict_folder(
    df: pd.DataFrame,
    folder_type_col: Optional[str] = 'folder_type'
) -> np.ndarray:
    """
    Predict folder type for emails.
    
    Uses existing folder_type if available, otherwise predicts based on patterns.
    
    Args:
        df: DataFrame with email data
        folder_type_col: Column name for existing folder type
        
    Returns:
        Array of predicted folder types
    """
    if folder_type_col and folder_type_col in df.columns and df[folder_type_col].notna().any():
        # Use existing folder types
        predicted_folders = df[folder_type_col].fillna('unknown').values
    else:
        # Predict based on patterns (simplified)
        # In real scenario, would train a classifier on known folder types
        predicted_folders = np.array(['inbox'] * len(df))
        
        # Basic heuristics
        for i, (_, row) in enumerate(df.iterrows()):
            text = str(row.get('text', '')).lower()
            subject = str(row.get('subject', '')).lower()
            
            # Draft indicators
            if any(word in text for word in ['draft', 'not sent', 'saved']):
                predicted_folders[i] = 'drafts'
            # Sent indicators
            elif any(word in subject for word in ['re:', 'fwd:', 'fw:']):
                predicted_folders[i] = 'sent'
            # Trash indicators
            elif any(word in text for word in ['deleted', 'trash']):
                predicted_folders[i] = 'deleted'
            else:
                predicted_folders[i] = 'inbox'
    
    return predicted_folders


def classify_emails(
    df: pd.DataFrame,
    classifier: UnifiedClassifier,
    text_col: str = 'text',
    subject_col: str = 'subject'
) -> EmailClassifications:
    """
    Run unified classification on emails.
    
    Args:
        df: DataFrame with email data
        classifier: UnifiedClassifier with trained models
        text_col: Column name for email text
        subject_col: Column name for subject
        
    Returns:
        EmailClassifications with all predictions
    """
    logger.info(f"Classifying {len(df)} emails...")
    
    texts = df[text_col].fillna('').astype(str).tolist()
    
    # Action required
    if classifier.action_classifier:
        action_prob = predict_action_required(
            classifier.action_classifier,
            texts,
            return_proba=True
        )
        action_pred = (action_prob > 0.5).astype(int)
    else:
        action_pred = np.zeros(len(df))
        action_prob = np.zeros(len(df))
    
    # Priority
    if classifier.priority_classifier:
        priority_pred = predict_priority(
            classifier.priority_classifier,
            df,
            text_col=text_col,
            return_proba=False
        )
        priority_proba = predict_priority(
            classifier.priority_classifier,
            df,
            text_col=text_col,
            return_proba=True
        )
    else:
        priority_pred = np.array(['medium'] * len(df))
        priority_proba = np.zeros((len(df), 3))  # low, medium, high
    
    # Category
    if classifier.category_classifier:
        category_pred = predict_category(
            classifier.category_classifier,
            texts,
            return_proba=False
        )
        category_proba = predict_category(
            classifier.category_classifier,
            texts,
            return_proba=True
        )
    else:
        category_pred = np.array(['general'] * len(df))
        category_proba = np.zeros((len(df), len(['general'])))
    
    # Urgency
    urgency_scores = detect_urgency(df, text_col=text_col, subject_col=subject_col)
    
    # Spam
    spam_pred, spam_prob = detect_spam(df, text_col=text_col)
    
    # Folder
    predicted_folder = predict_folder(df)
    
    return EmailClassifications(
        action_required=action_pred,
        action_probability=action_prob,
        priority=priority_pred,
        priority_probability=priority_proba,
        category=category_pred,
        category_probability=category_proba,
        urgency_score=urgency_scores,
        is_spam=spam_pred,
        spam_probability=spam_prob,
        predicted_folder=predicted_folder
    )


def add_classifications_to_dataframe(
    df: pd.DataFrame,
    classifications: EmailClassifications
) -> pd.DataFrame:
    """
    Add classification results to DataFrame.
    
    Args:
        df: Original DataFrame
        classifications: EmailClassifications object
        
    Returns:
        DataFrame with added classification columns
    """
    df = df.copy()
    
    df['action_required'] = classifications.action_required
    df['action_required_probability'] = classifications.action_probability
    df['priority'] = classifications.priority
    df['category'] = classifications.category
    df['urgency_score'] = classifications.urgency_score
    df['is_spam'] = classifications.is_spam
    df['spam_probability'] = classifications.spam_probability
    df['predicted_folder'] = classifications.predicted_folder
    
    return df


if __name__ == "__main__":
    # Test unified classification
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing unified email classification...")
    df = load_emails('maildir', data_format='maildir', max_rows=100)
    df = extract_temporal_features(df)
    
    # Create unified classifier (with empty models for now)
    unified = UnifiedClassifier()
    
    # Classify
    classifications = classify_emails(df, unified)
    
    # Add to DataFrame
    df_classified = add_classifications_to_dataframe(df, classifications)
    
    print(f"\nClassification summary:")
    print(f"  Action required: {df_classified['action_required'].sum()}/{len(df_classified)}")
    print(f"  Priority distribution:\n{df_classified['priority'].value_counts()}")
    print(f"  Category distribution:\n{df_classified['category'].value_counts()}")
    print(f"  Spam detected: {df_classified['is_spam'].sum()}/{len(df_classified)}")
    print(f"  Average urgency score: {df_classified['urgency_score'].mean():.2f}")

