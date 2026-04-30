"""
Content anomaly detection.

Detects unusual language, topics, or content patterns in emails.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContentAnomalyResult:
    """Container for content anomaly detection results."""
    anomalous_emails: pd.DataFrame
    anomaly_scores: Dict[str, float]
    anomaly_types: Dict[str, List[str]]  # anomaly_type -> [email_ids]
    metrics: Dict


def detect_unusual_language(
    df: pd.DataFrame,
    text_col: str = 'text',
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect emails with unusual language patterns.
    
    Indicators:
    - Unusual vocabulary (rare words)
    - Excessive use of special characters
    - Unusual sentence structure
    - Language inconsistencies
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly scores
    """
    logger.info("Detecting unusual language patterns...")
    
    df = df.copy()
    
    # Feature extraction
    texts = df[text_col].fillna('').astype(str)
    
    # Average word length
    df['avg_word_length'] = texts.apply(
        lambda x: np.mean([len(w) for w in re.findall(r'\b\w+\b', x)]) if x else 0
    )
    
    # Special character ratio
    df['special_char_ratio'] = texts.apply(
        lambda x: len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>?/~`]', x)) / max(len(x), 1)
    )
    
    # Capitalization ratio
    df['caps_ratio'] = texts.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    
    # Sentence length variance
    df['sentence_length_variance'] = texts.apply(
        lambda x: np.var([len(s.split()) for s in re.split(r'[.!?]+', x) if s.strip()]) if x else 0
    )
    
    # Calculate z-scores
    features = ['avg_word_length', 'special_char_ratio', 'caps_ratio', 'sentence_length_variance']
    anomaly_scores = []
    
    for feature in features:
        if feature in df.columns and df[feature].std() > 0:
            z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
            anomaly_scores.append(z_scores)
    
    if anomaly_scores:
        df['language_anomaly_score'] = np.mean(anomaly_scores, axis=0)
        df['is_language_anomaly'] = (df['language_anomaly_score'] > threshold).astype(int)
    else:
        df['language_anomaly_score'] = 0.0
        df['is_language_anomaly'] = 0
    
    logger.info(f"Detected {df['is_language_anomaly'].sum()} language anomalies")
    return df


def detect_topic_anomalies(
    df: pd.DataFrame,
    topic_col: Optional[str] = None,
    text_col: str = 'text',
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Detect emails with unusual topics.
    
    Args:
        df: DataFrame with email data
        topic_col: Column name for topic assignments
        text_col: Column name for email text
        threshold: Minimum frequency threshold for rare topics
        
    Returns:
        DataFrame with topic anomaly flags
    """
    logger.info("Detecting topic anomalies...")
    
    df = df.copy()
    
    if topic_col and topic_col in df.columns:
        # Use existing topic assignments
        topic_counts = df[topic_col].value_counts()
        rare_topics = set(topic_counts[topic_counts / len(df) < threshold].index)
        
        df['is_topic_anomaly'] = df[topic_col].isin(rare_topics).astype(int)
        df['topic_anomaly_score'] = df['is_topic_anomaly'].astype(float)
    else:
        # Simple keyword-based topic detection
        common_words = Counter()
        for text in df[text_col].fillna('').astype(str):
            words = re.findall(r'\b\w{4,}\b', text.lower())
            common_words.update(words)
        
        # Find rare words
        total_words = sum(common_words.values())
        rare_words = {word for word, count in common_words.items() if count / total_words < threshold}
        
        def has_rare_words(text):
            words = set(re.findall(r'\b\w{4,}\b', text.lower()))
            return len(words.intersection(rare_words)) > 0
        
        df['is_topic_anomaly'] = df[text_col].fillna('').astype(str).apply(has_rare_words).astype(int)
        df['topic_anomaly_score'] = df['is_topic_anomaly'].astype(float)
    
    logger.info(f"Detected {df['is_topic_anomaly'].sum()} topic anomalies")
    return df


def detect_content_anomalies(
    df: pd.DataFrame,
    text_col: str = 'text',
    subject_col: str = 'subject',
    language_threshold: float = 2.0,
    topic_threshold: float = 0.05
) -> ContentAnomalyResult:
    """
    Comprehensive content anomaly detection.
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        subject_col: Column name for subject
        language_threshold: Threshold for language anomalies
        topic_threshold: Threshold for topic anomalies
        
    Returns:
        ContentAnomalyResult with all content anomalies
    """
    logger.info("Detecting content anomalies...")
    
    df = df.copy()
    
    # Detect unusual language
    df = detect_unusual_language(df, text_col=text_col, threshold=language_threshold)
    
    # Detect topic anomalies
    df = detect_topic_anomalies(df, text_col=text_col, threshold=topic_threshold)
    
    # Combined anomaly score
    df['content_anomaly_score'] = (
        df.get('language_anomaly_score', 0) * 0.5 +
        df.get('topic_anomaly_score', 0) * 0.5
    )
    df['is_content_anomaly'] = (df['content_anomaly_score'] > 0.5).astype(int)
    
    # Filter anomalies
    anomalous_emails = df[df['is_content_anomaly'] == 1].copy()
    
    # Categorize anomaly types
    anomaly_types = {
        'language': df[df['is_language_anomaly'] == 1].index.tolist(),
        'topic': df[df['is_topic_anomaly'] == 1].index.tolist(),
    }
    
    anomaly_scores = dict(zip(df.index, df['content_anomaly_score']))
    
    metrics = {
        'total_anomalies': len(anomalous_emails),
        'language_anomalies': df['is_language_anomaly'].sum(),
        'topic_anomalies': df['is_topic_anomaly'].sum(),
        'avg_anomaly_score': df['content_anomaly_score'].mean(),
    }
    
    logger.info(f"Content anomalies: {metrics}")
    
    return ContentAnomalyResult(
        anomalous_emails=anomalous_emails,
        anomaly_scores=anomaly_scores,
        anomaly_types=anomaly_types,
        metrics=metrics
    )


if __name__ == "__main__":
    # Test content anomaly detection
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing content anomaly detection...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    
    result = detect_content_anomalies(df)
    print(f"\nContent anomalies: {result.metrics}")
    print(f"Anomalous emails: {len(result.anomalous_emails)}")

