"""
Real-time email processing.

Processes emails in real-time as they arrive in the stream.
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RealTimeProcessor:
    """Container for real-time processing configuration."""
    classify: bool = True
    extract_features: bool = True
    detect_anomalies: bool = True
    max_processing_time: float = 1.0  # seconds per email


def process_email_stream(
    email_batch: pd.DataFrame,
    processor: Optional[RealTimeProcessor] = None
) -> pd.DataFrame:
    """
    Process a batch of emails in real-time.
    
    Args:
        email_batch: DataFrame with email batch
        processor: Optional RealTimeProcessor configuration
        
    Returns:
        Processed DataFrame with classifications and features
    """
    if processor is None:
        processor = RealTimeProcessor()
    
    logger.info(f"Real-time processing: {len(email_batch)} emails")
    start_time = time.time()
    
    result_df = email_batch.copy()
    
    # Quick classification
    if processor.classify:
        try:
            from workplace_email_utils.classification.unified import classify_emails, UnifiedClassifier
            classifier = UnifiedClassifier()
            classifications = classify_emails(result_df, classifier)
            from workplace_email_utils.classification.unified import add_classifications_to_dataframe
            result_df = add_classifications_to_dataframe(result_df, classifications)
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
    
    # Quick feature extraction
    if processor.extract_features:
        try:
            from workplace_email_utils.temporal_features.extractors import extract_temporal_features
            result_df = extract_temporal_features(result_df)
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
    
    # Anomaly detection (lightweight)
    if processor.detect_anomalies:
        try:
            from workplace_email_utils.anomaly_detection.temporal import detect_volume_spikes
            # Only run lightweight checks
            result_df = detect_volume_spikes(result_df)
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
    
    processing_time = time.time() - start_time
    avg_time = processing_time / len(email_batch) if len(email_batch) > 0 else 0
    
    logger.info(f"Processing complete: {processing_time:.2f}s ({avg_time:.3f}s per email)")
    
    return result_df


def real_time_classify(
    email_text: str,
    classifier = None
) -> Dict[str, Any]:
    """
    Classify a single email in real-time.
    
    Args:
        email_text: Email text to classify
        classifier: Optional classifier object
        
    Returns:
        Dictionary with classification results
    """
    start_time = time.time()
    
    # Quick classification
    result = {
        'text': email_text[:100],  # Preview
        'classification_time': 0.0
    }
    
    try:
        from workplace_email_utils.classification.unified import UnifiedClassifier, classify_emails
        from workplace_email_utils.classification.unified import add_classifications_to_dataframe
        
        if classifier is None:
            classifier = UnifiedClassifier()
        
        # Create single-row DataFrame
        df = pd.DataFrame({'text': [email_text]})
        classifications = classify_emails(df, classifier)
        
        result.update({
            'action_required': int(classifications.action_required[0]) if len(classifications.action_required) > 0 else 0,
            'priority': str(classifications.priority[0]) if len(classifications.priority) > 0 else 'medium',
            'category': str(classifications.category[0]) if len(classifications.category) > 0 else 'general',
            'urgency_score': float(classifications.urgency_score[0]) if len(classifications.urgency_score) > 0 else 0.0,
        })
        
    except Exception as e:
        logger.warning(f"Real-time classification failed: {e}")
    
    result['classification_time'] = time.time() - start_time
    return result


if __name__ == "__main__":
    # Test real-time processing
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing real-time processing...")
    df = load_emails('maildir', data_format='maildir', max_rows=10)
    
    processor = RealTimeProcessor()
    result = process_email_stream(df, processor)
    
    print(f"✓ Processed {len(result)} emails in real-time")
    if 'priority' in result.columns:
        print(f"  Priorities: {result['priority'].value_counts().to_dict()}")

