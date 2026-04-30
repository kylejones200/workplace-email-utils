"""
Action-required detection classifier.

Detects whether an email or email sentence requires action from the recipient.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActionClassifier:
    """Container for action detection classifier."""
    model: any
    vectorizer: any
    model_type: str = "logistic_regression"
    accuracy: float = 0.0
    training_samples: int = 0


def train_action_classifier(
    train_df: pd.DataFrame,
    text_col: str = 'text',
    label_col: str = 'label',
    model_type: str = 'logistic_regression',
    tfidf_max_features: int = 5000,
    random_state: int = 42,
    cv_folds: int = 5
) -> ActionClassifier:
    """
    Train action-required detection classifier.
    
    Args:
        train_df: Training DataFrame with text and label columns
        text_col: Column name containing text
        label_col: Column name containing labels (1=action required, 0=no action)
        model_type: Type of classifier ('logistic_regression', 'random_forest', 'svm')
        tfidf_max_features: Maximum TF-IDF features
        random_state: Random seed
        cv_folds: Cross-validation folds
        
    Returns:
        Trained ActionClassifier
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for action classification. Install with: pip install scikit-learn")
    
    logger.info(f"Training action classifier (model: {model_type})")
    logger.info(f"Training samples: {len(train_df)}")
    
    # Prepare data
    X_train = train_df[text_col].fillna('').astype(str).values
    y_train = train_df[label_col].values
    
    # Create TF-IDF features
    logger.info("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    logger.info(f"TF-IDF features shape: {X_train_tfidf.shape}")
    
    # Train classifier
    logger.info(f"Training {model_type} classifier...")
    
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == 'svm':
        model = SVC(
            kernel='linear',
            probability=True,
            random_state=random_state,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train_tfidf, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv_folds, scoring='accuracy')
    accuracy = cv_scores.mean()
    logger.info(f"Cross-validation accuracy: {accuracy:.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return ActionClassifier(
        model=model,
        vectorizer=vectorizer,
        model_type=model_type,
        accuracy=accuracy,
        training_samples=len(train_df)
    )


def predict_action_required(
    classifier: ActionClassifier,
    texts: List[str],
    return_proba: bool = False
) -> np.ndarray:
    """
    Predict if texts require action.
    
    Args:
        classifier: Trained ActionClassifier
        texts: List of text strings to classify
        return_proba: If True, return probabilities instead of binary predictions
        
    Returns:
        Array of predictions (1=action required, 0=no action) or probabilities
    """
    # Vectorize texts
    X = classifier.vectorizer.transform(texts)
    
    # Predict
    if return_proba:
        predictions = classifier.model.predict_proba(X)[:, 1]  # Probability of action required
    else:
        predictions = classifier.model.predict(X)
    
    return predictions


def evaluate_action_classifier(
    classifier: ActionClassifier,
    test_df: pd.DataFrame,
    text_col: str = 'text',
    label_col: str = 'label'
) -> Dict:
    """
    Evaluate action classifier on test set.
    
    Args:
        classifier: Trained ActionClassifier
        test_df: Test DataFrame
        text_col: Column name containing text
        label_col: Column name containing labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating action classifier...")
    
    X_test = test_df[text_col].fillna('').astype(str).values
    y_test = test_df[label_col].values
    
    # Predict
    y_pred = predict_action_required(classifier, X_test.tolist())
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Test accuracy: {accuracy:.3f}")
    logger.info(f"Precision (action): {report['1']['precision']:.3f}")
    logger.info(f"Recall (action): {report['1']['recall']:.3f}")
    logger.info(f"F1-score (action): {report['1']['f1-score']:.3f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def save_action_classifier(classifier: ActionClassifier, filepath: str):
    """
    Save action classifier to disk.
    
    Args:
        classifier: ActionClassifier to save
        filepath: Path to save file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(classifier, f)
    
    logger.info(f"Saved action classifier to {filepath}")


def load_action_classifier(filepath: str) -> ActionClassifier:
    """
    Load action classifier from disk.
    
    Args:
        filepath: Path to classifier file
        
    Returns:
        Loaded ActionClassifier
    """
    with open(filepath, 'rb') as f:
        classifier = pickle.load(f)
    
    logger.info(f"Loaded action classifier from {filepath}")
    return classifier


if __name__ == "__main__":
    # Test action classifier
    from .dataset_loader import load_enron_intent_dataset, prepare_classification_data
    
    print("Testing action classifier...")
    
    # Load dataset
    df = load_enron_intent_dataset(download_if_missing=True)
    
    # Prepare data
    train_df, test_df = prepare_classification_data(df, balance_classes=True)
    
    # Train classifier
    classifier = train_action_classifier(
        train_df,
        model_type='logistic_regression'
    )
    
    # Evaluate
    metrics = evaluate_action_classifier(classifier, test_df)
    
    # Test prediction
    test_texts = [
        "I need you to finish the report by Friday",
        "Hope things are well with you",
        "Please review the contract and let me know your thoughts"
    ]
    
    predictions = predict_action_required(classifier, test_texts, return_proba=True)
    
    print("\nTest predictions:")
    for text, proba in zip(test_texts, predictions):
        print(f"  '{text[:50]}...': {proba:.2f} (action required)")

