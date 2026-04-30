"""
Email category classification.

Classifies emails into categories: sales, support, HR, legal, finance, operations, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CategoryClassifier:
    """Container for category classification model."""
    model: any
    vectorizer: any
    categories: List[str]
    accuracy: float = 0.0


# Category keyword patterns
CATEGORY_PATTERNS = {
    'sales': [
        r'\b(sale|purchase|buy|order|quote|price|cost|invoice|contract|deal|proposal|client|customer)\b',
        r'\b(revenue|commission|discount|pricing|negotiat)\b',
    ],
    'support': [
        r'\b(support|help|issue|problem|bug|error|fix|resolve|troubleshoot|ticket)\b',
        r'\b(customer service|technical support|assistance|resolve|broken)\b',
    ],
    'hr': [
        r'\b(hire|hiring|job|position|resume|cv|interview|employee|staff|recruit)\b',
        r'\b(vacation|pto|leave|benefit|salary|payroll|performance review)\b',
    ],
    'legal': [
        r'\b(legal|law|attorney|lawyer|contract|agreement|compliance|regulation|litigation)\b',
        r'\b(nda|terms|conditions|liability|disclaimer|copyright)\b',
    ],
    'finance': [
        r'\b(budget|expense|cost|payment|invoice|accounting|financial|revenue|profit)\b',
        r'\b(tax|audit|account|balance|transaction|refund|payment|billing)\b',
    ],
    'operations': [
        r'\b(meeting|schedule|calendar|project|task|deadline|deliverable|status)\b',
        r'\b(process|procedure|workflow|operations|production|supply|logistics)\b',
    ],
    'general': [],  # Default category
}


def extract_category_features(text: str) -> Dict[str, int]:
    """
    Extract category indicator features from text.
    
    Args:
        text: Email text
        
    Returns:
        Dictionary with category match counts
    """
    text_lower = text.lower()
    features = {}
    
    for category, patterns in CATEGORY_PATTERNS.items():
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            count += len(matches)
        features[f'{category}_keyword_count'] = count
        features[f'has_{category}'] = 1 if count > 0 else 0
    
    return features


def auto_categorize_emails(
    df: pd.DataFrame,
    text_col: str = 'text',
    subject_col: str = 'subject'
) -> pd.DataFrame:
    """
    Auto-categorize emails based on keyword patterns.
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        subject_col: Column name for subject
        
    Returns:
        DataFrame with added 'category' column
    """
    logger.info("Auto-categorizing emails based on keyword patterns...")
    
    df = df.copy()
    
    # Combine subject and body for analysis
    df['full_text'] = (
        df.get(subject_col, '').fillna('').astype(str) + ' ' + 
        df.get(text_col, '').fillna('').astype(str)
    )
    
    categories = []
    category_scores = {cat: [] for cat in CATEGORY_PATTERNS.keys()}
    
    for _, row in df.iterrows():
        text = row['full_text']
        
        # Calculate scores for each category
        scores = {}
        for category, patterns in CATEGORY_PATTERNS.items():
            if category == 'general':
                continue
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            scores[category] = score
            category_scores[category].append(score)
        
        # Assign category with highest score
        if max(scores.values()) > 0:
            category = max(scores.items(), key=lambda x: x[1])[0]
        else:
            category = 'general'
        
        categories.append(category)
    
    df['category'] = categories
    
    # Add category scores as features
    for category in CATEGORY_PATTERNS.keys():
        if category != 'general':
            df[f'{category}_score'] = category_scores[category]
    
    logger.info(f"Category distribution:\n{df['category'].value_counts()}")
    
    return df


def train_category_classifier(
    df: pd.DataFrame,
    category_col: Optional[str] = None,
    auto_generate_labels: bool = True,
    text_col: str = 'text',
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42
) -> CategoryClassifier:
    """
    Train category classification model.
    
    Args:
        df: DataFrame with email data
        category_col: Column name with category labels
        auto_generate_labels: Auto-generate labels based on keywords if category_col not provided
        text_col: Column name for email text
        model_type: Type of classifier ('random_forest', 'naive_bayes')
        test_size: Proportion for testing
        random_state: Random seed
        
    Returns:
        Trained CategoryClassifier
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for category classification")
    
    logger.info("Training category classifier")
    
    # Auto-categorize if needed
    if category_col and category_col in df.columns:
        categories = df[category_col].values
        df_labeled = df.copy()
    elif auto_generate_labels:
        df_labeled = auto_categorize_emails(df, text_col=text_col)
        categories = df_labeled['category'].values
    else:
        raise ValueError("Must provide category_col or set auto_generate_labels=True")
    
    # Get unique categories
    unique_categories = sorted(list(set(categories)))
    logger.info(f"Categories: {unique_categories}")
    
    # Prepare texts
    texts = df_labeled[text_col].fillna('').astype(str).tolist()
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(texts)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(categories)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
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
    elif model_type == 'naive_bayes':
        model = MultinomialNB(alpha=1.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.3f}")
    
    category_names = label_encoder.inverse_transform(range(len(unique_categories)))
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=category_names)}")
    
    # Store label encoder in model for later use
    model.label_encoder_ = label_encoder
    
    return CategoryClassifier(
        model=model,
        vectorizer=vectorizer,
        categories=unique_categories,
        accuracy=accuracy
    )


def predict_category(
    classifier: CategoryClassifier,
    texts: List[str],
    return_proba: bool = False
) -> np.ndarray:
    """
    Predict categories for emails.
    
    Args:
        classifier: Trained CategoryClassifier
        texts: List of email texts
        return_proba: If True, return probability distributions
        
    Returns:
        Array of category predictions or probability distributions
    """
    # Vectorize texts
    X = classifier.vectorizer.transform(texts)
    
    # Predict
    if return_proba:
        predictions = classifier.model.predict_proba(X)
    else:
        predictions = classifier.model.predict(X)
        # Map back to category names
        if hasattr(classifier.model, 'label_encoder_'):
            predictions = classifier.model.label_encoder_.inverse_transform(predictions)
        else:
            predictions = np.array([classifier.categories[int(p)] for p in predictions])
    
    return predictions


if __name__ == "__main__":
    # Test category classifier
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing category classification...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    
    # Auto-categorize
    df_categorized = auto_categorize_emails(df)
    print(f"\nCategory distribution:")
    print(df_categorized['category'].value_counts())
    
    # Train classifier
    classifier = train_category_classifier(df, auto_generate_labels=True)
    
    # Test prediction
    test_texts = [
        "I need help with my account",
        "Please review this contract",
        "We're hiring for a new position",
        "The invoice is ready for payment"
    ]
    
    predictions = predict_category(classifier, test_texts)
    print("\nCategory predictions:")
    for text, category in zip(test_texts, predictions):
        print(f"  '{text[:40]}...': {category}")

