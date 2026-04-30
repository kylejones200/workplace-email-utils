"""
Classification module for email classification.

Includes:
- Action-required detection
- Priority classification
- Category classification
- Email routing
"""

from .action_detection import (
    train_action_classifier,
    predict_action_required,
    ActionClassifier
)
from .priority import (
    train_priority_classifier,
    predict_priority,
    PriorityClassifier,
    extract_priority_features
)
from .category import (
    train_category_classifier,
    predict_category,
    CategoryClassifier,
    auto_categorize_emails
)
from .unified import (
    UnifiedClassifier,
    EmailClassifications,
    classify_emails,
    detect_urgency,
    detect_spam,
    predict_folder,
    add_classifications_to_dataframe
)
from .dataset_loader import (
    load_enron_intent_dataset,
    download_intent_dataset,
    prepare_classification_data
)

__all__ = [
    'train_action_classifier',
    'predict_action_required',
    'ActionClassifier',
    'train_priority_classifier',
    'predict_priority',
    'PriorityClassifier',
    'extract_priority_features',
    'train_category_classifier',
    'predict_category',
    'CategoryClassifier',
    'auto_categorize_emails',
    'UnifiedClassifier',
    'EmailClassifications',
    'classify_emails',
    'detect_urgency',
    'detect_spam',
    'predict_folder',
    'add_classifications_to_dataframe',
    'load_enron_intent_dataset',
    'download_intent_dataset',
    'prepare_classification_data',
]

