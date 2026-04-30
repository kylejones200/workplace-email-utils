"""
Dataset loader for Enron Intent Dataset.

Downloads and loads the verified Enron Intent Dataset for training classification models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
import urllib.request
import zipfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset URLs
DATASET_REPO = "https://github.com/Charlie9/enron_intent_dataset_verified"
RAW_BASE_URL = "https://raw.githubusercontent.com/Charlie9/enron_intent_dataset_verified/master"


def download_intent_dataset(output_dir: str = "data/intent_dataset") -> str:
    """
    Download the Enron Intent Dataset from GitHub.
    
    Args:
        output_dir: Directory to save dataset files
        
    Returns:
        Path to dataset directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading Enron Intent Dataset...")
    
    # Download positive intent file
    pos_url = f"{RAW_BASE_URL}/intent_pos"
    pos_file = output_path / "intent_pos.txt"
    
    try:
        logger.info(f"Downloading positive intent data from {pos_url}")
        urllib.request.urlretrieve(pos_url, pos_file)
        logger.info(f"✓ Downloaded {pos_file}")
    except Exception as e:
        logger.warning(f"Failed to download positive intent file: {e}")
        logger.info("You can manually download from: https://github.com/Charlie9/enron_intent_dataset_verified")
        return str(output_path)
    
    # Download negative intent file
    neg_url = f"{RAW_BASE_URL}/intent_neg"
    neg_file = output_path / "intent_neg.txt"
    
    try:
        logger.info(f"Downloading negative intent data from {neg_url}")
        urllib.request.urlretrieve(neg_url, neg_file)
        logger.info(f"✓ Downloaded {neg_file}")
    except Exception as e:
        logger.warning(f"Failed to download negative intent file: {e}")
        logger.info("You can manually download from: https://github.com/Charlie9/enron_intent_dataset_verified")
        return str(output_path)
    
    logger.info(f"Dataset downloaded to {output_path}")
    return str(output_path)


def load_enron_intent_dataset(
    dataset_dir: Optional[str] = None,
    pos_file: Optional[str] = None,
    neg_file: Optional[str] = None,
    download_if_missing: bool = True
) -> pd.DataFrame:
    """
    Load the Enron Intent Dataset into a DataFrame.
    
    Args:
        dataset_dir: Directory containing intent_pos.txt and intent_neg.txt
        pos_file: Path to positive intent file (overrides dataset_dir)
        neg_file: Path to negative intent file (overrides dataset_dir)
        download_if_missing: Download dataset if files not found
        
    Returns:
        DataFrame with 'text' and 'label' columns
        - label: 1 for positive intent (action required), 0 for negative
    """
    # Determine file paths
    if pos_file is None or neg_file is None:
        if dataset_dir is None:
            dataset_dir = "data/intent_dataset"
        
        dataset_path = Path(dataset_dir)
        pos_file = pos_file or (dataset_path / "intent_pos.txt")
        neg_file = neg_file or (dataset_path / "intent_neg.txt")
    
    pos_path = Path(pos_file)
    neg_path = Path(neg_file)
    
    # Download if missing
    if download_if_missing and (not pos_path.exists() or not neg_path.exists()):
        logger.info("Dataset files not found. Downloading...")
        dataset_dir = download_intent_dataset(str(pos_path.parent))
        pos_path = Path(dataset_dir) / "intent_pos.txt"
        neg_path = Path(dataset_dir) / "intent_neg.txt"
    
    # Load positive examples
    if not pos_path.exists():
        raise FileNotFoundError(
            f"Positive intent file not found: {pos_path}\n"
            f"Download from: {DATASET_REPO}"
        )
    
    logger.info(f"Loading positive intent examples from {pos_path}")
    with open(pos_path, 'r', encoding='utf-8', errors='ignore') as f:
        positive_texts = [line.strip() for line in f if line.strip()]
    
    # Load negative examples
    if not neg_path.exists():
        raise FileNotFoundError(
            f"Negative intent file not found: {neg_path}\n"
            f"Download from: {DATASET_REPO}"
        )
    
    logger.info(f"Loading negative intent examples from {neg_path}")
    with open(neg_path, 'r', encoding='utf-8', errors='ignore') as f:
        negative_texts = [line.strip() for line in f if line.strip()]
    
    # Create DataFrame
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'intent_type': ['action_required'] * len(positive_texts) + ['no_action'] * len(negative_texts)
    })
    
    logger.info(f"Loaded Enron Intent Dataset:")
    logger.info(f"  Total examples: {len(df)}")
    logger.info(f"  Positive (action required): {len(positive_texts)} ({len(positive_texts)/len(df)*100:.1f}%)")
    logger.info(f"  Negative (no action): {len(negative_texts)} ({len(negative_texts)/len(df)*100:.1f}%)")
    
    return df


def prepare_classification_data(
    df: pd.DataFrame,
    text_col: str = 'text',
    test_size: float = 0.2,
    random_state: int = 42,
    balance_classes: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare classification data with train/test split and optional class balancing.
    
    Args:
        df: DataFrame with text and label columns
        text_col: Column name containing text
        test_size: Proportion of data for testing
        random_state: Random seed
        balance_classes: Whether to balance classes (undersample majority)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Preparing classification data...")
    
    # Separate features and labels
    X = df[text_col].values
    y = df['label'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_df = pd.DataFrame({text_col: X_train, 'label': y_train})
    test_df = pd.DataFrame({text_col: X_test, 'label': y_test})
    
    # Balance classes if requested
    if balance_classes:
        # Undersample majority class
        pos_train = train_df[train_df['label'] == 1]
        neg_train = train_df[train_df['label'] == 0]
        
        min_samples = min(len(pos_train), len(neg_train))
        
        if len(pos_train) > min_samples:
            pos_train = pos_train.sample(n=min_samples, random_state=random_state)
        if len(neg_train) > min_samples:
            neg_train = neg_train.sample(n=min_samples, random_state=random_state)
        
        train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        logger.info(f"Balanced training set: {len(train_df[train_df['label']==1])} positive, {len(train_df[train_df['label']==0])} negative")
    
    logger.info(f"Train set: {len(train_df)} examples")
    logger.info(f"Test set: {len(test_df)} examples")
    
    return train_df, test_df


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Enron Intent Dataset loader...")
    
    try:
        df = load_enron_intent_dataset(download_if_missing=True)
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nSample positive examples:")
        print(df[df['label']==1]['text'].head(3).tolist())
        print(f"\nSample negative examples:")
        print(df[df['label']==0]['text'].head(3).tolist())
        
        # Test train/test split
        train_df, test_df = prepare_classification_data(df)
        print(f"\n✓ Train/test split created:")
        print(f"  Train: {len(train_df)} examples")
        print(f"  Test: {len(test_df)} examples")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"\nTo download manually:")
        print(f"  1. Clone: git clone {DATASET_REPO}")
        print(f"  2. Copy intent_pos and intent_neg to data/intent_dataset/")

