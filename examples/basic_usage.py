"""
Basic usage example for the Enterprise Email Analytics Platform.

Demonstrates how to:
1. Load and parse emails
2. Build analytics model
3. Access classification and analysis results
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from workplace_email_utils.pipeline import build_knowledge_model

def main():
    """Basic example of processing emails and building analytics model."""
    
    print("Building email analytics model...")
    print("This may take a few minutes...\n")
    
    # Build the model
    model = build_knowledge_model(
        data_path='maildir',
        data_format='maildir',
        sample_size=5000,
        enable_threading=True,
        enable_classification=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total emails processed: {len(model.df)}")
    
    # Thread analysis
    if model.thread_data:
        print(f"\nThread Analysis:")
        print(f"  Total threads: {len(model.thread_data['thread_trees'])}")
        avg_thread_size = sum(
            t.message_count for t in model.thread_data['thread_trees'].values()
        ) / len(model.thread_data['thread_trees']) if model.thread_data['thread_trees'] else 0
        print(f"  Average thread size: {avg_thread_size:.1f} messages")
    
    # Classification results
    if model.classifications:
        print(f"\nClassification Results:")
        print(f"  High priority: {(model.classifications.priority == 'high').sum()}")
        print(f"  Action required: {model.classifications.action_required.sum()}")
        print(f"  Spam detected: {model.classifications.is_spam.sum()}")
    
    # Top categories
    if 'category' in model.df.columns:
        print(f"\nCategory Distribution:")
        top_categories = model.df['category'].value_counts().head(5)
        for category, count in top_categories.items():
            print(f"  {category}: {count}")
    
    print("\n" + "=" * 60)
    print("Model building complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

