"""
Main pipeline for knowledge management system.

Orchestrates all layers: ingestion -> features -> fusion -> clustering -> indexing
"""

import pandas as pd
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
import logging
import pickle
import os

from workplace_email_utils.ingest.email_parser import load_enron_csv, load_emails
from workplace_email_utils.content_features.extractors import build_content_features
from workplace_email_utils.graph_features.extractors import build_email_graph, compute_graph_features
from workplace_email_utils.graph_features.communities import detect_tight_knit_groups
from workplace_email_utils.graph_features.executive_analysis import analyze_executive_network, identify_key_executives
from workplace_email_utils.anomaly_detection.communication_patterns import detect_communication_anomalies
from workplace_email_utils.temporal_features.extractors import extract_temporal_features
from workplace_email_utils.entity_extraction.extractors import extract_entities_from_dataframe
from workplace_email_utils.threading.reconstruct import reconstruct_threads
from workplace_email_utils.threading.analysis import compute_thread_metrics, analyze_all_threads
from workplace_email_utils.classification.unified import (
    UnifiedClassifier,
    classify_emails,
    add_classifications_to_dataframe
)
from workplace_email_utils.classification.action_detection import train_action_classifier
from workplace_email_utils.classification.priority import train_priority_classifier
from workplace_email_utils.classification.category import train_category_classifier
from workplace_email_utils.fusion.combine import fuse_features
from workplace_email_utils.clustering.cluster import cluster_documents
from workplace_email_utils.vector_index.index import build_vector_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeModel:
    """Complete knowledge management model."""
    df: pd.DataFrame
    content_features: any
    graph_features: any
    fusion_result: any
    cluster_result: any
    vector_index: any
    executive_network: Optional[any] = None
    communication_anomalies: Optional[any] = None
    thread_data: Optional[any] = None
    classifications: Optional[any] = None
    classifier: Optional[any] = None


def build_knowledge_model(data_path: str,
                         data_format: str = 'auto',
                         max_rows: Optional[int] = None,
                         sample_size: Optional[int] = 10000,
                         n_topics_lda: int = 30,
                         n_topics_nmf: int = 30,
                         n_topics_plsa: int = 30,
                         embed_model_name: str = "all-MiniLM-L6-v2",
                         min_cluster_size: int = 30,
                         kmeans_k: int = 50,
                         umap_n_components: int = 50,
                         use_plsa: bool = True,
                         use_gibbs_lda: bool = False,
                         user_filter: Optional[list] = None,
                         folder_filter: Optional[list] = None,
                         knowledge_base: Optional[object] = None,
                         knowledge_base_path: Optional[str] = None,
                         enable_executive_analysis: bool = False,
                         enable_anomaly_detection: bool = False,
                         key_executives: Optional[list] = None,
                         enable_threading: bool = True,
                         enable_classification: bool = True,
                         train_classifiers: bool = False,
                         intent_dataset_path: Optional[str] = None) -> KnowledgeModel:
    """
    Build complete knowledge management model from raw email data (CSV or maildir).
    
    Args:
        data_path: Path to emails.csv file or maildir directory
        data_format: 'csv', 'maildir', or 'auto' (detects based on path)
        max_rows: Maximum rows/emails to process (None for all)
        sample_size: Random sample size (None for no sampling)
        n_topics_lda: Number of LDA topics
        n_topics_nmf: Number of NMF topics
        n_topics_plsa: Number of pLSA topics
        embed_model_name: Sentence transformer model name
        min_cluster_size: Minimum cluster size for HDBSCAN
        kmeans_k: Number of KMeans clusters
        umap_n_components: UMAP target dimensionality
        use_plsa: Include pLSA topic modeling
        use_gibbs_lda: Use Gibbs sampling LDA (requires lda package)
        user_filter: Optional list of user folders to include (maildir only)
        folder_filter: Optional list of folder types to include (maildir only)
        knowledge_base: Optional KnowledgeBase object with known entities
        knowledge_base_path: Optional path to knowledge base JSON file
        enable_executive_analysis: Enable executive network analysis (based on Enron paper)
        enable_anomaly_detection: Enable communication pattern anomaly detection
        key_executives: Optional list of key executive email addresses
        enable_threading: Enable email threading and conversation analysis (Phase 1.3)
        enable_classification: Enable email classification (Phase 1.4)
        train_classifiers: Train classifiers from data (if False, uses heuristics)
        intent_dataset_path: Path to Enron Intent Dataset for training action classifier
        
    Returns:
        KnowledgeModel with all features and results
    """
    logger.info("=" * 60)
    logger.info("Building Knowledge Management Model")
    logger.info("=" * 60)
    
    # Layer 1: Ingestion
    logger.info("\n[Layer 1] Ingestion")
    df = load_emails(
        data_path, 
        data_format=data_format,
        max_rows=max_rows, 
        sample_size=sample_size,
        user_filter=user_filter,
        folder_filter=folder_filter
    )
    logger.info(f"Loaded {len(df)} emails")
    
    # Log folder type distribution if available (from maildir)
    if 'folder_type' in df.columns:
        logger.info(f"Folder type distribution:\n{df['folder_type'].value_counts()}")
    
    # Extract temporal features
    logger.info("\n[Layer 1.5] Temporal Features")
    df = extract_temporal_features(df)
    logger.info(f"Extracted temporal features for {len(df)} emails")
    
    # Load knowledge base if path provided
    if knowledge_base_path and knowledge_base is None:
        try:
            from workplace_email_utils.entity_extraction.knowledge_base import KnowledgeBase
            knowledge_base = KnowledgeBase.load(knowledge_base_path)
            logger.info(f"Loaded knowledge base from {knowledge_base_path}")
        except Exception as e:
            logger.warning(f"Failed to load knowledge base from {knowledge_base_path}: {e}")
    
    # Extract entities
    logger.info("\n[Layer 1.6] Entity Extraction")
    df = extract_entities_from_dataframe(df, text_col='text', knowledge_base=knowledge_base)
    logger.info(f"Extracted entities for {len(df)} emails")
    
    # Layer 1.7: Thread Reconstruction (Phase 1.3)
    thread_trees = {}
    thread_metrics = {}
    if enable_threading:
        logger.info("\n[Layer 1.7] Email Threading & Conversation Analysis")
        try:
            df, thread_trees = reconstruct_threads(df)
            logger.info(f"Reconstructed {len(thread_trees)} threads")
            
            # Compute thread metrics
            thread_metrics = compute_thread_metrics(df, thread_trees)
            logger.info(f"Computed metrics for {len(thread_metrics)} threads")
            
            # Get thread summary statistics
            if thread_metrics:
                avg_thread_size = sum(m.message_count for m in thread_metrics.values()) / len(thread_metrics)
                max_thread_size = max(m.message_count for m in thread_metrics.values())
                logger.info(f"Thread statistics: avg_size={avg_thread_size:.1f}, max_size={max_thread_size}")
        except Exception as e:
            logger.warning(f"Thread reconstruction failed: {e}")
            thread_trees = {}
            thread_metrics = {}
    
    # Layer 1.8: Email Classification (Phase 1.4)
    unified_classifier = None
    classifications_result = None
    if enable_classification:
        logger.info("\n[Layer 1.8] Email Classification")
        try:
            from workplace_email_utils.classification.unified import UnifiedClassifier, classify_emails, add_classifications_to_dataframe
            
            # Create unified classifier (heuristic-based if not training)
            unified_classifier = UnifiedClassifier()
            
            # Optionally train classifiers if requested
            if train_classifiers:
                logger.info("Training classification models...")
                
                # Train action classifier if dataset available
                if intent_dataset_path:
                    try:
                        from workplace_email_utils.classification.dataset_loader import load_enron_intent_dataset, prepare_classification_data
                        from workplace_email_utils.classification.action_detection import train_action_classifier
                        
                        intent_df = load_enron_intent_dataset(dataset_dir=intent_dataset_path, download_if_missing=False)
                        train_df, _ = prepare_classification_data(intent_df)
                        
                        unified_classifier.action_classifier = train_action_classifier(
                            train_df,
                            model_type='logistic_regression'
                        )
                        logger.info("✓ Action classifier trained")
                    except Exception as e:
                        logger.warning(f"Failed to train action classifier: {e}")
                
                # Train priority classifier
                try:
                    unified_classifier.priority_classifier = train_priority_classifier(
                        df,
                        auto_generate_labels=True
                    )
                    logger.info("✓ Priority classifier trained")
                except Exception as e:
                    logger.warning(f"Failed to train priority classifier: {e}")
                
                # Train category classifier
                try:
                    unified_classifier.category_classifier = train_category_classifier(
                        df,
                        auto_generate_labels=True
                    )
                    logger.info("✓ Category classifier trained")
                except Exception as e:
                    logger.warning(f"Failed to train category classifier: {e}")
            
            # Classify emails
            classifications_result = classify_emails(df, unified_classifier)
            
            # Add classifications to DataFrame
            df = add_classifications_to_dataframe(df, classifications_result)
            
            logger.info(f"Classification summary:")
            logger.info(f"  Action required: {classifications_result.action_required.sum()}/{len(df)}")
            logger.info(f"  High priority: {(classifications_result.priority == 'high').sum()}")
            logger.info(f"  Spam detected: {classifications_result.is_spam.sum()}")
            
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            unified_classifier = None
            classifications_result = None
    
    # Create doc_ids if not present
    if 'doc_id' not in df.columns:
        df['doc_id'] = [f"doc_{i}" for i in range(len(df))]
    
    # Prepare texts
    texts = df['text'].fillna('').astype(str).tolist()
    doc_ids = df['doc_id'].astype(str).tolist()
    
    # Layer 2: Content Features
    logger.info("\n[Layer 2] Content Features")
    content_features = build_content_features(
        texts,
        embed_model_name=embed_model_name,
        n_topics_lda=n_topics_lda,
        n_topics_nmf=n_topics_nmf,
        n_topics_plsa=n_topics_plsa,
        use_plsa=use_plsa,
        use_gibbs_lda=use_gibbs_lda
    )
    
    # Layer 3: Graph Features
    logger.info("\n[Layer 3] Graph Features")
    graph = build_email_graph(df)
    graph_features = compute_graph_features(df, graph)
    
    # Layer 3.5: Executive Network Analysis (Optional)
    executive_network = None
    if enable_executive_analysis:
        logger.info("\n[Layer 3.5] Executive Network Analysis")
        key_execs = None
        if key_executives:
            key_execs = {email.lower().strip() for email in key_executives}
        elif knowledge_base:
            # Try to get executives from knowledge base
            try:
                key_execs = knowledge_base.get_known_people()
            except:
                pass
        
        try:
            executive_network = analyze_executive_network(
                df,
                key_executives=key_execs,
                auto_identify=(key_execs is None),
                folder_filter='sent'
            )
            logger.info(f"Executive network: {executive_network.network_metrics}")
        except Exception as e:
            logger.warning(f"Executive network analysis failed: {e}")
    
    # Layer 3.6: Communication Pattern Anomaly Detection (Optional)
    communication_anomalies = None
    if enable_anomaly_detection:
        logger.info("\n[Layer 3.6] Communication Pattern Anomaly Detection")
        key_execs = None
        if key_executives:
            key_execs = {email.lower().strip() for email in key_executives}
        elif executive_network:
            # Use executives from executive network analysis
            try:
                key_execs = set(executive_network.executive_graph.nodes())
            except:
                pass
        
        try:
            communication_anomalies = detect_communication_anomalies(
                df,
                key_executives=key_execs,
                anomaly_threshold=0.3,
                folder_filter='sent'
            )
            logger.info(f"Communication anomalies: {communication_anomalies.metrics}")
        except Exception as e:
            logger.warning(f"Communication anomaly detection failed: {e}")
    
    # Layer 4: Feature Fusion
    logger.info("\n[Layer 4] Feature Fusion")
    fusion_result = fuse_features(
        content_features,
        graph_features,
        umap_n_components=umap_n_components
    )
    
    # Layer 5: Clustering
    logger.info("\n[Layer 5] Clustering")
    cluster_result = cluster_documents(
        fusion_result.X_reduced,
        min_cluster_size=min_cluster_size,
        kmeans_k=kmeans_k
    )
    
    # Layer 6: Vector Index
    logger.info("\n[Layer 6] Vector Index")
    vector_index = build_vector_index(fusion_result.X_scaled, doc_ids)
    
    logger.info("\n" + "=" * 60)
    logger.info("Model building complete!")
    logger.info("=" * 60)
    
    # Package thread data
    thread_data = None
    if enable_threading and thread_trees:
        thread_data = {
            'thread_trees': thread_trees,
            'thread_metrics': thread_metrics,
            'df_with_threads': df  # DataFrame with thread_id column
        }
    
    return KnowledgeModel(
        df=df,
        content_features=content_features,
        graph_features=graph_features,
        fusion_result=fusion_result,
        cluster_result=cluster_result,
        vector_index=vector_index,
        executive_network=executive_network,
        communication_anomalies=communication_anomalies,
        thread_data=thread_data,
        classifications=classifications_result,
        classifier=unified_classifier
    )


def save_model(model: KnowledgeModel, output_dir: str):
    """
    Save model to disk.
    
    Args:
        model: KnowledgeModel to save
        output_dir: Directory to save model files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    
    # Save DataFrame
    model.df.to_parquet(os.path.join(output_dir, 'emails.parquet'), index=False)
    
    # Save models and results
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump({
            'content_features': model.content_features,
            'graph_features': model.graph_features,
            'fusion_result': model.fusion_result,
            'cluster_result': model.cluster_result,
            'vector_index': model.vector_index
        }, f)
    
    logger.info("Model saved successfully")


def load_model(output_dir: str) -> KnowledgeModel:
    """
    Load model from disk.
    
    Args:
        output_dir: Directory containing model files
        
    Returns:
        Loaded KnowledgeModel
    """
    logger.info(f"Loading model from {output_dir}")
    
    df = pd.read_parquet(os.path.join(output_dir, 'emails.parquet'))
    
    with open(os.path.join(output_dir, 'model.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    return KnowledgeModel(
        df=df,
        content_features=data['content_features'],
        graph_features=data['graph_features'],
        fusion_result=data['fusion_result'],
        cluster_result=data['cluster_result'],
        vector_index=data['vector_index']
    )


if __name__ == "__main__":
    # Example usage
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Build knowledge management model from email data')
    parser.add_argument('data_path', nargs='?', default='emails.csv', 
                       help='Path to emails.csv file or maildir directory')
    parser.add_argument('--format', choices=['auto', 'csv', 'maildir'], default='auto',
                       help='Data format: auto (detect), csv, or maildir')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Random sample size (None for no sampling)')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum rows/emails to process')
    parser.add_argument('--folder-filter', nargs='+', default=None,
                       help='Filter specific folder types (maildir only, e.g., inbox sent)')
    
    args = parser.parse_args()
    
    logger.info(f"Processing {args.data_path} (format: {args.format}) with sample size {args.sample_size}")
    
    model = build_knowledge_model(
        data_path=args.data_path,
        data_format=args.format,
        sample_size=args.sample_size,
        max_rows=args.max_rows,
        folder_filter=args.folder_filter,
        n_topics_lda=30,
        n_topics_nmf=30,
        n_topics_plsa=30,
        min_cluster_size=30,
        kmeans_k=50
    )
    
    # Save model
    save_model(model, "model_output")
    
    # Print summary
    logger.info("\nModel Summary:")
    logger.info(f"  Documents: {len(model.df)}")
    logger.info(f"  HDBSCAN clusters: {len(set(model.cluster_result.hdbscan_labels)) - (1 if -1 in model.cluster_result.hdbscan_labels else 0)}")
    logger.info(f"  KMeans clusters: {model.cluster_result.kmeans_model.n_clusters}")
    logger.info(f"  Vector index available: {model.vector_index.available}")
    
    # Print folder metadata if available
    if 'folder_type' in model.df.columns:
        logger.info(f"  Folder types: {model.df['folder_type'].unique()}")

