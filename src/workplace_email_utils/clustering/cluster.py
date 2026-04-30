"""
Clustering module for document grouping.

Uses HDBSCAN and KMeans for clustering documents.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from sklearn.cluster import KMeans, DBSCAN

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available. Will use DBSCAN as fallback.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Container for clustering results."""
    hdbscan_labels: np.ndarray
    hdbscan_probs: np.ndarray
    kmeans_labels: np.ndarray
    kmeans_model: KMeans
    hdbscan_model: hdbscan.HDBSCAN


def cluster_documents(X_reduced: np.ndarray,
                     min_cluster_size: int = 30,
                     kmeans_k: int = 50,
                     hdbscan_min_samples: Optional[int] = None) -> ClusterResult:
    """
    Cluster documents using HDBSCAN and KMeans.
    
    Args:
        X_reduced: Reduced feature matrix (n_docs, n_features)
        min_cluster_size: Minimum cluster size for HDBSCAN
        kmeans_k: Number of clusters for KMeans
        hdbscan_min_samples: Minimum samples for HDBSCAN (defaults to min_cluster_size)
        
    Returns:
        ClusterResult with cluster labels and models
    """
    logger.info("Clustering documents")
    
    # HDBSCAN clustering (or DBSCAN fallback)
    if HDBSCAN_AVAILABLE:
        logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size})")
        if hdbscan_min_samples is None:
            hdbscan_min_samples = min_cluster_size
        
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        hdb_labels = hdb.fit_predict(X_reduced)
        hdb_probs = hdb.probabilities_ if hasattr(hdb, 'probabilities_') else np.ones(len(hdb_labels))
        
        n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = list(hdb_labels).count(-1)
        logger.info(f"HDBSCAN found {n_clusters_hdb} clusters, {n_noise} noise points")
    else:
        logger.info(f"Running DBSCAN (HDBSCAN not available)")
        # Use DBSCAN as fallback
        hdb = DBSCAN(eps=0.5, min_samples=min_cluster_size)
        hdb_labels = hdb.fit_predict(X_reduced)
        hdb_probs = np.ones(len(hdb_labels))  # DBSCAN doesn't have probabilities
        n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = list(hdb_labels).count(-1)
        logger.info(f"DBSCAN found {n_clusters_hdb} clusters, {n_noise} noise points")
    
    # KMeans clustering
    logger.info(f"Running KMeans (k={kmeans_k})")
    km = KMeans(
        n_clusters=kmeans_k,
        random_state=42,
        n_init='auto'
    )
    km_labels = km.fit_predict(X_reduced)
    
    logger.info(f"KMeans found {kmeans_k} clusters")
    
    return ClusterResult(
        hdbscan_labels=hdb_labels,
        hdbscan_probs=hdb_probs,
        kmeans_labels=km_labels,
        kmeans_model=km,
        hdbscan_model=hdb
    )

