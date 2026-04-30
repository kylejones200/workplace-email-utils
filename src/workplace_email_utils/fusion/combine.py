"""
Feature fusion and dimensionality reduction module.

Combines content, graph, and topic features, then reduces dimensionality.
"""

import numpy as np
from typing import List
from dataclasses import dataclass
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Will use PCA for dimensionality reduction.")

from workplace_email_utils.content_features.extractors import ContentFeatures
from workplace_email_utils.graph_features.extractors import GraphFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Container for fused and reduced features."""
    X_raw: np.ndarray          # Concatenated features
    X_scaled: np.ndarray       # Standardized features
    X_reduced: np.ndarray      # UMAP-reduced features
    scaler: StandardScaler
    reducer: umap.UMAP
    fused_feature_names: List[str]


def fuse_features(content: ContentFeatures,
                 graph: GraphFeatures,
                 umap_n_components: int = 50,
                 umap_n_neighbors: int = 30,
                 umap_min_dist: float = 0.1) -> FusionResult:
    """
    Fuse all feature types and reduce dimensionality.
    
    Args:
        content: Content features (embeddings, topics)
        graph: Graph features
        umap_n_components: Target dimensionality for UMAP
        umap_n_neighbors: UMAP neighborhood size
        umap_min_dist: UMAP minimum distance
        
    Returns:
        FusionResult with fused and reduced features
    """
    logger.info("Fusing features")
    
    # Concatenate all feature blocks
    X_blocks = [content.embeddings]
    
    if content.lda_topics is not None:
        X_blocks.append(content.lda_topics)
    
    if content.nmf_topics is not None:
        X_blocks.append(content.nmf_topics)
    
    if content.plsa_topics is not None:
        X_blocks.append(content.plsa_topics)
    
    if graph.feature_matrix is not None:
        X_blocks.append(graph.feature_matrix)
    
    X_raw = np.hstack(X_blocks).astype(np.float32)
    logger.info(f"Raw fused features shape: {X_raw.shape}")
    
    # Standardize features
    logger.info("Standardizing features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Reduce dimensionality with UMAP or PCA fallback
    if UMAP_AVAILABLE:
        logger.info(f"Reducing to {umap_n_components} dimensions with UMAP")
        reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        X_reduced = reducer.fit_transform(X_scaled)
    else:
        logger.info(f"Reducing to {umap_n_components} dimensions with PCA (UMAP not available)")
        reducer = PCA(n_components=umap_n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
    logger.info(f"Reduced features shape: {X_reduced.shape}")
    
    # Build feature names
    n_emb = content.embeddings.shape[1]
    fused_feature_names = [f'emb_{i}' for i in range(n_emb)]
    
    if content.lda_topics is not None:
        n_lda = content.lda_topics.shape[1]
        fused_feature_names.extend([f'lda_{i}' for i in range(n_lda)])
    
    if content.nmf_topics is not None:
        n_nmf = content.nmf_topics.shape[1]
        fused_feature_names.extend([f'nmf_{i}' for i in range(n_nmf)])
    
    if content.plsa_topics is not None:
        n_plsa = content.plsa_topics.shape[1]
        fused_feature_names.extend([f'plsa_{i}' for i in range(n_plsa)])
    
    if graph.feature_matrix is not None:
        fused_feature_names.extend(graph.feature_names)
    
    return FusionResult(
        X_raw=X_raw,
        X_scaled=X_scaled,
        X_reduced=X_reduced,
        scaler=scaler,
        reducer=reducer,
        fused_feature_names=fused_feature_names
    )

