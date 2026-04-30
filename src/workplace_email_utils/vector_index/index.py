"""
Vector index module for semantic search.

Uses FAISS for efficient similarity search.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Vector index will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorIndex:
    """Container for vector index."""
    index: Optional[any]  # FAISS index
    dim: int
    doc_ids: List[str]
    available: bool = False


def build_vector_index(X: np.ndarray, doc_ids: List[str]) -> VectorIndex:
    """
    Build FAISS vector index for similarity search.
    
    Args:
        X: Feature matrix (n_docs, n_features)
        doc_ids: List of document IDs
        
    Returns:
        VectorIndex object
    """
    if not FAISS_AVAILABLE:
        logger.warning("FAISS not available. Returning empty index.")
        return VectorIndex(
            index=None,
            dim=X.shape[1],
            doc_ids=doc_ids,
            available=False
        )
    
    logger.info(f"Building FAISS index for {len(doc_ids)} documents")
    
    dim = X.shape[1]
    X_norm = X.astype(np.float32).copy()
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(X_norm)
    
    # Use inner product index (for normalized vectors, inner product = cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(X_norm)
    
    logger.info("FAISS index built successfully")
    
    return VectorIndex(
        index=index,
        dim=dim,
        doc_ids=doc_ids,
        available=True
    )


def query_index(vindex: VectorIndex,
               query_vec: np.ndarray,
               top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Query the vector index for similar documents.
    
    Args:
        vindex: VectorIndex object
        query_vec: Query vector (n_features,)
        top_k: Number of results to return
        
    Returns:
        List of (doc_id, similarity_score) tuples
    """
    if not vindex.available or vindex.index is None:
        raise RuntimeError("Vector index is not available. Install faiss-cpu.")
    
    # Normalize query vector
    q = query_vec.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    
    # Search
    scores, indices = vindex.index.search(q, top_k)
    
    # Format results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(vindex.doc_ids):
            continue
        results.append((vindex.doc_ids[int(idx)], float(score)))
    
    return results

