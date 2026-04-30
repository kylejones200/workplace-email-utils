"""
Content feature extraction module.

Extracts embeddings, topic models (LDA, pLSA, NMF), and TF-IDF features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sentence_transformers import SentenceTransformer

try:
    import lda
    LDA_AVAILABLE = True
except ImportError:
    LDA_AVAILABLE = False
    logging.warning("lda package not available. Gibbs sampling LDA will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContentFeatures:
    """Container for all content-based features."""
    embeddings: np.ndarray          # (n_docs, emb_dim)
    lda_topics: np.ndarray          # (n_docs, n_topics_lda)
    nmf_topics: np.ndarray          # (n_docs, n_topics_nmf)
    plsa_topics: Optional[np.ndarray] = None  # (n_docs, n_topics_plsa)
    tfidf_vocab: List[str] = None
    lda_model: Optional[LatentDirichletAllocation] = None
    nmf_model: Optional[NMF] = None
    embed_model_name: str = ""


def build_embeddings(texts: List[str], 
                    model_name: str = "all-MiniLM-L6-v2",
                    batch_size: int = 64) -> np.ndarray:
    """
    Generate sentence embeddings using sentence transformers.
    
    Args:
        texts: List of text documents
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        
    Returns:
        Array of embeddings (n_docs, emb_dim)
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Encoding {len(texts)} documents")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def build_tfidf(texts: List[str], 
                max_features: int = 20000,
                min_df: int = 5,
                max_df: float = 0.8) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Build TF-IDF vectors for documents.
    
    Returns:
        Tuple of (vectorizer, tfidf_matrix)
    """
    logger.info("Building TF-IDF vectors")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)  # Include bigrams
    )
    
    tfidf = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF shape: {tfidf.shape}")
    
    return vectorizer, tfidf


def build_lda(tfidf_tuple, n_topics: int = 50, max_iter: int = 20) -> Tuple[LatentDirichletAllocation, np.ndarray]:
    """
    Build LDA topic model using scikit-learn (variational Bayes).
    
    Args:
        tfidf_tuple: Tuple of (tfidf_vectorizer, tfidf_matrix)
        n_topics: Number of topics
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (lda_model, doc_topic_matrix)
    """
    logger.info(f"Fitting LDA with {n_topics} topics")
    
    tfidf_vectorizer, tfidf_matrix = tfidf_tuple
    
    # Use TF-IDF as is (LDA can work with TF-IDF)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )
    
    doc_topics = lda_model.fit_transform(tfidf_matrix)
    logger.info(f"LDA doc-topics shape: {doc_topics.shape}")
    
    return lda_model, doc_topics


def build_lda_gibbs(count_matrix: np.ndarray, 
                    n_topics: int = 50,
                    n_iter: int = 1000) -> Tuple[any, np.ndarray]:
    """
    Build LDA topic model using Gibbs sampling (lda package).
    
    Args:
        count_matrix: Document-term count matrix
        n_topics: Number of topics
        n_iter: Number of Gibbs iterations
        
    Returns:
        Tuple of (lda_model, doc_topic_matrix)
    """
    if not LDA_AVAILABLE:
        raise ImportError("lda package required for Gibbs sampling LDA")
    
    logger.info(f"Fitting LDA (Gibbs) with {n_topics} topics, {n_iter} iterations")
    
    model = lda.LDA(
        n_topics=n_topics,
        n_iter=n_iter,
        random_state=42
    )
    
    model.fit(count_matrix)
    doc_topics = model.doc_topic_
    
    logger.info(f"LDA (Gibbs) doc-topics shape: {doc_topics.shape}")
    return model, doc_topics


def build_nmf(tfidf_tuple, n_topics: int = 50, max_iter: int = 200) -> Tuple[NMF, np.ndarray]:
    """
    Build NMF topic model.
    
    Args:
        tfidf_tuple: Tuple of (tfidf_vectorizer, tfidf_matrix)
        n_topics: Number of topics
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (nmf_model, doc_topic_matrix)
    """
    logger.info(f"Fitting NMF with {n_topics} topics")
    
    tfidf_vectorizer, tfidf_matrix = tfidf_tuple
    
    nmf_model = NMF(
        n_components=n_topics,
        init='nndsvd',
        max_iter=max_iter,
        random_state=42
    )
    
    doc_topics = nmf_model.fit_transform(tfidf_matrix)
    logger.info(f"NMF doc-topics shape: {doc_topics.shape}")
    
    return nmf_model, doc_topics


def plsa_em(count_matrix: np.ndarray,
            n_topics: int = 50,
            n_iter: int = 100,
            random_state: int = 42,
            eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit pLSA using Expectation-Maximization algorithm.
    
    Args:
        count_matrix: Document-term count matrix (n_docs, n_terms)
        n_topics: Number of topics
        n_iter: Number of EM iterations
        random_state: Random seed
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (P_z_d, P_w_z) where:
          P_z_d: (n_docs, n_topics) - P(topic | document)
          P_w_z: (n_topics, n_terms) - P(word | topic)
    """
    logger.info(f"Fitting pLSA with {n_topics} topics, {n_iter} iterations")
    
    rng = np.random.default_rng(random_state)
    n_docs, n_terms = count_matrix.shape
    
    # Initialize P(z|d) and P(w|z) randomly
    P_z_d = rng.random((n_docs, n_topics), dtype=np.float32)
    P_z_d /= P_z_d.sum(axis=1, keepdims=True) + eps
    
    P_w_z = rng.random((n_topics, n_terms), dtype=np.float32)
    P_w_z /= P_w_z.sum(axis=1, keepdims=True) + eps
    
    # EM loop
    for it in range(n_iter):
        # E-step: compute P(z | d, w)
        N_z_d = np.zeros_like(P_z_d)
        N_z_w = np.zeros_like(P_w_z)
        
        for w in range(n_terms):
            col = count_matrix[:, w]
            if not np.any(col):
                continue
            
            Pwz = P_w_z[:, w] + eps
            Pzd = P_z_d + eps
            
            # Unnormalized P(z, d, w) ∝ P(z|d) * P(w|z)
            joint = Pzd * Pwz[np.newaxis, :]
            
            # Normalize over topics
            denom = joint.sum(axis=1, keepdims=True) + eps
            Pz_dw = joint / denom
            
            # Expected counts
            weighted = col[:, np.newaxis] * Pz_dw
            N_z_d += weighted
            N_z_w[:, w] = weighted.sum(axis=0)
        
        # M-step: update parameters
        P_z_d = N_z_d + eps
        P_z_d /= P_z_d.sum(axis=1, keepdims=True) + eps
        
        P_w_z = N_z_w + eps
        P_w_z /= P_w_z.sum(axis=1, keepdims=True) + eps
        
        if (it + 1) % 10 == 0 or it == n_iter - 1:
            # Log-likelihood estimate
            Pw_d = P_z_d @ P_w_z
            Pw_d = np.clip(Pw_d, eps, 1.0)
            ll = np.sum(count_matrix * np.log(Pw_d))
            logger.info(f"pLSA iter {it + 1}/{n_iter}, log-likelihood: {ll:.3f}")
    
    logger.info("pLSA training complete")
    return P_z_d, P_w_z


def build_content_features(texts: List[str],
                          embed_model_name: str = "all-MiniLM-L6-v2",
                          n_topics_lda: int = 30,
                          n_topics_nmf: int = 30,
                          n_topics_plsa: int = 30,
                          max_features_tfidf: int = 20000,
                          use_gibbs_lda: bool = False,
                          use_plsa: bool = True) -> ContentFeatures:
    """
    Build all content features: embeddings, topics, TF-IDF.
    
    Args:
        texts: List of document texts
        embed_model_name: Sentence transformer model
        n_topics_lda: Number of LDA topics
        n_topics_nmf: Number of NMF topics
        n_topics_plsa: Number of pLSA topics
        max_features_tfidf: Max TF-IDF features
        use_gibbs_lda: Use Gibbs sampling LDA (requires lda package)
        use_plsa: Include pLSA topic modeling
        
    Returns:
        ContentFeatures object with all features
    """
    logger.info("Building content features")
    
    # 1. Embeddings
    embeddings = build_embeddings(texts, embed_model_name)
    
    # 2. TF-IDF
    tfidf_vectorizer, tfidf_matrix = build_tfidf(texts, max_features_tfidf)
    tfidf_vocab = list(tfidf_vectorizer.get_feature_names_out())
    
    # 3. LDA
    if use_gibbs_lda and LDA_AVAILABLE:
        # Convert TF-IDF to counts for Gibbs LDA
        count_vectorizer = CountVectorizer(
            vocabulary=tfidf_vectorizer.get_feature_names_out(),
            max_features=max_features_tfidf
        )
        count_matrix = count_vectorizer.fit_transform(texts).astype(np.float32).toarray()
        lda_model, lda_topics = build_lda_gibbs(count_matrix, n_topics_lda)
    else:
        lda_model, lda_topics = build_lda((tfidf_vectorizer, tfidf_matrix), n_topics_lda)
    
    # 4. NMF
    nmf_model, nmf_topics = build_nmf((tfidf_vectorizer, tfidf_matrix), n_topics_nmf)
    
    # 5. pLSA (optional)
    plsa_topics = None
    if use_plsa:
        count_vectorizer = CountVectorizer(
            vocabulary=tfidf_vectorizer.get_feature_names_out(),
            max_features=max_features_tfidf
        )
        count_matrix = count_vectorizer.fit_transform(texts).astype(np.float32).toarray()
        plsa_topics, _ = plsa_em(count_matrix, n_topics_plsa, n_iter=80)
    
    return ContentFeatures(
        embeddings=embeddings,
        lda_topics=lda_topics,
        nmf_topics=nmf_topics,
        plsa_topics=plsa_topics,
        tfidf_vocab=tfidf_vocab,
        lda_model=lda_model,
        nmf_model=nmf_model,
        embed_model_name=embed_model_name
    )

