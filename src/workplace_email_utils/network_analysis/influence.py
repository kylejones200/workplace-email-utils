"""
Influence metrics computation.

Computes PageRank, HITS, and other influence metrics for network nodes.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InfluenceMetrics:
    """Container for influence metrics."""
    pagerank: Dict[str, float]
    authority: Dict[str, float]  # HITS authority scores
    hub: Dict[str, float]  # HITS hub scores
    eigenvector_centrality: Dict[str, float]
    closeness_centrality: Dict[str, float]
    betweenness_centrality: Dict[str, float]


def compute_pagerank(
    G: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 100,
    weight: str = 'weight'
) -> Dict[str, float]:
    """
    Compute PageRank scores for network nodes.
    
    PageRank measures the importance of nodes based on the importance
    of nodes that link to them.
    
    Args:
        G: NetworkX graph
        alpha: Damping parameter (probability of following links)
        max_iter: Maximum iterations
        weight: Edge weight attribute
        
    Returns:
        Dictionary mapping node -> PageRank score
    """
    logger.info("Computing PageRank scores...")
    
    # Convert to undirected if needed
    if isinstance(G, nx.DiGraph):
        G_undir = G.to_undirected()
    else:
        G_undir = G.copy()
    
    try:
        if weight in next(iter(G_undir.edges(data=True)))[2]:
            # Use weighted PageRank
            pagerank = nx.pagerank(G_undir, alpha=alpha, max_iter=max_iter, weight=weight)
        else:
            # Unweighted PageRank
            pagerank = nx.pagerank(G_undir, alpha=alpha, max_iter=max_iter)
    except Exception as e:
        logger.warning(f"PageRank computation failed: {e}")
        # Fallback to degree centrality
        pagerank = nx.degree_centrality(G_undir)
    
    logger.info(f"Computed PageRank for {len(pagerank)} nodes")
    return pagerank


def compute_hits(
    G: nx.DiGraph,
    max_iter: int = 100,
    normalized: bool = True
) -> tuple:
    """
    Compute HITS (Hyperlink-Induced Topic Search) scores.
    
    HITS identifies:
    - Authorities: Nodes with high in-degree (many links to them)
    - Hubs: Nodes with high out-degree (links to many authorities)
    
    Args:
        G: Directed NetworkX graph
        max_iter: Maximum iterations
        normalized: Whether to normalize scores
        
    Returns:
        Tuple of (authority_scores, hub_scores) dictionaries
    """
    logger.info("Computing HITS scores...")
    
    # HITS requires directed graph
    if not isinstance(G, nx.DiGraph):
        G = G.to_directed()
    
    try:
        hits = nx.hits(G, max_iter=max_iter, normalized=normalized)
        authority_scores, hub_scores = hits
        
        logger.info(f"Computed HITS for {len(authority_scores)} nodes")
        return authority_scores, hub_scores
    except Exception as e:
        logger.warning(f"HITS computation failed: {e}")
        # Fallback: use in/out degree as proxies
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        max_in = max(in_degree.values()) if in_degree.values() else 1
        max_out = max(out_degree.values()) if out_degree.values() else 1
        
        authority_scores = {n: in_degree.get(n, 0) / max_in for n in G.nodes()}
        hub_scores = {n: out_degree.get(n, 0) / max_out for n in G.nodes()}
        
        return authority_scores, hub_scores


def compute_influence_metrics(
    G: nx.Graph,
    weight: Optional[str] = 'weight',
    sample_size: Optional[int] = None
) -> InfluenceMetrics:
    """
    Compute comprehensive influence metrics for network nodes.
    
    Args:
        G: NetworkX graph
        weight: Edge weight attribute
        sample_size: Optional sample size for expensive computations
        
    Returns:
        InfluenceMetrics object with all metrics
    """
    logger.info("Computing comprehensive influence metrics...")
    
    # PageRank
    pagerank = compute_pagerank(G, weight=weight)
    
    # HITS (requires directed graph)
    if isinstance(G, nx.DiGraph):
        authority, hub = compute_hits(G)
    else:
        # Convert to directed for HITS
        G_dir = G.to_directed()
        authority, hub = compute_hits(G_dir)
    
    # Eigenvector centrality
    logger.info("Computing eigenvector centrality...")
    try:
        if len(G) > 1000:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight=weight, max_iter=100)
        else:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight=weight)
    except Exception as e:
        logger.warning(f"Eigenvector centrality failed: {e}")
        eigenvector = {}
    
    # Closeness centrality
    logger.info("Computing closeness centrality...")
    try:
        if len(G) > 500:
            # Use sample for large graphs
            nodes_sample = list(G.nodes())[:min(sample_size or 500, len(G))]
            closeness = {}
            for node in nodes_sample:
                try:
                    closeness[node] = nx.closeness_centrality(G, node, distance=weight if weight else None)
                except:
                    closeness[node] = 0.0
        else:
            closeness = nx.closeness_centrality(G, distance=weight if weight else None)
    except Exception as e:
        logger.warning(f"Closeness centrality failed: {e}")
        closeness = {}
    
    # Betweenness centrality
    logger.info("Computing betweenness centrality...")
    try:
        if len(G) > 500:
            # Sample for large graphs
            betweenness = nx.betweenness_centrality(
                G,
                k=min(sample_size or 500, len(G)),
                weight=weight,
                normalized=True,
                seed=42
            )
        else:
            betweenness = nx.betweenness_centrality(G, weight=weight, normalized=True)
    except Exception as e:
        logger.warning(f"Betweenness centrality failed: {e}")
        betweenness = {}
    
    return InfluenceMetrics(
        pagerank=pagerank,
        authority=authority,
        hub=hub,
        eigenvector_centrality=eigenvector,
        closeness_centrality=closeness,
        betweenness_centrality=betweenness
    )


def rank_influencers(
    metrics: InfluenceMetrics,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Rank nodes by influence across multiple metrics.
    
    Args:
        metrics: InfluenceMetrics object
        top_n: Number of top influencers to return
        
    Returns:
        DataFrame with ranked influencers
    """
    # Get all nodes
    all_nodes = set(metrics.pagerank.keys())
    all_nodes.update(metrics.authority.keys())
    all_nodes.update(metrics.hub.keys())
    
    rows = []
    for node in all_nodes:
        rows.append({
            'node': node,
            'pagerank': metrics.pagerank.get(node, 0.0),
            'authority': metrics.authority.get(node, 0.0),
            'hub': metrics.hub.get(node, 0.0),
            'eigenvector': metrics.eigenvector_centrality.get(node, 0.0),
            'closeness': metrics.closeness_centrality.get(node, 0.0),
            'betweenness': metrics.betweenness_centrality.get(node, 0.0),
        })
    
    df = pd.DataFrame(rows)
    
    # Compute composite influence score (normalized average)
    score_cols = ['pagerank', 'authority', 'hub', 'eigenvector', 'closeness', 'betweenness']
    for col in score_cols:
        if df[col].max() > 0:
            df[col + '_norm'] = df[col] / df[col].max()
        else:
            df[col + '_norm'] = 0.0
    
    df['influence_score'] = df[[c + '_norm' for c in score_cols]].mean(axis=1)
    df = df.sort_values('influence_score', ascending=False).head(top_n)
    
    return df


if __name__ == "__main__":
    # Test influence metrics
    from workplace_email_utils.graph_features.extractors import build_email_graph
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing influence metrics...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    G = build_email_graph(df)
    
    metrics = compute_influence_metrics(G)
    
    # Rank influencers
    influencers = rank_influencers(metrics, top_n=10)
    print(f"\nTop 10 influencers:")
    print(influencers[['node', 'pagerank', 'authority', 'influence_score']])

