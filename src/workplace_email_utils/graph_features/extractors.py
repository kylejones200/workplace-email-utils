"""
Social graph feature extraction module.

Builds email communication graph and extracts network features.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphFeatures:
    """Container for graph-based features."""
    feature_matrix: np.ndarray      # (n_docs, n_graph_features)
    feature_names: List[str]
    graph: nx.DiGraph


def build_email_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph from email communications.
    
    Nodes are email addresses, edges represent email communications.
    Edge weights represent frequency of communication.
    
    Args:
        df: DataFrame with 'sender' and 'recipients' columns
        
    Returns:
        Directed NetworkX graph
    """
    logger.info("Building email communication graph")
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        sender = row.get('sender', '')
        recipients = row.get('recipients', [])
        
        if not sender or pd.isna(sender):
            continue
        
        # Handle recipients as list or string
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(',') if r.strip()]
        elif not isinstance(recipients, list):
            recipients = []
        
        # Add edges from sender to each recipient
        for recipient in recipients:
            if not recipient or pd.isna(recipient):
                continue
            
            # Normalize email addresses
            sender = str(sender).lower().strip()
            recipient = str(recipient).lower().strip()
            
            if sender and recipient and sender != recipient:
                if G.has_edge(sender, recipient):
                    G[sender][recipient]['weight'] += 1
                else:
                    G.add_edge(sender, recipient, weight=1)
    
    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_graph_features(df: pd.DataFrame, G: nx.DiGraph) -> GraphFeatures:
    """
    Compute graph-based features for each email/document.
    
    Features include:
    - Sender degree (in, out, total)
    - Sender centrality measures
    - Recipient statistics
    
    Args:
        df: DataFrame with email data
        G: Email communication graph
        
    Returns:
        GraphFeatures object
    """
    logger.info("Computing graph features")
    
    # Compute node-level metrics
    in_degree = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    total_degree = dict(G.degree(weight='weight'))
    
    # Centrality measures (compute only if graph is not too large)
    logger.info("Computing centrality measures")
    try:
        degree_centrality = nx.degree_centrality(G)
    except:
        degree_centrality = {}
    
    try:
        # Betweenness centrality (sample nodes if graph is large)
        if len(G) > 500:
            betweenness = nx.betweenness_centrality(
                G, k=min(500, len(G)), weight='weight', normalized=True, seed=42
            )
        else:
            betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    except Exception as e:
        logger.warning(f"Betweenness centrality failed: {e}")
        betweenness = {n: 0.0 for n in G.nodes()}
    
    try:
        # Eigenvector centrality
        if len(G) > 1000:
            # Use power method for large graphs
            eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=100)
        else:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception as e:
        logger.warning(f"Eigenvector centrality failed: {e}")
        eigenvector = {n: 0.0 for n in G.nodes()}
    
    # Compute features for each document
    rows = []
    for _, row in df.iterrows():
        sender = str(row.get('sender', '')).lower().strip()
        recipients = row.get('recipients', [])
        
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(',') if r.strip()]
        elif not isinstance(recipients, list):
            recipients = []
        
        # Sender features
        sender_degree = float(total_degree.get(sender, 0))
        sender_in_degree = float(in_degree.get(sender, 0))
        sender_out_degree = float(out_degree.get(sender, 0))
        sender_deg_centrality = float(degree_centrality.get(sender, 0))
        sender_betweenness = float(betweenness.get(sender, 0))
        sender_eigenvector = float(eigenvector.get(sender, 0))
        
        # Recipient features (averages)
        recipient_degrees = [float(total_degree.get(str(r).lower().strip(), 0)) for r in recipients]
        recipient_centralities = [float(degree_centrality.get(str(r).lower().strip(), 0)) for r in recipients]
        
        mean_recipient_degree = float(np.mean(recipient_degrees)) if recipient_degrees else 0.0
        mean_recipient_centrality = float(np.mean(recipient_centralities)) if recipient_centralities else 0.0
        n_recipients = len(recipients)
        
        rows.append([
            sender_degree,
            sender_in_degree,
            sender_out_degree,
            sender_deg_centrality,
            sender_betweenness,
            sender_eigenvector,
            mean_recipient_degree,
            mean_recipient_centrality,
            n_recipients,
        ])
    
    feature_names = [
        'sender_degree',
        'sender_in_degree',
        'sender_out_degree',
        'sender_deg_centrality',
        'sender_betweenness',
        'sender_eigenvector',
        'mean_recipient_degree',
        'mean_recipient_centrality',
        'n_recipients',
    ]
    
    feature_matrix = np.asarray(rows, dtype=np.float32)
    logger.info(f"Graph features shape: {feature_matrix.shape}")
    
    return GraphFeatures(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        graph=G
    )

