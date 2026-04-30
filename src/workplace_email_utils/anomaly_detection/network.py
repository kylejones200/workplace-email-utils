"""
Network structure anomaly detection.

Detects anomalies in network topology, community structure, and connectivity patterns.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import logging

from workplace_email_utils.graph_features.communities import detect_tight_knit_groups, CommunityResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkAnomalyResult:
    """Container for network structure anomaly detection results."""
    anomalous_nodes: List[str]
    anomalous_communities: List[Set[str]]
    anomaly_scores: Dict[str, float]
    anomaly_types: Dict[str, List[str]]
    metrics: Dict


def detect_structural_anomalies(
    G: nx.Graph,
    communities: Optional[CommunityResult] = None,
    density_threshold: float = 0.8,
    size_threshold: int = 10
) -> List[Set[str]]:
    """
    Detect anomalous network structures.
    
    Based on Enron paper: "tight balls with spikes"
    - Unusually dense communities
    - Small tight-knit groups
    - Isolated clusters
    
    Args:
        G: NetworkX graph
        communities: Optional CommunityResult
        density_threshold: Minimum density for suspicious communities
        size_threshold: Maximum size for suspicious tight-knit groups
        
    Returns:
        List of anomalous community sets
    """
    logger.info("Detecting structural anomalies...")
    
    if communities is None:
        communities = detect_tight_knit_groups(G)
    
    anomalous_communities = []
    
    for comm in communities.communities:
        if len(comm) > size_threshold:
            continue
        
        # Calculate density
        subgraph = G.subgraph(comm)
        if isinstance(subgraph, nx.DiGraph):
            n = len(comm)
            if n < 2:
                continue
            max_edges = n * (n - 1)
            density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0
        else:
            density = nx.density(subgraph) if len(subgraph) > 1 else 0
        
        # High density + small size = suspicious
        if density >= density_threshold:
            anomalous_communities.append(comm)
    
    logger.info(f"Detected {len(anomalous_communities)} structurally anomalous communities")
    return anomalous_communities


def detect_connectivity_anomalies(
    G: nx.Graph,
    threshold: float = 3.0
) -> List[str]:
    """
    Detect nodes with anomalous connectivity patterns.
    
    Anomalies:
    - Nodes with unusually high/low degree
    - Nodes with unusual clustering coefficient
    - Isolated nodes or hubs
    
    Args:
        G: NetworkX graph
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        List of anomalous node identifiers
    """
    logger.info("Detecting connectivity anomalies...")
    
    # Calculate node metrics
    degrees = dict(G.degree())
    clustering = nx.clustering(G.to_undirected() if isinstance(G, nx.DiGraph) else G)
    
    if not degrees:
        return []
    
    # Calculate z-scores
    degree_values = list(degrees.values())
    clustering_values = list(clustering.values())
    
    degree_mean = np.mean(degree_values) if degree_values else 0
    degree_std = np.std(degree_values) if degree_values and len(degree_values) > 1 else 1
    clustering_mean = np.mean(clustering_values) if clustering_values else 0
    clustering_std = np.std(clustering_values) if clustering_values and len(clustering_values) > 1 else 1
    
    anomalous_nodes = []
    
    for node in G.nodes():
        degree_z = abs((degrees.get(node, 0) - degree_mean) / degree_std) if degree_std > 0 else 0
        clustering_z = abs((clustering.get(node, 0) - clustering_mean) / clustering_std) if clustering_std > 0 else 0
        
        # Anomalous if either metric is extreme
        if degree_z > threshold or clustering_z > threshold:
            anomalous_nodes.append(node)
    
    logger.info(f"Detected {len(anomalous_nodes)} connectivity anomalies")
    return anomalous_nodes


def detect_community_evolution_anomalies(
    temporal_communities: Dict[str, CommunityResult],
    threshold: float = 0.5
) -> List[Dict]:
    """
    Detect anomalies in how communities evolve over time.
    
    Anomalies:
    - Sudden appearance of tight-knit groups
    - Rapid community growth/shrinkage
    - Community merging/splitting
    
    Args:
        temporal_communities: Dictionary of period -> CommunityResult
        threshold: Threshold for significant change
        
    Returns:
        List of anomaly events
    """
    logger.info("Detecting community evolution anomalies...")
    
    periods = sorted(temporal_communities.keys())
    anomalies = []
    
    for i in range(1, len(periods)):
        prev_period = periods[i-1]
        curr_period = periods[i]
        
        prev_comm = temporal_communities[prev_period]
        curr_comm = temporal_communities[curr_period]
        
        # Check for sudden appearance of tight-knit groups
        prev_suspicious = set(frozenset(c) for c in prev_comm.suspicious_communities)
        curr_suspicious = set(frozenset(c) for c in curr_comm.suspicious_communities)
        
        new_groups = curr_suspicious - prev_suspicious
        
        if len(new_groups) > 0:
            anomalies.append({
                'period': curr_period,
                'type': 'new_suspicious_group',
                'count': len(new_groups),
                'groups': [list(g) for g in new_groups]
            })
        
        # Check for rapid community size changes
        prev_sizes = [len(c) for c in prev_comm.communities]
        curr_sizes = [len(c) for c in curr_comm.communities]
        
        if prev_sizes and curr_sizes:
            prev_avg = np.mean(prev_sizes)
            curr_avg = np.mean(curr_sizes)
            
            if prev_avg > 0:
                change_ratio = abs(curr_avg - prev_avg) / prev_avg
                if change_ratio > threshold:
                    anomalies.append({
                        'period': curr_period,
                        'type': 'rapid_community_change',
                        'change_ratio': change_ratio,
                        'prev_avg_size': prev_avg,
                        'curr_avg_size': curr_avg
                    })
    
    logger.info(f"Detected {len(anomalies)} community evolution anomalies")
    return anomalies


def detect_network_anomalies(
    G: nx.Graph,
    communities: Optional[CommunityResult] = None,
    temporal_communities: Optional[Dict[str, CommunityResult]] = None
) -> NetworkAnomalyResult:
    """
    Comprehensive network structure anomaly detection.
    
    Args:
        G: NetworkX graph
        communities: Optional CommunityResult
        temporal_communities: Optional temporal community data
        
    Returns:
        NetworkAnomalyResult with all network anomalies
    """
    logger.info("Detecting network structure anomalies...")
    
    # Detect structural anomalies
    anomalous_communities = detect_structural_anomalies(G, communities)
    
    # Detect connectivity anomalies
    anomalous_nodes = detect_connectivity_anomalies(G)
    
    # Calculate anomaly scores for nodes
    anomaly_scores = {}
    degrees = dict(G.degree())
    degree_mean = np.mean(list(degrees.values())) if degrees else 0
    degree_std = np.std(list(degrees.values())) if degrees and len(degrees) > 1 else 1
    
    for node in G.nodes():
        score = 0.0
        
        # High/low degree anomaly
        if node in anomalous_nodes:
            degree_z = abs((degrees.get(node, 0) - degree_mean) / degree_std) if degree_std > 0 else 0
            score += min(degree_z / 3.0, 1.0)
        
        # Community membership anomaly
        for comm in anomalous_communities:
            if node in comm:
                score += 0.5
        
        anomaly_scores[node] = min(score, 1.0)
    
    # Categorize anomalies
    anomaly_types = {
        'high_degree': [n for n in anomalous_nodes if degrees.get(n, 0) > degree_mean],
        'low_degree': [n for n in anomalous_nodes if degrees.get(n, 0) < degree_mean],
        'suspicious_community': list(set().union(*anomalous_communities)) if anomalous_communities else []
    }
    
    # Evolution anomalies
    evolution_anomalies = []
    if temporal_communities:
        evolution_anomalies = detect_community_evolution_anomalies(temporal_communities)
    
    metrics = {
        'anomalous_nodes': len(anomalous_nodes),
        'anomalous_communities': len(anomalous_communities),
        'evolution_anomalies': len(evolution_anomalies),
        'avg_anomaly_score': np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0
    }
    
    logger.info(f"Network anomalies: {metrics}")
    
    return NetworkAnomalyResult(
        anomalous_nodes=anomalous_nodes,
        anomalous_communities=anomalous_communities,
        anomaly_scores=anomaly_scores,
        anomaly_types=anomaly_types,
        metrics=metrics
    )


if __name__ == "__main__":
    # Test network anomaly detection
    from workplace_email_utils.graph_features.extractors import build_email_graph
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing network anomaly detection...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    G = build_email_graph(df)
    
    result = detect_network_anomalies(G)
    print(f"\nNetwork anomalies: {result.metrics}")
    print(f"Anomalous nodes: {len(result.anomalous_nodes)}")
    print(f"Anomalous communities: {len(result.anomalous_communities)}")

