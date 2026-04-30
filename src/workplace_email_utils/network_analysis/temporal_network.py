"""
Temporal network analysis.

Analyzes how network structure evolves over time.
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TemporalNetworkAnalysis:
    """Container for temporal network analysis results."""
    time_periods: List[str]
    networks: List[nx.Graph]  # Network for each time period
    node_metrics_over_time: Dict[str, List[float]]  # node -> [metric values]
    network_size_over_time: List[int]
    network_density_over_time: List[float]
    community_evolution: Dict[int, List[Set[str]]]  # community_id -> [membership over time]


def build_temporal_network(
    df: pd.DataFrame,
    date_col: str = 'date_parsed',
    time_period: str = 'month',
    sender_col: str = 'sender',
    recipient_col: str = 'recipients'
) -> Dict[str, nx.DiGraph]:
    """
    Build network graphs for different time periods.
    
    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        time_period: Time period ('day', 'week', 'month', 'quarter', 'year')
        sender_col: Column name for sender
        recipient_col: Column name for recipients
        
    Returns:
        Dictionary mapping time_period -> NetworkX graph
    """
    logger.info(f"Building temporal network (period: {time_period})")
    
    # Ensure dates are datetime
    if date_col not in df.columns or df[date_col].isna().all():
        logger.warning(f"Date column '{date_col}' not available or all null")
        return {}
    
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col])
    
    if len(df_copy) == 0:
        logger.warning("No valid dates found")
        return {}
    
    # Group by time period
    if time_period == 'day':
        df_copy['period'] = df_copy[date_col].dt.date
    elif time_period == 'week':
        df_copy['period'] = df_copy[date_col].dt.to_period('W').astype(str)
    elif time_period == 'month':
        df_copy['period'] = df_copy[date_col].dt.to_period('M').astype(str)
    elif time_period == 'quarter':
        df_copy['period'] = df_copy[date_col].dt.to_period('Q').astype(str)
    elif time_period == 'year':
        df_copy['period'] = df_copy[date_col].dt.to_period('Y').astype(str)
    else:
        raise ValueError(f"Unknown time period: {time_period}")
    
    temporal_networks = {}
    
    for period in sorted(df_copy['period'].unique()):
        period_df = df_copy[df_copy['period'] == period]
        
        # Build graph for this period
        G = nx.DiGraph()
        
        for _, row in period_df.iterrows():
            sender = str(row.get(sender_col, '')).lower().strip()
            recipients = row.get(recipient_col, [])
            
            if not sender or pd.isna(sender):
                continue
            
            if isinstance(recipients, str):
                recipients = [r.strip() for r in recipients.split(',') if r.strip()]
            elif not isinstance(recipients, list):
                recipients = []
            
            for recipient in recipients:
                if not recipient or pd.isna(recipient):
                    continue
                
                recipient = str(recipient).lower().strip()
                
                if sender and recipient and sender != recipient:
                    if G.has_edge(sender, recipient):
                        G[sender][recipient]['weight'] += 1
                    else:
                        G.add_edge(sender, recipient, weight=1)
        
        if len(G.nodes()) > 0:
            temporal_networks[str(period)] = G
            logger.info(f"  Period {period}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    logger.info(f"Built {len(temporal_networks)} temporal networks")
    return temporal_networks


def analyze_network_evolution(
    temporal_networks: Dict[str, nx.Graph],
    node_subset: Optional[List[str]] = None
) -> TemporalNetworkAnalysis:
    """
    Analyze how network structure evolves over time.
    
    Args:
        temporal_networks: Dictionary of period -> network graph
        node_subset: Optional subset of nodes to track
        
    Returns:
        TemporalNetworkAnalysis with evolution metrics
    """
    logger.info("Analyzing network evolution...")
    
    time_periods = sorted(temporal_networks.keys())
    networks = [temporal_networks[p] for p in time_periods]
    
    # Track metrics over time
    network_size_over_time = [len(G.nodes()) for G in networks]
    network_density_over_time = []
    
    for G in networks:
        if isinstance(G, nx.DiGraph):
            n = len(G)
            if n > 1:
                max_edges = n * (n - 1)
                density = G.number_of_edges() / max_edges if max_edges > 0 else 0
            else:
                density = 0.0
        else:
            density = nx.density(G) if len(G) > 0 else 0.0
        network_density_over_time.append(density)
    
    # Track node metrics over time
    all_nodes = set()
    for G in networks:
        all_nodes.update(G.nodes())
    
    if node_subset:
        tracked_nodes = [n for n in node_subset if n in all_nodes]
    else:
        # Track top nodes by average degree
        node_degrees = {}
        for G in networks:
            for node in G.nodes():
                if node not in node_degrees:
                    node_degrees[node] = []
                node_degrees[node].append(G.degree(node))
        
        avg_degrees = {node: np.mean(degrees) for node, degrees in node_degrees.items()}
        tracked_nodes = sorted(avg_degrees.items(), key=lambda x: x[1], reverse=True)[:50]
        tracked_nodes = [node for node, _ in tracked_nodes]
    
    node_metrics_over_time = {}
    for node in tracked_nodes:
        metrics = []
        for G in networks:
            if node in G:
                metrics.append(G.degree(node))
            else:
                metrics.append(0.0)
        node_metrics_over_time[node] = metrics
    
    # Community evolution (simplified - would need community detection per period)
    community_evolution = {}
    
    return TemporalNetworkAnalysis(
        time_periods=time_periods,
        networks=networks,
        node_metrics_over_time=node_metrics_over_time,
        network_size_over_time=network_size_over_time,
        network_density_over_time=network_density_over_time,
        community_evolution=community_evolution
    )


def detect_network_changes(
    temporal_analysis: TemporalNetworkAnalysis,
    threshold: float = 0.2
) -> List[Dict]:
    """
    Detect significant changes in network structure over time.
    
    Args:
        temporal_analysis: TemporalNetworkAnalysis object
        threshold: Threshold for significant change (relative change)
        
    Returns:
        List of change events
    """
    changes = []
    
    for i in range(1, len(temporal_analysis.networks)):
        prev_G = temporal_analysis.networks[i-1]
        curr_G = temporal_analysis.networks[i-1]
        
        prev_size = len(prev_G.nodes())
        curr_size = len(curr_G.nodes())
        
        if prev_size > 0:
            size_change = abs(curr_size - prev_size) / prev_size
            if size_change > threshold:
                changes.append({
                    'period': temporal_analysis.time_periods[i],
                    'type': 'size_change',
                    'change': size_change,
                    'prev_size': prev_size,
                    'curr_size': curr_size
                })
    
    return changes


if __name__ == "__main__":
    # Test temporal network analysis
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing temporal network analysis...")
    df = load_emails('maildir', data_format='maildir', max_rows=1000)
    df = extract_temporal_features(df)
    
    temporal_networks = build_temporal_network(df, time_period='month')
    
    if temporal_networks:
        analysis = analyze_network_evolution(temporal_networks)
        print(f"\nNetwork evolution over {len(analysis.time_periods)} periods")
        print(f"Network size: {min(analysis.network_size_over_time)} -> {max(analysis.network_size_over_time)}")
        
        changes = detect_network_changes(analysis)
        print(f"Significant changes: {len(changes)}")

