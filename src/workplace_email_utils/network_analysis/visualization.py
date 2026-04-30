"""
Network visualization tools.

Provides visualization functions for network graphs, communities, and influence.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_network_graph(
    G: nx.Graph,
    pos: Optional[Dict] = None,
    node_size: int = 300,
    node_color: str = 'lightblue',
    edge_color: str = 'gray',
    with_labels: bool = False,
    ax=None
):
    """
    Plot a network graph.
    
    Args:
        G: NetworkX graph
        pos: Optional node positions (will compute if None)
        node_size: Size of nodes
        node_color: Color of nodes
        edge_color: Color of edges
        with_labels: Whether to show node labels
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot plot network.")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Compute layout if not provided
    if pos is None:
        if len(G.nodes()) > 0:
            try:
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
        else:
            logger.warning("No nodes to plot")
            return ax
    
    # Draw graph
    nx.draw(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color,
        alpha=0.7,
        with_labels=with_labels and len(G.nodes()) <= 50,
        font_size=8,
        arrows=True if isinstance(G, nx.DiGraph) else False,
        arrowsize=15
    )
    
    ax.set_title(f"Network Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)")
    ax.axis('off')
    
    return ax


def plot_community_structure(
    G: nx.Graph,
    communities: List[Set[str]],
    pos: Optional[Dict] = None,
    ax=None
):
    """
    Plot network with communities highlighted.
    
    Args:
        G: NetworkX graph
        communities: List of community sets
        pos: Optional node positions
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not available. Cannot plot communities.")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    
    # Compute layout
    if pos is None:
        if len(G.nodes()) > 0:
            try:
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
        else:
            return ax
    
    # Assign colors to communities
    node_to_community = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = comm_id
    
    n_communities = len(communities)
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, max(n_communities, 1)))
    
    node_colors = [colors[node_to_community.get(node, 0)] for node in G.nodes()]
    
    # Draw graph
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edge_color='gray',
        alpha=0.7,
        node_size=300,
        with_labels=False,
        arrows=True if isinstance(G, nx.DiGraph) else False
    )
    
    ax.set_title(f"Community Structure ({n_communities} communities)")
    ax.axis('off')
    
    return ax


def plot_influence_network(
    G: nx.Graph,
    influence_scores: Dict[str, float],
    top_n: int = 20,
    pos: Optional[Dict] = None,
    ax=None
):
    """
    Plot network with node sizes proportional to influence.
    
    Args:
        G: NetworkX graph
        influence_scores: Dictionary of node -> influence score
        top_n: Number of top influencers to highlight
        pos: Optional node positions
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot plot influence network.")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    
    # Compute layout
    if pos is None:
        if len(G.nodes()) > 0:
            try:
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
    
    # Get top influencers
    sorted_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = {node for node, _ in sorted_influencers}
    
    # Node sizes based on influence
    max_influence = max(influence_scores.values()) if influence_scores.values() else 1.0
    node_sizes = [
        300 + (influence_scores.get(node, 0.0) / max_influence * 1000) if node in top_nodes else 50
        for node in G.nodes()
    ]
    
    # Node colors (top influencers in red)
    node_colors = ['red' if node in top_nodes else 'lightblue' for node in G.nodes()]
    
    # Draw graph
    nx.draw(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color='gray',
        alpha=0.6,
        with_labels=False,
        arrows=True if isinstance(G, nx.DiGraph) else False
    )
    
    # Label top influencers
    if top_nodes:
        labels = {node: node[:15] for node in top_nodes if node in G}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(f"Influence Network (Top {top_n} influencers highlighted)")
    ax.axis('off')
    
    return ax


def plot_temporal_network(
    temporal_analysis,
    metric: str = 'network_size',
    ax=None
):
    """
    Plot network metrics over time.
    
    Args:
        temporal_analysis: TemporalNetworkAnalysis object
        metric: Metric to plot ('network_size', 'network_density', 'node_degree')
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot plot temporal network.")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    time_periods = temporal_analysis.time_periods
    
    if metric == 'network_size':
        values = temporal_analysis.network_size_over_time
        ylabel = 'Number of Nodes'
    elif metric == 'network_density':
        values = temporal_analysis.network_density_over_time
        ylabel = 'Network Density'
    else:
        logger.warning(f"Unknown metric: {metric}")
        return ax
    
    ax.plot(range(len(time_periods)), values, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Time Period')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Network {metric.replace("_", " ").title()} Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(time_periods)))
    ax.set_xticklabels([p[:10] for p in time_periods], rotation=45, ha='right')
    
    return ax

