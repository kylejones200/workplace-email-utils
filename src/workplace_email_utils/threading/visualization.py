"""
Thread visualization functions.

Provides visualization tools for email threads and conversation flow.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Optional
import logging

from .reconstruct import ThreadTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_thread_tree(
    thread_tree: ThreadTree,
    df: Optional = None,
    ax: Optional = None,
    show_labels: bool = True,
    max_depth: int = 10
) -> None:
    """
    Visualize a thread as a tree structure.
    
    Args:
        thread_tree: ThreadTree object to visualize
        df: Optional DataFrame with email data
        ax: Optional matplotlib axes
        show_labels: Whether to show message labels
        max_depth: Maximum depth to show
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot plot thread tree.")
        return
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Build tree graph
    G = nx.DiGraph()
    
    # Add nodes
    for msg_id in thread_tree.messages:
        G.add_node(msg_id)
    
    # Add edges from parent_map
    for child, parent in thread_tree.parent_map.items():
        if parent in G and child in G:
            G.add_edge(parent, child)
    
    # If no edges, create a linear chain (for visualization)
    if len(G.edges()) == 0 and len(thread_tree.messages) > 1:
        for i in range(len(thread_tree.messages) - 1):
            G.add_edge(thread_tree.messages[i], thread_tree.messages[i+1])
    
    # Layout
    if len(G.nodes()) == 0:
        logger.warning("No nodes to plot")
        return
    
    if len(G.nodes()) == 1:
        pos = {list(G.nodes())[0]: (0, 0)}
    else:
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.planar_layout(G) if nx.is_planar(G) else nx.random_layout(G)
    
    # Draw tree
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=show_labels and len(G.nodes()) <= 20,
        node_color='lightblue',
        node_size=500,
        font_size=8,
        arrows=True,
        arrowsize=20,
        edge_color='gray',
        alpha=0.7
    )
    
    ax.set_title(f"Thread: {thread_tree.subject[:50]}...\n"
                 f"Messages: {thread_tree.message_count}, Depth: {thread_tree.depth}")
    ax.axis('off')
    
    return ax


def plot_thread_timeline(
    thread_tree: ThreadTree,
    df: Optional = None,
    ax: Optional = None
) -> None:
    """
    Plot thread timeline showing message order and timing.
    
    Args:
        thread_tree: ThreadTree object
        df: Optional DataFrame with email data
        ax: Optional matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/pandas not available. Cannot plot timeline.")
        return
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Get message dates if df provided
    if df is not None:
        thread_df = df[df.get('thread_id') == thread_tree.thread_id].copy()
        if 'date' in thread_df.columns:
            thread_df['date_parsed'] = pd.to_datetime(
                thread_df['date'],
                errors='coerce',
                utc=True
            )
            thread_df = thread_df.sort_values('date_parsed')
            
            # Plot timeline
            dates = thread_df['date_parsed'].dropna()
            if len(dates) > 0:
                y_pos = range(len(dates))
                ax.plot(dates, y_pos, 'o-', markersize=8)
                ax.set_xlabel('Date')
                ax.set_ylabel('Message Order')
                ax.set_title(f"Thread Timeline: {thread_tree.subject[:50]}...")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
    
    return ax


def plot_thread_statistics(
    thread_metrics_dict: Dict,
    ax: Optional = None
) -> None:
    """
    Plot statistics across threads.
    
    Args:
        thread_metrics_dict: Dictionary of thread_id -> ThreadMetrics
        ax: Optional matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/pandas not available. Cannot plot statistics.")
        return
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to DataFrame
    rows = []
    for thread_id, metrics in thread_metrics_dict.items():
        rows.append({
            'message_count': metrics.message_count,
            'participant_count': metrics.participant_count,
            'depth': metrics.depth,
            'duration_days': metrics.duration_days,
        })
    
    if not rows:
        logger.warning("No thread metrics to plot")
        return
    
    df = pd.DataFrame(rows)
    
    # Plot histograms
    df[['message_count', 'participant_count']].hist(ax=ax, bins=20, alpha=0.7)
    ax.set_title("Thread Statistics Distribution")
    
    return ax

