"""
Community detection and network structure analysis.

Based on insights from "Uncovering Wrongdoing in the Enron Email Corpus" paper:
- Detects tight-knit communities (suspicious groups)
- Analyzes network structure patterns
- Identifies "tight balls with spikes" network structures
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CommunityResult:
    """Container for community detection results."""
    communities: List[Set[str]]
    community_assignments: Dict[str, int]  # node -> community_id
    community_metrics: Dict[int, Dict]  # community_id -> metrics
    suspicious_communities: List[Set[str]]


try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not available. Install with: pip install python-louvain")

try:
    from networkx.algorithms import community as nx_community
    NETWORKX_COMMUNITY_AVAILABLE = True
except ImportError:
    NETWORKX_COMMUNITY_AVAILABLE = False


def detect_communities_louvain(G: nx.Graph, random_state: int = 42) -> Dict[str, int]:
    """
    Detect communities using Louvain algorithm.
    
    Args:
        G: NetworkX graph (will convert to undirected if needed)
        random_state: Random seed
        
    Returns:
        Dictionary mapping node -> community_id
    """
    if not LOUVAIN_AVAILABLE:
        logger.warning("python-louvain not available. Using greedy modularity instead.")
        return detect_communities_greedy(G)
    
    # Convert to undirected for Louvain
    if isinstance(G, nx.DiGraph):
        G_undir = G.to_undirected()
    else:
        G_undir = G.copy()
    
    # Detect communities
    partition = community_louvain.best_partition(G_undir, random_state=random_state)
    return partition


def detect_communities_greedy(G: nx.Graph) -> Dict[str, int]:
    """
    Detect communities using greedy modularity maximization.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping node -> community_id
    """
    # Convert to undirected if needed
    if isinstance(G, nx.DiGraph):
        G_undir = G.to_undirected()
    else:
        G_undir = G.copy()
    
    try:
        communities = nx_community.greedy_modularity_communities(G_undir, weight='weight')
        partition = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                partition[node] = comm_id
        return partition
    except Exception as e:
        logger.warning(f"Greedy modularity failed: {e}")
        # Fallback: each node is its own community
        return {node: i for i, node in enumerate(G_undir.nodes())}


def detect_suspicious_communities(
    G: nx.DiGraph,
    min_size: int = 3,
    max_size: int = 10,
    min_density: float = 0.5,
    partition: Optional[Dict[str, int]] = None
) -> List[Set[str]]:
    """
    Detect tight-knit communities that may indicate suspicious groups.
    
    Based on paper's insight: corrupt networks are "tight balls with spikes"
    - Small groups (3-10 people)
    - High internal density
    - Tight-knit connections
    
    Args:
        G: Email communication graph
        min_size: Minimum community size
        max_size: Maximum community size
        min_density: Minimum density threshold
        
    Returns:
        List of suspicious communities (sets of nodes)
    """
    logger.info(f"Detecting suspicious communities (size: {min_size}-{max_size}, density > {min_density})")
    
    # Detect communities if partition not provided
    if partition is None:
        partition = detect_communities_louvain(G)
    
    # Group nodes by community
    communities_dict = defaultdict(set)
    for node, comm_id in partition.items():
        communities_dict[comm_id].add(node)
    
    suspicious = []
    
    for comm_id, nodes in communities_dict.items():
        if not (min_size <= len(nodes) <= max_size):
            continue
        
        # Calculate community density
        subgraph = G.subgraph(nodes)
        
        # For directed graph, calculate density
        if isinstance(subgraph, nx.DiGraph):
            n = len(nodes)
            if n < 2:
                continue
            max_possible_edges = n * (n - 1)  # Directed: n*(n-1)
            actual_edges = subgraph.number_of_edges()
            density = actual_edges / max_possible_edges if max_possible_edges > 0 else 0
        else:
            density = nx.density(subgraph)
        
        # Check if tight-knit
        if density >= min_density:
            suspicious.append(nodes)
            logger.info(f"Found suspicious community: {len(nodes)} nodes, density={density:.3f}")
    
    logger.info(f"Detected {len(suspicious)} suspicious communities")
    return suspicious


def analyze_community_structure(
    G: nx.DiGraph,
    partition: Optional[Dict[str, int]] = None
) -> Dict:
    """
    Analyze overall community structure of the network.
    
    Args:
        G: Email communication graph
        partition: Optional community partition (will detect if None)
        
    Returns:
        Dictionary with community structure metrics
    """
    if partition is None:
        partition = detect_communities_louvain(G)
    
    # Group nodes by community
    communities_dict = defaultdict(set)
    for node, comm_id in partition.items():
        communities_dict[comm_id].add(node)
    
    # Calculate metrics
    community_sizes = [len(nodes) for nodes in communities_dict.values()]
    
    # Calculate modularity
    if isinstance(G, nx.DiGraph):
        G_undir = G.to_undirected()
    else:
        G_undir = G.copy()
    
    try:
        modularity = nx_community.modularity(G_undir, communities_dict.values(), weight='weight')
    except:
        modularity = 0.0
    
    # Calculate average density
    densities = []
    for nodes in communities_dict.values():
        if len(nodes) > 1:
            subgraph = G.subgraph(nodes)
            if isinstance(subgraph, nx.DiGraph):
                n = len(nodes)
                max_edges = n * (n - 1)
                density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0
            else:
                density = nx.density(subgraph)
            densities.append(density)
    
    return {
        'n_communities': len(communities_dict),
        'community_sizes': community_sizes,
        'avg_community_size': np.mean(community_sizes) if community_sizes else 0,
        'modularity': modularity,
        'avg_density': np.mean(densities) if densities else 0,
        'communities': communities_dict
    }


def detect_tight_knit_groups(
    G: nx.DiGraph,
    min_size: int = 3,
    max_size: int = 10,
    min_density: float = 0.5
) -> CommunityResult:
    """
    Comprehensive community detection with suspicious group identification.
    
    Args:
        G: Email communication graph
        min_size: Minimum community size for suspicious detection
        max_size: Maximum community size for suspicious detection
        min_density: Minimum density for suspicious groups
        
    Returns:
        CommunityResult with all community information
    """
    logger.info("Detecting communities and tight-knit groups")
    
    # Detect all communities
    partition = detect_communities_louvain(G)
    
    # Group nodes by community
    communities_dict = defaultdict(set)
    for node, comm_id in partition.items():
        communities_dict[comm_id].add(node)
    
    communities_list = list(communities_dict.values())
    
    # Calculate metrics for each community
    community_metrics = {}
    for comm_id, nodes in communities_dict.items():
        subgraph = G.subgraph(nodes)
        n = len(nodes)
        
        # Calculate density
        if n > 1:
            if isinstance(subgraph, nx.DiGraph):
                max_edges = n * (n - 1)
                density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0
            else:
                density = nx.density(subgraph)
        else:
            density = 0.0
        
        community_metrics[comm_id] = {
            'size': n,
            'density': density,
            'n_edges': subgraph.number_of_edges(),
            'is_suspicious': (min_size <= n <= max_size and density >= min_density)
        }
    
    # Identify suspicious communities
    suspicious = [
        nodes for comm_id, nodes in communities_dict.items()
        if community_metrics[comm_id]['is_suspicious']
    ]
    
    return CommunityResult(
        communities=communities_list,
        community_assignments=partition,
        community_metrics=community_metrics,
        suspicious_communities=suspicious
    )


if __name__ == "__main__":
    # Test community detection
    from workplace_email_utils.graph_features.extractors import build_email_graph
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing community detection...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    G = build_email_graph(df)
    
    result = detect_tight_knit_groups(G)
    print(f"\nDetected {len(result.communities)} communities")
    print(f"Found {len(result.suspicious_communities)} suspicious tight-knit groups")
    
    if result.suspicious_communities:
        print("\nSuspicious communities:")
        for i, comm in enumerate(result.suspicious_communities[:5]):
            print(f"  {i+1}. {len(comm)} members: {list(comm)[:3]}...")

