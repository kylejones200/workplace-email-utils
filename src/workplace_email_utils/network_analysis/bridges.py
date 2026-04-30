"""
Bridge node detection.

Identifies bridge nodes that connect different communities or network regions.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
import logging

from workplace_email_utils.graph_features.communities import detect_tight_knit_groups, CommunityResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BridgeAnalysis:
    """Container for bridge node analysis results."""
    bridge_nodes: List[str]
    bridge_scores: Dict[str, float]
    inter_community_bridges: Dict[str, List[str]]  # bridge_node -> [communities it connects]
    isolated_nodes: List[str]


def detect_bridge_nodes(
    G: nx.Graph,
    communities: Optional[CommunityResult] = None,
    min_betweenness: float = 0.1,
    method: str = 'betweenness'
) -> List[str]:
    """
    Detect bridge nodes that connect different parts of the network.
    
    Bridge nodes have high betweenness centrality and connect
    different communities or network regions.
    
    Args:
        G: NetworkX graph
        communities: Optional CommunityResult from community detection
        min_betweenness: Minimum betweenness centrality threshold
        method: Detection method ('betweenness', 'community_cut', 'both')
        
    Returns:
        List of bridge node identifiers
    """
    logger.info("Detecting bridge nodes...")
    
    bridge_nodes = []
    
    if method in ['betweenness', 'both']:
        # Method 1: High betweenness centrality
        try:
            if len(G) > 500:
                betweenness = nx.betweenness_centrality(G, k=min(500, len(G)), normalized=True, seed=42)
            else:
                betweenness = nx.betweenness_centrality(G, normalized=True)
            
            # Nodes with high betweenness are potential bridges
            max_betweenness = max(betweenness.values()) if betweenness.values() else 0
            threshold = max_betweenness * min_betweenness
            
            betweenness_bridges = [
                node for node, score in betweenness.items()
                if score >= threshold
            ]
            
            bridge_nodes.extend(betweenness_bridges)
            logger.info(f"Found {len(betweenness_bridges)} bridge nodes by betweenness centrality")
        except Exception as e:
            logger.warning(f"Betweenness-based bridge detection failed: {e}")
    
    if method in ['community_cut', 'both'] and communities:
        # Method 2: Nodes connecting different communities
        community_bridges = find_inter_community_bridges(G, communities)
        bridge_nodes.extend(community_bridges)
        logger.info(f"Found {len(community_bridges)} bridge nodes connecting communities")
    
    # Remove duplicates
    bridge_nodes = list(set(bridge_nodes))
    
    logger.info(f"Total unique bridge nodes: {len(bridge_nodes)}")
    return bridge_nodes


def find_inter_community_bridges(
    G: nx.Graph,
    communities: CommunityResult
) -> List[str]:
    """
    Find nodes that connect different communities.
    
    Args:
        G: NetworkX graph
        communities: CommunityResult from community detection
        
    Returns:
        List of bridge nodes connecting communities
    """
    bridge_nodes = []
    
    # Build community membership map
    node_to_community = {}
    for comm_id, nodes in enumerate(communities.communities):
        for node in nodes:
            node_to_community[node] = comm_id
    
    # Find nodes with neighbors in different communities
    for node in G.nodes():
        if node not in node_to_community:
            continue
        
        node_community = node_to_community[node]
        neighbor_communities = set()
        
        for neighbor in G.neighbors(node):
            if neighbor in node_to_community:
                neighbor_communities.add(node_to_community[neighbor])
        
        # If node has neighbors in different communities, it's a bridge
        if len(neighbor_communities) > 1:
            bridge_nodes.append(node)
    
    return bridge_nodes


def analyze_bridge_structure(
    G: nx.Graph,
    communities: Optional[CommunityResult] = None
) -> BridgeAnalysis:
    """
    Comprehensive bridge node analysis.
    
    Args:
        G: NetworkX graph
        communities: Optional CommunityResult
        
    Returns:
        BridgeAnalysis with all bridge information
    """
    logger.info("Analyzing bridge structure...")
    
    # Detect communities if not provided
    if communities is None:
        from workplace_email_utils.graph_features.communities import detect_tight_knit_groups
        communities = detect_tight_knit_groups(G)
    
    # Detect bridge nodes
    bridge_nodes = detect_bridge_nodes(G, communities, method='both')
    
    # Compute bridge scores (betweenness centrality)
    try:
        if len(G) > 500:
            betweenness = nx.betweenness_centrality(G, k=min(500, len(G)), normalized=True, seed=42)
        else:
            betweenness = nx.betweenness_centrality(G, normalized=True)
        
        bridge_scores = {node: betweenness.get(node, 0.0) for node in bridge_nodes}
    except:
        bridge_scores = {node: 1.0 for node in bridge_nodes}
    
    # Find which communities each bridge connects
    node_to_community = {}
    for comm_id, comm_nodes in enumerate(communities.communities):
        for node in comm_nodes:
            node_to_community[node] = comm_id
    
    inter_community_bridges = {}
    for bridge in bridge_nodes:
        if bridge not in node_to_community:
            continue
        
        connected_communities = set([node_to_community[bridge]])
        
        for neighbor in G.neighbors(bridge):
            if neighbor in node_to_community:
                connected_communities.add(node_to_community[neighbor])
        
        if len(connected_communities) > 1:
            inter_community_bridges[bridge] = list(connected_communities)
    
    # Find isolated nodes (no edges or only self-loops)
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    
    return BridgeAnalysis(
        bridge_nodes=bridge_nodes,
        bridge_scores=bridge_scores,
        inter_community_bridges=inter_community_bridges,
        isolated_nodes=isolated_nodes
    )


def identify_critical_bridges(
    bridge_analysis: BridgeAnalysis,
    top_n: int = 10
) -> List[tuple]:
    """
    Identify the most critical bridge nodes.
    
    Critical bridges:
    - High bridge scores (betweenness)
    - Connect many communities
    - High degree centrality
    
    Args:
        bridge_analysis: BridgeAnalysis object
        top_n: Number of top bridges to return
        
    Returns:
        List of (node, score, communities) tuples
    """
    bridge_info = []
    
    for bridge in bridge_analysis.bridge_nodes:
        score = bridge_analysis.bridge_scores.get(bridge, 0.0)
        communities = bridge_analysis.inter_community_bridges.get(bridge, [])
        
        # Composite score: bridge score * number of communities connected
        composite_score = score * (1 + len(communities))
        
        bridge_info.append((bridge, composite_score, len(communities), communities))
    
    # Sort by composite score
    bridge_info.sort(key=lambda x: x[1], reverse=True)
    
    return bridge_info[:top_n]


if __name__ == "__main__":
    # Test bridge detection
    from workplace_email_utils.graph_features.extractors import build_email_graph
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing bridge detection...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    G = build_email_graph(df)
    
    # Detect communities
    from workplace_email_utils.graph_features.communities import detect_tight_knit_groups
    communities = detect_tight_knit_groups(G)
    
    # Analyze bridges
    bridge_analysis = analyze_bridge_structure(G, communities)
    
    print(f"\nBridge nodes: {len(bridge_analysis.bridge_nodes)}")
    print(f"Inter-community bridges: {len(bridge_analysis.inter_community_bridges)}")
    
    # Get critical bridges
    critical = identify_critical_bridges(bridge_analysis, top_n=5)
    print(f"\nTop 5 critical bridges:")
    for bridge, score, n_communities, comms in critical:
        print(f"  {bridge}: score={score:.3f}, connects {n_communities} communities")

