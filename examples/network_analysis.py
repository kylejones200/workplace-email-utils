"""
Network analysis example.

Demonstrates how to analyze communication networks, identify key influencers,
and detect communities.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from workplace_email_utils.pipeline import build_knowledge_model
from workplace_email_utils.network_analysis.influence import compute_influence_metrics, rank_influencers
from workplace_email_utils.network_analysis.bridges import analyze_bridge_structure
from workplace_email_utils.graph_features.communities import detect_tight_knit_groups
from workplace_email_utils.graph_features.extractors import build_email_graph

def main():
    """Example of network analysis capabilities."""
    
    print("Building model for network analysis...")
    model = build_knowledge_model(
        data_path='maildir',
        data_format='maildir',
        sample_size=5000
    )
    
    # Build network graph
    print("\nBuilding communication network...")
    graph = build_email_graph(model.df)
    print(f"Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Compute influence metrics
    print("\nComputing influence metrics...")
    metrics = compute_influence_metrics(graph)
    
    # Rank influencers
    print("\nTop 10 Influencers:")
    influencers = rank_influencers(metrics, top_n=10)
    for idx, row in influencers.iterrows():
        print(f"  {idx+1}. {row['node'][:40]:40s} "
              f"Score: {row['influence_score']:.3f} "
              f"(PageRank: {row['pagerank']:.3f})")
    
    # Detect communities
    print("\nDetecting communities...")
    communities = detect_tight_knit_groups(graph)
    print(f"Total communities: {len(communities.communities)}")
    print(f"Suspicious tight-knit groups: {len(communities.suspicious_communities)}")
    
    # Analyze bridges
    print("\nAnalyzing bridge nodes...")
    bridges = analyze_bridge_structure(graph, communities)
    print(f"Bridge nodes: {len(bridges.bridge_nodes)}")
    
    if bridges.bridge_nodes:
        print("\nTop 5 Critical Bridges:")
        from workplace_email_utils.network_analysis.bridges import identify_critical_bridges
        critical = identify_critical_bridges(bridges, top_n=5)
        for bridge, score, n_comm, comms in critical:
            print(f"  {bridge[:40]:40s} Score: {score:.3f} (connects {n_comm} communities)")

if __name__ == "__main__":
    main()

