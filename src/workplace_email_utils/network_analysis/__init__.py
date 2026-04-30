"""
Advanced network analysis module.

Includes influence metrics, bridge detection, temporal network analysis,
and visualization tools.
"""

from .influence import (
    compute_pagerank,
    compute_hits,
    compute_influence_metrics,
    InfluenceMetrics
)
from .bridges import (
    detect_bridge_nodes,
    find_inter_community_bridges,
    BridgeAnalysis
)
from .temporal_network import (
    build_temporal_network,
    analyze_network_evolution,
    TemporalNetworkAnalysis
)
from .visualization import (
    plot_network_graph,
    plot_community_structure,
    plot_influence_network,
    plot_temporal_network
)

__all__ = [
    'compute_pagerank',
    'compute_hits',
    'compute_influence_metrics',
    'InfluenceMetrics',
    'detect_bridge_nodes',
    'find_inter_community_bridges',
    'BridgeAnalysis',
    'build_temporal_network',
    'analyze_network_evolution',
    'TemporalNetworkAnalysis',
    'plot_network_graph',
    'plot_community_structure',
    'plot_influence_network',
    'plot_temporal_network',
]

