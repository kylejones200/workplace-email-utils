"""
Social graph feature extraction module.

Builds email communication graph and extracts network features, including
community detection and executive network analysis.
"""

from .extractors import build_email_graph, compute_graph_features, GraphFeatures
from .communities import (
    detect_communities_louvain,
    detect_suspicious_communities,
    analyze_community_structure,
    detect_tight_knit_groups,
    CommunityResult
)
from .executive_analysis import (
    identify_key_executives,
    filter_executive_communications,
    analyze_executive_network,
    ExecutiveNetworkAnalysis
)

__all__ = [
    'build_email_graph',
    'compute_graph_features',
    'GraphFeatures',
    'detect_communities_louvain',
    'detect_suspicious_communities',
    'analyze_community_structure',
    'detect_tight_knit_groups',
    'CommunityResult',
    'identify_key_executives',
    'filter_executive_communications',
    'analyze_executive_network',
    'ExecutiveNetworkAnalysis',
]

