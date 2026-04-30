"""
Executive network analysis based on Enron paper insights.

Filters and analyzes communications between key executives to identify
suspicious patterns and tight-knit groups.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import logging

from .extractors import build_email_graph
from .communities import detect_tight_knit_groups, CommunityResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutiveNetworkAnalysis:
    """Container for executive network analysis results."""
    executive_emails: pd.DataFrame
    executive_graph: nx.DiGraph
    communities: CommunityResult
    network_metrics: Dict


def identify_key_executives(
    df: pd.DataFrame,
    method: str = 'centrality',
    top_n: int = 20,
    sender_col: str = 'sender',
    recipient_col: str = 'recipients'
) -> Set[str]:
    """
    Identify key executives based on communication patterns.
    
    Methods:
    - 'centrality': Top nodes by degree/centrality
    - 'volume': Most active senders/recipients
    - 'knowledge_base': From knowledge base (if provided)
    
    Args:
        df: DataFrame with email data
        method: Method for identification
        top_n: Number of executives to identify
        sender_col: Column name for sender
        recipient_col: Column name for recipients
        
    Returns:
        Set of key executive email addresses
    """
    logger.info(f"Identifying key executives using method: {method}")
    
    if method == 'centrality':
        # Build graph and use centrality
        G = build_email_graph(df)
        
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G)
        
        # Get top nodes by centrality
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        key_people = {node.lower().strip() for node, _ in sorted_nodes[:top_n]}
        
        logger.info(f"Identified {len(key_people)} key executives by centrality")
        return key_people
    
    elif method == 'volume':
        # Count email activity
        sender_counts = df[sender_col].value_counts()
        recipient_counts = pd.Series()
        
        # Count recipients
        all_recipients = []
        for recipients in df[recipient_col].dropna():
            if isinstance(recipients, list):
                all_recipients.extend(recipients)
            elif isinstance(recipients, str):
                all_recipients.extend([r.strip() for r in recipients.split(',')])
        
        if all_recipients:
            recipient_counts = pd.Series(all_recipients).value_counts()
        
        # Combine sender and recipient activity
        combined = sender_counts.add(recipient_counts, fill_value=0).sort_values(ascending=False)
        key_people = {email.lower().strip() for email in combined.head(top_n).index}
        
        logger.info(f"Identified {len(key_people)} key executives by volume")
        return key_people
    
    else:
        raise ValueError(f"Unknown method: {method}")


def filter_executive_communications(
    df: pd.DataFrame,
    key_executives: Set[str],
    sender_col: str = 'sender',
    recipient_col: str = 'recipients',
    folder_filter: Optional[str] = 'sent'
) -> pd.DataFrame:
    """
    Filter emails to only communications between key executives.
    
    Based on Enron paper methodology:
    - Emails sent by key executives
    - Where at least one recipient is also a key executive
    - Optionally filter to 'sent' folder only
    
    Args:
        df: DataFrame with email data
        key_executives: Set of key executive email addresses
        sender_col: Column name for sender
        recipient_col: Column name for recipients
        folder_filter: Optional folder type filter ('sent', 'inbox', None)
        
    Returns:
        Filtered DataFrame with executive communications only
    """
    logger.info(f"Filtering executive communications from {len(df)} emails")
    logger.info(f"Key executives: {len(key_executives)}")
    
    # Normalize key executives set
    key_executives_lower = {email.lower().strip() for email in key_executives}
    
    # Filter by folder type if specified
    if folder_filter and 'folder_type' in df.columns:
        df_filtered = df[df['folder_type'] == folder_filter].copy()
        logger.info(f"Filtered to {folder_filter} folder: {len(df_filtered)} emails")
    else:
        df_filtered = df.copy()
    
    # Filter: sender is key executive
    df_exec_sender = df_filtered[
        df_filtered[sender_col].str.lower().str.strip().isin(key_executives_lower)
    ].copy()
    
    logger.info(f"Emails from key executives: {len(df_exec_sender)}")
    
    # Filter: at least one recipient is also key executive
    def has_executive_recipient(recipients):
        if pd.isna(recipients):
            return False
        if isinstance(recipients, list):
            recipient_set = {str(r).lower().strip() for r in recipients}
        elif isinstance(recipients, str):
            recipient_set = {r.lower().strip() for r in recipients.split(',')}
        else:
            return False
        return bool(recipient_set.intersection(key_executives_lower))
    
    df_exec_comm = df_exec_sender[
        df_exec_sender[recipient_col].apply(has_executive_recipient)
    ].copy()
    
    logger.info(f"Executive-to-executive communications: {len(df_exec_comm)}")
    
    return df_exec_comm


def analyze_executive_network(
    df: pd.DataFrame,
    key_executives: Optional[Set[str]] = None,
    auto_identify: bool = True,
    top_n: int = 20,
    folder_filter: Optional[str] = 'sent'
) -> ExecutiveNetworkAnalysis:
    """
    Comprehensive executive network analysis.
    
    Based on Enron paper methodology:
    1. Identify key executives
    2. Filter communications between executives
    3. Build executive network
    4. Detect tight-knit communities
    
    Args:
        df: DataFrame with email data
        key_executives: Optional set of known key executives
        auto_identify: Auto-identify executives if not provided
        top_n: Number of executives to identify (if auto_identify)
        folder_filter: Optional folder filter ('sent' recommended)
        
    Returns:
        ExecutiveNetworkAnalysis with all results
    """
    logger.info("Analyzing executive network")
    
    # Identify executives if not provided
    if key_executives is None:
        if auto_identify:
            key_executives = identify_key_executives(df, method='centrality', top_n=top_n)
        else:
            raise ValueError("Must provide key_executives or set auto_identify=True")
    
    # Filter executive communications
    exec_emails = filter_executive_communications(
        df,
        key_executives,
        folder_filter=folder_filter
    )
    
    if len(exec_emails) == 0:
        logger.warning("No executive communications found")
        return ExecutiveNetworkAnalysis(
            executive_emails=pd.DataFrame(),
            executive_graph=nx.DiGraph(),
            communities=None,
            network_metrics={}
        )
    
    # Build executive network
    exec_graph = build_email_graph(exec_emails)
    
    logger.info(f"Executive network: {exec_graph.number_of_nodes()} nodes, {exec_graph.number_of_edges()} edges")
    
    # Detect communities
    communities = detect_tight_knit_groups(exec_graph)
    
    # Calculate network metrics
    network_metrics = {
        'n_nodes': exec_graph.number_of_nodes(),
        'n_edges': exec_graph.number_of_edges(),
        'density': nx.density(exec_graph.to_undirected()) if isinstance(exec_graph, nx.DiGraph) else nx.density(exec_graph),
        'n_communities': len(communities.communities),
        'n_suspicious_communities': len(communities.suspicious_communities),
        'avg_community_size': np.mean([len(c) for c in communities.communities]) if communities.communities else 0,
    }
    
    logger.info(f"Network metrics: {network_metrics}")
    
    return ExecutiveNetworkAnalysis(
        executive_emails=exec_emails,
        executive_graph=exec_graph,
        communities=communities,
        network_metrics=network_metrics
    )


if __name__ == "__main__":
    # Test executive network analysis
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing executive network analysis...")
    df = load_emails('maildir', data_format='maildir', max_rows=1000)
    
    # Analyze executive network
    analysis = analyze_executive_network(df, auto_identify=True, top_n=15)
    
    print(f"\nExecutive communications: {len(analysis.executive_emails)}")
    print(f"Network nodes: {analysis.executive_graph.number_of_nodes()}")
    print(f"Suspicious communities: {len(analysis.communities.suspicious_communities)}")
    
    if analysis.communities.suspicious_communities:
        print("\nSuspicious groups:")
        for i, comm in enumerate(analysis.communities.suspicious_communities[:3]):
            print(f"  {i+1}. {len(comm)} members")

