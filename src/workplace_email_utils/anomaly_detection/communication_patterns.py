"""
Communication pattern anomaly detection.

Based on insights from Enron paper:
- Detect unusual small-group communications
- Identify suspicious communication patterns
- Find anomalous executive interactions
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CommunicationAnomalyResult:
    """Container for communication anomaly detection results."""
    anomalous_emails: pd.DataFrame
    anomalous_groups: List[Set[str]]
    anomaly_scores: Dict[str, float]
    metrics: Dict


def detect_unusual_executive_patterns(
    df: pd.DataFrame,
    key_executives: Set[str],
    folder_filter: Optional[str] = 'sent'
) -> pd.DataFrame:
    """
    Detect unusual communication patterns among executives.
    
    Based on Enron paper: Look for:
    - Small tight-knit groups of executives
    - Unusual communication frequency
    - Off-hours communications
    - Cross-department communication anomalies
    
    Args:
        df: DataFrame with email data
        key_executives: Set of key executive email addresses
        folder_filter: Optional folder filter
        
    Returns:
        DataFrame with anomalous executive communications flagged
    """
    from workplace_email_utils.graph_features.executive_analysis import filter_executive_communications
    
    logger.info("Detecting unusual executive communication patterns")
    
    # Filter to executive communications
    exec_comm = filter_executive_communications(df, key_executives, folder_filter=folder_filter)
    
    if len(exec_comm) == 0:
        return pd.DataFrame()
    
    exec_comm = exec_comm.copy()
    exec_comm['is_anomaly'] = 0
    exec_comm['anomaly_score'] = 0.0
    exec_comm['anomaly_reason'] = ''
    
    # Check for small group communications (2-5 executives)
    for idx, row in exec_comm.iterrows():
        sender = str(row.get('sender', '')).lower().strip()
        recipients = row.get('recipients', [])
        
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(',')]
        elif not isinstance(recipients, list):
            recipients = []
        
        # Count executives in communication
        exec_recipients = [r for r in recipients if str(r).lower().strip() in key_executives]
        total_executives = 1 + len(exec_recipients)  # sender + recipients
        
        # Small group flag (2-5 executives)
        if 2 <= total_executives <= 5:
            exec_comm.at[idx, 'is_anomaly'] = 1
            exec_comm.at[idx, 'anomaly_score'] += 0.3
            exec_comm.at[idx, 'anomaly_reason'] += 'small_exec_group;'
        
        # Off-hours communication
        if 'hour' in row and pd.notna(row.get('hour')):
            hour = row['hour']
            if hour < 9 or hour > 17:
                exec_comm.at[idx, 'is_anomaly'] = 1
                exec_comm.at[idx, 'anomaly_score'] += 0.2
                exec_comm.at[idx, 'anomaly_reason'] += 'off_hours;'
        
        # Weekend communication
        if 'is_weekend' in row and row.get('is_weekend') == 1:
            exec_comm.at[idx, 'is_anomaly'] = 1
            exec_comm.at[idx, 'anomaly_score'] += 0.2
            exec_comm.at[idx, 'anomaly_reason'] += 'weekend;'
    
    n_anomalies = exec_comm['is_anomaly'].sum()
    logger.info(f"Detected {n_anomalies} anomalous executive communications")
    
    return exec_comm


def detect_communication_anomalies(
    df: pd.DataFrame,
    key_executives: Optional[Set[str]] = None,
    anomaly_threshold: float = 0.5,
    folder_filter: Optional[str] = 'sent'
) -> CommunicationAnomalyResult:
    """
    Detect communication pattern anomalies.
    
    Based on Enron paper insights:
    - Small tight-knit groups
    - Unusual communication patterns
    - Executive network anomalies
    
    Args:
        df: DataFrame with email data
        key_executives: Optional set of key executives (will auto-identify if None)
        anomaly_threshold: Minimum anomaly score to flag
        folder_filter: Optional folder filter
        
    Returns:
        CommunicationAnomalyResult with detected anomalies
    """
    logger.info("Detecting communication pattern anomalies")
    
    # Auto-identify executives if not provided
    if key_executives is None:
        from workplace_email_utils.graph_features.executive_analysis import identify_key_executives
        key_executives = identify_key_executives(df, method='centrality', top_n=20)
        logger.info(f"Auto-identified {len(key_executives)} key executives")
    
    # Detect unusual executive patterns
    anomalous_emails = detect_unusual_executive_patterns(
        df,
        key_executives,
        folder_filter=folder_filter
    )
    
    if len(anomalous_emails) == 0:
        return CommunicationAnomalyResult(
            anomalous_emails=pd.DataFrame(),
            anomalous_groups=[],
            anomaly_scores={},
            metrics={}
        )
    
    # Filter by threshold
    high_anomaly_emails = anomalous_emails[
        anomalous_emails['anomaly_score'] >= anomaly_threshold
    ]
    
    # Identify anomalous groups
    anomalous_groups = []
    if len(high_anomaly_emails) > 0:
        # Group emails by participants
        groups = {}
        for idx, row in high_anomaly_emails.iterrows():
            sender = str(row.get('sender', '')).lower().strip()
            recipients = row.get('recipients', [])
            
            if isinstance(recipients, str):
                recipients = [r.strip() for r in recipients.split(',')]
            elif not isinstance(recipients, list):
                recipients = []
            
            # Create participant set
            participants = {sender}
            participants.update({str(r).lower().strip() for r in recipients if str(r).lower().strip() in key_executives})
            
            # Group by participant set
            key = frozenset(participants)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        # Identify tight-knit groups (appear together frequently)
        for participants, email_indices in groups.items():
            if len(participants) >= 2 and len(email_indices) >= 3:
                anomalous_groups.append(set(participants))
    
    # Calculate anomaly scores by group
    anomaly_scores = {}
    for group in anomalous_groups:
        group_emails = high_anomaly_emails[
            high_anomaly_emails.apply(
                lambda r: any(p in str(r.get('sender', '')).lower() or p in [str(x).lower() for x in r.get('recipients', [])] for p in group),
                axis=1
            )
        ]
        score = group_emails['anomaly_score'].mean() if len(group_emails) > 0 else 0.0
        anomaly_scores[','.join(sorted(group))] = score
    
    metrics = {
        'total_anomalies': len(high_anomaly_emails),
        'anomalous_groups': len(anomalous_groups),
        'avg_anomaly_score': high_anomaly_emails['anomaly_score'].mean() if len(high_anomaly_emails) > 0 else 0.0,
    }
    
    logger.info(f"Communication anomalies: {metrics}")
    
    return CommunicationAnomalyResult(
        anomalous_emails=high_anomaly_emails,
        anomalous_groups=anomalous_groups,
        anomaly_scores=anomaly_scores,
        metrics=metrics
    )


if __name__ == "__main__":
    # Test anomaly detection
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.temporal_features.extractors import extract_temporal_features
    
    print("Testing communication anomaly detection...")
    df = load_emails('maildir', data_format='maildir', max_rows=1000)
    df = extract_temporal_features(df)
    
    result = detect_communication_anomalies(df, anomaly_threshold=0.3)
    
    print(f"\nAnomalous emails: {len(result.anomalous_emails)}")
    print(f"Anomalous groups: {len(result.anomalous_groups)}")
    print(f"Metrics: {result.metrics}")

