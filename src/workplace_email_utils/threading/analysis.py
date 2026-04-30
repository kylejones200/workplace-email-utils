"""
Thread analysis and metrics computation.

Analyzes conversation flow, computes thread metrics, and scores thread importance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .reconstruct import ThreadTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThreadMetrics:
    """Container for thread metrics."""
    thread_id: str
    message_count: int
    participant_count: int
    depth: int
    duration_days: float
    avg_response_time_hours: float
    message_rate_per_day: float
    subject: str
    participants: Set[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None


def compute_thread_metrics(
    df: pd.DataFrame,
    thread_trees: Dict[str, ThreadTree],
    date_column: str = 'date'
) -> Dict[str, ThreadMetrics]:
    """
    Compute comprehensive metrics for each thread.
    
    Metrics include:
    - Message count
    - Participant count
    - Thread depth
    - Duration (days)
    - Average response time
    - Message rate (messages per day)
    
    Args:
        df: DataFrame with email data and thread_id column
        thread_trees: Dictionary of thread_id -> ThreadTree
        date_column: Column name for dates
        
    Returns:
        Dictionary mapping thread_id -> ThreadMetrics
    """
    logger.info("Computing thread metrics")
    
    metrics = {}
    
    # Parse dates for response time calculation
    df_with_dates = df.copy()
    if date_column in df.columns:
        df_with_dates['date_parsed'] = pd.to_datetime(
            df_with_dates[date_column],
            errors='coerce',
            utc=True
        )
    else:
        logger.warning(f"Date column '{date_column}' not found. Response times will be 0.")
        df_with_dates['date_parsed'] = None
    
    for thread_id, thread_tree in thread_trees.items():
        thread_df = df_with_dates[df_with_dates.get('thread_id') == thread_id].copy()
        
        if len(thread_df) == 0:
            continue
        
        # Basic counts
        message_count = thread_tree.message_count
        participant_count = len(thread_tree.participants)
        depth = thread_tree.depth
        
        # Duration
        duration_days = 0.0
        if thread_tree.start_date and thread_tree.end_date:
            try:
                start = pd.to_datetime(thread_tree.start_date, errors='coerce', utc=True)
                end = pd.to_datetime(thread_tree.end_date, errors='coerce', utc=True)
                if pd.notna(start) and pd.notna(end):
                    duration_days = (end - start).total_seconds() / (24 * 3600)
                    duration_days = max(duration_days, 0.0)
            except:
                pass
        
        # Average response time
        avg_response_time_hours = 0.0
        if 'date_parsed' in thread_df.columns and thread_df['date_parsed'].notna().sum() > 1:
            dates = thread_df['date_parsed'].dropna().sort_values()
            if len(dates) > 1:
                response_times = []
                for i in range(1, len(dates)):
                    time_diff = (dates.iloc[i] - dates.iloc[i-1]).total_seconds() / 3600
                    if time_diff > 0:  # Only positive time differences
                        response_times.append(time_diff)
                
                if response_times:
                    avg_response_time_hours = np.mean(response_times)
        
        # Message rate
        message_rate_per_day = 0.0
        if duration_days > 0:
            message_rate_per_day = message_count / duration_days
        elif message_count > 0:
            # If duration is 0 (all messages same day), use 1 day as baseline
            message_rate_per_day = message_count
        
        metrics[thread_id] = ThreadMetrics(
            thread_id=thread_id,
            message_count=message_count,
            participant_count=participant_count,
            depth=depth,
            duration_days=duration_days,
            avg_response_time_hours=avg_response_time_hours,
            message_rate_per_day=message_rate_per_day,
            subject=thread_tree.subject,
            participants=thread_tree.participants,
            start_date=thread_tree.start_date,
            end_date=thread_tree.end_date
        )
    
    logger.info(f"Computed metrics for {len(metrics)} threads")
    return metrics


def analyze_conversation_flow(
    df: pd.DataFrame,
    thread_id: str,
    thread_tree: ThreadTree
) -> Dict:
    """
    Analyze the conversation flow within a thread.
    
    Analyzes:
    - Message order and timing
    - Participant engagement (who replied when)
    - Turn-taking patterns
    - Response patterns
    
    Args:
        df: DataFrame with email data
        thread_id: Thread ID to analyze
        thread_tree: ThreadTree object
        
    Returns:
        Dictionary with flow analysis metrics
    """
    thread_df = df[df.get('thread_id') == thread_id].copy()
    
    if len(thread_df) == 0:
        return {}
    
    # Parse dates
    thread_df['date_parsed'] = pd.to_datetime(
        thread_df.get('date', ''),
        errors='coerce',
        utc=True
    )
    thread_df = thread_df.sort_values('date_parsed')
    
    # Analyze participant engagement
    participant_turns = []
    participant_counts = defaultdict(int)
    
    for _, row in thread_df.iterrows():
        sender = row.get('sender', '')
        participant_turns.append(sender)
        participant_counts[sender] += 1
    
    # Calculate turn-taking metrics
    unique_turns = len(set(participant_turns))
    turn_concentration = max(participant_counts.values()) / len(participant_turns) if participant_turns else 0
    
    # Analyze response patterns
    response_times = []
    for i in range(1, len(thread_df)):
        prev_date = thread_df.iloc[i-1]['date_parsed']
        curr_date = thread_df.iloc[i]['date_parsed']
        if pd.notna(prev_date) and pd.notna(curr_date):
            time_diff = (curr_date - prev_date).total_seconds() / 3600
            if time_diff > 0:
                response_times.append(time_diff)
    
    return {
        'participant_turns': participant_turns,
        'participant_engagement': dict(participant_counts),
        'unique_senders': unique_turns,
        'turn_concentration': turn_concentration,
        'response_times': response_times,
        'avg_response_time_hours': np.mean(response_times) if response_times else 0.0,
        'min_response_time_hours': np.min(response_times) if response_times else 0.0,
        'max_response_time_hours': np.max(response_times) if response_times else 0.0,
    }


def score_thread_importance(
    thread_metrics: ThreadMetrics,
    flow_analysis: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Score thread importance based on various factors.
    
    Factors:
    - Message count (more messages = more important)
    - Participant count (more people = more important)
    - Depth (longer chains = more important)
    - Duration (longer discussions = more important)
    - Response rate (active discussions = more important)
    
    Args:
        thread_metrics: ThreadMetrics object
        flow_analysis: Optional conversation flow analysis
        weights: Optional weights for different factors
        
    Returns:
        Importance score (0.0 to 1.0)
    """
    if weights is None:
        weights = {
            'message_count': 0.3,
            'participant_count': 0.2,
            'depth': 0.2,
            'duration': 0.15,
            'message_rate': 0.15
        }
    
    # Normalize factors (assuming max values for normalization)
    max_message_count = 100
    max_participant_count = 20
    max_depth = 10
    max_duration_days = 30
    max_message_rate = 10  # messages per day
    
    # Compute normalized scores
    message_score = min(thread_metrics.message_count / max_message_count, 1.0)
    participant_score = min(thread_metrics.participant_count / max_participant_count, 1.0)
    depth_score = min(thread_metrics.depth / max_depth, 1.0)
    duration_score = min(thread_metrics.duration_days / max_duration_days, 1.0)
    rate_score = min(thread_metrics.message_rate_per_day / max_message_rate, 1.0)
    
    # Weighted sum
    importance_score = (
        weights['message_count'] * message_score +
        weights['participant_count'] * participant_score +
        weights['depth'] * depth_score +
        weights['duration'] * duration_score +
        weights['message_rate'] * rate_score
    )
    
    # Adjust based on flow analysis if available
    if flow_analysis:
        # High turn concentration might indicate focused discussion
        if flow_analysis.get('turn_concentration', 0) > 0.5:
            importance_score *= 1.1  # Boost for focused discussions
        importance_score = min(importance_score, 1.0)  # Cap at 1.0
    
    return importance_score


def analyze_all_threads(
    df: pd.DataFrame,
    thread_trees: Dict[str, ThreadTree],
    thread_metrics: Optional[Dict[str, ThreadMetrics]] = None
) -> pd.DataFrame:
    """
    Analyze all threads and return summary DataFrame.
    
    Args:
        df: DataFrame with email data
        thread_trees: Dictionary of thread trees
        thread_metrics: Optional pre-computed metrics
        
    Returns:
        DataFrame with thread analysis summary
    """
    if thread_metrics is None:
        thread_metrics = compute_thread_metrics(df, thread_trees)
    
    rows = []
    for thread_id, metrics in thread_metrics.items():
        flow = analyze_conversation_flow(
            df,
            thread_id,
            thread_trees[thread_id]
        )
        importance = score_thread_importance(metrics, flow)
        
        rows.append({
            'thread_id': thread_id,
            'message_count': metrics.message_count,
            'participant_count': metrics.participant_count,
            'depth': metrics.depth,
            'duration_days': metrics.duration_days,
            'avg_response_time_hours': metrics.avg_response_time_hours,
            'message_rate_per_day': metrics.message_rate_per_day,
            'importance_score': importance,
            'subject': metrics.subject,
            'start_date': metrics.start_date,
            'end_date': metrics.end_date,
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test thread analysis
    from workplace_email_utils.ingest.email_parser import load_emails
    from workplace_email_utils.threading.reconstruct import reconstruct_threads
    
    print("Testing thread analysis...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    
    df_with_threads, thread_trees = reconstruct_threads(df)
    
    metrics = compute_thread_metrics(df_with_threads, thread_trees)
    print(f"\nComputed metrics for {len(metrics)} threads")
    
    # Show top threads by importance
    summary = analyze_all_threads(df_with_threads, thread_trees, metrics)
    top_threads = summary.nlargest(5, 'importance_score')
    print("\nTop threads by importance:")
    print(top_threads[['thread_id', 'message_count', 'participant_count', 'importance_score']])

