"""
Email thread reconstruction algorithm.

Reconstructs email threads from Message-ID, In-Reply-To, and References headers.
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThreadTree:
    """Represents an email thread as a tree structure."""
    root_message_id: str
    messages: List[str]  # List of message IDs in chronological order
    participants: Set[str]  # All participants in thread
    thread_id: str  # Unique thread identifier
    depth: int = 0  # Maximum depth of thread
    message_count: int = 0  # Total number of messages
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    subject: str = ""
    parent_map: Dict[str, str] = field(default_factory=dict)  # message_id -> parent_message_id


def reconstruct_threads(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, ThreadTree]]:
    """
    Reconstruct email threads from Message-ID, In-Reply-To, and References headers.
    
    Algorithm:
    1. Build parent-child relationships from In-Reply-To
    2. Use References header to fill in missing links
    3. Handle missing Message-IDs (assign temporary IDs)
    4. Group messages into threads
    5. Assign thread IDs
    
    Args:
        df: DataFrame with email data including message_id, in_reply_to, references
        
    Returns:
        Tuple of (df_with_threads, thread_trees_dict)
        - df_with_threads: DataFrame with added 'thread_id' column
        - thread_trees_dict: Dictionary mapping thread_id -> ThreadTree
    """
    logger.info("Reconstructing email threads")
    
    df = df.copy()
    
    # Ensure required columns exist
    if 'message_id' not in df.columns:
        logger.warning("No message_id column found. Assigning temporary IDs.")
        df['message_id'] = [f"temp_{i}" for i in range(len(df))]
    
    # Normalize message IDs (remove angle brackets if present)
    df['message_id'] = df['message_id'].apply(normalize_message_id)
    df['in_reply_to'] = df.get('in_reply_to', pd.Series([''] * len(df))).apply(normalize_message_id)
    
    # Handle missing message IDs - assign unique IDs
    missing_msg_ids = df['message_id'].isna() | (df['message_id'] == '')
    if missing_msg_ids.any():
        logger.warning(f"Found {missing_msg_ids.sum()} emails without Message-ID. Assigning temporary IDs.")
        df.loc[missing_msg_ids, 'message_id'] = [
            f"temp_{i}" for i in range(missing_msg_ids.sum())
        ]
    
    # Build message lookup
    message_lookup = {}
    for idx, row in df.iterrows():
        msg_id = row['message_id']
        if msg_id and pd.notna(msg_id):
            message_lookup[msg_id] = {
                'idx': idx,
                'in_reply_to': row.get('in_reply_to', ''),
                'references_list': row.get('references_list', []),
                'subject': row.get('subject', ''),
                'date': row.get('date', ''),
                'sender': row.get('sender', ''),
                'recipients': row.get('recipients', [])
            }
    
    # Build parent-child relationships
    parent_map = {}  # child_message_id -> parent_message_id
    children_map = defaultdict(list)  # parent_message_id -> [child_ids]
    
    # Step 1: Use In-Reply-To to establish direct parent-child relationships
    for msg_id, msg_data in message_lookup.items():
        in_reply_to = msg_data['in_reply_to']
        if in_reply_to and pd.notna(in_reply_to):
            if in_reply_to in message_lookup:
                parent_map[msg_id] = in_reply_to
                children_map[in_reply_to].append(msg_id)
    
    # Step 2: Use References header to fill in missing links
    # References contains the full thread history, so we can use it to find parents
    for msg_id, msg_data in message_lookup.items():
        if msg_id not in parent_map and msg_data.get('references_list'):
            refs = msg_data['references_list']
            # The last reference in the list is typically the immediate parent
            if refs:
                for ref in reversed(refs):  # Check from most recent
                    if ref in message_lookup:
                        parent_map[msg_id] = ref
                        children_map[ref].append(msg_id)
                        break
    
    # Step 3: Build thread graph and find root messages
    # Build directed graph
    G = nx.DiGraph()
    for msg_id in message_lookup.keys():
        G.add_node(msg_id)
    
    for child, parent in parent_map.items():
        if parent in message_lookup:  # Ensure parent exists
            G.add_edge(parent, child)
    
    # Find root messages (no parent)
    root_messages = [msg_id for msg_id in message_lookup.keys() 
                     if msg_id not in parent_map]
    
    # For messages not connected to any root, create a root for them
    connected = set()
    for root in root_messages:
        connected.update(nx.descendants(G, root))
        connected.add(root)
    
    disconnected = set(message_lookup.keys()) - connected
    root_messages.extend(disconnected)
    
    logger.info(f"Found {len(root_messages)} root messages")
    
    # Step 4: Assign thread IDs and build thread trees
    thread_trees = {}
    df['thread_id'] = None
    
    for thread_idx, root_id in enumerate(root_messages):
        thread_id = f"thread_{thread_idx}"
        
        # Get all messages in this thread (descendants of root)
        thread_messages = set([root_id])
        if root_id in G:
            thread_messages.update(nx.descendants(G, root_id))
        
        # Sort messages by date if available
        thread_message_ids = sorted(
            thread_messages,
            key=lambda mid: message_lookup[mid].get('date', '') if mid in message_lookup else ''
        )
        
        # Assign thread_id to all messages in thread
        for msg_id in thread_message_ids:
            if msg_id in df.index or (df['message_id'] == msg_id).any():
                df.loc[df['message_id'] == msg_id, 'thread_id'] = thread_id
        
        # Build ThreadTree
        participants = set()
        dates = []
        subject = message_lookup[root_id].get('subject', '')
        
        for msg_id in thread_message_ids:
            msg_data = message_lookup[msg_id]
            participants.add(msg_data.get('sender', ''))
            if isinstance(msg_data.get('recipients'), list):
                participants.update(msg_data.get('recipients', []))
            date = msg_data.get('date', '')
            if date:
                dates.append(date)
        
        # Calculate depth
        depth = 0
        if root_id in G:
            try:
                depth = max([
                    len(nx.shortest_path(G, root_id, leaf))
                    for leaf in thread_message_ids
                    if nx.has_path(G, root_id, leaf)
                ], default=0) - 1
            except:
                depth = 0
        
        thread_trees[thread_id] = ThreadTree(
            root_message_id=root_id,
            messages=thread_message_ids,
            participants=participants - {''},  # Remove empty strings
            thread_id=thread_id,
            depth=depth,
            message_count=len(thread_message_ids),
            start_date=min(dates) if dates else None,
            end_date=max(dates) if dates else None,
            subject=subject,
            parent_map={k: v for k, v in parent_map.items() if k in thread_message_ids}
        )
    
    # Assign thread_id to messages that weren't assigned (orphans)
    unassigned = df['thread_id'].isna()
    if unassigned.any():
        logger.warning(f"Found {unassigned.sum()} unassigned messages. Creating individual threads.")
        next_thread_idx = len(thread_trees)
        for idx in df[unassigned].index:
            msg_id = df.loc[idx, 'message_id']
            thread_id = f"thread_{next_thread_idx}"
            df.loc[idx, 'thread_id'] = thread_id
            next_thread_idx += 1
            
            # Create single-message thread
            thread_trees[thread_id] = ThreadTree(
                root_message_id=msg_id,
                messages=[msg_id],
                participants=set([df.loc[idx, 'sender']]) | set(df.loc[idx].get('recipients', [])),
                thread_id=thread_id,
                depth=0,
                message_count=1,
                start_date=df.loc[idx].get('date'),
                end_date=df.loc[idx].get('date'),
                subject=df.loc[idx].get('subject', ''),
                parent_map={}
            )
    
    logger.info(f"Reconstructed {len(thread_trees)} threads")
    logger.info(f"Threads contain {df['thread_id'].notna().sum()} messages")
    
    return df, thread_trees


def normalize_message_id(msg_id: str) -> str:
    """
    Normalize Message-ID by removing angle brackets and whitespace.
    
    Args:
        msg_id: Raw Message-ID string
        
    Returns:
        Normalized Message-ID
    """
    if pd.isna(msg_id) or not msg_id:
        return ''
    
    msg_id = str(msg_id).strip()
    # Remove angle brackets
    msg_id = msg_id.strip('<>')
    return msg_id


def find_thread_root(df: pd.DataFrame, message_id: str) -> Optional[str]:
    """
    Find the root message of a thread containing the given message_id.
    
    Args:
        df: DataFrame with thread information
        message_id: Message-ID to find root for
        
    Returns:
        Root message ID or None
    """
    if message_id not in df['message_id'].values:
        return None
    
    thread_id = df[df['message_id'] == message_id]['thread_id'].values[0]
    if pd.isna(thread_id):
        return None
    
    thread_df = df[df['thread_id'] == thread_id]
    
    # Find root (message with no in_reply_to or in_reply_to not in thread)
    for _, row in thread_df.iterrows():
        in_reply_to = row.get('in_reply_to', '')
        if not in_reply_to or in_reply_to not in thread_df['message_id'].values:
            return row['message_id']
    
    # Fallback: earliest message
    if 'date' in thread_df.columns:
        return thread_df.sort_values('date')['message_id'].iloc[0]
    
    return thread_df['message_id'].iloc[0]


if __name__ == "__main__":
    # Test thread reconstruction
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing thread reconstruction...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    
    df_with_threads, thread_trees = reconstruct_threads(df)
    
    print(f"\nReconstructed {len(thread_trees)} threads")
    print(f"Messages in threads: {df_with_threads['thread_id'].notna().sum()}")
    
    # Show some threads
    for thread_id, thread_tree in list(thread_trees.items())[:5]:
        print(f"\nThread {thread_id}:")
        print(f"  Messages: {thread_tree.message_count}")
        print(f"  Depth: {thread_tree.depth}")
        print(f"  Participants: {len(thread_tree.participants)}")
        print(f"  Subject: {thread_tree.subject[:50]}")

