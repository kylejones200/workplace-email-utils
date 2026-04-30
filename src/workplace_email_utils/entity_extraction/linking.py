"""
Entity linking module.

Links entities across emails to build relationships and track entities over time.
"""

import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_entity_name(entity: str) -> str:
    """
    Normalize entity name for matching.
    
    Args:
        entity: Entity name to normalize
        
    Returns:
        Normalized entity name
    """
    # Lowercase, remove punctuation, strip whitespace
    normalized = entity.lower().strip()
    # Remove common punctuation
    normalized = normalized.replace('.', '').replace(',', '').replace(';', '')
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def compute_entity_similarity(entity1: str, entity2: str) -> float:
    """
    Compute similarity between two entity names.
    
    Uses simple string matching. For better results, use fuzzy string matching.
    
    Args:
        entity1: First entity name
        entity2: Second entity name
        
    Returns:
        Similarity score between 0 and 1
    """
    norm1 = normalize_entity_name(entity1)
    norm2 = normalize_entity_name(entity2)
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # One contains the other
    if norm1 in norm2 or norm2 in norm1:
        return 0.8
    
    # Word overlap
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard


def link_entities_across_emails(df: pd.DataFrame,
                                entity_col: str = 'entities_persons',
                                similarity_threshold: float = 0.7,
                                doc_id_col: str = 'doc_id') -> Dict[str, List[str]]:
    """
    Link entities across emails by finding similar entity mentions.
    
    Args:
        df: DataFrame with entity columns
        entity_col: Column name containing entity lists
        similarity_threshold: Minimum similarity to link entities
        doc_id_col: Column name for document ID
        
    Returns:
        Dictionary mapping canonical entity names to lists of document IDs
    """
    logger.info(f"Linking entities from column: {entity_col}")
    
    if entity_col not in df.columns:
        logger.warning(f"Column '{entity_col}' not found in DataFrame")
        return {}
    
    # Collect all entities with their document IDs
    entity_docs: Dict[str, Set[str]] = defaultdict(set)
    
    for idx, row in df.iterrows():
        doc_id = str(row.get(doc_id_col, idx))
        entities = row.get(entity_col, [])
        
        if not isinstance(entities, list):
            continue
        
        for entity in entities:
            if entity:
                entity_docs[entity].add(doc_id)
    
    logger.info(f"Found {len(entity_docs)} unique entity mentions")
    
    # Link similar entities
    entity_groups: Dict[str, str] = {}  # entity -> canonical name
    canonical_entities: Dict[str, List[str]] = defaultdict(list)  # canonical -> doc_ids
    
    entities_list = list(entity_docs.keys())
    
    for i, entity1 in enumerate(entities_list):
        # Find canonical name (or create new)
        canonical = None
        
        # Check if similar to existing canonical entity
        for existing_canonical in canonical_entities.keys():
            similarity = compute_entity_similarity(entity1, existing_canonical)
            if similarity >= similarity_threshold:
                canonical = existing_canonical
                break
        
        # If no match, this becomes a new canonical entity
        if canonical is None:
            canonical = entity1
        
        entity_groups[entity1] = canonical
        canonical_entities[canonical].extend(list(entity_docs[entity1]))
    
    # Remove duplicates in document lists
    for canonical in canonical_entities:
        canonical_entities[canonical] = list(set(canonical_entities[canonical]))
    
    logger.info(f"Linked to {len(canonical_entities)} canonical entities")
    
    return dict(canonical_entities)


def build_entity_network(df: pd.DataFrame,
                        entity_cols: List[str] = None,
                        doc_id_col: str = 'doc_id',
                        date_col: str = 'date_parsed') -> pd.DataFrame:
    """
    Build network of entities showing co-occurrence in emails.
    
    Args:
        df: DataFrame with entity columns
        entity_cols: List of entity column names to analyze
        doc_id_col: Column name for document ID
        date_col: Column name for date
        
    Returns:
        DataFrame with entity relationships (entity1, entity2, weight, emails)
    """
    if entity_cols is None:
        entity_cols = ['entities_persons', 'entities_organizations', 'entities_locations']
    
    logger.info(f"Building entity network from {len(entity_cols)} entity types")
    
    relationships = []
    
    for idx, row in df.iterrows():
        doc_id = str(row.get(doc_id_col, idx))
        date = row.get(date_col)
        
        # Collect all entities from this email
        all_entities = []
        for col in entity_cols:
            if col in df.columns:
                entities = row.get(col, [])
                if isinstance(entities, list):
                    all_entities.extend([(e, col) for e in entities if e])
        
        # Create pairs of co-occurring entities
        for i, (entity1, type1) in enumerate(all_entities):
            for j, (entity2, type2) in enumerate(all_entities[i+1:], start=i+1):
                # Normalize for matching
                norm1 = normalize_entity_name(entity1)
                norm2 = normalize_entity_name(entity2)
                
                # Skip if same entity
                if norm1 == norm2:
                    continue
                
                # Order entities for consistency
                if norm1 > norm2:
                    norm1, norm2 = norm2, norm1
                    entity1, entity2 = entity2, entity1
                    type1, type2 = type2, type1
                
                relationships.append({
                    'entity1': entity1,
                    'entity2': entity2,
                    'entity1_type': type1,
                    'entity2_type': type2,
                    'doc_id': doc_id,
                    'date': date
                })
    
    if not relationships:
        logger.warning("No entity relationships found")
        return pd.DataFrame()
    
    # Aggregate relationships
    rel_df = pd.DataFrame(relationships)
    
    # Group by entity pairs and count co-occurrences
    network = rel_df.groupby(['entity1', 'entity2', 'entity1_type', 'entity2_type']).agg({
        'doc_id': ['count', lambda x: list(set(x))],
        'date': ['min', 'max']
    }).reset_index()
    
    network.columns = ['entity1', 'entity2', 'entity1_type', 'entity2_type', 
                       'weight', 'doc_ids', 'first_seen', 'last_seen']
    
    network = network.sort_values('weight', ascending=False)
    
    logger.info(f"Built network with {len(network)} entity relationships")
    
    return network


if __name__ == "__main__":
    # Test entity linking
    from extractors import extract_entities_from_dataframe
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing entity linking...")
    
    # Load sample data
    df = load_emails('maildir', data_format='maildir', max_rows=100)
    
    # Extract entities
    df = extract_entities_from_dataframe(df)
    
    # Link entities
    person_links = link_entities_across_emails(df, entity_col='entities_persons')
    print(f"\nLinked {len(person_links)} person entities")
    print(f"Sample links: {list(person_links.items())[:5]}")
    
    # Build network
    network = build_entity_network(df)
    print(f"\nBuilt network with {len(network)} relationships")
    if len(network) > 0:
        print(f"Top relationships:\n{network.head()}")

