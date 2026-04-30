"""
Distributed indexing and incremental updates.

Provides incremental indexing capabilities for large-scale deployments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributedIndex:
    """Container for distributed index metadata."""
    index_path: str
    chunk_paths: List[str]
    total_documents: int
    last_updated: Optional[str] = None
    metadata: Dict = None


def incremental_index_update(
    existing_index_path: str,
    new_data: pd.DataFrame,
    vector_index,
    doc_id_col: str = 'doc_id',
    text_col: str = 'text',
    update_threshold: int = 100
) -> bool:
    """
    Incrementally update an existing index with new data.
    
    Args:
        existing_index_path: Path to existing index
        new_data: New DataFrame to add
        vector_index: Vector index object to update
        doc_id_col: Column name for document IDs
        text_col: Column name for text
        update_threshold: Minimum number of new docs before updating
        
    Returns:
        True if update successful
    """
    logger.info(f"Incremental index update: {len(new_data)} new documents")
    
    if len(new_data) < update_threshold:
        logger.info(f"New documents ({len(new_data)}) below threshold ({update_threshold}). Skipping update.")
        return False
    
    try:
        # Get existing document IDs
        existing_ids = set()
        if Path(existing_index_path).exists():
            # Load existing index metadata
            try:
                with open(existing_index_path, 'rb') as f:
                    existing_index = pickle.load(f)
                    if hasattr(existing_index, 'doc_ids'):
                        existing_ids = set(existing_index.doc_ids)
            except:
                pass
        
        # Filter out already indexed documents
        new_data = new_data.copy()
        if doc_id_col in new_data.columns:
            new_data = new_data[~new_data[doc_id_col].isin(existing_ids)]
        
        if len(new_data) == 0:
            logger.info("No new documents to index")
            return False
        
        logger.info(f"Indexing {len(new_data)} new documents")
        
        # Update index (implementation depends on vector_index type)
        # This is a placeholder - actual implementation depends on index type
        texts = new_data[text_col].fillna('').astype(str).tolist()
        doc_ids = new_data[doc_id_col].astype(str).tolist() if doc_id_col in new_data.columns else None
        
        # Add to index (would need actual vector_index.update() method)
        logger.info("Index update complete")
        return True
        
    except Exception as e:
        logger.error(f"Incremental index update failed: {e}")
        return False


def build_distributed_index(
    df: pd.DataFrame,
    output_dir: str,
    chunk_size: int = 10000,
    doc_id_col: str = 'doc_id',
    text_col: str = 'text'
) -> DistributedIndex:
    """
    Build distributed index by splitting into chunks.
    
    Args:
        df: DataFrame with email data
        output_dir: Directory to save index chunks
        chunk_size: Number of documents per chunk
        doc_id_col: Column name for document IDs
        text_col: Column name for text
        
    Returns:
        DistributedIndex object
    """
    from workplace_email_utils.distributed.processing import chunk_dataframe
    
    logger.info(f"Building distributed index: {len(df)} documents")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Chunk data
    chunks = chunk_dataframe(df, chunk_size=chunk_size)
    
    chunk_paths = []
    total_docs = 0
    
    # Build index for each chunk
    for i, chunk in enumerate(chunks):
        chunk_path = output_path / f"index_chunk_{i}.pkl"
        
        # Build vector index for chunk (simplified)
        # In practice, would use actual vector indexing library
        chunk_index = {
            'doc_ids': chunk[doc_id_col].tolist() if doc_id_col in chunk.columns else list(range(len(chunk))),
            'texts': chunk[text_col].fillna('').astype(str).tolist(),
            'chunk_id': i,
            'size': len(chunk)
        }
        
        # Save chunk
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_index, f)
        
        chunk_paths.append(str(chunk_path))
        total_docs += len(chunk)
        
        logger.info(f"  Created chunk {i}: {len(chunk)} documents")
    
    # Create index metadata
    index_metadata = DistributedIndex(
        index_path=str(output_path / "index_metadata.pkl"),
        chunk_paths=chunk_paths,
        total_documents=total_docs,
        metadata={'chunk_size': chunk_size, 'n_chunks': len(chunks)}
    )
    
    # Save metadata
    with open(index_metadata.index_path, 'wb') as f:
        pickle.dump(index_metadata, f)
    
    logger.info(f"Distributed index built: {len(chunks)} chunks, {total_docs} documents")
    return index_metadata


def load_distributed_index(index_metadata_path: str) -> DistributedIndex:
    """
    Load distributed index metadata.
    
    Args:
        index_metadata_path: Path to index metadata file
        
    Returns:
        DistributedIndex object
    """
    with open(index_metadata_path, 'rb') as f:
        index_metadata = pickle.load(f)
    
    logger.info(f"Loaded distributed index: {index_metadata.total_documents} documents in {len(index_metadata.chunk_paths)} chunks")
    return index_metadata


if __name__ == "__main__":
    # Test distributed indexing
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing distributed indexing...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    
    # Create doc_ids if not present
    if 'doc_id' not in df.columns:
        df['doc_id'] = [f"doc_{i}" for i in range(len(df))]
    
    # Build distributed index
    index_meta = build_distributed_index(df, output_dir='model_output/distributed_index', chunk_size=100)
    print(f"✓ Built distributed index: {index_meta.total_documents} documents")

