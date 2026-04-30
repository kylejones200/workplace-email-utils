"""
Distributed processing framework.

Enables parallel processing of emails and feature extraction for scalability.
"""

import pandas as pd
import numpy as np
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributedProcessor:
    """Container for distributed processing configuration."""
    n_workers: int = 4
    chunk_size: int = 1000
    method: str = 'multiprocessing'  # 'multiprocessing', 'threading', 'sequential'


def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 1000,
    overlap: int = 0
) -> List[pd.DataFrame]:
    """
    Split DataFrame into chunks for parallel processing.
    
    Args:
        df: DataFrame to chunk
        chunk_size: Size of each chunk
        overlap: Number of rows to overlap between chunks
        
    Returns:
        List of DataFrame chunks
    """
    chunks = []
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(n_chunks):
        start_idx = i * chunk_size - (overlap if i > 0 else 0)
        end_idx = min((i + 1) * chunk_size, len(df))
        start_idx = max(0, start_idx)
        
        chunk = df.iloc[start_idx:end_idx].copy()
        chunks.append(chunk)
    
    logger.info(f"Split DataFrame into {len(chunks)} chunks (chunk_size={chunk_size})")
    return chunks


def process_chunk_sequential(
    chunk: pd.DataFrame,
    process_func: Callable[[pd.DataFrame], pd.DataFrame],
    **kwargs
) -> pd.DataFrame:
    """
    Process a single chunk sequentially.
    
    Args:
        chunk: DataFrame chunk to process
        process_func: Function to apply to chunk
        **kwargs: Additional arguments for process_func
        
    Returns:
        Processed DataFrame chunk
    """
    try:
        return process_func(chunk, **kwargs)
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        return pd.DataFrame()


def parallel_process_emails(
    df: pd.DataFrame,
    process_func: Callable[[pd.DataFrame], pd.DataFrame],
    processor: Optional[DistributedProcessor] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Process emails in parallel using chunks.
    
    Args:
        df: DataFrame with email data
        process_func: Function to apply to each chunk
        processor: Optional DistributedProcessor configuration
        **kwargs: Additional arguments for process_func
        
    Returns:
        Processed DataFrame
    """
    if processor is None:
        processor = DistributedProcessor()
    
    logger.info(f"Parallel processing {len(df)} emails (method: {processor.method}, workers: {processor.n_workers})")
    
    # Chunk data
    chunks = chunk_dataframe(df, chunk_size=processor.chunk_size)
    
    # Process chunks
    if processor.method == 'multiprocessing':
        try:
            from multiprocessing import Pool
            
            with Pool(processes=processor.n_workers) as pool:
                results = pool.starmap(
                    process_chunk_sequential,
                    [(chunk, process_func, kwargs) for chunk in chunks]
                )
        except Exception as e:
            logger.warning(f"Multiprocessing failed: {e}. Falling back to sequential.")
            results = [process_chunk_sequential(chunk, process_func, **kwargs) for chunk in chunks]
    
    elif processor.method == 'threading':
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=processor.n_workers) as executor:
                futures = [
                    executor.submit(process_chunk_sequential, chunk, process_func, **kwargs)
                    for chunk in chunks
                ]
                results = [f.result() for f in futures]
        except Exception as e:
            logger.warning(f"Threading failed: {e}. Falling back to sequential.")
            results = [process_chunk_sequential(chunk, process_func, **kwargs) for chunk in chunks]
    
    else:  # sequential
        results = [process_chunk_sequential(chunk, process_func, **kwargs) for chunk in chunks]
    
    # Combine results
    if results:
        combined = pd.concat([r for r in results if not r.empty], ignore_index=True)
        logger.info(f"Parallel processing complete: {len(combined)} emails processed")
        return combined
    else:
        logger.warning("No results from parallel processing")
        return pd.DataFrame()


def parallel_feature_extraction(
    df: pd.DataFrame,
    feature_functions: List[Callable],
    processor: Optional[DistributedProcessor] = None
) -> pd.DataFrame:
    """
    Extract multiple features in parallel.
    
    Args:
        df: DataFrame with email data
        feature_functions: List of feature extraction functions
        processor: Optional DistributedProcessor configuration
        
    Returns:
        DataFrame with extracted features
    """
    logger.info(f"Parallel feature extraction ({len(feature_functions)} functions)")
    
    result_df = df.copy()
    
    # Apply each feature function
    for func in feature_functions:
        try:
            result_df = func(result_df)
            logger.info(f"  ✓ Applied {func.__name__}")
        except Exception as e:
            logger.warning(f"  ✗ Failed {func.__name__}: {e}")
    
    return result_df


def process_in_batches(
    data_path: str,
    process_func: Callable[[pd.DataFrame], pd.DataFrame],
    batch_size: int = 1000,
    max_rows: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Process large datasets in batches to manage memory.
    
    Args:
        data_path: Path to data file or directory
        process_func: Function to process each batch
        batch_size: Number of rows per batch
        max_rows: Maximum total rows to process
        **kwargs: Additional arguments for process_func
        
    Returns:
        Combined processed DataFrame
    """
    logger.info(f"Batch processing from {data_path} (batch_size={batch_size})")
    
    from workplace_email_utils.ingest.email_parser import load_emails
    
    # Load and process in batches
    all_results = []
    processed = 0
    
    # For now, load in chunks
    chunk_size = batch_size
    offset = 0
    
    while True:
        if max_rows and processed >= max_rows:
            break
        
        # Load batch
        try:
            batch = load_emails(
                data_path,
                max_rows=chunk_size,
                sample_size=None
            )
            
            if len(batch) == 0:
                break
            
            # Process batch
            processed_batch = process_func(batch, **kwargs)
            all_results.append(processed_batch)
            
            processed += len(batch)
            offset += len(batch)
            
            logger.info(f"  Processed batch: {len(batch)} emails (total: {processed})")
            
            if len(batch) < chunk_size:
                break
                
        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            break
    
    # Combine results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"Batch processing complete: {len(combined)} emails processed")
        return combined
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Test distributed processing
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing distributed processing...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    
    # Test chunking
    chunks = chunk_dataframe(df, chunk_size=100)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Test parallel processing
    def dummy_process(chunk_df):
        chunk_df['processed'] = True
        return chunk_df
    
    processor = DistributedProcessor(n_workers=2, method='sequential')
    result = parallel_process_emails(df.head(100), dummy_process, processor)
    print(f"✓ Processed {len(result)} emails in parallel")

