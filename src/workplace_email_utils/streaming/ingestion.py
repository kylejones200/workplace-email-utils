"""
Streaming email ingestion.

Handles real-time email ingestion from various sources.
"""

import pandas as pd
from typing import Iterator, Optional, Callable, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmailStream:
    """Container for email stream configuration."""
    source: str  # 'file', 'directory', 'api'
    source_path: Optional[str] = None
    batch_size: int = 10
    poll_interval: float = 1.0  # seconds


@dataclass
class StreamProcessor:
    """Container for stream processing configuration."""
    process_func: Optional[Callable] = None
    batch_mode: bool = True
    max_queue_size: int = 1000


def stream_emails(
    stream: EmailStream,
    processor: Optional[StreamProcessor] = None,
    max_emails: Optional[int] = None
) -> Iterator[pd.DataFrame]:
    """
    Stream emails from a source.
    
    Args:
        stream: EmailStream configuration
        processor: Optional StreamProcessor
        max_emails: Maximum number of emails to stream
        
    Yields:
        DataFrame batches of emails
    """
    logger.info(f"Starting email stream from {stream.source}")
    
    if stream.source == 'directory':
        yield from _stream_from_directory(stream, processor, max_emails)
    elif stream.source == 'file':
        yield from _stream_from_file(stream, processor, max_emails)
    else:
        logger.warning(f"Unknown stream source: {stream.source}")
        yield pd.DataFrame()


def _stream_from_directory(
    stream: EmailStream,
    processor: Optional[StreamProcessor],
    max_emails: Optional[int]
) -> Iterator[pd.DataFrame]:
    """Stream emails from directory (e.g., maildir)."""
    from workplace_email_utils.ingest.email_parser import load_enron_maildir
    
    source_path = Path(stream.source_path) if stream.source_path else Path('.')
    
    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        return
    
    # For simplicity, stream in batches
    offset = 0
    batch_size = stream.batch_size
    total_processed = 0
    
    while True:
        if max_emails and total_processed >= max_emails:
            break
        
        # Load batch
        try:
            batch = load_enron_maildir(
                str(source_path),
                max_emails=batch_size,
                sample_size=None
            )
            
            if len(batch) == 0:
                time.sleep(stream.poll_interval)
                continue
            
            total_processed += len(batch)
            yield batch
            
            offset += len(batch)
            
        except Exception as e:
            logger.error(f"Error streaming from directory: {e}")
            break


def _stream_from_file(
    stream: EmailStream,
    processor: Optional[StreamProcessor],
    max_emails: Optional[int]
) -> Iterator[pd.DataFrame]:
    """Stream emails from file (e.g., CSV)."""
    from workplace_email_utils.ingest.email_parser import load_enron_csv
    
    source_path = Path(stream.source_path) if stream.source_path else None
    
    if not source_path or not source_path.exists():
        logger.error(f"Source file does not exist: {source_path}")
        return
    
    # Stream file in chunks
    chunk_size = stream.batch_size
    
    try:
        # Read file in chunks
        for chunk in pd.read_csv(source_path, chunksize=chunk_size):
            if max_emails and len(chunk) > max_emails:
                chunk = chunk.head(max_emails)
            
            yield chunk
            
            if max_emails:
                break
                
    except Exception as e:
        logger.error(f"Error streaming from file: {e}")


if __name__ == "__main__":
    # Test streaming
    stream = EmailStream(source='directory', source_path='maildir', batch_size=10)
    
    print("Testing email streaming...")
    count = 0
    for batch in stream_emails(stream, max_emails=50):
        count += len(batch)
        print(f"  Streamed batch: {len(batch)} emails (total: {count})")
        if count >= 50:
            break
    
    print(f"✓ Streamed {count} emails total")

