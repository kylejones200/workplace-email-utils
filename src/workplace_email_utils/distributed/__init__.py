"""
Distributed computing and scalability module.

Provides parallel processing, distributed indexing, and scalable processing frameworks.
"""

from .processing import (
    parallel_process_emails,
    chunk_dataframe,
    DistributedProcessor
)
from .indexing import (
    incremental_index_update,
    build_distributed_index,
    DistributedIndex
)

__all__ = [
    'parallel_process_emails',
    'chunk_dataframe',
    'DistributedProcessor',
    'incremental_index_update',
    'build_distributed_index',
    'DistributedIndex',
]

