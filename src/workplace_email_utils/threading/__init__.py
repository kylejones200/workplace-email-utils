"""
Email threading and conversation analysis module.

Reconstructs email threads, analyzes conversation flow, and provides
thread-based metrics and insights.
"""

from .reconstruct import reconstruct_threads, ThreadTree
from .analysis import (
    compute_thread_metrics,
    analyze_conversation_flow,
    score_thread_importance,
    ThreadMetrics
)
from .visualization import plot_thread_tree, plot_thread_timeline

__all__ = [
    'reconstruct_threads',
    'ThreadTree',
    'compute_thread_metrics',
    'analyze_conversation_flow',
    'score_thread_importance',
    'ThreadMetrics',
    'plot_thread_tree',
    'plot_thread_timeline',
]

