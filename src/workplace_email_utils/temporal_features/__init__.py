"""
Temporal and time series analysis module.

Extracts temporal features, analyzes time-based patterns, and detects anomalies.
"""

from .extractors import extract_temporal_features, get_temporal_feature_matrix, TemporalFeatures
from .analysis import (
    compute_email_volume_trends,
    analyze_response_times,
    compute_communication_velocity,
    detect_temporal_anomalies
)
from .visualization import (
    plot_email_volume_trends,
    plot_hourly_distribution,
    plot_day_of_week_distribution,
    plot_response_time_distribution
)

__all__ = [
    'extract_temporal_features',
    'get_temporal_feature_matrix',
    'TemporalFeatures',
    'compute_email_volume_trends',
    'analyze_response_times',
    'compute_communication_velocity',
    'detect_temporal_anomalies',
    'plot_email_volume_trends',
    'plot_hourly_distribution',
    'plot_day_of_week_distribution',
    'plot_response_time_distribution',
]

