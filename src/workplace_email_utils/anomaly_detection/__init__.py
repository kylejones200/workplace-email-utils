"""
Anomaly detection module.

Detects unusual patterns in email communications, network structures, and temporal behaviors.
"""

from .communication_patterns import (
    detect_communication_anomalies,
    detect_unusual_executive_patterns,
    CommunicationAnomalyResult
)
from .content import (
    detect_content_anomalies,
    detect_unusual_language,
    detect_topic_anomalies,
    ContentAnomalyResult
)
from .network import (
    detect_network_anomalies,
    detect_structural_anomalies,
    detect_connectivity_anomalies,
    NetworkAnomalyResult
)
from .temporal import (
    detect_temporal_anomalies,
    detect_volume_spikes,
    detect_off_hours_anomalies,
    detect_response_time_anomalies,
    TemporalAnomalyResult
)

__all__ = [
    'detect_communication_anomalies',
    'detect_unusual_executive_patterns',
    'CommunicationAnomalyResult',
    'detect_content_anomalies',
    'detect_unusual_language',
    'detect_topic_anomalies',
    'ContentAnomalyResult',
    'detect_network_anomalies',
    'detect_structural_anomalies',
    'detect_connectivity_anomalies',
    'NetworkAnomalyResult',
    'detect_temporal_anomalies',
    'detect_volume_spikes',
    'detect_off_hours_anomalies',
    'detect_response_time_anomalies',
    'TemporalAnomalyResult',
]

