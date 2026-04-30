"""
Real-time and streaming processing module.

Provides streaming email ingestion, real-time classification, and alerting.
"""

from .ingestion import (
    stream_emails,
    EmailStream,
    StreamProcessor
)
from .processing import (
    process_email_stream,
    real_time_classify,
    RealTimeProcessor
)
from .alerts import (
    AlertSystem,
    create_alert,
    check_alerts
)

__all__ = [
    'stream_emails',
    'EmailStream',
    'StreamProcessor',
    'process_email_stream',
    'real_time_classify',
    'RealTimeProcessor',
    'AlertSystem',
    'create_alert',
    'check_alerts',
]

