"""
Real-time alerting system.

Detects and triggers alerts for anomalies, urgent emails, and other events.
"""

import pandas as pd
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Container for alert information."""
    alert_id: str
    alert_type: str  # 'anomaly', 'urgent', 'escalation', 'volume_spike'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    email_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AlertSystem:
    """Container for alert system configuration."""
    alert_callbacks: List[Callable] = field(default_factory=list)
    enabled_alert_types: List[str] = field(default_factory=lambda: ['anomaly', 'urgent'])
    severity_threshold: str = 'medium'  # Minimum severity to trigger


def create_alert(
    alert_type: str,
    message: str,
    severity: str = 'medium',
    email_id: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Alert:
    """
    Create an alert.
    
    Args:
        alert_type: Type of alert
        message: Alert message
        severity: Alert severity
        email_id: Optional email ID
        metadata: Optional metadata dictionary
        
    Returns:
        Alert object
    """
    alert_id = f"alert_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    return Alert(
        alert_id=alert_id,
        alert_type=alert_type,
        severity=severity,
        message=message,
        timestamp=datetime.now(),
        email_id=email_id,
        metadata=metadata or {}
    )


def check_alerts(
    email_df: pd.DataFrame,
    alert_system: Optional[AlertSystem] = None
) -> List[Alert]:
    """
    Check emails and generate alerts.
    
    Args:
        email_df: DataFrame with email data
        alert_system: Optional AlertSystem configuration
        
    Returns:
        List of Alert objects
    """
    if alert_system is None:
        alert_system = AlertSystem()
    
    alerts = []
    
    # Check for urgent emails
    if 'urgent' in alert_system.enabled_alert_types:
        urgent_emails = email_df[
            (email_df.get('urgency_score', 0) > 0.8) |
            (email_df.get('priority', '') == 'high')
        ]
        
        for _, email in urgent_emails.iterrows():
            alert = create_alert(
                alert_type='urgent',
                message=f"Urgent email from {email.get('sender', 'unknown')}",
                severity='high',
                email_id=str(email.get('doc_id', email.name)),
                metadata={'subject': email.get('subject', '')}
            )
            alerts.append(alert)
    
    # Check for anomalies
    if 'anomaly' in alert_system.enabled_alert_types:
        anomaly_emails = email_df[
            (email_df.get('is_temporal_anomaly', 0) == 1) |
            (email_df.get('is_content_anomaly', 0) == 1)
        ]
        
        for _, email in anomaly_emails.iterrows():
            alert = create_alert(
                alert_type='anomaly',
                message=f"Anomalous email detected",
                severity='medium',
                email_id=str(email.get('doc_id', email.name)),
                metadata={'anomaly_type': 'content'}
            )
            alerts.append(alert)
    
    # Check for escalation risks
    if 'escalation' in alert_system.enabled_alert_types:
        if 'escalation_risk' in email_df.columns:
            high_risk = email_df[email_df['escalation_risk'] > 0.7]
            
            for _, email in high_risk.iterrows():
                alert = create_alert(
                    alert_type='escalation',
                    message=f"High escalation risk email",
                    severity='high',
                    email_id=str(email.get('doc_id', email.name)),
                    metadata={'risk_score': float(email['escalation_risk'])}
                )
                alerts.append(alert)
    
    # Filter by severity threshold
    severity_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
    threshold_level = severity_levels.get(alert_system.severity_threshold, 1)
    
    filtered_alerts = [
        alert for alert in alerts
        if severity_levels.get(alert.severity, 0) >= threshold_level
    ]
    
    # Trigger callbacks
    for alert in filtered_alerts:
        for callback in alert_system.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    logger.info(f"Generated {len(filtered_alerts)} alerts from {len(email_df)} emails")
    return filtered_alerts


if __name__ == "__main__":
    # Test alert system
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing alert system...")
    df = load_emails('maildir', data_format='maildir', max_rows=100)
    
    alert_system = AlertSystem(
        enabled_alert_types=['urgent', 'anomaly'],
        severity_threshold='medium'
    )
    
    alerts = check_alerts(df, alert_system)
    print(f"✓ Generated {len(alerts)} alerts")
    for alert in alerts[:5]:
        print(f"  - {alert.alert_type}: {alert.message}")

