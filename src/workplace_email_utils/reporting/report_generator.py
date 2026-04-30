"""
Report generation system.

Generates comprehensive reports from email analytics.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    report_type: str = 'summary'  # 'summary', 'detailed', 'executive'
    sections: List[str] = field(default_factory=lambda: ['overview', 'analytics', 'anomalies'])
    include_charts: bool = True
    date_range: Optional[Dict] = None


@dataclass
class ReportGenerator:
    """Report generator with configuration."""
    config: ReportConfig = None


def generate_report(
    df: pd.DataFrame,
    config: Optional[ReportConfig] = None,
    title: str = "Email Analytics Report"
) -> Dict[str, Any]:
    """
    Generate comprehensive email analytics report.
    
    Args:
        df: DataFrame with email data
        config: Optional ReportConfig
        title: Report title
        
    Returns:
        Dictionary with report data
    """
    if config is None:
        config = ReportConfig()
    
    logger.info(f"Generating {config.report_type} report: {title}")
    
    report = {
        'title': title,
        'generated_at': datetime.now().isoformat(),
        'sections': {}
    }
    
    # Overview section
    if 'overview' in config.sections:
        report['sections']['overview'] = {
            'total_emails': len(df),
            'date_range': _get_date_range(df),
            'unique_senders': df['sender'].nunique() if 'sender' in df.columns else 0,
            'unique_recipients': _count_unique_recipients(df),
        }
    
    # Analytics section
    if 'analytics' in config.sections:
        report['sections']['analytics'] = {
            'priority_distribution': df['priority'].value_counts().to_dict() if 'priority' in df.columns else {},
            'category_distribution': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            'sentiment_distribution': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
            'top_senders': df['sender'].value_counts().head(10).to_dict() if 'sender' in df.columns else {},
        }
    
    # Anomalies section
    if 'anomalies' in config.sections:
        anomaly_summary = {}
        anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower()]
        for col in anomaly_cols:
            if df[col].dtype in [int, bool]:
                anomaly_summary[col] = int(df[col].sum())
        
        report['sections']['anomalies'] = anomaly_summary
    
    # Metrics section
    if 'metrics' in config.sections:
        report['sections']['metrics'] = {
            'avg_response_time': df['response_time'].mean() if 'response_time' in df.columns else None,
            'avg_urgency_score': df['urgency_score'].mean() if 'urgency_score' in df.columns else None,
            'escalation_rate': (df['escalation_risk'] > 0.5).mean() if 'escalation_risk' in df.columns else None,
        }
    
    logger.info(f"Report generated: {len(report['sections'])} sections")
    return report


def _get_date_range(df: pd.DataFrame) -> Dict:
    """Get date range from DataFrame."""
    if 'date_parsed' not in df.columns:
        return {}
    
    dates = pd.to_datetime(df['date_parsed'], errors='coerce').dropna()
    if len(dates) == 0:
        return {}
    
    return {
        'start': str(dates.min().date()),
        'end': str(dates.max().date()),
        'days': (dates.max() - dates.min()).days
    }


def _count_unique_recipients(df: pd.DataFrame) -> int:
    """Count unique recipients."""
    if 'recipients' not in df.columns:
        return 0
    
    all_recipients = set()
    for recipients in df['recipients'].dropna():
        if isinstance(recipients, list):
            all_recipients.update(recipients)
        elif isinstance(recipients, str):
            all_recipients.update([r.strip() for r in recipients.split(',')])
    
    return len(all_recipients)


if __name__ == "__main__":
    # Test report generation
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing report generation...")
    df = load_emails('maildir', data_format='maildir', max_rows=100)
    
    config = ReportConfig(
        report_type='summary',
        sections=['overview', 'analytics', 'anomalies']
    )
    
    report = generate_report(df, config)
    print(f"✓ Generated report: {report['title']}")
    print(f"  Sections: {list(report['sections'].keys())}")

