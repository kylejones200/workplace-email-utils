"""
Dashboard framework for email analytics.

Provides structured dashboard creation and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Dashboard:
    """Container for dashboard configuration."""
    title: str
    widgets: List[Dict] = field(default_factory=list)
    layout: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


def create_email_analytics_dashboard(
    df: pd.DataFrame,
    title: str = "Email Analytics Dashboard"
) -> Dashboard:
    """
    Create a comprehensive email analytics dashboard.
    
    Args:
        df: DataFrame with email data
        title: Dashboard title
        
    Returns:
        Dashboard object with configured widgets
    """
    logger.info(f"Creating email analytics dashboard: {title}")
    
    widgets = []
    
    # Email volume widget
    if 'date_parsed' in df.columns:
        widgets.append({
            'type': 'email_volume',
            'title': 'Email Volume Over Time',
            'data': _get_volume_data(df)
        })
    
    # Priority distribution widget
    if 'priority' in df.columns:
        widgets.append({
            'type': 'priority_distribution',
            'title': 'Priority Distribution',
            'data': df['priority'].value_counts().to_dict()
        })
    
    # Category distribution widget
    if 'category' in df.columns:
        widgets.append({
            'type': 'category_distribution',
            'title': 'Category Distribution',
            'data': df['category'].value_counts().to_dict()
        })
    
    # Sentiment widget
    if 'sentiment' in df.columns:
        widgets.append({
            'type': 'sentiment_distribution',
            'title': 'Sentiment Distribution',
            'data': df['sentiment'].value_counts().to_dict()
        })
    
    # Top senders widget
    if 'sender' in df.columns:
        top_senders = df['sender'].value_counts().head(10).to_dict()
        widgets.append({
            'type': 'top_senders',
            'title': 'Top 10 Senders',
            'data': top_senders
        })
    
    # Anomaly summary widget
    anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() or 'is_' in col.lower()]
    if anomaly_cols:
        anomaly_summary = {}
        for col in anomaly_cols:
            if df[col].dtype in [int, bool]:
                anomaly_summary[col] = int(df[col].sum())
        
        widgets.append({
            'type': 'anomaly_summary',
            'title': 'Anomaly Summary',
            'data': anomaly_summary
        })
    
    return Dashboard(
        title=title,
        widgets=widgets,
        layout={'columns': 2, 'rows': len(widgets) // 2 + 1},
        metadata={'total_emails': len(df)}
    )


def _get_volume_data(df: pd.DataFrame) -> Dict:
    """Get email volume data for time series widget."""
    if 'date_parsed' not in df.columns:
        return {}
    
    df_copy = df.copy()
    df_copy['date_parsed'] = pd.to_datetime(df_copy['date_parsed'], errors='coerce')
    df_copy = df_copy.dropna(subset=['date_parsed'])
    
    # Group by day
    df_copy['date'] = df_copy['date_parsed'].dt.date
    volume = df_copy.groupby('date').size()
    
    return {
        'dates': [str(d) for d in volume.index],
        'volumes': volume.values.tolist()
    }


def export_dashboard_data(dashboard: Dashboard, format: str = 'json') -> str:
    """
    Export dashboard data to various formats.
    
    Args:
        dashboard: Dashboard object
        format: Export format ('json', 'csv', 'html')
        
    Returns:
        Exported data as string
    """
    import json
    
    if format == 'json':
        dashboard_dict = {
            'title': dashboard.title,
            'widgets': dashboard.widgets,
            'metadata': dashboard.metadata
        }
        return json.dumps(dashboard_dict, indent=2, default=str)
    
    elif format == 'html':
        html = f"<html><head><title>{dashboard.title}</title></head><body>"
        html += f"<h1>{dashboard.title}</h1>"
        
        for widget in dashboard.widgets:
            html += f"<h2>{widget['title']}</h2>"
            if isinstance(widget['data'], dict):
                html += "<ul>"
                for key, value in widget['data'].items():
                    html += f"<li>{key}: {value}</li>"
                html += "</ul>"
        
        html += "</body></html>"
        return html
    
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    # Test dashboard creation
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing dashboard creation...")
    df = load_emails('maildir', data_format='maildir', max_rows=100)
    
    dashboard = create_email_analytics_dashboard(df)
    print(f"✓ Created dashboard: {dashboard.title}")
    print(f"  Widgets: {len(dashboard.widgets)}")
    
    # Export
    json_data = export_dashboard_data(dashboard, format='json')
    print(f"✓ Exported dashboard (JSON: {len(json_data)} chars)")

