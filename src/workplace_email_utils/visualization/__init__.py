"""
Visualization and dashboard module.

Provides dashboard creation, visualization tools, and export capabilities.
"""

from .dashboards import (
    create_email_analytics_dashboard,
    export_dashboard_data,
    Dashboard
)

__all__ = [
    'create_email_analytics_dashboard',
    'export_dashboard_data',
    'Dashboard',
]

