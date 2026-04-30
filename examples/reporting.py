"""
Reporting example.

Demonstrates how to generate comprehensive reports and export to multiple formats.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from workplace_email_utils.pipeline import build_knowledge_model
from workplace_email_utils.reporting.report_generator import generate_report, ReportConfig
from workplace_email_utils.reporting.exports import export_report
from workplace_email_utils.visualization.dashboards import create_email_analytics_dashboard, export_dashboard_data

def main():
    """Example of reporting and dashboard capabilities."""
    
    print("Building model for reporting...")
    model = build_knowledge_model(
        data_path='maildir',
        data_format='maildir',
        sample_size=5000,
        enable_classification=True
    )
    
    # Generate comprehensive report
    print("\nGenerating report...")
    config = ReportConfig(
        report_type='summary',
        sections=['overview', 'analytics', 'anomalies', 'metrics']
    )
    report = generate_report(model.df, config)
    
    print("\nReport Sections:")
    for section_name, section_data in report['sections'].items():
        print(f"  - {section_name}")
    
    # Export report
    print("\nExporting report...")
    export_report(report, 'model_output/email_report.json', format='json')
    export_report(report, 'model_output/email_report.html', format='html')
    print("  ✓ Exported to model_output/email_report.json")
    print("  ✓ Exported to model_output/email_report.html")
    
    # Create dashboard
    print("\nCreating dashboard...")
    dashboard = create_email_analytics_dashboard(model.df, title="Email Analytics Dashboard")
    print(f"  ✓ Created dashboard with {len(dashboard.widgets)} widgets")
    
    # Export dashboard
    dashboard_json = export_dashboard_data(dashboard, format='json')
    print(f"  ✓ Dashboard exported (JSON: {len(dashboard_json)} characters)")
    
    print("\nReporting complete!")

if __name__ == "__main__":
    main()

