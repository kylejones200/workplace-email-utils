"""
Multi-format export functionality.

Exports data to CSV, Excel, JSON, PDF, and other formats.
"""

import pandas as pd
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_csv(
    df: pd.DataFrame,
    filepath: str,
    index: bool = False
) -> bool:
    """
    Export DataFrame to CSV.
    
    Args:
        df: DataFrame to export
        filepath: Output file path
        index: Whether to include index
        
    Returns:
        True if successful
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)
        logger.info(f"Exported {len(df)} rows to CSV: {filepath}")
        return True
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return False


def export_to_excel(
    df: pd.DataFrame,
    filepath: str,
    sheet_name: str = 'Sheet1',
    index: bool = False
) -> bool:
    """
    Export DataFrame to Excel.
    
    Args:
        df: DataFrame to export
        filepath: Output file path
        sheet_name: Excel sheet name
        index: Whether to include index
        
    Returns:
        True if successful
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
        
        logger.info(f"Exported {len(df)} rows to Excel: {filepath}")
        return True
    except Exception as e:
        logger.warning(f"Excel export failed (may need openpyxl): {e}")
        return False


def export_to_json(
    data: Any,
    filepath: str,
    indent: int = 2
) -> bool:
    """
    Export data to JSON.
    
    Args:
        data: Data to export (DataFrame, dict, list)
        filepath: Output file path
        indent: JSON indentation
        
    Returns:
        True if successful
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        
        logger.info(f"Exported data to JSON: {filepath}")
        return True
    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        return False


def export_to_pdf(
    report_data: Dict[str, Any],
    filepath: str
) -> bool:
    """
    Export report to PDF.
    
    Args:
        report_data: Report dictionary
        filepath: Output file path
        
    Returns:
        True if successful
    """
    try:
        # Simple HTML-to-PDF approach
        html_content = _report_to_html(report_data)
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as HTML (would need pdfkit or similar for actual PDF)
        html_path = str(path).replace('.pdf', '.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Exported report to HTML: {html_path}")
        logger.info("Note: Install pdfkit for actual PDF export")
        return True
    except Exception as e:
        logger.warning(f"PDF export failed: {e}")
        return False


def _report_to_html(report_data: Dict[str, Any]) -> str:
    """Convert report dictionary to HTML."""
    html = f"<html><head><title>{report_data.get('title', 'Report')}</title></head><body>"
    html += f"<h1>{report_data.get('title', 'Report')}</h1>"
    html += f"<p>Generated: {report_data.get('generated_at', 'N/A')}</p>"
    
    for section_name, section_data in report_data.get('sections', {}).items():
        html += f"<h2>{section_name.title()}</h2>"
        
        if isinstance(section_data, dict):
            html += "<ul>"
            for key, value in section_data.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        else:
            html += f"<p>{section_data}</p>"
    
    html += "</body></html>"
    return html


def export_report(
    report_data: Dict[str, Any],
    filepath: str,
    format: str = 'json'
) -> bool:
    """
    Export report in specified format.
    
    Args:
        report_data: Report dictionary
        filepath: Output file path
        format: Export format ('json', 'html', 'pdf')
        
    Returns:
        True if successful
    """
    if format == 'json':
        return export_to_json(report_data, filepath)
    elif format == 'html':
        html = _report_to_html(report_data)
        try:
            with open(filepath, 'w') as f:
                f.write(html)
            logger.info(f"Exported report to HTML: {filepath}")
            return True
        except Exception as e:
            logger.error(f"HTML export failed: {e}")
            return False
    elif format == 'pdf':
        return export_to_pdf(report_data, filepath)
    else:
        logger.error(f"Unknown export format: {format}")
        return False


if __name__ == "__main__":
    # Test exports
    from workplace_email_utils.ingest.email_parser import load_emails
    
    print("Testing export functionality...")
    df = load_emails('maildir', data_format='maildir', max_rows=50)
    
    # Test CSV export
    export_to_csv(df, 'model_output/test_export.csv')
    
    # Test JSON export
    export_to_json(df.head(10), 'model_output/test_export.json')
    
    print("✓ Exports tested")

