"""
Temporal visualization functions.

Creates visualizations for temporal patterns and trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_email_volume_trends(volume_df: pd.DataFrame,
                             figsize: tuple = (12, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot email volume trends over time.
    
    Args:
        volume_df: DataFrame from compute_email_volume_trends
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'date' not in volume_df.columns or len(volume_df) == 0:
        logger.warning("No data to plot")
        return fig
    
    ax.plot(volume_df['date'], volume_df['count'], label='Email Volume', alpha=0.7)
    
    if 'rolling_mean' in volume_df.columns:
        ax.plot(volume_df['date'], volume_df['rolling_mean'], 
                label='Rolling Mean (7 days)', linewidth=2, color='orange')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Emails')
    ax.set_title('Email Volume Trends Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_hourly_distribution(df: pd.DataFrame,
                            hour_col: str = 'hour',
                            figsize: tuple = (10, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of emails by hour of day.
    
    Args:
        df: DataFrame with temporal features
        hour_col: Name of hour column
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if hour_col not in df.columns:
        logger.warning(f"Column '{hour_col}' not found")
        return plt.figure()
    
    valid_hours = df[df[hour_col].notna()][hour_col].astype(int)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(valid_hours, bins=24, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Emails')
    ax.set_title('Email Distribution by Hour of Day')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_day_of_week_distribution(df: pd.DataFrame,
                                  day_col: str = 'day_of_week',
                                  figsize: tuple = (10, 6),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of emails by day of week.
    
    Args:
        df: DataFrame with temporal features
        day_col: Name of day_of_week column
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if day_col not in df.columns:
        logger.warning(f"Column '{day_col}' not found")
        return plt.figure()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    valid_days = df[df[day_col].notna()][day_col].astype(int)
    day_counts = valid_days.value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(7), [day_counts.get(i, 0) for i in range(7)], 
           edgecolor='black', alpha=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names, rotation=45, ha='right')
    ax.set_ylabel('Number of Emails')
    ax.set_title('Email Distribution by Day of Week')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_response_time_distribution(response_df: pd.DataFrame,
                                   figsize: tuple = (10, 6),
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of response times.
    
    Args:
        response_df: DataFrame from analyze_response_times
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if len(response_df) == 0 or 'response_time_hours' not in response_df.columns:
        logger.warning("No response time data to plot")
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    response_times = response_df['response_time_hours']
    
    # Use log scale for better visualization
    ax.hist(response_times[response_times > 0], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Response Time (hours)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Email Response Times')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualizations
    from workplace_email_utils.ingest.email_parser import load_emails
    from extractors import extract_temporal_features
    from analysis import compute_email_volume_trends, analyze_response_times
    
    print("Testing temporal visualizations...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    df = extract_temporal_features(df)
    
    # Volume trends
    volume = compute_email_volume_trends(df)
    if len(volume) > 0:
        plot_email_volume_trends(volume, save_path='temporal_volume_trends.png')
    
    # Hourly distribution
    plot_hourly_distribution(df, save_path='temporal_hourly_dist.png')
    
    # Day of week distribution
    plot_day_of_week_distribution(df, save_path='temporal_day_dist.png')
    
    print("Visualizations created!")

