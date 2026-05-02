"""
eda.py
Exploratory Data Analysis functions for CICIoT2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_summary(df):
    """Print dataset info and basic stats."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    print(f"\nNull values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")
    print(f"\nBasic Stats:")
    print(df.describe())

def plot_label_distribution(df, label_col='label', save_path=None):
    """Bar chart of attack type distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df[label_col].value_counts()
    
    # Bar chart
    counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Attack Type Distribution (Count)', fontsize=13)
    axes[0].set_xlabel('Attack Type')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Attack Type Distribution (%)', fontsize=13)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_correlation_heatmap(df, top_n=15, save_path=None):
    """Correlation heatmap of top N numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Pick top_n features by variance
    top_features = numeric_df.var().nlargest(top_n).index
    corr = numeric_df[top_features].corr()
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, square=True)
    plt.title(f'Correlation Heatmap (Top {top_n} Features)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_feature_distributions(df, features, label_col='label', save_path=None):
    """Box plots of key features split by attack category."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, feat in zip(axes, features):
        top_labels = df[label_col].value_counts().head(6).index
        subset = df[df[label_col].isin(top_labels)]
        subset.boxplot(column=feat, by=label_col, ax=ax)
        ax.set_title(feat)
        ax.set_xlabel('')
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    plt.suptitle('Feature Distribution by Attack Type')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_protocol_usage(df, save_path=None):
    """Show protocol usage (TCP, UDP, ICMP, HTTP, etc.)."""
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'ARP']
    available = [p for p in protocols if p in df.columns]
    
    proto_sums = df[available].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(9, 5))
    proto_sums.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Protocol Usage in IoT Traffic', fontsize=13)
    plt.xlabel('Protocol')
    plt.ylabel('Total Occurrences')
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()