"""
data_loader.py
Handles loading and initial cleaning of CICIoT2023 dataset
"""

import pandas as pd
import numpy as np
import os

def load_dataset(filepath):
    """Load a CSV file from CICIoT2023 dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded: {filepath}")
    print(f"   Shape: {df.shape}")
    return df

def load_multiple_files(folder_path, max_files=5):
    """Load and combine multiple CSV files from the dataset folder."""
    all_dfs = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for i, file in enumerate(files[:max_files]):
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        all_dfs.append(df)
        print(f"Loaded {file} — shape: {df.shape}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✅ Combined dataset shape: {combined.shape}")
    return combined

def clean_data(df):
    """Basic cleaning: drop duplicates, handle nulls, fix types."""
    initial_shape = df.shape
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop columns with >50% nulls
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    
    # Fill remaining nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    print(f"✅ Cleaned: {initial_shape} → {df.shape}")
    return df

def encode_labels(df, label_col='label'):
    """Encode string labels to integers."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df[label_col])
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"✅ Labels encoded: {label_map}")
    return df, label_map