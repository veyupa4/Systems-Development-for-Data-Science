"""
model.py
Machine Learning model training and evaluation for CICIoT2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

def prepare_features(df, label_col='label', drop_cols=None):
    """Prepare feature matrix X and target y."""
    if drop_cols is None:
        drop_cols = []
    
    # Encode label
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    
    # Drop non-feature columns
    exclude = [label_col] + drop_cols + ['label_encoded']
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    
    print(f"✅ Features: {X.shape[1]} | Samples: {X.shape[0]} | Classes: {len(le.classes_)}")
    return X, y, le

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=300, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'model': model, 'accuracy': acc, 'f1': f1, 'y_pred': y_pred}
        print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    return results

def plot_confusion_matrix(y_test, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=13)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importances from Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(11, 6))
    plt.bar(range(top_n), importances[indices], color='teal')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=13)
    plt.ylabel('Importance Score')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_model_comparison(results, save_path=None):
    """Bar chart comparing model accuracy and F1."""
    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    f1s = [results[n]['f1'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
    ax.bar(x + width/2, f1s, width, label='F1 Score', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title('Model Comparison', fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()