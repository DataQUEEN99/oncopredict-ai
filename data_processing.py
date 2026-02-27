"""
Data Processing Module for OncoPredict AI
Handles data loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_and_prepare_data():
    """
    Load breast cancer dataset and add synthetic lifestyle/genetic features
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    
    # Load UCI Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0 = malignant (cancer), 1 = benign (no cancer)
    
    # Invert target to make 1 = cancer risk
    y = 1 - y
    
    # Create synthetic dataset with medical features
    np.random.seed(42)
    n_samples = len(X)
    
    # Create synthetic medical features
    synthetic_features = pd.DataFrame({
        'age': np.random.randint(20, 81, n_samples),
        'bmi': np.random.normal(27, 5, n_samples).clip(15, 45),
        'smoking': np.random.binomial(1, 0.25, n_samples),
        'alcohol': np.random.binomial(1, 0.35, n_samples),
        'physical_activity': np.random.randint(0, 11, n_samples),
        'family_history': np.random.binomial(1, 0.15, n_samples),
        'brca1_mutation': np.random.binomial(1, 0.05, n_samples),
        'tp53_mutation': np.random.binomial(1, 0.08, n_samples),
        'tumor_marker': np.random.gamma(2, 2, n_samples).clip(0, 50)
    })
    
    # Use only the most important original features to avoid high dimensionality
    important_original_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'worst radius', 'worst texture', 'worst perimeter',
        'worst area', 'worst concave points'
    ]
    
    X_reduced = X[important_original_features]
    
    # Combine with synthetic features
    X_combined = pd.concat([X_reduced, synthetic_features], axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_combined.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_combined.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_combined.columns.tolist(), scaler


def create_input_features(user_inputs, feature_names, scaler):
    """
    Convert user inputs to scaled feature array
    
    Args:
        user_inputs: dict with user input values
        feature_names: list of all feature names
        scaler: fitted StandardScaler object
    
    Returns:
        Scaled feature array ready for prediction
    """
    
    # Create a DataFrame with all features
    # Set original features to median values (they're not user-provided)
    input_df = pd.DataFrame(columns=feature_names)
    
    # Use median values for original breast cancer features
    # These will be the median of the training data
    median_values = {
        'mean radius': 13.37,
        'mean texture': 18.84,
        'mean perimeter': 86.24,
        'mean area': 551.1,
        'mean smoothness': 0.096,
        'worst radius': 16.27,
        'worst texture': 25.68,
        'worst perimeter': 107.26,
        'worst area': 880.6,
        'worst concave points': 0.114
    }
    
    # Create input row
    input_row = {}
    for feature in feature_names:
        if feature in user_inputs:
            input_row[feature] = user_inputs[feature]
        elif feature in median_values:
            input_row[feature] = median_values[feature]
        else:
            input_row[feature] = 0
    
    input_df = pd.DataFrame([input_row])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    return input_scaled


def get_feature_description():
    """
    Return descriptions for user-input features
    """
    return {
        'age': 'Patient age in years (20-80)',
        'bmi': 'Body Mass Index (15-45)',
        'smoking': 'Current smoker (0=No, 1=Yes)',
        'alcohol': 'Regular alcohol consumption (0=No, 1=Yes)',
        'physical_activity': 'Weekly physical activity hours (0-10)',
        'family_history': 'Family history of cancer (0=No, 1=Yes)',
        'brca1_mutation': 'BRCA1 gene mutation (0=No, 1=Yes)',
        'tp53_mutation': 'TP53 gene mutation (0=No, 1=Yes)',
        'tumor_marker': 'Tumor marker level (0-50 ng/mL)'
    }
