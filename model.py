"""
Model Training Module for OncoPredict AI
Handles model training, evaluation, and persistence
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib
import os


class CancerRiskModel:
    """
    Cancer risk prediction model with multiple algorithms
    """
    
    def __init__(self):
        self.logistic_model = None
        self.rf_model = None
        self.feature_names = None
        self.metrics = {}
        
    def train_models(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Train both Logistic Regression and Random Forest models
        """
        
        self.feature_names = feature_names
        
        # Train Logistic Regression
        print("Training Logistic Regression model...")
        self.logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        self.logistic_model.fit(X_train, y_train)
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate both models
        self.metrics['logistic'] = self._evaluate_model(
            self.logistic_model, X_test, y_test, "Logistic Regression"
        )
        self.metrics['random_forest'] = self._evaluate_model(
            self.rf_model, X_test, y_test, "Random Forest"
        )
        
        print("\n=== Model Training Complete ===")
        print(f"Logistic Regression - Accuracy: {self.metrics['logistic']['accuracy']:.4f}")
        print(f"Random Forest - Accuracy: {self.metrics['random_forest']['accuracy']:.4f}")
        
        return self.metrics
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model and return metrics
        """
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        
        return metrics
    
    def predict(self, X, model_type='logistic'):
        """
        Make prediction using specified model
        
        Args:
            X: Input features
            model_type: 'logistic' or 'random_forest'
        
        Returns:
            Prediction probability (0-1)
        """
        
        model = self.logistic_model if model_type == 'logistic' else self.rf_model
        
        if model is None:
            raise ValueError("Model not trained yet!")
        
        # Return probability of cancer (class 1)
        return model.predict_proba(X)[0][1]
    
    def get_feature_importance(self, model_type='logistic'):
        """
        Get feature importance/coefficients
        
        Returns:
            DataFrame with feature names and importance scores
        """
        
        if model_type == 'logistic':
            importance = np.abs(self.logistic_model.coef_[0])
        else:
            importance = self.rf_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, directory='models'):
        """
        Save trained models to disk
        """
        
        os.makedirs(directory, exist_ok=True)
        
        joblib.dump(self.logistic_model, f'{directory}/logistic_model.pkl')
        joblib.dump(self.rf_model, f'{directory}/random_forest_model.pkl')
        joblib.dump(self.feature_names, f'{directory}/feature_names.pkl')
        joblib.dump(self.metrics, f'{directory}/metrics.pkl')
        
        print(f"Models saved to {directory}/")
    
    def load_models(self, directory='models'):
        """
        Load trained models from disk
        """
        
        self.logistic_model = joblib.load(f'{directory}/logistic_model.pkl')
        self.rf_model = joblib.load(f'{directory}/random_forest_model.pkl')
        self.feature_names = joblib.load(f'{directory}/feature_names.pkl')
        self.metrics = joblib.load(f'{directory}/metrics.pkl')
        
        print(f"Models loaded from {directory}/")
        
        return self


def train_and_save_models():
    """
    Complete pipeline: load data, train models, save to disk
    """
    
    from data_processing import load_and_prepare_data
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data()
    
    # Train models
    model = CancerRiskModel()
    model.train_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Save models
    model.save_models('models')
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ All models and preprocessors saved successfully!")
    
    return model, scaler


if __name__ == "__main__":
    # Train models when run directly
    train_and_save_models()
