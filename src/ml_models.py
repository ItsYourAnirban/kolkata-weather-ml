#!/usr/bin/env python3
"""
Machine Learning Models Module

Implements multiple ML models for weather prediction:
- Linear Regression for continuous predictions
- Random Forest for both regression and classification
- Support Vector Machine for complex pattern recognition
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, precision_score, recall_score, f1_score,
                            classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

class WeatherPredictionModels:
    """Comprehensive ML models for weather prediction tasks."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def initialize_models(self, problem_type='regression'):
        """Initialize ML models based on problem type."""
        print(f"🔧 Initializing {problem_type} models...")
        
        if problem_type == 'regression':
            self.models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Lasso Regression': Lasso(alpha=0.1, random_state=42),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    n_jobs=-1
                ),
                'Support Vector Regression': SVR(
                    kernel='rbf', 
                    C=1.0, 
                    epsilon=0.1
                )
            }
        else:  # classification
            self.models = {
                'Random Forest Classifier': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'Support Vector Classifier': SVC(
                    kernel='rbf',
                    C=1.0,
                    random_state=42,
                    probability=True
                )
            }
        
        print(f"✅ Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train, X_val, y_val, feature_names):
        """Train all models and evaluate on validation set."""
        print("🚀 Training models...")
        print("=" * 50)
        
        self.feature_names = feature_names
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'y_train_pred': y_train_pred,
                    'y_val_pred': y_val_pred,
                    'trained': True
                }
                
                # Calculate and display metrics
                if hasattr(model, 'predict_proba'):  # Classification
                    self._calculate_classification_metrics(name, y_train, y_val, y_train_pred, y_val_pred)
                else:  # Regression
                    self._calculate_regression_metrics(name, y_train, y_val, y_train_pred, y_val_pred)
                
                print(f"✅ {name} training completed")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                self.results[name] = {'trained': False, 'error': str(e)}
    
    def _calculate_regression_metrics(self, model_name, y_train, y_val, y_train_pred, y_val_pred):
        """Calculate regression metrics."""
        # Training metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Validation metrics
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Store metrics
        self.results[model_name].update({
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'val_rmse': np.sqrt(val_mse)
        })
        
        # Display metrics
        print(f"   Training   - MSE: {train_mse:.3f}, MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
        print(f"   Validation - MSE: {val_mse:.3f}, MAE: {val_mae:.3f}, R²: {val_r2:.3f}")
    
    def _calculate_classification_metrics(self, model_name, y_train, y_val, y_train_pred, y_val_pred):
        """Calculate classification metrics."""
        # Training metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        # Validation metrics
        val_acc = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        
        # Store metrics
        self.results[model_name].update({
            'train_accuracy': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })
        
        # Display metrics
        print(f"   Training   - Acc: {train_acc:.3f}, Precision: {train_precision:.3f}, Recall: {train_recall:.3f}, F1: {train_f1:.3f}")
        print(f"   Validation - Acc: {val_acc:.3f}, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, F1: {val_f1:.3f}")
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from tree-based models."""
        importance_data = {}
        
        for name, result in self.results.items():
            if result.get('trained', False):
                model = result['model']
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(top_n)
                    
                    importance_data[name] = importance_df
        
        return importance_data
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Evaluate all trained models on test set."""
        print("\n🧪 Evaluating models on test set...")
        print("=" * 40)
        
        test_results = {}
        
        for name, result in self.results.items():
            if result.get('trained', False):
                model = result['model']
                y_test_pred = model.predict(X_test)
                
                if hasattr(model, 'predict_proba'):  # Classification
                    test_acc = accuracy_score(y_test, y_test_pred)
                    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                    
                    test_results[name] = {
                        'accuracy': test_acc,
                        'f1_score': test_f1,
                        'predictions': y_test_pred
                    }
                    
                    print(f"🎯 {name}: Accuracy = {test_acc:.3f}, F1 = {test_f1:.3f}")
                    
                else:  # Regression
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    test_results[name] = {
                        'mse': test_mse,
                        'mae': test_mae,
                        'r2': test_r2,
                        'rmse': np.sqrt(test_mse),
                        'predictions': y_test_pred
                    }
                    
                    print(f"🎯 {name}: RMSE = {np.sqrt(test_mse):.3f}, MAE = {test_mae:.3f}, R² = {test_r2:.3f}")
        
        return test_results
    
    def get_best_model(self, metric='val_r2'):
        """Get the best performing model based on specified metric."""
        best_model = None
        best_score = -np.inf if 'r2' in metric or 'accuracy' in metric else np.inf
        best_name = None
        
        for name, result in self.results.items():
            if result.get('trained', False) and metric in result:
                score = result[metric]
                
                if 'r2' in metric or 'accuracy' in metric or 'f1' in metric:
                    if score > best_score:
                        best_score = score
                        best_model = result['model']
                        best_name = name
                else:  # For error metrics (lower is better)
                    if score < best_score:
                        best_score = score
                        best_model = result['model']
                        best_name = name
        
        return best_model, best_name, best_score
    
    def save_models(self, save_dir="models"):
        """Save trained models to disk."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        saved_models = []
        
        for name, result in self.results.items():
            if result.get('trained', False):
                model = result['model']
                filename = f"{save_dir}/{name.replace(' ', '_').lower()}_model.pkl"
                
                try:
                    joblib.dump(model, filename)
                    saved_models.append(filename)
                    print(f"✅ Saved {name} to {filename}")
                except Exception as e:
                    print(f"❌ Error saving {name}: {e}")
        
        return saved_models
    
    def load_model(self, filename):
        """Load a saved model from disk."""
        try:
            model = joblib.load(filename)
            print(f"✅ Loaded model from {filename}")
            return model
        except Exception as e:
            print(f"❌ Error loading model from {filename}: {e}")
            return None
    
    def predict_future(self, model, X_input, feature_names=None):
        """Make predictions using a trained model."""
        if hasattr(model, 'predict'):
            predictions = model.predict(X_input)
            
            # If it's a classifier with probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_input)
                return predictions, probabilities
            
            return predictions
        else:
            print("❌ Model doesn't have predict method")
            return None


def main():
    """Test the ML models module."""
    print("🧪 Testing Weather Prediction ML Models")
    print("=" * 50)
    
    # Import preprocessing to get data
    from data_preprocessing import WeatherDataPreprocessor
    
    # Test regression models (temperature prediction)
    print("\n🌡️  Testing Regression Models (Temperature Prediction):")
    preprocessor = WeatherDataPreprocessor()
    temp_data = preprocessor.preprocess_pipeline(prediction_type='temperature', days_ahead=1)
    
    if temp_data:
        # Initialize and train regression models
        regression_models = WeatherPredictionModels()
        regression_models.initialize_models(problem_type='regression')
        
        regression_models.train_models(
            temp_data['X_train'], temp_data['y_train'],
            temp_data['X_val'], temp_data['y_val'],
            temp_data['feature_names']
        )
        
        # Get best model
        best_model, best_name, best_score = regression_models.get_best_model('val_r2')
        print(f"\n🏆 Best regression model: {best_name} (R² = {best_score:.3f})")
        
        # Test on test set
        test_results = regression_models.evaluate_on_test_set(
            temp_data['X_test'], temp_data['y_test']
        )
    
    # Test classification models (rainfall prediction)
    print("\n\n🌧️  Testing Classification Models (Rainfall Prediction):")
    rain_data = preprocessor.preprocess_pipeline(prediction_type='rainfall', days_ahead=1)
    
    if rain_data:
        # Initialize and train classification models
        classification_models = WeatherPredictionModels()
        classification_models.initialize_models(problem_type='classification')
        
        classification_models.train_models(
            rain_data['X_train'], rain_data['y_train'],
            rain_data['X_val'], rain_data['y_val'],
            rain_data['feature_names']
        )
        
        # Get best model
        best_model, best_name, best_score = classification_models.get_best_model('val_accuracy')
        print(f"\n🏆 Best classification model: {best_name} (Accuracy = {best_score:.3f})")
        
        # Test on test set
        test_results = classification_models.evaluate_on_test_set(
            rain_data['X_test'], rain_data['y_test']
        )
    
    print("\n✅ ML Models testing completed!")


if __name__ == "__main__":
    main()
