#!/usr/bin/env python3
"""
Visualization and Model Evaluation Module

Creates comprehensive visualizations and analysis for weather prediction models:
- Performance comparison charts
- Feature importance plots
- Prediction accuracy visualizations
- Time series analysis
- Model evaluation reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeatherVisualization:
    """Comprehensive visualization tools for weather prediction analysis."""
    
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_model_performance_comparison(self, models_results, metric='r2', problem_type='regression'):
        """Create bar plot comparing model performance."""
        print("📊 Creating model performance comparison...")
        
        model_names = []
        scores = []
        
        for name, result in models_results.results.items():
            if result.get('trained', False):
                if problem_type == 'regression':
                    metric_key = f'val_{metric}'
                else:
                    metric_key = f'val_{metric}'
                
                if metric_key in result:
                    model_names.append(name)
                    scores.append(result[metric_key])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(model_names, scores, color=sns.color_palette("husl", len(model_names)))
        
        # Customize plot
        ax.set_title(f'Model Performance Comparison - {metric.upper()}', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        filename = f"{self.save_dir}/model_comparison_{metric}_{problem_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {filename}")
        
        plt.show()
        return filename
    
    def plot_feature_importance(self, importance_data, top_n=15):
        """Plot feature importance for tree-based models."""
        print("🔍 Creating feature importance plots...")
        
        n_models = len(importance_data)
        if n_models == 0:
            print("⚠️  No feature importance data available")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance_df) in enumerate(importance_data.items()):
            ax = axes[idx]
            
            # Get top features
            top_features = importance_df.head(top_n)
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{model_name}\nTop {top_n} Features', fontweight='bold')
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{importance:.3f}', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{self.save_dir}/feature_importance.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {filename}")
        
        plt.show()
        return filename
    
    def plot_prediction_accuracy(self, y_true, y_pred, model_name, problem_type='regression'):
        """Plot prediction accuracy visualization."""
        print(f"🎯 Creating prediction accuracy plot for {model_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if problem_type == 'regression':
            # Actual vs Predicted scatter plot
            axes[0].scatter(y_true, y_pred, alpha=0.6, color='blue')
            axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title(f'{model_name}: Actual vs Predicted')
            
            # Calculate R² for annotation
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residuals plot
            residuals = y_true - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6, color='green')
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title(f'{model_name}: Residual Plot')
            
        else:  # classification
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            axes[0].set_title(f'{model_name}: Confusion Matrix')
            
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            axes[1].bar(metrics.keys(), metrics.values(), color=sns.color_palette("husl", len(metrics)))
            axes[1].set_ylabel('Score')
            axes[1].set_title(f'{model_name}: Classification Metrics')
            axes[1].set_ylim([0, 1])
            
            # Add value labels
            for i, (metric, value) in enumerate(metrics.items()):
                axes[1].text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{self.save_dir}/prediction_accuracy_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {filename}")
        
        plt.show()
        return filename
    
    def plot_weather_data_analysis(self, data, save_plots=True):
        """Create comprehensive weather data analysis plots."""
        print("🌤️  Creating weather data analysis...")
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Temperature vs Humidity scatter plot
        scatter = axes[0, 0].scatter(data['temperature'], data['humidity'], 
                                   c=data['rainfall'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Humidity (%)')
        axes[0, 0].set_title('Temperature vs Humidity (colored by Rainfall)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Rainfall (mm)')
        
        # 2. Seasonal temperature distribution
        seasonal_data = []
        for season in data['season'].unique():
            seasonal_data.append(data[data['season'] == season]['temperature'])
        
        axes[0, 1].boxplot(seasonal_data, labels=data['season'].unique())
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].set_title('Temperature Distribution by Season')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Monthly rainfall pattern
        monthly_rainfall = data.groupby('month')['rainfall'].mean()
        axes[1, 0].bar(monthly_rainfall.index, monthly_rainfall.values, 
                      color=sns.color_palette("Blues_r", len(monthly_rainfall)))
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Rainfall (mm)')
        axes[1, 0].set_title('Average Monthly Rainfall Pattern')
        
        # 4. Time series of temperature
        if 'date' in data.columns:
            data_sample = data.iloc[::30]  # Sample every 30th point for clarity
            axes[1, 1].plot(pd.to_datetime(data_sample['date']), data_sample['temperature'])
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].set_title('Temperature Time Series (Sampled)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{self.save_dir}/weather_data_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot: {filename}")
        
        plt.show()
        return filename if save_plots else None
    
    def create_model_evaluation_report(self, models_results, test_results, problem_type='regression'):
        """Create comprehensive evaluation report."""
        print("📋 Creating model evaluation report...")
        
        report_data = []
        
        for name, result in models_results.results.items():
            if result.get('trained', False):
                model_info = {'Model': name, 'Status': 'Trained'}
                
                if problem_type == 'regression':
                    # Validation metrics
                    model_info.update({
                        'Validation_R2': result.get('val_r2', 'N/A'),
                        'Validation_RMSE': result.get('val_rmse', 'N/A'),
                        'Validation_MAE': result.get('val_mae', 'N/A')
                    })
                    
                    # Test metrics
                    if name in test_results:
                        model_info.update({
                            'Test_R2': test_results[name].get('r2', 'N/A'),
                            'Test_RMSE': test_results[name].get('rmse', 'N/A'),
                            'Test_MAE': test_results[name].get('mae', 'N/A')
                        })
                
                else:  # classification
                    # Validation metrics
                    model_info.update({
                        'Validation_Accuracy': result.get('val_accuracy', 'N/A'),
                        'Validation_F1': result.get('val_f1', 'N/A'),
                        'Validation_Precision': result.get('val_precision', 'N/A')
                    })
                    
                    # Test metrics
                    if name in test_results:
                        model_info.update({
                            'Test_Accuracy': test_results[name].get('accuracy', 'N/A'),
                            'Test_F1': test_results[name].get('f1_score', 'N/A')
                        })
                
                report_data.append(model_info)
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data)
        
        # Format numeric columns
        numeric_cols = report_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report_df[col] = report_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
        
        # Save to CSV
        filename = f"{self.save_dir}/model_evaluation_report_{problem_type}.csv"
        report_df.to_csv(filename, index=False)
        print(f"✅ Saved report: {filename}")
        
        # Display summary
        print("\n📊 Model Evaluation Summary:")
        print("=" * 50)
        print(report_df.to_string(index=False))
        
        return report_df, filename
    
    def plot_learning_curves(self, models_results, X_train, y_train):
        """Plot learning curves for model performance analysis."""
        print("📈 Creating learning curves...")
        
        from sklearn.model_selection import learning_curve
        
        trained_models = [(name, result['model']) for name, result in models_results.results.items() 
                         if result.get('trained', False)]
        
        if len(trained_models) == 0:
            print("⚠️  No trained models available for learning curves")
            return
        
        fig, axes = plt.subplots(1, len(trained_models), figsize=(6*len(trained_models), 5))
        if len(trained_models) == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(trained_models):
            try:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_train, y_train, cv=3, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
                )
                
                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                # Plot learning curves
                axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
                axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
                
                axes[idx].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
                axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
                
                axes[idx].set_xlabel('Training Set Size')
                axes[idx].set_ylabel('Score')
                axes[idx].set_title(f'{name}\nLearning Curve')
                axes[idx].legend()
                axes[idx].grid(True)
                
            except Exception as e:
                print(f"⚠️  Could not create learning curve for {name}: {e}")
                axes[idx].text(0.5, 0.5, f"Learning curve\nnot available\nfor {name}", 
                              ha='center', va='center', transform=axes[idx].transAxes)
        
        plt.tight_layout()
        
        filename = f"{self.save_dir}/learning_curves.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {filename}")
        
        plt.show()
        return filename


def main():
    """Test the visualization module."""
    print("🧪 Testing Weather Data Visualization")
    print("=" * 50)
    
    # Import required modules
    from data_preprocessing import WeatherDataPreprocessor
    from ml_models import WeatherPredictionModels
    
    # Initialize visualization
    viz = WeatherVisualization()
    
    # Get data and train models
    preprocessor = WeatherDataPreprocessor()
    temp_data = preprocessor.preprocess_pipeline(prediction_type='temperature', days_ahead=1)
    
    if temp_data:
        print("\n🌡️  Testing Temperature Prediction Visualizations:")
        
        # Train models
        regression_models = WeatherPredictionModels()
        regression_models.initialize_models(problem_type='regression')
        regression_models.train_models(
            temp_data['X_train'], temp_data['y_train'],
            temp_data['X_val'], temp_data['y_val'],
            temp_data['feature_names']
        )
        
        # Evaluate on test set
        test_results = regression_models.evaluate_on_test_set(
            temp_data['X_test'], temp_data['y_test']
        )
        
        # Create visualizations
        viz.plot_model_performance_comparison(regression_models, 'r2', 'regression')
        viz.plot_weather_data_analysis(temp_data['original_data'])
        
        # Feature importance
        importance_data = regression_models.get_feature_importance()
        if importance_data:
            viz.plot_feature_importance(importance_data)
        
        # Prediction accuracy for best model
        best_model, best_name, _ = regression_models.get_best_model('val_r2')
        best_result = regression_models.results[best_name]
        viz.plot_prediction_accuracy(
            temp_data['y_test'], 
            test_results[best_name]['predictions'],
            best_name,
            'regression'
        )
        
        # Create evaluation report
        report_df, report_file = viz.create_model_evaluation_report(
            regression_models, test_results, 'regression'
        )
        
        print(f"\n📁 All visualizations saved in: {viz.save_dir}/")
    
    print("\n✅ Visualization testing completed!")


if __name__ == "__main__":
    main()
