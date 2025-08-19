#!/usr/bin/env python3
"""
Generate All Weather Prediction Visualizations
Creates comprehensive graphs and charts to showcase the ML project.
"""

import sys
sys.path.append('src')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from data_preprocessing import WeatherDataPreprocessor
from ml_models import WeatherPredictionModels
from visualization import WeatherVisualization

def generate_all_visualizations():
    """Generate comprehensive visualizations for the weather prediction project."""
    
    print("🎨 Generating comprehensive weather prediction visualizations...")
    print("=" * 60)
    
    # Initialize components
    preprocessor = WeatherDataPreprocessor()
    viz = WeatherVisualization()
    
    # 1. Generate weather data analysis
    print("\n📊 Creating weather data analysis...")
    df = preprocessor.load_data()
    if df is not None:
        viz.plot_weather_data_analysis(df, save_plots=True)
        print("✅ Weather data analysis saved")
    
    # 2. Temperature prediction models
    print("\n🌡️ Training temperature prediction models...")
    temp_data = preprocessor.preprocess_pipeline(prediction_type='temperature', days_ahead=1)
    
    if temp_data:
        # Train models
        temp_models = WeatherPredictionModels()
        temp_models.initialize_models(problem_type='regression')
        temp_models.train_models(
            temp_data['X_train'], temp_data['y_train'],
            temp_data['X_val'], temp_data['y_val'],
            temp_data['feature_names']
        )
        
        # Evaluate on test set
        temp_test_results = temp_models.evaluate_on_test_set(
            temp_data['X_test'], temp_data['y_test']
        )
        
        # Generate visualizations
        viz.plot_model_performance_comparison(temp_models, 'r2', 'regression')
        
        # Feature importance
        importance_data = temp_models.get_feature_importance()
        if importance_data:
            viz.plot_feature_importance(importance_data)
        
        # Best model prediction accuracy
        best_model, best_name, _ = temp_models.get_best_model('val_r2')
        viz.plot_prediction_accuracy(
            temp_data['y_test'], 
            temp_test_results[best_name]['predictions'],
            best_name,
            'regression'
        )
        
        # Create evaluation report
        viz.create_model_evaluation_report(temp_models, temp_test_results, 'regression')
        
        print("✅ Temperature prediction visualizations completed")
    
    # 3. Rainfall prediction models
    print("\n🌧️ Training rainfall prediction models...")
    rain_data = preprocessor.preprocess_pipeline(prediction_type='rainfall', days_ahead=1)
    
    if rain_data:
        # Train models
        rain_models = WeatherPredictionModels()
        rain_models.initialize_models(problem_type='classification')
        rain_models.train_models(
            rain_data['X_train'], rain_data['y_train'],
            rain_data['X_val'], rain_data['y_val'],
            rain_data['feature_names']
        )
        
        # Evaluate on test set
        rain_test_results = rain_models.evaluate_on_test_set(
            rain_data['X_test'], rain_data['y_test']
        )
        
        # Generate visualizations
        viz.plot_model_performance_comparison(rain_models, 'accuracy', 'classification')
        
        # Feature importance for rainfall
        rain_importance_data = rain_models.get_feature_importance()
        if rain_importance_data:
            # Create separate plot for rainfall feature importance
            fig, axes = plt.subplots(1, len(rain_importance_data), figsize=(6*len(rain_importance_data), 8))
            if len(rain_importance_data) == 1:
                axes = [axes]
            
            for idx, (model_name, importance_df) in enumerate(rain_importance_data.items()):
                ax = axes[idx] if len(rain_importance_data) > 1 else axes[0]
                
                # Get top features
                top_features = importance_df.head(15)
                
                # Create horizontal bar plot
                bars = ax.barh(range(len(top_features)), top_features['importance'], 
                              color=sns.color_palette("viridis", len(top_features)))
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'], fontsize=8)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'Rainfall Prediction\n{model_name} - Top 15 Features', fontweight='bold')
                
                # Add value labels
                for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{importance:.3f}', va='center', ha='left', fontsize=7)
            
            plt.tight_layout()
            filename = f"results/rainfall_feature_importance.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot: {filename}")
            plt.close()
        
        # Best model prediction accuracy for rainfall
        best_model, best_name, _ = rain_models.get_best_model('val_accuracy')
        viz.plot_prediction_accuracy(
            rain_data['y_test'], 
            rain_test_results[best_name]['predictions'],
            best_name,
            'classification'
        )
        
        # Create evaluation report
        viz.create_model_evaluation_report(rain_models, rain_test_results, 'classification')
        
        print("✅ Rainfall prediction visualizations completed")
    
    # 4. Create combined model comparison
    print("\n📈 Creating combined model comparison...")
    
    if temp_data and rain_data:
        # Create a comprehensive comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temperature models comparison
        temp_names = []
        temp_scores = []
        for name, result in temp_models.results.items():
            if result.get('trained', False) and 'val_r2' in result:
                temp_names.append(name.replace(' ', '\n'))
                temp_scores.append(result['val_r2'])
        
        bars1 = axes[0].bar(temp_names, temp_scores, color=sns.color_palette("Blues_r", len(temp_names)))
        axes[0].set_title('Temperature Prediction Models\n(R² Score)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_ylim([0, 1])
        
        # Add value labels
        for bar, score in zip(bars1, temp_scores):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rainfall models comparison
        rain_names = []
        rain_scores = []
        for name, result in rain_models.results.items():
            if result.get('trained', False) and 'val_accuracy' in result:
                rain_names.append(name.replace(' ', '\n'))
                rain_scores.append(result['val_accuracy'])
        
        bars2 = axes[1].bar(rain_names, rain_scores, color=sns.color_palette("Greens_r", len(rain_names)))
        axes[1].set_title('Rainfall Prediction Models\n(Accuracy Score)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Accuracy Score', fontsize=12)
        axes[1].set_ylim([0, 1])
        
        # Add value labels
        for bar, score in zip(bars2, rain_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = "results/combined_model_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {filename}")
        plt.close()
    
    # 5. Create project summary infographic
    print("\n🎯 Creating project summary infographic...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🌤️ Kolkata Weather Prediction System - ML Project Summary', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Dataset statistics
    axes[0, 0].text(0.5, 0.9, '📊 Dataset Statistics', ha='center', va='top', 
                    fontsize=16, fontweight='bold', transform=axes[0, 0].transAxes)
    
    stats_text = f"""
📅 Time Period: 2020-2022 (3 years)
📈 Total Records: 1,095 days
🌡️ Temperature Range: 13.8°C to 46.3°C  
💧 Humidity Range: 31% to 107%
🌧️ Rainfall Days: 361 (33%)
🔢 Features Created: 41 engineered features
🎯 Prediction Horizons: 1, 3, 7 days ahead
"""
    
    axes[0, 0].text(0.05, 0.75, stats_text, ha='left', va='top', 
                    fontsize=11, transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axis('off')
    
    # ML Models used
    axes[0, 1].text(0.5, 0.9, '🤖 Machine Learning Models', ha='center', va='top', 
                    fontsize=16, fontweight='bold', transform=axes[0, 1].transAxes)
    
    models_text = """
🔹 Linear Regression - Baseline model
🔹 Ridge Regression - L2 regularization  
🔹 Lasso Regression - L1 regularization
🔹 Random Forest - Ensemble method
🔹 Support Vector Machine - Kernel-based
🔹 Classification for rainfall (binary)
🔹 Regression for temperature/humidity
"""
    
    axes[0, 1].text(0.05, 0.75, models_text, ha='left', va='top', 
                    fontsize=11, transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].axis('off')
    
    # Features engineered
    axes[1, 0].text(0.5, 0.9, '⚙️ Feature Engineering', ha='center', va='top', 
                    fontsize=16, fontweight='bold', transform=axes[1, 0].transAxes)
    
    features_text = """
🔸 Lag Features: 1, 2, 3, 7 day history
🔸 Moving Averages: 3, 7, 14 day windows
🔸 Cyclical Encoding: Seasonal patterns
🔸 Interaction Features: Temp × Humidity
🔸 Weather Derivatives: Heat index, comfort
🔸 Seasonal Dummies: One-hot encoding
🔸 Data Scaling: StandardScaler normalization
"""
    
    axes[1, 0].text(0.05, 0.75, features_text, ha='left', va='top', 
                    fontsize=11, transform=axes[1, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].axis('off')
    
    # Results summary
    axes[1, 1].text(0.5, 0.9, '🏆 Performance Results', ha='center', va='top', 
                    fontsize=16, fontweight='bold', transform=axes[1, 1].transAxes)
    
    # Get best results if available
    if 'temp_models' in locals() and 'rain_models' in locals():
        best_temp_model, best_temp_name, best_temp_score = temp_models.get_best_model('val_r2')
        best_rain_model, best_rain_name, best_rain_score = rain_models.get_best_model('val_accuracy')
        
        results_text = f"""
🌡️ Best Temperature Model: 
   {best_temp_name}
   R² Score: {best_temp_score:.3f}

🌧️ Best Rainfall Model:
   {best_rain_name}  
   Accuracy: {best_rain_score:.3f}

📈 Interactive System with:
   • Model comparison
   • Visualizations  
   • Future predictions
"""
    else:
        results_text = """
🌡️ Temperature Prediction:
   Multiple regression models
   Feature importance analysis
   
🌧️ Rainfall Prediction:
   Binary classification models
   Confusion matrix evaluation
   
📈 Comprehensive Evaluation:
   • Performance metrics
   • Visualization suite
   • Interactive interface
"""
    
    axes[1, 1].text(0.05, 0.75, results_text, ha='left', va='top', 
                    fontsize=11, transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    filename = "results/project_summary_infographic.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {filename}")
    plt.close()
    
    print(f"\n🎨 All visualizations completed!")
    print(f"📁 Check the results/ directory for all generated charts")
    
    # List all created files
    import os
    result_files = os.listdir('results')
    print(f"\n📊 Generated {len(result_files)} visualization files:")
    for file in sorted(result_files):
        print(f"   • {file}")


if __name__ == "__main__":
    generate_all_visualizations()
