#!/usr/bin/env python3
"""
Kolkata Weather Prediction System

Main application that combines data processing, machine learning models,
and visualization to create a comprehensive weather prediction system.

Features:
- Interactive command-line interface
- Multiple prediction types (temperature, rainfall, humidity)
- Model comparison and evaluation
- Visualization and reporting
- Future prediction capabilities
"""

import argparse
import sys
import os
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_preprocessing import WeatherDataPreprocessor
from ml_models import WeatherPredictionModels
from visualization import WeatherVisualization

warnings.filterwarnings('ignore')

class WeatherPredictionSystem:
    """Main weather prediction system."""
    
    def __init__(self):
        self.preprocessor = WeatherDataPreprocessor()
        self.models = {}
        self.visualizer = WeatherVisualization()
        self.data_cache = {}
    
    def print_header(self):
        """Print application header."""
        print("=" * 70)
        print("🌤️  KOLKATA WEATHER PREDICTION SYSTEM")
        print("    Advanced Machine Learning for Weather Forecasting")
        print("=" * 70)
        print()
    
    def print_menu(self):
        """Display main menu options."""
        print("📋 MAIN MENU")
        print("-" * 30)
        print("1. 🌡️  Temperature Prediction")
        print("2. 🌧️  Rainfall Prediction") 
        print("3. 💧 Humidity Prediction")
        print("4. 📊 Model Performance Comparison")
        print("5. 📈 Generate Visualizations")
        print("6. 📋 Model Evaluation Report")
        print("7. 🔮 Future Weather Prediction")
        print("8. ❓ Help & Information")
        print("9. 🚪 Exit")
        print("-" * 30)
    
    def preprocess_data(self, prediction_type, days_ahead=1, force_reload=False):
        """Preprocess data for specified prediction type."""
        cache_key = f"{prediction_type}_{days_ahead}"
        
        if cache_key in self.data_cache and not force_reload:
            print(f"📦 Using cached data for {prediction_type} prediction ({days_ahead} days ahead)")
            return self.data_cache[cache_key]
        
        print(f"🔄 Preprocessing data for {prediction_type} prediction ({days_ahead} days ahead)...")
        
        try:
            data = self.preprocessor.preprocess_pipeline(
                prediction_type=prediction_type, 
                days_ahead=days_ahead
            )
            
            if data:
                self.data_cache[cache_key] = data
                print(f"✅ Data preprocessing completed for {prediction_type}")
                return data
            else:
                print(f"❌ Failed to preprocess data for {prediction_type}")
                return None
                
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")
            return None
    
    def train_models(self, data, prediction_type):
        """Train ML models for specified prediction type."""
        problem_type = 'classification' if prediction_type == 'rainfall' else 'regression'
        
        print(f"\n🚀 Training {problem_type} models for {prediction_type} prediction...")
        
        try:
            # Initialize models
            model_system = WeatherPredictionModels()
            model_system.initialize_models(problem_type=problem_type)
            
            # Train models
            model_system.train_models(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                data['feature_names']
            )
            
            # Evaluate on test set
            test_results = model_system.evaluate_on_test_set(
                data['X_test'], data['y_test']
            )
            
            # Store models
            self.models[prediction_type] = {
                'model_system': model_system,
                'test_results': test_results,
                'problem_type': problem_type,
                'data': data
            }
            
            return model_system, test_results
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return None, None
    
    def predict_temperature(self):
        """Handle temperature prediction workflow."""
        print("\n🌡️  TEMPERATURE PREDICTION")
        print("=" * 40)
        
        # Get user input for prediction horizon
        try:
            days = int(input("Enter days ahead for prediction (1, 3, or 7): "))
            if days not in [1, 3, 7]:
                days = 1
                print("Invalid input. Using 1 day ahead.")
        except ValueError:
            days = 1
            print("Invalid input. Using 1 day ahead.")
        
        # Preprocess data
        data = self.preprocess_data('temperature', days_ahead=days)
        if not data:
            return
        
        # Train models
        model_system, test_results = self.train_models(data, 'temperature')
        if not model_system:
            return
        
        # Get best model
        best_model, best_name, best_score = model_system.get_best_model('val_r2')
        print(f"\n🏆 Best Model: {best_name} (R² = {best_score:.3f})")
        
        # Display test results summary
        print("\n📊 Test Set Performance Summary:")
        for name, results in test_results.items():
            print(f"   {name}: RMSE = {results['rmse']:.2f}°C, R² = {results['r2']:.3f}")
        
        return model_system, test_results
    
    def predict_rainfall(self):
        """Handle rainfall prediction workflow."""
        print("\n🌧️  RAINFALL PREDICTION")
        print("=" * 40)
        
        # Get user input for prediction horizon
        try:
            days = int(input("Enter days ahead for prediction (1, 3, or 7): "))
            if days not in [1, 3, 7]:
                days = 1
                print("Invalid input. Using 1 day ahead.")
        except ValueError:
            days = 1
            print("Invalid input. Using 1 day ahead.")
        
        # Preprocess data
        data = self.preprocess_data('rainfall', days_ahead=days)
        if not data:
            return
        
        # Train models
        model_system, test_results = self.train_models(data, 'rainfall')
        if not model_system:
            return
        
        # Get best model
        best_model, best_name, best_score = model_system.get_best_model('val_accuracy')
        print(f"\n🏆 Best Model: {best_name} (Accuracy = {best_score:.3f})")
        
        # Display test results summary
        print("\n📊 Test Set Performance Summary:")
        for name, results in test_results.items():
            print(f"   {name}: Accuracy = {results['accuracy']:.3f}, F1 = {results['f1_score']:.3f}")
        
        return model_system, test_results
    
    def predict_humidity(self):
        """Handle humidity prediction workflow."""
        print("\n💧 HUMIDITY PREDICTION")
        print("=" * 40)
        
        # Get user input for prediction horizon
        try:
            days = int(input("Enter days ahead for prediction (1, 3, or 7): "))
            if days not in [1, 3, 7]:
                days = 1
                print("Invalid input. Using 1 day ahead.")
        except ValueError:
            days = 1
            print("Invalid input. Using 1 day ahead.")
        
        # Preprocess data
        data = self.preprocess_data('humidity', days_ahead=days)
        if not data:
            return
        
        # Train models
        model_system, test_results = self.train_models(data, 'humidity')
        if not model_system:
            return
        
        # Get best model
        best_model, best_name, best_score = model_system.get_best_model('val_r2')
        print(f"\n🏆 Best Model: {best_name} (R² = {best_score:.3f})")
        
        # Display test results summary
        print("\n📊 Test Set Performance Summary:")
        for name, results in test_results.items():
            print(f"   {name}: RMSE = {results['rmse']:.2f}%, R² = {results['r2']:.3f}")
        
        return model_system, test_results
    
    def compare_models(self):
        """Compare performance across all trained models."""
        print("\n📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 50)
        
        if not self.models:
            print("⚠️  No models trained yet. Please train some models first.")
            return
        
        # Display comparison for each prediction type
        for pred_type, model_info in self.models.items():
            print(f"\n🎯 {pred_type.capitalize()} Prediction Models:")
            print("-" * 30)
            
            model_system = model_info['model_system']
            test_results = model_info['test_results']
            problem_type = model_info['problem_type']
            
            if problem_type == 'regression':
                metric = 'R²'
                for name, results in test_results.items():
                    print(f"   {name:<25} | RMSE: {results['rmse']:.3f} | {metric}: {results['r2']:.3f}")
            else:
                for name, results in test_results.items():
                    print(f"   {name:<25} | Accuracy: {results['accuracy']:.3f} | F1: {results['f1_score']:.3f}")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n📈 GENERATING VISUALIZATIONS")
        print("=" * 40)
        
        if not self.models:
            print("⚠️  No models trained yet. Please train some models first.")
            return
        
        print("Creating visualizations...")
        
        for pred_type, model_info in self.models.items():
            print(f"\n📊 Creating plots for {pred_type} prediction...")
            
            model_system = model_info['model_system']
            test_results = model_info['test_results']
            problem_type = model_info['problem_type']
            data = model_info['data']
            
            try:
                # Model performance comparison
                metric = 'r2' if problem_type == 'regression' else 'accuracy'
                self.visualizer.plot_model_performance_comparison(
                    model_system, metric, problem_type
                )
                
                # Feature importance
                importance_data = model_system.get_feature_importance()
                if importance_data:
                    self.visualizer.plot_feature_importance(importance_data)
                
                # Best model prediction accuracy
                best_metric = 'val_r2' if problem_type == 'regression' else 'val_accuracy'
                best_model, best_name, _ = model_system.get_best_model(best_metric)
                
                self.visualizer.plot_prediction_accuracy(
                    data['y_test'], 
                    test_results[best_name]['predictions'],
                    best_name,
                    problem_type
                )
                
                # Weather data analysis (only once)
                if pred_type == list(self.models.keys())[0]:
                    self.visualizer.plot_weather_data_analysis(data['original_data'])
                
                print(f"✅ Visualizations created for {pred_type}")
                
            except Exception as e:
                print(f"❌ Error creating visualizations for {pred_type}: {e}")
        
        print(f"\n📁 All visualizations saved in: {self.visualizer.save_dir}/")
    
    def generate_reports(self):
        """Generate comprehensive evaluation reports."""
        print("\n📋 GENERATING EVALUATION REPORTS")
        print("=" * 45)
        
        if not self.models:
            print("⚠️  No models trained yet. Please train some models first.")
            return
        
        for pred_type, model_info in self.models.items():
            print(f"\n📊 Creating report for {pred_type} prediction...")
            
            model_system = model_info['model_system']
            test_results = model_info['test_results']
            problem_type = model_info['problem_type']
            
            try:
                report_df, report_file = self.visualizer.create_model_evaluation_report(
                    model_system, test_results, problem_type
                )
                print(f"✅ Report saved: {report_file}")
                
            except Exception as e:
                print(f"❌ Error creating report for {pred_type}: {e}")
        
        print(f"\n📁 All reports saved in: {self.visualizer.save_dir}/")
    
    def predict_future(self):
        """Make future weather predictions."""
        print("\n🔮 FUTURE WEATHER PREDICTION")
        print("=" * 40)
        
        if not self.models:
            print("⚠️  No models trained yet. Please train some models first.")
            return
        
        print("Available prediction types:")
        for i, pred_type in enumerate(self.models.keys(), 1):
            print(f"   {i}. {pred_type.capitalize()}")
        
        try:
            choice = int(input(f"\nSelect prediction type (1-{len(self.models)}): ")) - 1
            pred_types = list(self.models.keys())
            
            if 0 <= choice < len(pred_types):
                pred_type = pred_types[choice]
                model_info = self.models[pred_type]
                
                # Get best model
                model_system = model_info['model_system']
                problem_type = model_info['problem_type']
                best_metric = 'val_r2' if problem_type == 'regression' else 'val_accuracy'
                best_model, best_name, best_score = model_system.get_best_model(best_metric)
                
                print(f"\n🎯 Using best model: {best_name}")
                print(f"Model performance: {best_metric.replace('val_', '').upper()} = {best_score:.3f}")
                
                # Use recent data for prediction (last few rows of test set)
                data = model_info['data']
                recent_data = data['X_test'][-5:]  # Last 5 samples
                
                # Make predictions
                predictions = model_system.predict_future(best_model, recent_data)
                
                print(f"\n🔮 Future {pred_type} predictions (last 5 test samples):")
                print("-" * 40)
                
                if problem_type == 'regression':
                    unit = "°C" if pred_type == "temperature" else "%"
                    for i, pred in enumerate(predictions):
                        print(f"Sample {i+1}: {pred:.2f} {unit}")
                else:
                    for i, pred in enumerate(predictions):
                        result = "Rain" if pred == 1 else "No Rain"
                        print(f"Sample {i+1}: {result}")
                
            else:
                print("❌ Invalid choice")
                
        except (ValueError, IndexError):
            print("❌ Invalid input")
    
    def show_help(self):
        """Display help and information."""
        print("\n❓ HELP & INFORMATION")
        print("=" * 30)
        print("""
🌤️  ABOUT THIS SYSTEM:
This is an advanced machine learning system for weather prediction in Kolkata.
It uses multiple algorithms to forecast temperature, rainfall, and humidity.

🔧 AVAILABLE MODELS:
• Linear Regression - Simple linear relationships
• Ridge Regression - Regularized linear model
• Lasso Regression - Feature selection via regularization
• Random Forest - Ensemble of decision trees
• Support Vector Machine - Complex pattern recognition

📊 PREDICTION TYPES:
• Temperature: Continuous values in Celsius
• Rainfall: Binary classification (Rain/No Rain)
• Humidity: Continuous values as percentage

📈 FEATURES USED:
• Historical weather patterns
• Seasonal trends
• Lag features (previous days' weather)
• Moving averages
• Interaction features

🎯 EVALUATION METRICS:
• Regression: R², RMSE, MAE
• Classification: Accuracy, Precision, Recall, F1-Score

📁 OUTPUT FILES:
• Visualizations: PNG plots saved in results/
• Reports: CSV files with detailed metrics
• Models: Trained models can be saved for reuse

💡 TIPS FOR BEST RESULTS:
• Try different prediction horizons (1, 3, 7 days)
• Compare multiple models to find the best one
• Generate visualizations to understand model behavior
• Use evaluation reports for detailed analysis
        """)
    
    def run(self):
        """Run the main application loop."""
        self.print_header()
        
        while True:
            try:
                self.print_menu()
                choice = input("\nEnter your choice (1-9): ").strip()
                
                if choice == '1':
                    self.predict_temperature()
                
                elif choice == '2':
                    self.predict_rainfall()
                
                elif choice == '3':
                    self.predict_humidity()
                
                elif choice == '4':
                    self.compare_models()
                
                elif choice == '5':
                    self.generate_visualizations()
                
                elif choice == '6':
                    self.generate_reports()
                
                elif choice == '7':
                    self.predict_future()
                
                elif choice == '8':
                    self.show_help()
                
                elif choice == '9':
                    print("\n👋 Thank you for using the Kolkata Weather Prediction System!")
                    print("Stay weather-aware! 🌤️")
                    break
                
                else:
                    print("❌ Invalid choice. Please select a number from 1-9.")
                
                input("\n⏸️  Press Enter to continue...")
                print("\n" + "="*70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {e}")
                print("Please try again or contact support.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kolkata Weather Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python weather_predictor.py              # Interactive mode
  
For more information, run the program and select 'Help & Information'
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='Kolkata Weather Prediction System v1.0'
    )
    
    args = parser.parse_args()
    
    # Initialize and run the system
    system = WeatherPredictionSystem()
    system.run()


if __name__ == "__main__":
    main()
