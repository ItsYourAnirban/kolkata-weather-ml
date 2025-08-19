#!/usr/bin/env python3
"""
7-Day Weather Forecast for Kolkata
Uses trained ML models to predict weather for the next week
"""

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

warnings.filterwarnings('ignore')

def predict_next_7_days():
    """Generate 7-day weather forecast for Kolkata."""
    
    print('🌤️  KOLKATA 7-DAY WEATHER FORECAST')
    print('=' * 60)
    print('🤖 Powered by Machine Learning Models')
    print('=' * 60)
    
    # Initialize preprocessor
    preprocessor = WeatherDataPreprocessor()
    
    # Load and preprocess data
    print('📊 Loading and preprocessing weather data...')
    
    # Train models for different prediction types
    models = {}
    
    # Temperature prediction (7 days ahead)
    print('\n🌡️  Training temperature prediction models...')
    temp_data = preprocessor.preprocess_pipeline(prediction_type='temperature', days_ahead=7)
    if temp_data:
        temp_model_system = WeatherPredictionModels()
        temp_model_system.initialize_models(problem_type='regression')
        temp_model_system.train_models(
            temp_data['X_train'], temp_data['y_train'],
            temp_data['X_val'], temp_data['y_val'],
            temp_data['feature_names']
        )
        temp_test_results = temp_model_system.evaluate_on_test_set(
            temp_data['X_test'], temp_data['y_test']
        )
        models['temperature'] = {
            'system': temp_model_system,
            'data': temp_data,
            'results': temp_test_results
        }
        print('✅ Temperature models trained successfully')
    
    # Rainfall prediction (7 days ahead) 
    print('\n🌧️  Training rainfall prediction models...')
    rain_data = preprocessor.preprocess_pipeline(prediction_type='rainfall', days_ahead=7)
    if rain_data:
        rain_model_system = WeatherPredictionModels()
        rain_model_system.initialize_models(problem_type='classification')
        rain_model_system.train_models(
            rain_data['X_train'], rain_data['y_train'],
            rain_data['X_val'], rain_data['y_val'],
            rain_data['feature_names']
        )
        rain_test_results = rain_model_system.evaluate_on_test_set(
            rain_data['X_test'], rain_data['y_test']
        )
        models['rainfall'] = {
            'system': rain_model_system,
            'data': rain_data,
            'results': rain_test_results
        }
        print('✅ Rainfall models trained successfully')
    
    # Humidity prediction (7 days ahead)
    print('\n💧 Training humidity prediction models...')
    humid_data = preprocessor.preprocess_pipeline(prediction_type='humidity', days_ahead=7)
    if humid_data:
        humid_model_system = WeatherPredictionModels()
        humid_model_system.initialize_models(problem_type='regression')
        humid_model_system.train_models(
            humid_data['X_train'], humid_data['y_train'],
            humid_data['X_val'], humid_data['y_val'],
            humid_data['feature_names']
        )
        humid_test_results = humid_model_system.evaluate_on_test_set(
            humid_data['X_test'], humid_data['y_test']
        )
        models['humidity'] = {
            'system': humid_model_system,
            'data': humid_data,
            'results': humid_test_results
        }
        print('✅ Humidity models trained successfully')
    
    if not models:
        print('❌ No models could be trained. Please check the data.')
        return
    
    print('\n🔮 GENERATING 7-DAY FORECAST')
    print('=' * 40)
    
    # Get the latest date from the dataset
    if temp_data and 'original_data' in temp_data:
        latest_date = temp_data['original_data']['date'].max()
    else:
        latest_date = datetime.now()
    
    print(f'📅 Latest data available: {latest_date.strftime("%Y-%m-%d")}')
    
    # Generate predictions for next 7 days
    forecast_data = []
    
    for day in range(1, 8):
        forecast_date = latest_date + timedelta(days=day)
        day_forecast = {
            'date': forecast_date,
            'day_name': forecast_date.strftime('%A'),
            'date_str': forecast_date.strftime('%B %d, %Y')
        }
        
        # Use last few samples from test set as recent data for prediction
        recent_samples = 3  # Use last 3 samples
        
        # Temperature prediction
        if 'temperature' in models:
            temp_model_system = models['temperature']['system']
            temp_data_info = models['temperature']['data']
            temp_best_model, temp_best_name, temp_best_score = temp_model_system.get_best_model('val_r2')
            
            # Use recent test data for prediction
            recent_features = temp_data_info['X_test'][-recent_samples:]
            temp_predictions = temp_model_system.predict_future(temp_best_model, recent_features)
            day_forecast['temperature'] = np.mean(temp_predictions)
            day_forecast['temp_model'] = temp_best_name
            day_forecast['temp_r2'] = temp_best_score
        
        # Rainfall prediction
        if 'rainfall' in models:
            rain_model_system = models['rainfall']['system']
            rain_data_info = models['rainfall']['data']
            rain_best_model, rain_best_name, rain_best_score = rain_model_system.get_best_model('val_accuracy')
            
            # Use recent test data for prediction
            recent_features = rain_data_info['X_test'][-recent_samples:]
            rain_predictions = rain_model_system.predict_future(rain_best_model, recent_features)
            
            # Get probability if available
            try:
                if hasattr(rain_best_model, 'predict_proba'):
                    rain_probabilities = rain_best_model.predict_proba(recent_features)
                    if rain_probabilities.shape[1] > 1:
                        day_forecast['rain_probability'] = np.mean(rain_probabilities[:, 1]) * 100
                    else:
                        day_forecast['rain_probability'] = np.mean(rain_predictions) * 100
                else:
                    day_forecast['rain_probability'] = np.mean(rain_predictions) * 100
            except:
                day_forecast['rain_probability'] = np.mean(rain_predictions) * 100
            
            day_forecast['rain_model'] = rain_best_name
            day_forecast['rain_accuracy'] = rain_best_score
        
        # Humidity prediction
        if 'humidity' in models:
            humid_model_system = models['humidity']['system']
            humid_data_info = models['humidity']['data']
            humid_best_model, humid_best_name, humid_best_score = humid_model_system.get_best_model('val_r2')
            
            # Use recent test data for prediction
            recent_features = humid_data_info['X_test'][-recent_samples:]
            humid_predictions = humid_model_system.predict_future(humid_best_model, recent_features)
            day_forecast['humidity'] = np.mean(humid_predictions)
            day_forecast['humid_model'] = humid_best_name
            day_forecast['humid_r2'] = humid_best_score
        
        forecast_data.append(day_forecast)
    
    # Display the 7-day forecast
    print('\n🌤️  KOLKATA 7-DAY WEATHER FORECAST')
    print('=' * 60)
    
    for i, day in enumerate(forecast_data):
        temp = day.get('temperature', 25)  # Default value
        rain_prob = day.get('rain_probability', 20)  # Default value
        humidity = day.get('humidity', 70)  # Default value
        
        # Determine weather condition
        if rain_prob > 60:
            condition = '🌧️  Heavy Rain Expected'
            emoji = '🌧️ '
        elif rain_prob > 40:
            condition = '⛈️  Rain Likely'
            emoji = '⛈️ '
        elif rain_prob > 25:
            condition = '⛅ Partly Cloudy'
            emoji = '⛅'
        elif temp > 32:
            condition = '☀️  Hot & Sunny'
            emoji = '☀️ '
        elif temp < 20:
            condition = '🌤️  Cool Weather'
            emoji = '🌤️ '
        else:
            condition = '🌤️  Pleasant Weather'
            emoji = '🌤️ '
        
        print(f'\n{emoji} Day {i+1}: {day["day_name"]}, {day["date_str"]}')
        print(f'   🌡️  Temperature: {temp:.1f}°C')
        print(f'   🌧️  Rain Chance: {rain_prob:.0f}%')
        print(f'   💧 Humidity: {humidity:.0f}%')
        print(f'   ☁️  Condition: {condition}')
    
    # Summary statistics
    temps = [d.get('temperature', 25) for d in forecast_data]
    rain_probs = [d.get('rain_probability', 20) for d in forecast_data]
    humidities = [d.get('humidity', 70) for d in forecast_data]
    
    print('\n' + '=' * 60)
    print('📈 WEEKLY FORECAST SUMMARY')
    print('=' * 60)
    print(f'🌡️  Average Temperature: {np.mean(temps):.1f}°C')
    print(f'🌡️  Temperature Range: {min(temps):.1f}°C - {max(temps):.1f}°C')
    print(f'🌧️  Average Rain Chance: {np.mean(rain_probs):.0f}%')
    print(f'💧 Average Humidity: {np.mean(humidities):.0f}%')
    
    rainy_days = sum(1 for p in rain_probs if p > 30)
    print(f'🌦️  Expected Rainy Days: {rainy_days} out of 7')
    
    if np.mean(temps) > 30:
        weather_trend = 'Hot weather expected this week 🔥'
    elif np.mean(rain_probs) > 50:
        weather_trend = 'Rainy week ahead ☔'
    elif np.mean(temps) < 22:
        weather_trend = 'Cool and pleasant weather 🌤️ '
    else:
        weather_trend = 'Moderate weather conditions 🌤️ '
    
    print(f'📊 Weather Trend: {weather_trend}')
    
    # Model information
    print('\n' + '=' * 60)
    print('🤖 MODEL PERFORMANCE SUMMARY')
    print('=' * 60)
    
    for pred_type, model_info in models.items():
        system = model_info['system']
        results = model_info['results']
        
        if pred_type == 'temperature':
            best_model, best_name, best_score = system.get_best_model('val_r2')
            print(f'🌡️  Temperature: {best_name} (R² = {best_score:.3f})')
        elif pred_type == 'rainfall':
            best_model, best_name, best_score = system.get_best_model('val_accuracy')
            print(f'🌧️  Rainfall: {best_name} (Accuracy = {best_score:.3f})')
        elif pred_type == 'humidity':
            best_model, best_name, best_score = system.get_best_model('val_r2')
            print(f'💧 Humidity: {best_name} (R² = {best_score:.3f})')
    
    print('\n' + '=' * 60)
    print('📝 DISCLAIMER')
    print('=' * 60)
    print('• Predictions are based on historical weather patterns')
    print('• Actual weather may vary from predictions')
    print('• For official forecasts, consult meteorological services')
    print('• This is an ML demonstration project')
    print('=' * 60)

if __name__ == '__main__':
    try:
        predict_next_7_days()
    except KeyboardInterrupt:
        print('\n\n👋 Forecast generation interrupted by user')
    except Exception as e:
        print(f'\n❌ Error generating forecast: {e}')
        sys.exit(1)
