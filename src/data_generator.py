#!/usr/bin/env python3
"""
Kolkata Weather Data Generator

Creates synthetic but realistic weather data for Kolkata based on:
- Seasonal patterns (monsoon, winter, summer)
- Historical climate trends
- Natural weather variations and correlations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class KolkataWeatherGenerator:
    """Generate synthetic weather data for Kolkata with realistic seasonal patterns."""
    
    def __init__(self, start_date="2020-01-01", num_days=1095):  # 3 years of data
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.num_days = num_days
        self.dates = [self.start_date + timedelta(days=i) for i in range(num_days)]
        
        # Kolkata climate parameters based on historical data
        self.climate_params = {
            'winter': {  # Dec-Feb
                'temp_min': 12, 'temp_max': 28, 'temp_mean': 20,
                'humidity_min': 40, 'humidity_max': 80, 'humidity_mean': 60,
                'rainfall_prob': 0.1, 'rainfall_max': 30,
                'wind_min': 2, 'wind_max': 15, 'wind_mean': 8
            },
            'summer': {  # Mar-May
                'temp_min': 25, 'temp_max': 42, 'temp_mean': 34,
                'humidity_min': 35, 'humidity_max': 70, 'humidity_mean': 50,
                'rainfall_prob': 0.15, 'rainfall_max': 60,
                'wind_min': 3, 'wind_max': 20, 'wind_mean': 12
            },
            'monsoon': {  # Jun-Sep
                'temp_min': 24, 'temp_max': 35, 'temp_mean': 29,
                'humidity_min': 70, 'humidity_max': 95, 'humidity_mean': 85,
                'rainfall_prob': 0.7, 'rainfall_max': 150,
                'wind_min': 5, 'wind_max': 25, 'wind_mean': 15
            },
            'post_monsoon': {  # Oct-Nov
                'temp_min': 20, 'temp_max': 32, 'temp_mean': 26,
                'humidity_min': 50, 'humidity_max': 80, 'humidity_mean': 65,
                'rainfall_prob': 0.2, 'rainfall_max': 40,
                'wind_min': 2, 'wind_max': 18, 'wind_mean': 10
            }
        }
    
    def get_season(self, date):
        """Determine season based on month."""
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'summer'
        elif month in [6, 7, 8, 9]:
            return 'monsoon'
        else:
            return 'post_monsoon'
    
    def add_noise(self, value, noise_factor=0.1):
        """Add realistic random noise to weather parameters."""
        noise = np.random.normal(0, value * noise_factor)
        return max(0, value + noise)
    
    def generate_temperature(self, season_params, day_of_year):
        """Generate temperature with seasonal and daily variations."""
        base_temp = season_params['temp_mean']
        
        # Add seasonal variation (sinusoidal)
        seasonal_variation = 3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Add daily temperature variation
        daily_variation = np.random.normal(0, 2)
        
        temp = base_temp + seasonal_variation + daily_variation
        temp = np.clip(temp, season_params['temp_min'], season_params['temp_max'])
        
        return round(self.add_noise(temp, 0.05), 1)
    
    def generate_humidity(self, season_params, temperature, rainfall):
        """Generate humidity correlated with temperature and rainfall."""
        base_humidity = season_params['humidity_mean']
        
        # Inverse correlation with temperature
        temp_effect = -(temperature - 25) * 0.8
        
        # Positive correlation with rainfall
        rain_effect = rainfall * 0.3
        
        humidity = base_humidity + temp_effect + rain_effect + np.random.normal(0, 5)
        humidity = np.clip(humidity, season_params['humidity_min'], season_params['humidity_max'])
        
        return round(self.add_noise(humidity, 0.05), 1)
    
    def generate_rainfall(self, season_params):
        """Generate rainfall with realistic patterns."""
        if np.random.random() < season_params['rainfall_prob']:
            # Exponential distribution for rainfall amounts
            rainfall = np.random.exponential(season_params['rainfall_max'] / 3)
            rainfall = min(rainfall, season_params['rainfall_max'])
        else:
            rainfall = 0
        
        return round(rainfall, 1)
    
    def generate_wind_speed(self, season_params, rainfall):
        """Generate wind speed correlated with rainfall."""
        base_wind = season_params['wind_mean']
        
        # Higher wind speed during rainfall
        rain_effect = rainfall * 0.1
        
        wind = base_wind + rain_effect + np.random.normal(0, 2)
        wind = np.clip(wind, season_params['wind_min'], season_params['wind_max'])
        
        return round(self.add_noise(wind, 0.1), 1)
    
    def generate_pressure(self, season, temperature, humidity):
        """Generate atmospheric pressure with seasonal and weather correlations."""
        # Base pressure for Kolkata (sea level)
        base_pressure = 1013.25
        
        # Seasonal adjustments
        seasonal_adj = {
            'winter': 5, 'summer': -3, 'monsoon': -8, 'post_monsoon': 2
        }
        
        # Temperature and humidity effects
        temp_effect = -(temperature - 25) * 0.3
        humidity_effect = -(humidity - 60) * 0.1
        
        pressure = base_pressure + seasonal_adj[season] + temp_effect + humidity_effect
        pressure += np.random.normal(0, 3)
        
        return round(pressure, 1)
    
    def add_weather_correlations(self, df):
        """Add derived features based on weather correlations."""
        # Heat index (feels-like temperature)
        df['heat_index'] = df.apply(
            lambda row: self.calculate_heat_index(row['temperature'], row['humidity']), 
            axis=1
        )
        
        # Weather comfort index (0-100)
        df['comfort_index'] = df.apply(
            lambda row: self.calculate_comfort_index(row), axis=1
        )
        
        # Rainfall category
        df['rainfall_category'] = pd.cut(
            df['rainfall'], 
            bins=[-0.1, 0, 5, 25, 100, float('inf')],
            labels=['No Rain', 'Light', 'Moderate', 'Heavy', 'Very Heavy']
        )
        
        return df
    
    def calculate_heat_index(self, temp, humidity):
        """Calculate heat index (simplified formula)."""
        if temp < 27:
            return temp
        
        hi = -42.379 + 2.04901523 * temp + 10.14333127 * humidity
        hi += -0.22475541 * temp * humidity - 6.83783e-3 * temp**2
        hi += -5.481717e-2 * humidity**2 + 1.22874e-3 * temp**2 * humidity
        hi += 8.5282e-4 * temp * humidity**2 - 1.99e-6 * temp**2 * humidity**2
        
        return round(hi, 1)
    
    def calculate_comfort_index(self, row):
        """Calculate weather comfort index based on multiple factors."""
        temp_score = max(0, 100 - abs(row['temperature'] - 25) * 3)
        humidity_score = max(0, 100 - abs(row['humidity'] - 50) * 2)
        rain_score = 100 if row['rainfall'] == 0 else max(0, 100 - row['rainfall'] * 2)
        wind_score = max(0, min(100, row['wind_speed'] * 8))  # Optimal wind 5-12 km/h
        
        comfort = (temp_score + humidity_score + rain_score + wind_score) / 4
        return round(comfort, 1)
    
    def generate_dataset(self):
        """Generate complete weather dataset."""
        weather_data = []
        
        for i, date in enumerate(self.dates):
            season = self.get_season(date)
            season_params = self.climate_params[season]
            day_of_year = date.timetuple().tm_yday
            
            # Generate correlated weather parameters
            rainfall = self.generate_rainfall(season_params)
            temperature = self.generate_temperature(season_params, day_of_year)
            humidity = self.generate_humidity(season_params, temperature, rainfall)
            wind_speed = self.generate_wind_speed(season_params, rainfall)
            pressure = self.generate_pressure(season, temperature, humidity)
            
            weather_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'day_of_year': day_of_year,
                'month': date.month,
                'season': season,
                'temperature': temperature,
                'humidity': humidity,
                'rainfall': rainfall,
                'wind_speed': wind_speed,
                'pressure': pressure
            })
        
        df = pd.DataFrame(weather_data)
        df = self.add_weather_correlations(df)
        
        return df
    
    def save_dataset(self, df, filename="kolkata_weather_data.csv"):
        """Save dataset to CSV file."""
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False)
        print(f"✅ Dataset saved to {filepath}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        return filepath


def main():
    """Generate and save Kolkata weather dataset."""
    print("🌤️  Generating Kolkata Weather Dataset...")
    print("=" * 50)
    
    # Generate 3 years of data (2020-2022)
    generator = KolkataWeatherGenerator(start_date="2020-01-01", num_days=1095)
    weather_df = generator.generate_dataset()
    
    # Display basic statistics
    print("\n📈 Dataset Statistics:")
    print("-" * 30)
    print(f"Total records: {len(weather_df)}")
    print(f"Temperature range: {weather_df['temperature'].min():.1f}°C to {weather_df['temperature'].max():.1f}°C")
    print(f"Humidity range: {weather_df['humidity'].min():.1f}% to {weather_df['humidity'].max():.1f}%")
    print(f"Total rainfall days: {(weather_df['rainfall'] > 0).sum()}")
    print(f"Average wind speed: {weather_df['wind_speed'].mean():.1f} km/h")
    
    # Seasonal breakdown
    print("\n🌍 Seasonal Distribution:")
    print(weather_df['season'].value_counts())
    
    # Save dataset
    print("\n💾 Saving dataset...")
    generator.save_dataset(weather_df)
    
    # Show sample data
    print("\n🔍 Sample Data:")
    print(weather_df.head(10).to_string(index=False))
    
    print("\n✅ Weather dataset generation completed!")


if __name__ == "__main__":
    main()
