#!/usr/bin/env python3
"""
Data Preprocessing Module

Handles data cleaning, feature engineering, and preparation for ML models.
Includes functions for:
- Data loading and validation
- Feature engineering (lag features, moving averages, etc.)
- Data scaling and normalization
- Train/validation/test splits
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class WeatherDataPreprocessor:
    """Comprehensive data preprocessing for weather prediction models."""
    
    def __init__(self, data_path="data/kolkata_weather_data.csv"):
        self.data_path = data_path
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.target_columns = None
        
    def load_data(self):
        """Load and perform initial data validation."""
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {df.shape}")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Basic data validation
            print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"❌ Missing values: {df.isnull().sum().sum()}")
            print(f"🔢 Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
            
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def create_lag_features(self, df, columns=['temperature', 'humidity', 'rainfall'], lags=[1, 2, 3, 7]):
        """Create lagged features for time series prediction."""
        print(f"🔄 Creating lag features for {columns} with lags {lags}")
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_moving_averages(self, df, columns=['temperature', 'humidity'], windows=[3, 7, 14]):
        """Create moving average features."""
        print(f"📊 Creating moving averages for {columns} with windows {windows}")
        
        for col in columns:
            for window in windows:
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        
        return df
    
    def create_cyclical_features(self, df):
        """Create cyclical features from temporal data."""
        print("🔄 Creating cyclical features...")
        
        # Cyclical encoding for day of year
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between weather variables."""
        print("🔗 Creating interaction features...")
        
        # Temperature-humidity interaction (important for heat index)
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        # Pressure-wind interaction
        df['pressure_wind_interaction'] = df['pressure'] * df['wind_speed']
        
        # Temperature gradient (change from previous day)
        df['temp_gradient'] = df['temperature'] - df['temperature'].shift(1)
        
        # Rainfall intensity (rainfall per wind speed)
        df['rainfall_intensity'] = df['rainfall'] / (df['wind_speed'] + 1)  # +1 to avoid division by zero
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features."""
        print("🏷️  Encoding categorical features...")
        
        # One-hot encode season
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # One-hot encode rainfall category
        if 'rainfall_category' in df.columns:
            rain_dummies = pd.get_dummies(df['rainfall_category'], prefix='rain_cat')
            df = pd.concat([df, rain_dummies], axis=1)
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values appropriately."""
        print("🔧 Handling missing values...")
        
        # Forward fill for lag features (common in time series)
        lag_columns = [col for col in df.columns if 'lag_' in col]
        for col in lag_columns:
            df[col] = df[col].fillna(method='ffill')
        
        # Fill remaining missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        print(f"❌ Remaining missing values: {df.isnull().sum().sum()}")
        return df
    
    def create_prediction_targets(self, df, target_days=[1, 3, 7]):
        """Create future prediction targets."""
        print(f"🎯 Creating prediction targets for {target_days} days ahead...")
        
        for days in target_days:
            # Future temperature
            df[f'temp_future_{days}d'] = df['temperature'].shift(-days)
            
            # Future rainfall (binary: will it rain?)
            df[f'rainfall_future_{days}d'] = (df['rainfall'].shift(-days) > 0).astype(int)
            
            # Future humidity
            df[f'humidity_future_{days}d'] = df['humidity'].shift(-days)
        
        return df
    
    def prepare_features_and_targets(self, df, prediction_type='temperature', days_ahead=1):
        """Prepare feature matrix and target vector for a specific prediction task."""
        
        # Define target column based on prediction type
        if prediction_type == 'temperature':
            target_col = f'temp_future_{days_ahead}d'
        elif prediction_type == 'rainfall':
            target_col = f'rainfall_future_{days_ahead}d'
        elif prediction_type == 'humidity':
            target_col = f'humidity_future_{days_ahead}d'
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        
        # Remove rows where target is NaN
        df_clean = df.dropna(subset=[target_col]).copy()
        
        # Define feature columns (exclude target columns and identifiers)
        exclude_cols = ['date', 'rainfall_category', 'season'] + [col for col in df.columns if 'future_' in col]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Ensure only numeric columns are selected
        numeric_cols = df_clean[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        print(f"🎯 Prepared {prediction_type} prediction for {days_ahead} days ahead:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        
        return X, y, feature_cols
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler."""
        print("⚖️  Scaling features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation, and test sets."""
        print(f"✂️  Splitting data: train={1-test_size-val_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Validation set: {X_val.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_importance_data(self, feature_names, importance_scores):
        """Create feature importance dataframe for analysis."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def preprocess_pipeline(self, prediction_type='temperature', days_ahead=1):
        """Complete preprocessing pipeline."""
        print("🔄 Starting complete preprocessing pipeline...")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Feature engineering
        df = self.create_lag_features(df)
        df = self.create_moving_averages(df)
        df = self.create_cyclical_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df)
        
        # Create prediction targets
        df = self.create_prediction_targets(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Prepare features and targets
        X, y, feature_names = self.prepare_features_and_targets(df, prediction_type, days_ahead)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        print("✅ Preprocessing pipeline completed!")
        
        return {
            'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': feature_names,
            'scaler': self.scaler,
            'original_data': df
        }


def main():
    """Test the preprocessing pipeline."""
    print("🧪 Testing Weather Data Preprocessing Pipeline")
    print("=" * 50)
    
    preprocessor = WeatherDataPreprocessor()
    
    # Test temperature prediction preprocessing
    print("\n🌡️  Testing Temperature Prediction Preprocessing:")
    temp_data = preprocessor.preprocess_pipeline(prediction_type='temperature', days_ahead=1)
    
    if temp_data:
        print(f"\n📊 Feature Information:")
        print(f"   Number of features: {len(temp_data['feature_names'])}")
        print(f"   Sample features: {temp_data['feature_names'][:10]}")
        print(f"   Target range: {temp_data['y_train'].min():.1f} to {temp_data['y_train'].max():.1f}")
    
    # Test rainfall prediction preprocessing
    print("\n🌧️  Testing Rainfall Prediction Preprocessing:")
    rain_data = preprocessor.preprocess_pipeline(prediction_type='rainfall', days_ahead=3)
    
    if rain_data:
        print(f"   Rainfall prediction (binary): {rain_data['y_train'].unique()}")
        print(f"   Class distribution: {pd.Series(rain_data['y_train']).value_counts()}")
    
    print("\n✅ Preprocessing tests completed!")


if __name__ == "__main__":
    main()
