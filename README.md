# 🌤️ Kolkata Weather Prediction System

An advanced machine learning system for weather prediction in Kolkata using multiple ML algorithms and comprehensive data analysis.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

- **Multiple ML Models**: Linear Regression, Ridge, Lasso, Random Forest, Support Vector Machine
- **Multi-target Prediction**: Temperature, Rainfall (binary), Humidity
- **Comprehensive Preprocessing**: Feature engineering, scaling, lag features, moving averages
- **Interactive Interface**: User-friendly command-line application
- **Rich Visualizations**: Performance comparisons, feature importance, prediction accuracy
- **Detailed Reporting**: CSV reports with all evaluation metrics
- **Future Predictions**: Make predictions on new data using trained models

## 📁 Project Structure

```
kolkata-weather-ml/
├── data/                          # Generated weather datasets
│   └── kolkata_weather_data.csv   # 3 years of synthetic weather data
├── src/                           # Source code modules
│   ├── data_generator.py          # Weather data generation
│   ├── data_preprocessing.py      # Data preprocessing and feature engineering
│   ├── ml_models.py              # Machine learning models implementation
│   └── visualization.py          # Visualization and evaluation tools
├── models/                        # Saved ML models (created during runtime)
├── results/                       # Generated plots and reports
├── notebooks/                     # Jupyter notebooks (for development)
├── venv/                         # Python virtual environment
├── weather_predictor.py          # Main application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- macOS, Linux, or Windows

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kolkata-weather-ml
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate weather data**
   ```bash
   python src/data_generator.py
   ```

5. **Run the main application**
   ```bash
   python weather_predictor.py
   ```

## 🖥️ Usage

### Interactive Mode

The main application provides an interactive menu-driven interface:

```bash
python weather_predictor.py
```

**Available Options:**
1. 🌡️ Temperature Prediction
2. 🌧️ Rainfall Prediction  
3. 💧 Humidity Prediction
4. 📊 Model Performance Comparison
5. 📈 Generate Visualizations
6. 📋 Model Evaluation Report
7. 🔮 Future Weather Prediction
8. ❓ Help & Information
9. 🚪 Exit

### Command Line Usage

```bash
# Show version
python weather_predictor.py --version

# Get help
python weather_predictor.py --help
```

### Module Usage

You can also import and use individual modules:

```python
from src.data_preprocessing import WeatherDataPreprocessor
from src.ml_models import WeatherPredictionModels
from src.visualization import WeatherVisualization

# Preprocess data
preprocessor = WeatherDataPreprocessor()
data = preprocessor.preprocess_pipeline('temperature', days_ahead=1)

# Train models
models = WeatherPredictionModels()
models.initialize_models('regression')
models.train_models(data['X_train'], data['y_train'], 
                   data['X_val'], data['y_val'], data['feature_names'])

# Generate visualizations
viz = WeatherVisualization()
viz.plot_model_performance_comparison(models, 'r2', 'regression')
```

## 🧠 Machine Learning Models

### Regression Models (Temperature & Humidity)

1. **Linear Regression**
   - Simple linear relationships
   - Fast training and prediction
   - Good baseline model

2. **Ridge Regression**
   - L2 regularization to prevent overfitting
   - Handles multicollinearity well
   - Good for high-dimensional data

3. **Lasso Regression**
   - L1 regularization for feature selection
   - Automatic feature selection
   - Sparse model interpretation

4. **Random Forest Regressor**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Feature importance analysis
   - Robust to outliers

5. **Support Vector Regression (SVR)**
   - Kernel-based learning
   - Effective for complex patterns
   - Good generalization capability

### Classification Models (Rainfall)

1. **Random Forest Classifier**
   - Ensemble method with high accuracy
   - Built-in feature importance
   - Handles class imbalance

2. **Support Vector Classifier (SVC)**
   - Effective for binary classification
   - Good performance with RBF kernel
   - Probability estimates available

## 📊 Features & Data Processing

### Generated Features

- **Lag Features**: Previous 1, 2, 3, and 7 days weather data
- **Moving Averages**: 3, 7, and 14-day rolling averages
- **Cyclical Features**: Sine/cosine encoding for day of year and month
- **Interaction Features**: Temperature-humidity, pressure-wind interactions
- **Seasonal Encoding**: One-hot encoded seasons
- **Derived Metrics**: Heat index, comfort index, weather categories

### Data Pipeline

1. **Data Loading**: Load and validate weather data
2. **Feature Engineering**: Create lag, moving average, and interaction features
3. **Encoding**: One-hot encode categorical variables
4. **Target Creation**: Generate future prediction targets
5. **Missing Value Handling**: Forward fill and median imputation
6. **Data Splitting**: Train/validation/test splits (60%/20%/20%)
7. **Feature Scaling**: StandardScaler normalization

## 📈 Evaluation Metrics

### Regression Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error  
- **MAE**: Mean Absolute Error

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 🎨 Visualizations

The system generates comprehensive visualizations:

1. **Model Performance Comparison**: Bar charts comparing model metrics
2. **Feature Importance**: Horizontal bar plots for tree-based models
3. **Prediction Accuracy**: Actual vs predicted scatter plots and residual plots
4. **Weather Data Analysis**: Multi-panel exploration of weather patterns
5. **Learning Curves**: Training vs validation performance over dataset size

All plots are saved as high-resolution PNG files in the `results/` directory.

## 📋 Reports

Detailed CSV reports include:
- Model names and training status
- Validation metrics for all models
- Test set performance
- Best model identification
- Formatted numeric precision

## 🌍 Weather Data

The synthetic dataset includes realistic weather patterns for Kolkata:

- **Temperature**: 13°C to 46°C with seasonal variations
- **Humidity**: 31% to 107% with monsoon peaks
- **Rainfall**: Binary classification with seasonal probability
- **Wind Speed**: 2-25 km/h correlated with weather events
- **Pressure**: Sea-level adjusted with weather correlations
- **Seasons**: Winter, Summer, Monsoon, Post-monsoon patterns

**Dataset Statistics:**
- 3 years of daily data (1,095 records)
- 361 rainfall days across seasons
- Realistic correlations between weather variables
- Seasonal variations matching Kolkata climate

## 🔧 Technical Details

### Dependencies

```
scikit-learn>=1.7.1      # Machine learning algorithms
pandas>=2.3.1            # Data manipulation
numpy>=2.3.2             # Numerical computing
matplotlib>=3.10.5       # Plotting
seaborn>=0.13.2         # Statistical visualization
plotly>=6.3.0           # Interactive plots
jupyter>=1.1.1          # Notebook environment
```

### System Requirements

- **Memory**: Minimum 4GB RAM recommended
- **Storage**: ~100MB for full installation
- **CPU**: Multi-core recommended for Random Forest training
- **Python**: Version 3.8 or higher

### Performance Benchmarks

Typical performance on synthetic data:

| Model | Temperature R² | Rainfall Accuracy | Training Time |
|-------|---------------|------------------|---------------|
| Linear Regression | 0.67 | N/A | <1s |
| Random Forest | 0.69 | 0.72 | 5-10s |
| SVM | 0.69 | 0.72 | 10-20s |

## 🚨 Troubleshooting

### Common Issues

1. **ImportError: No module named 'src'**
   ```bash
   # Ensure you're running from the project root directory
   cd kolkata-weather-ml
   python weather_predictor.py
   ```

2. **Virtual environment issues**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Memory issues with large datasets**
   ```python
   # Reduce Random Forest parameters in ml_models.py
   RandomForestRegressor(n_estimators=50, max_depth=5)
   ```

4. **Plotting issues on headless systems**
   ```python
   # Add to the beginning of visualization.py
   import matplotlib
   matplotlib.use('Agg')
   ```

## 🔬 Development

### Running Individual Modules

```bash
# Test data generation
python src/data_generator.py

# Test preprocessing pipeline
python src/data_preprocessing.py

# Test ML models
python src/ml_models.py

# Test visualization
python src/visualization.py
```

### Adding New Models

1. Extend the `WeatherPredictionModels` class in `src/ml_models.py`
2. Add model to `initialize_models()` method
3. Ensure compatibility with existing evaluation pipeline

### Custom Features

Add new features in `data_preprocessing.py`:

```python
def create_custom_features(self, df):
    # Add your feature engineering here
    df['custom_feature'] = df['temperature'] * df['pressure']
    return df
```

## 📚 Educational Value

This project demonstrates:

### Machine Learning Concepts
- **Supervised Learning**: Regression and classification
- **Feature Engineering**: Creating meaningful predictors
- **Model Selection**: Comparing multiple algorithms
- **Cross-validation**: Proper train/validation/test splits
- **Evaluation Metrics**: Understanding model performance
- **Overfitting**: Regularization techniques

### Software Engineering
- **Modular Design**: Separated concerns in different modules
- **Object-Oriented Programming**: Classes for reusable components
- **Error Handling**: Robust error management
- **Documentation**: Comprehensive code documentation
- **Testing**: Module-level testing functions

### Data Science Workflow
- **Data Generation**: Synthetic data with realistic patterns
- **Exploratory Analysis**: Understanding data characteristics
- **Preprocessing**: Cleaning and transforming data
- **Model Training**: Systematic approach to ML
- **Evaluation**: Multiple metrics and visualizations
- **Reporting**: Professional result presentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Create Pull Request
---

**Made with ❤️ for learning Machine Learning and Data Science**

*Stay weather-aware! 🌤️*
