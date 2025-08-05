# Feature Engineering Module

This module implements basic feature extraction from individual weather data sources for the Kalshi Weather Predictor system.

## Overview

The feature engineering module transforms raw weather data from multiple APIs into structured features suitable for machine learning models. It extracts temperature, atmospheric, date-based, and quality features while handling data cleaning and outlier detection.

## Components

### BasicFeatureExtractor

The main class that handles feature extraction from weather data sources.

#### Key Features:
- **Temperature Features**: High/low temperatures, temperature range, averages, trends
- **Atmospheric Features**: Humidity, pressure, wind speed/direction, cloud cover, precipitation
- **Date-Based Features**: Day of year, month, season, cyclical encodings, LA-specific patterns
- **Quality Features**: Data quality scores, completeness metrics, missing data indicators
- **Outlier Detection**: IQR and Z-score methods with intelligent cleaning

#### Usage Example:
```python
from src.feature_engineering.basic_features import BasicFeatureExtractor
from src.utils.data_manager import DataManager

# Initialize components
data_manager = DataManager()
extractor = BasicFeatureExtractor()

# Load weather data
source_data = data_manager.load_all_sources()

# Extract features from all sources
features = extractor.extract_all_source_features(source_data, clean_outliers=True)

print(f"Extracted {len(features.columns)-1} features for {len(features)} records")
```

### FeaturePipeline

A complete pipeline for feature engineering operations.

#### Key Features:
- **Date Range Processing**: Create features for specific date ranges
- **Training Dataset Creation**: Generate complete datasets with features and targets
- **Feature Analysis**: Importance analysis, correlations, quality validation
- **File I/O**: Save and load features to/from files
- **Pipeline Status**: Monitor data availability and pipeline health

#### Usage Example:
```python
from src.feature_engineering.feature_pipeline import FeaturePipeline
from datetime import date, timedelta

# Initialize pipeline
pipeline = FeaturePipeline()

# Create features for recent data
end_date = date.today()
start_date = end_date - timedelta(days=30)
features = pipeline.create_features_for_date_range(start_date, end_date)

# Create training dataset
features, targets = pipeline.create_training_dataset(start_date, end_date)

# Validate feature quality
validation = pipeline.validate_feature_quality(features)
print(f"Features valid: {validation['is_valid']}")
```

## Feature Categories

### 1. Temperature Features
- `{source}_temp_high`: Predicted high temperature
- `{source}_temp_low`: Predicted low temperature  
- `{source}_temp_range`: Diurnal temperature variation (high - low)
- `{source}_temp_avg`: Average temperature ((high + low) / 2)
- `{source}_temp_high_change`: Day-over-day temperature change
- `{source}_temp_high_3day_avg`: 3-day rolling average

### 2. Atmospheric Features
- `{source}_humidity`: Relative humidity percentage
- `{source}_pressure`: Atmospheric pressure (hPa)
- `{source}_wind_speed`: Wind speed (mph)
- `{source}_wind_direction`: Wind direction (degrees)
- `{source}_wind_u`: East-west wind component
- `{source}_wind_v`: North-south wind component
- `{source}_cloud_cover`: Cloud cover percentage
- `{source}_precipitation_prob`: Precipitation probability
- `{source}_pressure_change`: Day-over-day pressure change
- `{source}_humidity_low`: Binary indicator for low humidity (<30%)
- `{source}_humidity_high`: Binary indicator for high humidity (>80%)

### 3. Date-Based Features
- `day_of_year`: Day of year (1-366)
- `month`: Month (1-12)
- `day_of_month`: Day of month (1-31)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `week_of_year`: Week of year
- `season`: Meteorological season (winter/spring/summer/fall)
- `season_numeric`: Numeric season encoding (0-3)
- `day_of_year_sin/cos`: Cyclical encoding for day of year
- `month_sin/cos`: Cyclical encoding for month
- `day_of_week_sin/cos`: Cyclical encoding for day of week
- `is_weekend`: Weekend indicator
- `is_heat_season`: LA heat season (June-September)
- `is_fire_season`: LA fire season (October-April)
- `is_marine_layer_season`: LA marine layer season (May-August)

### 4. Quality Features
- `{source}_quality_score`: Overall data quality score (0-1)
- `{source}_completeness`: Percentage of non-missing values
- `{source}_{feature}_missing`: Binary indicators for missing data

## Data Cleaning and Outlier Detection

### Outlier Detection Methods

#### IQR Method (Default)
- Calculates Q1, Q3, and IQR for each feature
- Identifies outliers as values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
- Configurable threshold multiplier

#### Z-Score Method
- Calculates standard deviations from the mean
- Identifies outliers beyond specified Z-score threshold (default: 3)

### Outlier Handling Strategy

#### Temperature Data
- **Capping**: Outliers are capped to reasonable bounds rather than removed
- **Quality Reduction**: Quality scores are reduced for capped values
- **Preservation**: Temperature data is never completely removed

#### Other Features
- **NaN Replacement**: Non-temperature outliers are set to NaN
- **Quality Impact**: Quality scores are reduced for affected records
- **Graceful Degradation**: Models can handle missing values

## LA-Specific Features

The system includes features specifically designed for Los Angeles weather patterns:

### Seasonal Patterns
- **Heat Season** (June-September): Peak temperature period
- **Fire Season** (October-April): Santa Ana wind period, dry conditions
- **Marine Layer Season** (May-August): Coastal fog influence

### Weather Phenomena
- **Marine Layer Detection**: Humidity and cloud patterns indicating marine layer
- **Santa Ana Indicators**: Wind direction and speed patterns for Santa Ana events
- **Heat Island Effects**: Urban temperature variations

## Data Quality Management

### Quality Scoring
- **Range Validation**: Check values against reasonable LA weather ranges
- **Consistency Checks**: Verify high >= low temperatures
- **Completeness**: Track missing data percentages
- **Outlier Impact**: Reduce scores for outlier-affected records

### Quality Thresholds
- Temperature: -20°F to 130°F (reasonable for LA)
- Humidity: 0% to 100%
- Pressure: 900 to 1100 hPa
- Wind Speed: 0 to 200 mph
- Cloud Cover: 0% to 100%
- Precipitation Probability: 0% to 100%

## Testing

The module includes comprehensive unit tests covering:

### Test Coverage
- Feature extraction for all categories
- Outlier detection and cleaning
- Data validation and quality checks
- Edge cases and error handling
- Empty data handling
- Missing column handling

### Running Tests
```bash
# Run all feature engineering tests
python -m pytest src/feature_engineering/test_basic_features.py -v

# Run specific test
python -m pytest src/feature_engineering/test_basic_features.py::TestBasicFeatureExtractor::test_extract_temperature_features -v

# Run with coverage
python -m pytest src/feature_engineering/test_basic_features.py --cov=src.feature_engineering
```

## Performance Considerations

### Efficient Processing
- **Vectorized Operations**: Uses pandas vectorized operations for speed
- **Memory Management**: Processes data in chunks for large datasets
- **Caching**: Caches computed features to avoid recomputation

### Scalability
- **Incremental Processing**: Can process new data incrementally
- **Source Isolation**: Each data source processed independently
- **Parallel Processing**: Ready for parallel processing of multiple sources

## Integration with ML Pipeline

### Feature Preparation
```python
# Create training dataset
features, targets = pipeline.create_training_dataset(start_date, end_date)

# Validate features
is_valid, issues = extractor.validate_features(features)

# Get feature importance analysis
analysis = pipeline.get_feature_importance_analysis(features)
```

### Model Input Format
Features are provided as a pandas DataFrame with:
- `date` column for temporal indexing
- Numeric features ready for ML models
- Consistent naming convention across sources
- Quality indicators for ensemble weighting

## Configuration

### Customizable Parameters
- Outlier detection thresholds
- Quality score weights
- Feature selection criteria
- Date range processing windows

### Source Configuration
The system supports these weather data sources:
- NWS (National Weather Service)
- OpenWeatherMap
- Tomorrow.io
- Weatherbit
- Visual Crossing

## Future Enhancements

### Planned Features
- **Advanced Ensemble Features**: Cross-source consensus metrics
- **LA-Specific Models**: Weather pattern recognition
- **Real-time Processing**: Streaming feature extraction
- **Feature Selection**: Automated feature importance ranking
- **Seasonal Adjustments**: Dynamic feature weighting by season

### Performance Optimizations
- **Parallel Processing**: Multi-threaded feature extraction
- **Caching Layer**: Redis-based feature caching
- **Incremental Updates**: Only process new/changed data
- **Memory Optimization**: Streaming processing for large datasets

## Error Handling

The module includes robust error handling for:
- Missing data files
- Corrupted data
- API failures
- Invalid date ranges
- Memory constraints
- File I/O errors

All errors are logged with appropriate detail levels and the system degrades gracefully when data sources are unavailable.