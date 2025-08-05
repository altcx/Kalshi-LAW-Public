#!/usr/bin/env python3
"""Test script for DataManager Parquet storage system."""

import sys
import os
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_manager import DataManager
from loguru import logger

def create_sample_weather_data():
    """Create sample weather data for testing."""
    return pd.DataFrame([
        {
            'date': '2025-01-15',
            'forecast_date': '2025-01-14',
            'predicted_high': 75.2,
            'predicted_low': 58.1,
            'humidity': 65.0,
            'pressure': 1013.2,
            'wind_speed': 8.5,
            'wind_direction': 225.0,
            'cloud_cover': 30.0,
            'precipitation_prob': 10.0
        },
        {
            'date': '2025-01-16',
            'forecast_date': '2025-01-14',
            'predicted_high': 78.5,
            'predicted_low': 60.3,
            'humidity': 70.0,
            'pressure': 1015.8,
            'wind_speed': 12.2,
            'wind_direction': 180.0,
            'cloud_cover': 45.0,
            'precipitation_prob': 25.0
        }
    ])

def create_invalid_weather_data():
    """Create invalid weather data for testing validation."""
    return pd.DataFrame([
        {
            'date': '2025-01-17',
            'forecast_date': '2025-01-16',
            'predicted_high': 150.0,  # Invalid: too high
            'predicted_low': 45.0,
            'humidity': 120.0,  # Invalid: > 100%
            'pressure': 1013.2,
            'wind_speed': -5.0,  # Invalid: negative
            'wind_direction': 225.0,
            'cloud_cover': 30.0,
            'precipitation_prob': 10.0
        },
        {
            'date': '2025-01-18',
            'forecast_date': '2025-01-16',
            'predicted_high': 65.0,
            'predicted_low': 70.0,  # Invalid: low > high
            'humidity': 60.0,
            'pressure': 1015.8,
            'wind_speed': 8.0,
            'wind_direction': 180.0,
            'cloud_cover': 45.0,
            'precipitation_prob': 25.0
        }
    ])

def test_data_manager():
    """Test DataManager functionality."""
    logger.info("Starting DataManager tests...")
    
    # Initialize DataManager
    dm = DataManager()
    
    # Test 1: Schema validation with valid data
    logger.info("Test 1: Schema validation with valid data")
    valid_data = create_sample_weather_data()
    is_valid, errors = dm.validate_weather_data_schema(valid_data, 'test_source')
    print(f"Valid data schema check: {is_valid}, errors: {errors}")
    assert is_valid, f"Valid data should pass schema validation, but got errors: {errors}"
    
    # Test 2: Schema validation with missing columns
    logger.info("Test 2: Schema validation with missing required columns")
    invalid_data = valid_data.drop(columns=['date'])
    is_valid, errors = dm.validate_weather_data_schema(invalid_data, 'test_source')
    print(f"Invalid data schema check: {is_valid}, errors: {errors}")
    assert not is_valid, "Data missing required columns should fail validation"
    
    # Test 3: Data quality checks
    logger.info("Test 3: Data quality checks")
    quality_checked_data = dm.perform_data_quality_checks(valid_data, 'test_source')
    print(f"Quality scores: {quality_checked_data['data_quality_score'].tolist()}")
    assert 'data_quality_score' in quality_checked_data.columns
    assert all(score >= 0.8 for score in quality_checked_data['data_quality_score']), "Valid data should have high quality scores"
    
    # Test 4: Data quality checks with invalid data
    logger.info("Test 4: Data quality checks with invalid data")
    invalid_data = create_invalid_weather_data()
    quality_checked_invalid = dm.perform_data_quality_checks(invalid_data, 'test_source')
    print(f"Invalid data quality scores: {quality_checked_invalid['data_quality_score'].tolist()}")
    assert all(score < 0.8 for score in quality_checked_invalid['data_quality_score']), "Invalid data should have low quality scores"
    
    # Test 5: Outlier detection
    logger.info("Test 5: Outlier detection")
    # Create more data points for meaningful outlier detection
    extended_data = pd.concat([valid_data] * 3, ignore_index=True)  # 6 data points
    extended_data.loc[0, 'predicted_high'] = 120.0  # Clear outlier for LA
    outliers = dm.detect_outliers(extended_data, 'predicted_high')
    print(f"Outliers detected: {outliers.tolist()}")
    assert outliers.iloc[0], "Should detect temperature outlier"
    
    # Test 6: Append daily data
    logger.info("Test 6: Append daily data")
    success = dm.append_daily_data('nws', valid_data, validate=True)
    print(f"Append daily data success: {success}")
    assert success, "Should successfully append valid data"
    
    # Test 7: Load data back
    logger.info("Test 7: Load data back")
    loaded_data = dm.load_source_data('nws')
    print(f"Loaded {len(loaded_data)} records")
    assert len(loaded_data) == 2, "Should load back the 2 records we stored"
    assert 'data_quality_score' in loaded_data.columns, "Loaded data should include quality scores"
    
    # Test 8: Store and retrieve prediction
    logger.info("Test 8: Store and retrieve prediction")
    dm.store_prediction(
        prediction=76.5,
        confidence=0.85,
        target_date=date(2025, 1, 20),
        model_contributions={'xgboost': 0.6, 'lightgbm': 0.4},
        feature_importance={'temperature_consensus': 0.3, 'pressure': 0.2}
    )
    
    predictions = dm.load_source_data('predictions')
    print(f"Stored predictions: {len(predictions)}")
    assert len(predictions) >= 1, "Should have stored at least one prediction"
    
    # Test 9: Update actual temperature
    logger.info("Test 9: Update actual temperature")
    dm.update_actual_temperature(date(2025, 1, 20), 77.2)
    
    updated_predictions = dm.load_source_data('predictions')
    actual_temp = updated_predictions[updated_predictions['date'].dt.date == date(2025, 1, 20)]['actual_temperature'].iloc[0]
    print(f"Updated actual temperature: {actual_temp}")
    assert actual_temp == 77.2, "Should have updated the actual temperature"
    
    # Test 10: Data quality summary
    logger.info("Test 10: Data quality summary")
    quality_summary = dm.get_data_quality_summary('nws', days=30)
    print(f"Quality summary: {quality_summary}")
    assert 'avg_quality_score' in quality_summary, "Should return quality metrics"
    
    # Test 11: Data summary
    logger.info("Test 11: Overall data summary")
    summary = dm.get_data_summary()
    print(f"Data summary: {summary}")
    assert 'nws' in summary, "Should include our test source in summary"
    
    logger.info("All DataManager tests passed!")

if __name__ == "__main__":
    test_data_manager()