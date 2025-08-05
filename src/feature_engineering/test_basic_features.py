"""Tests for basic feature extraction."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from unittest.mock import patch

from src.feature_engineering.basic_features import BasicFeatureExtractor


class TestBasicFeatureExtractor:
    """Test cases for BasicFeatureExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = BasicFeatureExtractor()
        
        # Create sample weather data
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17']),
            'forecast_date': pd.to_datetime(['2025-01-14', '2025-01-15', '2025-01-16']),
            'predicted_high': [75.0, 78.0, 72.0],
            'predicted_low': [58.0, 62.0, 55.0],
            'humidity': [65.0, 70.0, 60.0],
            'pressure': [1013.2, 1015.1, 1011.8],
            'wind_speed': [8.5, 12.0, 6.2],
            'wind_direction': [225.0, 180.0, 270.0],
            'cloud_cover': [30.0, 45.0, 20.0],
            'precipitation_prob': [10.0, 25.0, 5.0],
            'data_quality_score': [1.0, 0.95, 1.0]
        })
    
    def test_extract_temperature_features(self):
        """Test temperature feature extraction."""
        features = self.extractor.extract_temperature_features(self.sample_data, 'nws')
        
        # Check basic temperature features
        assert 'nws_temp_high' in features.columns
        assert 'nws_temp_low' in features.columns
        assert 'nws_temp_range' in features.columns
        assert 'nws_temp_avg' in features.columns
        
        # Check derived features
        assert 'nws_temp_high_change' in features.columns
        assert 'nws_temp_high_3day_avg' in features.columns
        
        # Verify calculations
        assert features['nws_temp_range'].iloc[0] == 17.0  # 75 - 58
        assert features['nws_temp_avg'].iloc[0] == 66.5   # (75 + 58) / 2
        assert features['nws_temp_high_change'].iloc[1] == 3.0  # 78 - 75
    
    def test_extract_atmospheric_features(self):
        """Test atmospheric feature extraction."""
        features = self.extractor.extract_atmospheric_features(self.sample_data, 'nws')
        
        # Check basic atmospheric features
        assert 'nws_humidity' in features.columns
        assert 'nws_pressure' in features.columns
        assert 'nws_wind_speed' in features.columns
        assert 'nws_wind_direction' in features.columns
        
        # Check derived features
        assert 'nws_wind_u' in features.columns  # East-west component
        assert 'nws_wind_v' in features.columns  # North-south component
        assert 'nws_pressure_change' in features.columns
        assert 'nws_humidity_low' in features.columns
        assert 'nws_humidity_high' in features.columns
        
        # Verify wind components calculation
        # For wind_speed=8.5, wind_direction=225° (southwest)
        expected_u = 8.5 * np.cos(np.radians(225))  # Should be negative (westward)
        expected_v = 8.5 * np.sin(np.radians(225))  # Should be negative (southward)
        assert abs(features['nws_wind_u'].iloc[0] - expected_u) < 0.01
        assert abs(features['nws_wind_v'].iloc[0] - expected_v) < 0.01
    
    def test_extract_date_features(self):
        """Test date-based feature extraction."""
        dates = pd.Series(['2025-01-15', '2025-07-15', '2025-12-15'])
        features = self.extractor.extract_date_features(dates)
        
        # Check basic date features
        assert 'day_of_year' in features.columns
        assert 'month' in features.columns
        assert 'day_of_month' in features.columns
        assert 'day_of_week' in features.columns
        assert 'season' in features.columns
        
        # Check cyclical encoding
        assert 'day_of_year_sin' in features.columns
        assert 'day_of_year_cos' in features.columns
        assert 'month_sin' in features.columns
        assert 'month_cos' in features.columns
        
        # Check LA-specific features
        assert 'is_heat_season' in features.columns
        assert 'is_fire_season' in features.columns
        assert 'is_marine_layer_season' in features.columns
        
        # Verify season assignment
        assert features['season'].iloc[0] == 'winter'  # January
        assert features['season'].iloc[1] == 'summer'  # July
        assert features['season'].iloc[2] == 'winter'  # December
        
        # Verify LA-specific seasons
        assert features['is_heat_season'].iloc[1] == 1  # July is heat season
        assert features['is_fire_season'].iloc[0] == 1  # January is fire season
        assert features['is_marine_layer_season'].iloc[1] == 1  # July is marine layer season
    
    def test_extract_quality_features(self):
        """Test quality feature extraction."""
        features = self.extractor.extract_quality_features(self.sample_data, 'nws')
        
        # Check quality features
        assert 'nws_quality_score' in features.columns
        assert 'nws_completeness' in features.columns
        
        # Check missing data indicators
        assert 'nws_predicted_high_missing' in features.columns
        assert 'nws_predicted_low_missing' in features.columns
        assert 'nws_humidity_missing' in features.columns
        
        # Verify completeness calculation (should be 1.0 for complete data)
        assert all(features['nws_completeness'] == 1.0)
    
    def test_detect_and_clean_outliers_iqr(self):
        """Test outlier detection and cleaning using IQR method."""
        # Create data with more points and clear outliers
        extended_data = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19']),
            'forecast_date': pd.to_datetime(['2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18']),
            'predicted_high': [75.0, 78.0, 72.0, 76.0, 150.0],  # Last one is extreme outlier
            'predicted_low': [58.0, 62.0, 55.0, 59.0, 60.0],
            'humidity': [65.0, 70.0, 60.0, 68.0, 150.0],  # Last one is impossible humidity
            'pressure': [1013.2, 1015.1, 1011.8, 1014.0, 1012.5],
            'wind_speed': [8.5, 12.0, 6.2, 9.1, 10.0],
            'wind_direction': [225.0, 180.0, 270.0, 200.0, 190.0],
            'cloud_cover': [30.0, 45.0, 20.0, 35.0, 40.0],
            'precipitation_prob': [10.0, 25.0, 5.0, 15.0, 20.0],
            'data_quality_score': [1.0, 0.95, 1.0, 0.98, 1.0]
        })
        
        cleaned_data = self.extractor.detect_and_clean_outliers(extended_data, 'nws', method='iqr')
        
        # Temperature outliers should be capped, not removed
        assert not pd.isna(cleaned_data.loc[4, 'predicted_high'])
        assert cleaned_data.loc[4, 'predicted_high'] < 150.0
        
        # Other outliers should be set to NaN
        assert pd.isna(cleaned_data.loc[4, 'humidity'])
        
        # Quality scores should be reduced for outliers
        assert cleaned_data.loc[4, 'data_quality_score'] < extended_data.loc[4, 'data_quality_score']
    
    def test_detect_and_clean_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        # Create data with outliers
        outlier_data = self.sample_data.copy()
        outlier_data.loc[0, 'predicted_high'] = 150.0  # Extreme outlier
        
        cleaned_data = self.extractor.detect_and_clean_outliers(outlier_data, 'nws', method='zscore', threshold=2.0)
        
        # Temperature outliers should be capped
        assert not pd.isna(cleaned_data.loc[0, 'predicted_high'])
        assert cleaned_data.loc[0, 'predicted_high'] < 150.0
    
    def test_extract_source_features(self):
        """Test extracting all features from a single source."""
        features = self.extractor.extract_source_features(self.sample_data, 'nws')
        
        # Should have date column plus features
        assert 'date' in features.columns
        assert len(features.columns) > 10  # Should have many features
        
        # Should have features from all categories
        temp_features = [col for col in features.columns if 'temp' in col.lower()]
        atmo_features = [col for col in features.columns if any(x in col.lower() for x in ['humidity', 'pressure', 'wind'])]
        quality_features = [col for col in features.columns if 'quality' in col.lower()]
        
        assert len(temp_features) > 0
        assert len(atmo_features) > 0
        assert len(quality_features) > 0
    
    def test_extract_all_source_features(self):
        """Test extracting features from multiple sources."""
        source_data = {
            'nws': self.sample_data,
            'openweather': self.sample_data.copy()  # Simulate second source
        }
        
        features = self.extractor.extract_all_source_features(source_data)
        
        # Should have date column plus features from both sources
        assert 'date' in features.columns
        assert len(features.columns) > 20  # Should have many features from both sources
        
        # Should have date-based features
        assert 'day_of_year' in features.columns
        assert 'month' in features.columns
        assert 'season' in features.columns
        
        # Should have features from both sources
        nws_features = [col for col in features.columns if 'nws_' in col]
        openweather_features = [col for col in features.columns if 'openweather_' in col]
        
        assert len(nws_features) > 0
        assert len(openweather_features) > 0
    
    def test_get_feature_summary(self):
        """Test feature summary generation."""
        features = self.extractor.extract_source_features(self.sample_data, 'nws')
        summary = self.extractor.get_feature_summary(features)
        
        # Check summary structure
        assert 'total_features' in summary
        assert 'total_records' in summary
        assert 'numeric_features' in summary
        assert 'missing_data_percentage' in summary
        assert 'date_range' in summary
        assert 'feature_categories' in summary
        
        # Check values
        assert summary['total_records'] == 3
        assert summary['total_features'] > 0
        assert 'start' in summary['date_range']
        assert 'end' in summary['date_range']
    
    def test_validate_features(self):
        """Test feature validation."""
        features = self.extractor.extract_source_features(self.sample_data, 'nws')
        is_valid, issues = self.extractor.validate_features(features)
        
        # Should be valid for good data
        assert is_valid
        assert len(issues) == 0
        
        # Test with empty DataFrame
        empty_features = pd.DataFrame()
        is_valid, issues = self.extractor.validate_features(empty_features)
        assert not is_valid
        assert len(issues) > 0
        
        # Test with missing date column
        bad_features = features.drop('date', axis=1)
        is_valid, issues = self.extractor.validate_features(bad_features)
        assert not is_valid
        assert any('date' in issue for issue in issues)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        # All methods should handle empty data gracefully
        temp_features = self.extractor.extract_temperature_features(empty_data, 'test')
        assert temp_features.empty
        
        atmo_features = self.extractor.extract_atmospheric_features(empty_data, 'test')
        assert atmo_features.empty
        
        quality_features = self.extractor.extract_quality_features(empty_data, 'test')
        assert quality_features.empty
        
        source_features = self.extractor.extract_source_features(empty_data, 'test')
        assert source_features.empty
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        # Data with only some columns
        partial_data = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-16']),
            'predicted_high': [75.0, 78.0],
            # Missing other columns
        })
        
        features = self.extractor.extract_source_features(partial_data, 'test')
        
        # Should still extract available features
        assert not features.empty
        assert 'test_temp_high' in features.columns
        # Should not have features for missing columns
        assert 'test_humidity' not in features.columns


if __name__ == '__main__':
    # Run a simple test
    extractor = BasicFeatureExtractor()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'date': pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17']),
        'forecast_date': pd.to_datetime(['2025-01-14', '2025-01-15', '2025-01-16']),
        'predicted_high': [75.0, 78.0, 72.0],
        'predicted_low': [58.0, 62.0, 55.0],
        'humidity': [65.0, 70.0, 60.0],
        'pressure': [1013.2, 1015.1, 1011.8],
        'wind_speed': [8.5, 12.0, 6.2],
        'wind_direction': [225.0, 180.0, 270.0],
        'cloud_cover': [30.0, 45.0, 20.0],
        'precipitation_prob': [10.0, 25.0, 5.0],
        'data_quality_score': [1.0, 0.95, 1.0]
    })
    
    print("Testing BasicFeatureExtractor...")
    
    # Test single source feature extraction
    features = extractor.extract_source_features(sample_data, 'nws')
    print(f"Extracted {len(features.columns)-1} features from single source")
    print(f"Feature columns: {list(features.columns)}")
    
    # Test feature summary
    summary = extractor.get_feature_summary(features)
    print(f"Feature summary: {summary}")
    
    # Test validation
    is_valid, issues = extractor.validate_features(features)
    print(f"Features valid: {is_valid}, Issues: {issues}")
    
    print("Basic tests passed!")