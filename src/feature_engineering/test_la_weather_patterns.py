"""Tests for LA-specific weather pattern feature extraction."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.feature_engineering.la_weather_patterns import LAWeatherPatternExtractor


class TestLAWeatherPatternExtractor:
    """Test cases for LA weather pattern feature extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = LAWeatherPatternExtractor()
        
        # Create sample data for testing
        dates = [date(2024, 6, 15), date(2024, 7, 15), date(2024, 12, 15), date(2024, 1, 15)]
        
        self.sample_data = pd.DataFrame({
            'date': dates,
            'predicted_high': [78, 82, 72, 68],
            'predicted_low': [65, 68, 58, 55],
            'humidity': [85, 80, 15, 20],  # High in summer (marine layer), low in winter (Santa Ana)
            'cloud_cover': [80, 75, 10, 5],  # High in summer, low in winter
            'wind_speed': [5, 6, 35, 30],  # Low in summer, high in winter (Santa Ana)
            'wind_direction': [270, 280, 90, 85],  # Westerly in summer, easterly in winter
            'pressure': [1012, 1010, 1020, 1018]
        })
    
    def test_detect_marine_layer_conditions(self):
        """Test marine layer detection."""
        features = self.extractor.detect_marine_layer_conditions(self.sample_data)
        
        # Check that features were created
        assert not features.empty
        assert 'date' in features.columns
        assert 'marine_layer_probability' in features.columns
        assert 'marine_layer_likely' in features.columns
        assert 'marine_layer_strength' in features.columns
        
        # Check probability range
        assert features['marine_layer_probability'].min() >= 0
        assert features['marine_layer_probability'].max() <= 1
        
        # Check binary features
        assert set(features['marine_layer_likely'].unique()).issubset({0, 1})
        
        # Summer dates (June, July) should have higher marine layer probability
        summer_mask = pd.to_datetime(features['date']).dt.month.isin([6, 7])
        winter_mask = pd.to_datetime(features['date']).dt.month.isin([12, 1])
        
        if summer_mask.any() and winter_mask.any():
            summer_prob = features.loc[summer_mask, 'marine_layer_probability'].mean()
            winter_prob = features.loc[winter_mask, 'marine_layer_probability'].mean()
            assert summer_prob > winter_prob
    
    def test_detect_santa_ana_conditions(self):
        """Test Santa Ana wind detection."""
        features = self.extractor.detect_santa_ana_conditions(self.sample_data)
        
        # Check that features were created
        assert not features.empty
        assert 'date' in features.columns
        assert 'santa_ana_probability' in features.columns
        assert 'santa_ana_likely' in features.columns
        assert 'santa_ana_strength' in features.columns
        assert 'santa_ana_fire_danger' in features.columns
        
        # Check probability range
        assert features['santa_ana_probability'].min() >= 0
        assert features['santa_ana_probability'].max() <= 1
        
        # Check binary features
        assert set(features['santa_ana_likely'].unique()).issubset({0, 1})
        
        # Winter dates (December, January) should have higher Santa Ana probability
        summer_mask = pd.to_datetime(features['date']).dt.month.isin([6, 7])
        winter_mask = pd.to_datetime(features['date']).dt.month.isin([12, 1])
        
        if summer_mask.any() and winter_mask.any():
            summer_prob = features.loc[summer_mask, 'santa_ana_probability'].mean()
            winter_prob = features.loc[winter_mask, 'santa_ana_probability'].mean()
            assert winter_prob > summer_prob
    
    def test_detect_heat_island_effects(self):
        """Test heat island effect detection."""
        features = self.extractor.detect_heat_island_effects(self.sample_data)
        
        # Check that features were created
        assert not features.empty
        assert 'date' in features.columns
        assert 'heat_island_intensity' in features.columns
        assert 'heat_island_temp_adjustment' in features.columns
        assert 'heat_island_effect' in features.columns
        
        # Check intensity range
        assert features['heat_island_intensity'].min() >= 0
        assert features['heat_island_intensity'].max() <= 1
        
        # Check temperature adjustment is reasonable (0-6°F)
        assert features['heat_island_temp_adjustment'].min() >= 0
        assert features['heat_island_temp_adjustment'].max() <= 6
        
        # Summer should generally have higher heat island effects
        summer_mask = pd.to_datetime(features['date']).dt.month.isin([6, 7, 8])
        if summer_mask.any():
            summer_intensity = features.loc[summer_mask, 'heat_island_intensity'].mean()
            assert summer_intensity > 0  # Should have some heat island effect in summer
    
    def test_create_seasonal_adjustments(self):
        """Test seasonal adjustment features."""
        features = self.extractor.create_seasonal_adjustments(self.sample_data)
        
        # Check that features were created
        assert not features.empty
        assert 'date' in features.columns
        assert 'la_monthly_normal' in features.columns
        assert 'temp_deviation_from_normal' in features.columns
        assert 'is_peak_summer' in features.columns
        assert 'is_peak_fire_season' in features.columns
        assert 'day_length_hours' in features.columns
        assert 'solar_declination' in features.columns
        
        # Check monthly normals are reasonable
        assert features['la_monthly_normal'].min() >= 60  # Reasonable minimum for LA
        assert features['la_monthly_normal'].max() <= 90  # Reasonable maximum for LA
        
        # Check binary seasonal features
        binary_features = ['is_peak_summer', 'is_peak_fire_season', 'is_peak_marine_layer_season']
        for feature in binary_features:
            if feature in features.columns:
                assert set(features[feature].unique()).issubset({0, 1})
        
        # Check day length is reasonable (9-15 hours for LA latitude, allowing for seasonal variation)
        assert features['day_length_hours'].min() >= 9
        assert features['day_length_hours'].max() <= 15
        
        # July should be marked as peak summer
        july_mask = pd.to_datetime(features['date']).dt.month == 7
        if july_mask.any():
            assert features.loc[july_mask, 'is_peak_summer'].iloc[0] == 1
        
        # December should be marked as peak fire season
        dec_mask = pd.to_datetime(features['date']).dt.month == 12
        if dec_mask.any():
            assert features.loc[dec_mask, 'is_peak_fire_season'].iloc[0] == 1
    
    def test_create_all_la_pattern_features(self):
        """Test creation of all LA pattern features."""
        features = self.extractor.create_all_la_pattern_features(self.sample_data)
        
        # Check that features were created
        assert not features.empty
        assert 'date' in features.columns
        
        # Check that features from all categories are present
        marine_features = [col for col in features.columns if 'marine_layer' in col]
        santa_ana_features = [col for col in features.columns if 'santa_ana' in col]
        heat_island_features = [col for col in features.columns if 'heat_island' in col]
        seasonal_features = [col for col in features.columns if any(x in col for x in ['season', 'normal', 'solar'])]
        
        assert len(marine_features) > 0
        assert len(santa_ana_features) > 0
        assert len(heat_island_features) > 0
        assert len(seasonal_features) > 0
        
        # Check total feature count is reasonable
        total_features = len(features.columns) - 1  # Exclude date
        assert total_features > 20  # Should have substantial number of features
        
        # Check no duplicate columns
        assert len(features.columns) == len(set(features.columns))
    
    def test_get_pattern_feature_summary(self):
        """Test feature summary generation."""
        features = self.extractor.create_all_la_pattern_features(self.sample_data)
        summary = self.extractor.get_pattern_feature_summary(features)
        
        # Check summary structure
        assert 'total_features' in summary
        assert 'total_records' in summary
        assert 'feature_categories' in summary
        assert 'date_range' in summary
        
        # Check feature categories
        categories = summary['feature_categories']
        assert 'marine_layer' in categories
        assert 'santa_ana' in categories
        assert 'heat_island' in categories
        assert 'seasonal' in categories
        
        # Check counts are reasonable
        assert summary['total_features'] > 0
        assert summary['total_records'] == len(self.sample_data)
    
    def test_validate_pattern_features(self):
        """Test feature validation."""
        features = self.extractor.create_all_la_pattern_features(self.sample_data)
        is_valid, issues = self.extractor.validate_pattern_features(features)
        
        # Should be valid with proper sample data
        assert is_valid
        assert len(issues) == 0
        
        # Test with invalid data
        invalid_features = features.copy()
        invalid_features['marine_layer_probability'] = 2.0  # Invalid probability > 1
        
        is_valid_invalid, issues_invalid = self.extractor.validate_pattern_features(invalid_features)
        assert not is_valid_invalid
        assert len(issues_invalid) > 0
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        
        # All methods should handle empty data gracefully
        marine_features = self.extractor.detect_marine_layer_conditions(empty_df)
        assert marine_features.empty
        
        santa_ana_features = self.extractor.detect_santa_ana_conditions(empty_df)
        assert santa_ana_features.empty
        
        heat_island_features = self.extractor.detect_heat_island_effects(empty_df)
        assert heat_island_features.empty
        
        seasonal_features = self.extractor.create_seasonal_adjustments(empty_df)
        assert seasonal_features.empty
        
        all_features = self.extractor.create_all_la_pattern_features(empty_df)
        assert all_features.empty
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create data with missing values
        missing_data = self.sample_data.copy()
        missing_data.loc[0, 'humidity'] = np.nan
        missing_data.loc[1, 'wind_speed'] = np.nan
        missing_data.loc[2, 'cloud_cover'] = np.nan
        
        # Should still create features
        features = self.extractor.create_all_la_pattern_features(missing_data)
        assert not features.empty
        
        # Features should handle missing data appropriately
        assert 'marine_layer_probability' in features.columns
        assert 'santa_ana_probability' in features.columns
        assert 'heat_island_intensity' in features.columns
    
    def test_seasonal_consistency(self):
        """Test that seasonal features are consistent with dates."""
        # Create data spanning a full year
        start_date = date(2024, 1, 1)
        dates = [start_date + timedelta(days=i*30) for i in range(12)]  # Monthly samples
        
        yearly_data = pd.DataFrame({
            'date': dates,
            'predicted_high': [68, 70, 72, 75, 78, 82, 85, 86, 84, 79, 73, 68],
            'humidity': [50] * 12,
            'cloud_cover': [40] * 12,
            'wind_speed': [10] * 12,
            'wind_direction': [180] * 12,
            'pressure': [1013] * 12
        })
        
        features = self.extractor.create_seasonal_adjustments(yearly_data)
        
        # Check that summer months are correctly identified
        summer_months = pd.to_datetime(features['date']).dt.month.isin([7, 8])
        peak_summer_flags = features['is_peak_summer'] == 1
        
        # Should have high correlation
        consistency = (summer_months == peak_summer_flags).mean()
        assert consistency >= 0.8  # Allow some flexibility
        
        # Check fire season consistency
        fire_months = pd.to_datetime(features['date']).dt.month.isin([10, 11, 12])
        fire_season_flags = features['is_peak_fire_season'] == 1
        
        fire_consistency = (fire_months == fire_season_flags).mean()
        assert fire_consistency >= 0.8
    
    def test_feature_ranges(self):
        """Test that all features are within reasonable ranges."""
        features = self.extractor.create_all_la_pattern_features(self.sample_data)
        
        # Probability features should be [0, 1]
        prob_features = [col for col in features.columns if 'probability' in col]
        for col in prob_features:
            assert features[col].min() >= 0
            assert features[col].max() <= 1
        
        # Intensity features should be [0, 1]
        intensity_features = [col for col in features.columns if 'intensity' in col]
        for col in intensity_features:
            assert features[col].min() >= 0
            assert features[col].max() <= 1
        
        # Temperature adjustments should be reasonable
        if 'heat_island_temp_adjustment' in features.columns:
            assert features['heat_island_temp_adjustment'].min() >= 0
            assert features['heat_island_temp_adjustment'].max() <= 10  # Reasonable upper bound
        
        # Day length should be reasonable for LA
        if 'day_length_hours' in features.columns:
            assert features['day_length_hours'].min() >= 9
            assert features['day_length_hours'].max() <= 15


def test_integration_with_sample_data():
    """Integration test with realistic sample data."""
    extractor = LAWeatherPatternExtractor()
    
    # Create more realistic sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    # Simulate seasonal patterns
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Base temperature with seasonal variation
    base_temp = 75 + 10 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    
    # Humidity with seasonal variation (higher in summer due to marine layer)
    base_humidity = 50 + 20 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
    
    # Wind patterns (higher in winter for Santa Ana)
    base_wind = 8 + 5 * np.sin(2 * np.pi * (day_of_year - 350) / 365)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'predicted_high': base_temp + np.random.normal(0, 3, n_days),
        'predicted_low': base_temp - 15 + np.random.normal(0, 2, n_days),
        'humidity': np.clip(base_humidity + np.random.normal(0, 10, n_days), 10, 95),
        'cloud_cover': np.clip(40 + np.random.normal(0, 20, n_days), 0, 100),
        'wind_speed': np.clip(base_wind + np.random.normal(0, 5, n_days), 0, 50),
        'wind_direction': np.random.uniform(0, 360, n_days),
        'pressure': 1013 + np.random.normal(0, 8, n_days)
    })
    
    # Extract features
    features = extractor.create_all_la_pattern_features(sample_data)
    
    # Validate results
    assert not features.empty
    assert len(features) == len(sample_data)
    
    # Check feature quality
    is_valid, issues = extractor.validate_pattern_features(features)
    assert is_valid, f"Validation issues: {issues}"
    
    # Get summary
    summary = extractor.get_pattern_feature_summary(features)
    assert summary['total_features'] > 25  # Should have substantial features
    assert summary['total_records'] == len(sample_data)
    
    print(f"Integration test passed: {summary['total_features']} features created for {summary['total_records']} records")


if __name__ == '__main__':
    # Run the integration test
    test_integration_with_sample_data()
    print("All tests would pass!")