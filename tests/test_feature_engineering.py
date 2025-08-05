"""
Comprehensive unit tests for feature engineering components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.basic_features import BasicFeatureExtractor
from src.feature_engineering.ensemble_features import EnsembleFeatureExtractor
from src.feature_engineering.la_weather_patterns import LAWeatherPatternExtractor
from src.feature_engineering.feature_pipeline import FeaturePipeline


class TestBasicFeatureExtractor:
    """Test BasicFeatureExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = BasicFeatureExtractor()
        
        # Create comprehensive sample data
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19']),
            'forecast_date': pd.to_datetime(['2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18']),
            'predicted_high': [75.0, 78.0, 72.0, 76.0, 74.0],
            'predicted_low': [58.0, 62.0, 55.0, 59.0, 57.0],
            'humidity': [65.0, 70.0, 60.0, 68.0, 63.0],
            'pressure': [1013.2, 1015.1, 1011.8, 1014.0, 1012.5],
            'wind_speed': [8.5, 12.0, 6.2, 9.1, 7.8],
            'wind_direction': [225.0, 180.0, 270.0, 200.0, 190.0],
            'cloud_cover': [30.0, 45.0, 20.0, 35.0, 40.0],
            'precipitation_prob': [10.0, 25.0, 5.0, 15.0, 20.0],
            'data_quality_score': [1.0, 0.95, 1.0, 0.98, 0.97]
        })
    
    def test_extract_temperature_features(self):
        """Test temperature feature extraction."""
        features = self.extractor.extract_temperature_features(self.sample_data, 'test')
        
        # Check basic features
        assert 'test_temp_high' in features.columns
        assert 'test_temp_low' in features.columns
        assert 'test_temp_range' in features.columns
        assert 'test_temp_avg' in features.columns
        
        # Check derived features
        assert 'test_temp_high_change' in features.columns
        assert 'test_temp_high_3day_avg' in features.columns
        assert 'test_temp_high_7day_avg' in features.columns
        
        # Verify calculations
        assert features['test_temp_range'].iloc[0] == 17.0  # 75 - 58
        assert features['test_temp_avg'].iloc[0] == 66.5   # (75 + 58) / 2
        assert features['test_temp_high_change'].iloc[1] == 3.0  # 78 - 75
    
    def test_extract_atmospheric_features(self):
        """Test atmospheric feature extraction."""
        features = self.extractor.extract_atmospheric_features(self.sample_data, 'test')
        
        # Check basic features
        assert 'test_humidity' in features.columns
        assert 'test_pressure' in features.columns
        assert 'test_wind_speed' in features.columns
        assert 'test_wind_direction' in features.columns
        assert 'test_cloud_cover' in features.columns
        assert 'test_precipitation_prob' in features.columns
        
        # Check derived features
        assert 'test_wind_u' in features.columns
        assert 'test_wind_v' in features.columns
        assert 'test_pressure_change' in features.columns
        assert 'test_humidity_low' in features.columns
        assert 'test_humidity_high' in features.columns
        
        # Verify wind component calculations
        wind_speed = 8.5
        wind_dir = 225.0  # Southwest
        expected_u = wind_speed * np.cos(np.radians(wind_dir))
        expected_v = wind_speed * np.sin(np.radians(wind_dir))
        
        assert abs(features['test_wind_u'].iloc[0] - expected_u) < 0.01
        assert abs(features['test_wind_v'].iloc[0] - expected_v) < 0.01
    
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
        
        # Verify season assignments
        assert features['season'].iloc[0] == 'winter'  # January
        assert features['season'].iloc[1] == 'summer'  # July
        assert features['season'].iloc[2] == 'winter'  # December
    
    def test_extract_quality_features(self):
        """Test quality feature extraction."""
        features = self.extractor.extract_quality_features(self.sample_data, 'test')
        
        # Check quality features
        assert 'test_quality_score' in features.columns
        assert 'test_completeness' in features.columns
        
        # Check missing data indicators
        assert 'test_predicted_high_missing' in features.columns
        assert 'test_predicted_low_missing' in features.columns
        assert 'test_humidity_missing' in features.columns
        
        # Verify completeness (should be 1.0 for complete data)
        assert all(features['test_completeness'] == 1.0)
    
    def test_outlier_detection_iqr(self):
        """Test IQR-based outlier detection."""
        # Create data with outliers
        outlier_data = self.sample_data.copy()
        outlier_data.loc[0, 'predicted_high'] = 150.0  # Extreme outlier
        outlier_data.loc[1, 'humidity'] = 150.0  # Impossible humidity
        
        cleaned_data = self.extractor.detect_and_clean_outliers(
            outlier_data, 'test', method='iqr'
        )
        
        # Temperature outliers should be capped
        assert cleaned_data.loc[0, 'predicted_high'] < 150.0
        assert not pd.isna(cleaned_data.loc[0, 'predicted_high'])
        
        # Other outliers should be set to NaN
        assert pd.isna(cleaned_data.loc[1, 'humidity'])
        
        # Quality scores should be reduced
        assert cleaned_data.loc[0, 'data_quality_score'] < outlier_data.loc[0, 'data_quality_score']
    
    def test_outlier_detection_zscore(self):
        """Test Z-score based outlier detection."""
        outlier_data = self.sample_data.copy()
        outlier_data.loc[0, 'predicted_high'] = 150.0
        
        cleaned_data = self.extractor.detect_and_clean_outliers(
            outlier_data, 'test', method='zscore', threshold=2.0
        )
        
        # Should handle outliers
        assert cleaned_data.loc[0, 'predicted_high'] < 150.0
    
    def test_extract_source_features_complete(self):
        """Test complete source feature extraction."""
        features = self.extractor.extract_source_features(self.sample_data, 'test')
        
        # Should have date column
        assert 'date' in features.columns
        
        # Should have many features
        assert len(features.columns) > 15
        
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
            'openweather': self.sample_data.copy()
        }
        
        features = self.extractor.extract_all_source_features(source_data)
        
        # Should have date column
        assert 'date' in features.columns
        
        # Should have date-based features
        assert 'day_of_year' in features.columns
        assert 'season' in features.columns
        
        # Should have features from both sources
        nws_features = [col for col in features.columns if 'nws_' in col]
        openweather_features = [col for col in features.columns if 'openweather_' in col]
        
        assert len(nws_features) > 0
        assert len(openweather_features) > 0
    
    def test_feature_validation(self):
        """Test feature validation."""
        features = self.extractor.extract_source_features(self.sample_data, 'test')
        
        is_valid, issues = self.extractor.validate_features(features)
        assert is_valid
        assert len(issues) == 0
        
        # Test with invalid features
        bad_features = features.drop('date', axis=1)
        is_valid, issues = self.extractor.validate_features(bad_features)
        assert not is_valid
        assert len(issues) > 0
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        features = self.extractor.extract_source_features(empty_data, 'test')
        assert features.empty
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        partial_data = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-16']),
            'predicted_high': [75.0, 78.0]
        })
        
        features = self.extractor.extract_source_features(partial_data, 'test')
        
        # Should extract available features
        assert not features.empty
        assert 'test_temp_high' in features.columns
        # Should not have features for missing columns
        assert 'test_humidity' not in features.columns


class TestEnsembleFeatureExtractor:
    """Test EnsembleFeatureExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EnsembleFeatureExtractor()
        
        # Create sample multi-source data
        dates = pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17'])
        
        self.source_data = {
            'nws': pd.DataFrame({
                'date': dates,
                'predicted_high': [75.0, 78.0, 72.0],
                'predicted_low': [58.0, 62.0, 55.0],
                'humidity': [65.0, 70.0, 60.0]
            }),
            'openweather': pd.DataFrame({
                'date': dates,
                'predicted_high': [76.0, 77.0, 73.0],
                'predicted_low': [59.0, 61.0, 56.0],
                'humidity': [63.0, 72.0, 58.0]
            }),
            'tomorrow': pd.DataFrame({
                'date': dates,
                'predicted_high': [74.0, 79.0, 71.0],
                'predicted_low': [57.0, 63.0, 54.0],
                'humidity': [67.0, 68.0, 62.0]
            })
        }
    
    def test_create_consensus_features(self):
        """Test consensus feature creation."""
        features = self.extractor.create_consensus_features(self.source_data)
        
        # Check consensus features
        assert 'consensus_high_mean' in features.columns
        assert 'consensus_high_median' in features.columns
        assert 'consensus_high_std' in features.columns
        assert 'consensus_high_min' in features.columns
        assert 'consensus_high_max' in features.columns
        
        # Check low temperature consensus
        assert 'consensus_low_mean' in features.columns
        assert 'consensus_low_std' in features.columns
        
        # Verify calculations
        expected_mean = (75.0 + 76.0 + 74.0) / 3  # First row
        assert abs(features['consensus_high_mean'].iloc[0] - expected_mean) < 0.01
    
    def test_create_agreement_features(self):
        """Test agreement feature creation."""
        features = self.extractor.create_agreement_features(self.source_data)
        
        # Check agreement features
        assert 'high_temp_agreement' in features.columns
        assert 'low_temp_agreement' in features.columns
        assert 'humidity_agreement' in features.columns
        assert 'overall_agreement' in features.columns
        
        # Check source count
        assert 'active_sources_count' in features.columns
        
        # Verify source count
        assert all(features['active_sources_count'] == 3)
    
    def test_create_divergence_features(self):
        """Test divergence feature creation."""
        features = self.extractor.create_divergence_features(self.source_data)
        
        # Check divergence features
        assert 'high_temp_range' in features.columns
        assert 'high_temp_iqr' in features.columns
        assert 'high_temp_cv' in features.columns
        
        # Check outlier detection
        assert 'high_temp_outlier_count' in features.columns
        
        # Verify range calculation
        expected_range = 76.0 - 74.0  # max - min for first row
        assert abs(features['high_temp_range'].iloc[0] - expected_range) < 0.01
    
    def test_create_confidence_features(self):
        """Test confidence feature creation."""
        features = self.extractor.create_confidence_features(self.source_data)
        
        # Check confidence features
        assert 'prediction_confidence' in features.columns
        assert 'data_reliability' in features.columns
        assert 'forecast_stability' in features.columns
        
        # Confidence should be between 0 and 1
        assert all(features['prediction_confidence'] >= 0)
        assert all(features['prediction_confidence'] <= 1)
    
    def test_create_weighted_consensus(self):
        """Test weighted consensus creation."""
        # Create source weights
        source_weights = {'nws': 0.4, 'openweather': 0.35, 'tomorrow': 0.25}
        
        features = self.extractor.create_weighted_consensus(self.source_data, source_weights)
        
        # Check weighted features
        assert 'weighted_high_mean' in features.columns
        assert 'weighted_low_mean' in features.columns
        assert 'weighted_humidity_mean' in features.columns
        
        # Verify weighted calculation
        expected_weighted = (75.0 * 0.4 + 76.0 * 0.35 + 74.0 * 0.25)
        assert abs(features['weighted_high_mean'].iloc[0] - expected_weighted) < 0.01
    
    def test_create_all_ensemble_features(self):
        """Test creating all ensemble features."""
        features = self.extractor.create_all_ensemble_features(self.source_data)
        
        # Should have date column
        assert 'date' in features.columns
        
        # Should have features from all categories
        consensus_features = [col for col in features.columns if 'consensus' in col]
        agreement_features = [col for col in features.columns if 'agreement' in col]
        divergence_features = [col for col in features.columns if any(x in col for x in ['range', 'iqr', 'cv'])]
        confidence_features = [col for col in features.columns if 'confidence' in col]
        
        assert len(consensus_features) > 0
        assert len(agreement_features) > 0
        assert len(divergence_features) > 0
        assert len(confidence_features) > 0
    
    def test_empty_source_data_handling(self):
        """Test handling of empty source data."""
        empty_data = {}
        
        features = self.extractor.create_all_ensemble_features(empty_data)
        assert features.empty
    
    def test_single_source_handling(self):
        """Test handling of single source data."""
        single_source = {'nws': self.source_data['nws']}
        
        features = self.extractor.create_all_ensemble_features(single_source)
        
        # Should still create features, but with different values
        assert not features.empty
        assert 'consensus_high_mean' in features.columns
        assert all(features['active_sources_count'] == 1)


class TestLAWeatherPatternExtractor:
    """Test LAWeatherPatternExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = LAWeatherPatternExtractor()
        
        # Create sample data with LA-specific patterns
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-07-15', '2025-10-15']),
            'predicted_high': [75.0, 85.0, 90.0],
            'predicted_low': [58.0, 68.0, 65.0],
            'humidity': [80.0, 85.0, 30.0],  # High humidity for marine layer, low for Santa Ana
            'pressure': [1013.2, 1015.1, 1025.0],  # High pressure for Santa Ana
            'wind_speed': [5.0, 3.0, 25.0],  # Low for marine layer, high for Santa Ana
            'wind_direction': [270.0, 270.0, 45.0],  # West for marine layer, NE for Santa Ana
            'cloud_cover': [90.0, 95.0, 10.0],  # High for marine layer, low for Santa Ana
            'temperature_range': [17.0, 17.0, 25.0]  # Normal, normal, high for Santa Ana
        })
    
    def test_detect_marine_layer(self):
        """Test marine layer detection."""
        features = self.extractor.detect_marine_layer(self.sample_data)
        
        # Check marine layer features
        assert 'marine_layer_probability' in features.columns
        assert 'marine_layer_strength' in features.columns
        assert 'marine_layer_indicator' in features.columns
        
        # July should have high marine layer probability
        assert features['marine_layer_probability'].iloc[1] > 0.5
        
        # October (Santa Ana season) should have low marine layer probability
        assert features['marine_layer_probability'].iloc[2] < 0.3
    
    def test_detect_santa_ana_winds(self):
        """Test Santa Ana wind detection."""
        features = self.extractor.detect_santa_ana_winds(self.sample_data)
        
        # Check Santa Ana features
        assert 'santa_ana_probability' in features.columns
        assert 'santa_ana_strength' in features.columns
        assert 'santa_ana_indicator' in features.columns
        
        # October with high pressure, NE winds, low humidity should indicate Santa Ana
        assert features['santa_ana_probability'].iloc[2] > 0.7
        
        # Summer with marine layer conditions should have low Santa Ana probability
        assert features['santa_ana_probability'].iloc[1] < 0.3
    
    def test_detect_heat_waves(self):
        """Test heat wave detection."""
        # Create data with heat wave pattern
        heat_wave_data = pd.DataFrame({
            'date': pd.to_datetime(['2025-07-15', '2025-07-16', '2025-07-17']),
            'predicted_high': [95.0, 98.0, 102.0],  # Increasing high temps
            'predicted_low': [75.0, 78.0, 80.0],
            'humidity': [25.0, 20.0, 15.0],  # Low humidity
            'pressure': [1020.0, 1022.0, 1025.0],  # High pressure
            'wind_speed': [2.0, 1.0, 0.5],  # Light winds
            'temperature_range': [20.0, 20.0, 22.0]
        })
        
        features = self.extractor.detect_heat_waves(heat_wave_data)
        
        # Check heat wave features
        assert 'heat_wave_probability' in features.columns
        assert 'heat_wave_intensity' in features.columns
        assert 'heat_wave_indicator' in features.columns
        
        # Should detect increasing heat wave probability
        assert features['heat_wave_probability'].iloc[2] > features['heat_wave_probability'].iloc[0]
    
    def test_calculate_heat_island_effect(self):
        """Test heat island effect calculation."""
        features = self.extractor.calculate_heat_island_effect(self.sample_data)
        
        # Check heat island features
        assert 'heat_island_intensity' in features.columns
        assert 'urban_heat_factor' in features.columns
        
        # Heat island should be stronger in summer
        summer_heat_island = features['heat_island_intensity'].iloc[1]
        winter_heat_island = features['heat_island_intensity'].iloc[0]
        assert summer_heat_island >= winter_heat_island
    
    def test_create_seasonal_adjustments(self):
        """Test seasonal adjustment creation."""
        features = self.extractor.create_seasonal_adjustments(self.sample_data)
        
        # Check seasonal features
        assert 'seasonal_temp_adjustment' in features.columns
        assert 'seasonal_humidity_adjustment' in features.columns
        assert 'seasonal_pattern_strength' in features.columns
        
        # Should have different adjustments for different seasons
        winter_adj = features['seasonal_temp_adjustment'].iloc[0]
        summer_adj = features['seasonal_temp_adjustment'].iloc[1]
        assert winter_adj != summer_adj
    
    def test_create_all_la_patterns(self):
        """Test creating all LA weather patterns."""
        features = self.extractor.create_all_la_patterns(self.sample_data)
        
        # Should have date column
        assert 'date' in features.columns
        
        # Should have features from all pattern types
        marine_features = [col for col in features.columns if 'marine_layer' in col]
        santa_ana_features = [col for col in features.columns if 'santa_ana' in col]
        heat_wave_features = [col for col in features.columns if 'heat_wave' in col]
        heat_island_features = [col for col in features.columns if 'heat_island' in col]
        seasonal_features = [col for col in features.columns if 'seasonal' in col]
        
        assert len(marine_features) > 0
        assert len(santa_ana_features) > 0
        assert len(heat_wave_features) > 0
        assert len(heat_island_features) > 0
        assert len(seasonal_features) > 0
    
    def test_pattern_interaction_detection(self):
        """Test detection of pattern interactions."""
        features = self.extractor.create_all_la_patterns(self.sample_data)
        
        # Marine layer and Santa Ana should be inversely related
        marine_prob = features['marine_layer_probability']
        santa_ana_prob = features['santa_ana_probability']
        
        # Where marine layer is high, Santa Ana should be low
        high_marine_idx = marine_prob > 0.7
        if high_marine_idx.any():
            assert all(santa_ana_prob[high_marine_idx] < 0.5)


class TestFeaturePipeline:
    """Test FeaturePipeline integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = FeaturePipeline()
        
        # Create comprehensive test data
        dates = pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17'])
        
        self.source_data = {
            'nws': pd.DataFrame({
                'date': dates,
                'predicted_high': [75.0, 78.0, 72.0],
                'predicted_low': [58.0, 62.0, 55.0],
                'humidity': [65.0, 70.0, 60.0],
                'pressure': [1013.2, 1015.1, 1011.8],
                'wind_speed': [8.5, 12.0, 6.2],
                'wind_direction': [225.0, 180.0, 270.0],
                'cloud_cover': [30.0, 45.0, 20.0],
                'precipitation_prob': [10.0, 25.0, 5.0],
                'data_quality_score': [1.0, 0.95, 1.0]
            }),
            'openweather': pd.DataFrame({
                'date': dates,
                'predicted_high': [76.0, 77.0, 73.0],
                'predicted_low': [59.0, 61.0, 56.0],
                'humidity': [63.0, 72.0, 58.0],
                'pressure': [1014.0, 1016.0, 1012.0],
                'wind_speed': [9.0, 11.0, 7.0],
                'wind_direction': [230.0, 185.0, 275.0],
                'cloud_cover': [35.0, 50.0, 25.0],
                'precipitation_prob': [12.0, 28.0, 8.0],
                'data_quality_score': [0.98, 0.97, 0.99]
            })
        }
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, 'basic_extractor')
        assert hasattr(self.pipeline, 'ensemble_extractor')
        assert hasattr(self.pipeline, 'la_pattern_extractor')
    
    def test_create_basic_features(self):
        """Test basic feature creation through pipeline."""
        features = self.pipeline.create_basic_features(self.source_data)
        
        # Should have date column
        assert 'date' in features.columns
        
        # Should have features from all sources
        nws_features = [col for col in features.columns if 'nws_' in col]
        openweather_features = [col for col in features.columns if 'openweather_' in col]
        
        assert len(nws_features) > 0
        assert len(openweather_features) > 0
        
        # Should have date-based features
        assert 'day_of_year' in features.columns
        assert 'season' in features.columns
    
    def test_create_ensemble_features(self):
        """Test ensemble feature creation through pipeline."""
        features = self.pipeline.create_ensemble_features(self.source_data)
        
        # Should have consensus features
        assert 'consensus_high_mean' in features.columns
        assert 'consensus_high_std' in features.columns
        
        # Should have agreement features
        assert 'high_temp_agreement' in features.columns
        assert 'overall_agreement' in features.columns
    
    def test_create_la_pattern_features(self):
        """Test LA pattern feature creation through pipeline."""
        features = self.pipeline.create_la_pattern_features(self.source_data)
        
        # Should have LA-specific features
        assert 'marine_layer_probability' in features.columns
        assert 'santa_ana_probability' in features.columns
        assert 'heat_island_intensity' in features.columns
    
    def test_create_complete_features(self):
        """Test complete feature creation."""
        features = self.pipeline.create_complete_features(
            self.source_data,
            include_ensemble=True,
            include_la_patterns=True
        )
        
        # Should have date column
        assert 'date' in features.columns
        
        # Should have features from all categories
        basic_features = [col for col in features.columns if any(src in col for src in ['nws_', 'openweather_'])]
        ensemble_features = [col for col in features.columns if 'consensus' in col]
        la_features = [col for col in features.columns if any(x in col for x in ['marine_layer', 'santa_ana', 'heat_island'])]
        
        assert len(basic_features) > 0
        assert len(ensemble_features) > 0
        assert len(la_features) > 0
        
        # Should have many features total
        assert len(features.columns) > 50
    
    def test_create_complete_features_selective(self):
        """Test selective feature creation."""
        # Test with only basic features
        basic_only = self.pipeline.create_complete_features(
            self.source_data,
            include_ensemble=False,
            include_la_patterns=False
        )
        
        # Should not have ensemble or LA features
        ensemble_features = [col for col in basic_only.columns if 'consensus' in col]
        la_features = [col for col in basic_only.columns if 'marine_layer' in col]
        
        assert len(ensemble_features) == 0
        assert len(la_features) == 0
        
        # Test with ensemble but no LA patterns
        ensemble_only = self.pipeline.create_complete_features(
            self.source_data,
            include_ensemble=True,
            include_la_patterns=False
        )
        
        # Should have ensemble but not LA features
        ensemble_features = [col for col in ensemble_only.columns if 'consensus' in col]
        la_features = [col for col in ensemble_only.columns if 'marine_layer' in col]
        
        assert len(ensemble_features) > 0
        assert len(la_features) == 0
    
    def test_feature_summary(self):
        """Test feature summary generation."""
        features = self.pipeline.create_complete_features(self.source_data)
        summary = self.pipeline.get_feature_summary(features)
        
        # Check summary structure
        assert 'total_features' in summary
        assert 'total_records' in summary
        assert 'feature_categories' in summary
        assert 'missing_data_percentage' in summary
        
        # Check values
        assert summary['total_records'] == 3
        assert summary['total_features'] > 0
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = {}
        
        features = self.pipeline.create_complete_features(empty_data)
        assert features.empty
    
    def test_single_source_handling(self):
        """Test handling of single source."""
        single_source = {'nws': self.source_data['nws']}
        
        features = self.pipeline.create_complete_features(single_source)
        
        # Should still create features
        assert not features.empty
        assert 'date' in features.columns
        
        # Should have basic features
        nws_features = [col for col in features.columns if 'nws_' in col]
        assert len(nws_features) > 0


def run_feature_engineering_tests():
    """Run all feature engineering tests."""
    print("Running feature engineering tests...")
    
    # Test BasicFeatureExtractor
    print("✓ Testing BasicFeatureExtractor...")
    extractor = BasicFeatureExtractor()
    sample_data = pd.DataFrame({
        'date': pd.to_datetime(['2025-01-15', '2025-01-16']),
        'predicted_high': [75.0, 78.0],
        'predicted_low': [58.0, 62.0],
        'humidity': [65.0, 70.0]
    })
    features = extractor.extract_source_features(sample_data, 'test')
    assert not features.empty
    
    # Test EnsembleFeatureExtractor
    print("✓ Testing EnsembleFeatureExtractor...")
    ensemble_extractor = EnsembleFeatureExtractor()
    source_data = {
        'nws': sample_data,
        'openweather': sample_data.copy()
    }
    ensemble_features = ensemble_extractor.create_all_ensemble_features(source_data)
    assert not ensemble_features.empty
    
    # Test LAWeatherPatternExtractor
    print("✓ Testing LAWeatherPatternExtractor...")
    la_extractor = LAWeatherPatternExtractor()
    la_data = sample_data.copy()
    la_data['pressure'] = [1013.2, 1015.1]
    la_data['wind_speed'] = [8.5, 12.0]
    la_data['wind_direction'] = [225.0, 180.0]
    la_data['cloud_cover'] = [30.0, 45.0]
    la_features = la_extractor.create_all_la_patterns(la_data)
    assert not la_features.empty
    
    # Test FeaturePipeline
    print("✓ Testing FeaturePipeline...")
    pipeline = FeaturePipeline()
    complete_features = pipeline.create_complete_features(source_data)
    assert not complete_features.empty
    assert len(complete_features.columns) > 20
    
    print("All feature engineering tests passed!")


if __name__ == '__main__':
    run_feature_engineering_tests()