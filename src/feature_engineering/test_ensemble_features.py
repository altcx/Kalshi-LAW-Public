"""Test ensemble feature extraction."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from src.feature_engineering.ensemble_features import EnsembleFeatureExtractor


class TestEnsembleFeatureExtractor:
    """Test cases for ensemble feature extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EnsembleFeatureExtractor()
        
        # Create sample data for testing
        dates = [date.today() - timedelta(days=i) for i in range(10, 0, -1)]
        
        self.sample_data = {
            'nws': pd.DataFrame({
                'date': dates,
                'forecast_date': [d - timedelta(days=1) for d in dates],
                'predicted_high': [75 + i for i in range(10)],
                'predicted_low': [55 + i for i in range(10)],
                'humidity': [60 + i*2 for i in range(10)],
                'pressure': [1013 + i for i in range(10)],
                'wind_speed': [10 + i*0.5 for i in range(10)],
                'cloud_cover': [30 + i*3 for i in range(10)],
                'precipitation_prob': [20 + i*2 for i in range(10)],
                'data_quality_score': [0.9 + i*0.01 for i in range(10)]
            }),
            'openweather': pd.DataFrame({
                'date': dates,
                'forecast_date': [d - timedelta(days=1) for d in dates],
                'predicted_high': [76 + i for i in range(10)],  # Slightly different values
                'predicted_low': [56 + i for i in range(10)],
                'humidity': [62 + i*2 for i in range(10)],
                'pressure': [1014 + i for i in range(10)],
                'wind_speed': [11 + i*0.5 for i in range(10)],
                'cloud_cover': [32 + i*3 for i in range(10)],
                'precipitation_prob': [22 + i*2 for i in range(10)],
                'data_quality_score': [0.85 + i*0.01 for i in range(10)]
            }),
            'tomorrow': pd.DataFrame({
                'date': dates,
                'forecast_date': [d - timedelta(days=1) for d in dates],
                'predicted_high': [74 + i for i in range(10)],  # Different values for variety
                'predicted_low': [54 + i for i in range(10)],
                'humidity': [58 + i*2 for i in range(10)],
                'pressure': [1012 + i for i in range(10)],
                'wind_speed': [9 + i*0.5 for i in range(10)],
                'cloud_cover': [28 + i*3 for i in range(10)],
                'precipitation_prob': [18 + i*2 for i in range(10)],
                'data_quality_score': [0.88 + i*0.01 for i in range(10)]
            })
        }
    
    def test_create_consensus_features(self):
        """Test consensus feature creation."""
        consensus_features = self.extractor.create_consensus_features(self.sample_data)
        
        assert not consensus_features.empty
        assert 'date' in consensus_features.columns
        assert len(consensus_features) == 10
        
        # Check that consensus features are created for key parameters
        assert 'predicted_high_consensus_mean' in consensus_features.columns
        assert 'predicted_high_consensus_median' in consensus_features.columns
        assert 'predicted_high_consensus_std' in consensus_features.columns
        assert 'predicted_high_source_count' in consensus_features.columns
        
        # Verify consensus calculations are reasonable
        first_row = consensus_features.iloc[0]
        
        # For first date, predicted_high values are [75, 76, 74] from the three sources
        expected_mean = (75 + 76 + 74) / 3
        assert abs(first_row['predicted_high_consensus_mean'] - expected_mean) < 0.01
        
        expected_median = 75  # Median of [74, 75, 76]
        assert abs(first_row['predicted_high_consensus_median'] - expected_median) < 0.01
        
        assert first_row['predicted_high_source_count'] == 3
    
    def test_create_agreement_metrics(self):
        """Test agreement metrics creation."""
        agreement_features = self.extractor.create_agreement_metrics(self.sample_data)
        
        assert not agreement_features.empty
        assert 'date' in agreement_features.columns
        assert len(agreement_features) == 10
        
        # Check that agreement metrics are created
        assert 'predicted_high_agreement_cv' in agreement_features.columns
        assert 'predicted_high_agreement_score' in agreement_features.columns
        assert 'predicted_high_pairwise_diff' in agreement_features.columns
        assert 'overall_agreement_score' in agreement_features.columns
        assert 'total_active_sources' in agreement_features.columns
        
        # Verify agreement scores are in reasonable range
        first_row = agreement_features.iloc[0]
        assert 0 <= first_row['predicted_high_agreement_score'] <= 1
        assert first_row['predicted_high_agreement_sources'] == 3
    
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        rolling_features = self.extractor.create_rolling_features(self.sample_data)
        
        assert not rolling_features.empty
        assert 'date' in rolling_features.columns
        assert len(rolling_features) == 10
        
        # Check that rolling features are created
        assert 'predicted_high_rolling_3d_mean' in rolling_features.columns
        assert 'predicted_high_rolling_7d_mean' in rolling_features.columns
        assert 'predicted_high_rolling_3d_trend' in rolling_features.columns
        assert 'predicted_high_rolling_3d_momentum' in rolling_features.columns
        
        # Verify rolling calculations make sense
        # The 3-day rolling mean should be available from the 3rd row onwards
        third_row = rolling_features.iloc[2]
        assert pd.notna(third_row['predicted_high_rolling_3d_mean'])
    
    def test_create_trend_features(self):
        """Test trend feature creation."""
        trend_features = self.extractor.create_trend_features(self.sample_data)
        
        assert not trend_features.empty
        assert 'date' in trend_features.columns
        
        # Check that trend features are created
        assert 'predicted_high_trend_3d' in trend_features.columns
        assert 'predicted_high_trend_acceleration' in trend_features.columns
        assert 'predicted_high_trend_reversal' in trend_features.columns
        
        # Verify trend calculations
        # Since our test data has increasing temperatures, trend should be positive
        non_nan_trends = trend_features['predicted_high_trend_3d'].dropna()
        if len(non_nan_trends) > 0:
            assert non_nan_trends.iloc[-1] > 0  # Last trend should be positive
    
    def test_create_source_reliability_features(self):
        """Test source reliability feature creation."""
        reliability_features = self.extractor.create_source_reliability_features(self.sample_data)
        
        assert not reliability_features.empty
        assert 'date' in reliability_features.columns
        assert len(reliability_features) == 10
        
        # Check that reliability features are created for each source
        for source in ['nws', 'openweather', 'tomorrow']:
            assert f'{source}_completeness' in reliability_features.columns
            assert f'{source}_quality_score' in reliability_features.columns
            assert f'{source}_freshness' in reliability_features.columns
            assert f'{source}_available' in reliability_features.columns
        
        # Check aggregate metrics
        assert 'total_sources_available' in reliability_features.columns
        assert 'average_source_quality' in reliability_features.columns
        assert 'source_diversity_score' in reliability_features.columns
        
        # Verify reliability calculations
        first_row = reliability_features.iloc[0]
        assert first_row['total_sources_available'] == 3  # All three sources available
        assert 0 <= first_row['source_diversity_score'] <= 1
    
    def test_create_all_ensemble_features(self):
        """Test creation of all ensemble features together."""
        all_features = self.extractor.create_all_ensemble_features(self.sample_data)
        
        assert not all_features.empty
        assert 'date' in all_features.columns
        assert len(all_features) == 10
        
        # Should have features from all categories
        feature_names = all_features.columns.tolist()
        
        # Consensus features
        consensus_features = [col for col in feature_names if 'consensus' in col]
        assert len(consensus_features) > 0
        
        # Agreement features
        agreement_features = [col for col in feature_names if 'agreement' in col]
        assert len(agreement_features) > 0
        
        # Rolling features
        rolling_features = [col for col in feature_names if 'rolling' in col]
        assert len(rolling_features) > 0
        
        # Trend features
        trend_features = [col for col in feature_names if 'trend' in col]
        assert len(trend_features) > 0
        
        # Reliability features
        reliability_features = [col for col in feature_names if any(x in col for x in ['completeness', 'quality', 'available'])]
        assert len(reliability_features) > 0
        
        print(f"Created {len(all_features.columns)-1} total ensemble features")
    
    def test_get_ensemble_feature_summary(self):
        """Test ensemble feature summary."""
        all_features = self.extractor.create_all_ensemble_features(self.sample_data)
        summary = self.extractor.get_ensemble_feature_summary(all_features)
        
        assert 'total_features' in summary
        assert 'total_records' in summary
        assert 'feature_categories' in summary
        assert 'date_range' in summary
        
        assert summary['total_records'] == 10
        assert summary['total_features'] > 0
        
        # Check feature categories
        categories = summary['feature_categories']
        assert categories['consensus'] > 0
        assert categories['agreement'] > 0
        assert categories['rolling'] > 0
        assert categories['trend'] > 0
        assert categories['reliability'] > 0
        
        print(f"Feature summary: {summary}")
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = {'nws': pd.DataFrame(), 'openweather': pd.DataFrame()}
        
        consensus_features = self.extractor.create_consensus_features(empty_data)
        assert consensus_features.empty
        
        agreement_features = self.extractor.create_agreement_metrics(empty_data)
        assert agreement_features.empty
        
        all_features = self.extractor.create_all_ensemble_features(empty_data)
        assert all_features.empty
    
    def test_single_source_handling(self):
        """Test handling of single source data."""
        single_source_data = {'nws': self.sample_data['nws']}
        
        # Consensus features should still work with single source
        consensus_features = self.extractor.create_consensus_features(single_source_data)
        assert not consensus_features.empty
        assert 'predicted_high_consensus_mean' in consensus_features.columns
        
        # Agreement metrics should handle single source gracefully
        agreement_features = self.extractor.create_agreement_metrics(single_source_data)
        assert not agreement_features.empty
        # With single source, agreement scores should be NaN or default values
        assert 'predicted_high_agreement_sources' in agreement_features.columns
    
    def test_missing_data_handling(self):
        """Test handling of missing data in sources."""
        # Create data with some missing values
        incomplete_data = self.sample_data.copy()
        incomplete_data['nws'].loc[0, 'predicted_high'] = np.nan
        incomplete_data['openweather'].loc[1, 'humidity'] = np.nan
        
        all_features = self.extractor.create_all_ensemble_features(incomplete_data)
        
        assert not all_features.empty
        # Should handle missing data gracefully
        assert len(all_features) == 10


def main():
    """Run ensemble feature tests."""
    print("=== Testing Ensemble Feature Extraction ===\n")
    
    # Create test instance
    test_instance = TestEnsembleFeatureExtractor()
    test_instance.setup_method()
    
    # Run tests
    tests = [
        ('Consensus Features', test_instance.test_create_consensus_features),
        ('Agreement Metrics', test_instance.test_create_agreement_metrics),
        ('Rolling Features', test_instance.test_create_rolling_features),
        ('Trend Features', test_instance.test_create_trend_features),
        ('Source Reliability', test_instance.test_create_source_reliability_features),
        ('All Ensemble Features', test_instance.test_create_all_ensemble_features),
        ('Feature Summary', test_instance.test_get_ensemble_feature_summary),
        ('Empty Data Handling', test_instance.test_empty_data_handling),
        ('Single Source Handling', test_instance.test_single_source_handling),
        ('Missing Data Handling', test_instance.test_missing_data_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running {test_name}...")
            test_func()
            print(f"✓ {test_name} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1
        print()
    
    print(f"=== Test Results: {passed} passed, {failed} failed ===")
    
    if failed == 0:
        print("\n=== Demonstrating Ensemble Features ===")
        
        # Create and show sample ensemble features
        extractor = EnsembleFeatureExtractor()
        all_features = extractor.create_all_ensemble_features(test_instance.sample_data)
        
        print(f"Created {len(all_features.columns)-1} ensemble features:")
        print(f"- Data shape: {all_features.shape}")
        print(f"- Date range: {all_features['date'].min()} to {all_features['date'].max()}")
        
        # Show feature categories
        summary = extractor.get_ensemble_feature_summary(all_features)
        print(f"\nFeature categories:")
        for category, count in summary['feature_categories'].items():
            print(f"- {category}: {count} features")
        
        # Show sample of features
        print(f"\nSample features (first 5 columns):")
        print(all_features.iloc[:3, :6].to_string())


if __name__ == '__main__':
    main()