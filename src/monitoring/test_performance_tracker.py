"""Tests for the performance tracking system."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from .performance_tracker import PerformanceTracker
from ..utils.data_manager import DataManager


class TestPerformanceTracker:
    """Test cases for PerformanceTracker."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_data_manager(self, temp_data_dir):
        """Create mock DataManager with test data."""
        # Create a mock DataManager
        data_manager = DataManager()
        data_manager.data_dir = temp_data_dir
        
        # Update file paths to use temp directory
        for key in data_manager.data_files:
            data_manager.data_files[key] = temp_data_dir / f"{key}_data.parquet"
        
        return data_manager
    
    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather forecast data."""
        # Use recent dates so they fall within the tracking window
        end_date = date.today()
        start_date = end_date - timedelta(days=14)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for i, date_val in enumerate(dates):
            # Create realistic temperature predictions with some variation
            base_temp = 70 + 10 * np.sin(i * 0.2)  # Seasonal variation
            noise = np.random.normal(0, 2)  # Random noise
            
            data.append({
                'date': date_val,
                'forecast_date': date_val - timedelta(days=1),
                'predicted_high': base_temp + noise,
                'predicted_low': base_temp - 15 + noise,
                'humidity': 60 + np.random.normal(0, 10),
                'pressure': 1013 + np.random.normal(0, 5),
                'wind_speed': 8 + np.random.normal(0, 3),
                'wind_direction': np.random.uniform(0, 360),
                'cloud_cover': np.random.uniform(0, 100),
                'precipitation_prob': np.random.uniform(0, 30),
                'data_quality_score': 0.9 + np.random.normal(0, 0.05)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_actual_temps(self):
        """Create sample actual temperature data."""
        # Use recent dates so they fall within the tracking window
        end_date = date.today()
        start_date = end_date - timedelta(days=14)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for i, date_val in enumerate(dates):
            # Create actual temperatures that are close to but not identical to predictions
            base_temp = 70 + 10 * np.sin(i * 0.2)
            actual_noise = np.random.normal(0, 1.5)  # Slightly less noise than predictions
            
            data.append({
                'date': date_val,
                'actual_high': base_temp + actual_noise,
                'actual_low': base_temp - 15 + actual_noise,
                'source': 'NOAA'
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def performance_tracker(self, mock_data_manager):
        """Create PerformanceTracker instance with mock data."""
        return PerformanceTracker(mock_data_manager)
    
    def test_calculate_accuracy_metrics(self, performance_tracker):
        """Test accuracy metrics calculation."""
        # Create test data
        predictions = pd.Series([70.0, 75.0, 80.0, 72.0, 68.0])
        actuals = pd.Series([71.0, 74.0, 82.0, 70.0, 69.0])
        
        metrics = performance_tracker.calculate_accuracy_metrics(predictions, actuals)
        
        # Check that all expected metrics are present
        expected_metrics = ['count', 'mae', 'rmse', 'mape', 'bias', 'std_error',
                          'accuracy_1f', 'accuracy_2f', 'accuracy_3f', 'accuracy_5f',
                          'max_error', 'min_error', 'median_error']
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check specific values
        assert metrics['count'] == 5
        assert abs(metrics['mae'] - 1.4) < 0.1  # Mean absolute error should be ~1.4
        assert metrics['accuracy_3f'] == 1.0  # All predictions within 3°F
        assert metrics['accuracy_1f'] == 0.6  # 3 out of 5 within 1°F (errors: 1, 1, 2, 2, 1)
    
    def test_calculate_accuracy_metrics_with_nans(self, performance_tracker):
        """Test accuracy metrics calculation with NaN values."""
        predictions = pd.Series([70.0, np.nan, 80.0, 72.0, np.nan])
        actuals = pd.Series([71.0, 74.0, np.nan, 70.0, 69.0])
        
        metrics = performance_tracker.calculate_accuracy_metrics(predictions, actuals)
        
        # Should only count valid pairs (70,71) and (72,70)
        assert metrics['count'] == 2
        assert abs(metrics['mae'] - 1.5) < 0.1  # (1 + 2) / 2 = 1.5
    
    def test_calculate_accuracy_metrics_empty_data(self, performance_tracker):
        """Test accuracy metrics with empty data."""
        predictions = pd.Series([])
        actuals = pd.Series([])
        
        metrics = performance_tracker.calculate_accuracy_metrics(predictions, actuals)
        assert metrics == {}
    
    def test_track_source_performance(self, performance_tracker, sample_weather_data, sample_actual_temps):
        """Test tracking performance for a specific source."""
        # Store sample data
        performance_tracker.data_manager.save_source_data('nws', sample_weather_data)
        performance_tracker.data_manager.save_source_data('actual_temperatures', sample_actual_temps)
        
        # Track performance
        metrics = performance_tracker.track_source_performance('nws', window_days=30)
        
        # Check that metrics were calculated
        assert 'error' not in metrics
        assert metrics['source'] == 'nws'
        assert metrics['count'] > 0
        assert 'mae' in metrics
        assert 'accuracy_3f' in metrics
        assert 'performance_category' in metrics
    
    def test_track_source_performance_no_data(self, performance_tracker):
        """Test tracking performance when no data is available."""
        metrics = performance_tracker.track_source_performance('nonexistent_source')
        assert 'error' in metrics
    
    def test_calculate_source_weights(self, performance_tracker):
        """Test source weight calculation."""
        # Create mock performance data
        performance_data = {
            'nws': {'accuracy_3f': 0.85, 'data_quality_avg': 0.9, 'count': 15},
            'openweather': {'accuracy_3f': 0.80, 'data_quality_avg': 0.85, 'count': 12},
            'tomorrow': {'accuracy_3f': 0.75, 'data_quality_avg': 0.8, 'count': 10},
            'weatherbit': {'accuracy_3f': 0.70, 'data_quality_avg': 0.75, 'count': 8},
            'visual_crossing': {'accuracy_3f': 0.65, 'data_quality_avg': 0.7, 'count': 5}
        }
        
        weights = performance_tracker.calculate_source_weights(performance_data)
        
        # Check that weights sum to approximately 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Check that better performing sources get higher weights
        assert weights['nws'] > weights['visual_crossing']
        
        # Check that all weights are within bounds
        for weight in weights.values():
            assert performance_tracker.weight_adjustment['min_weight'] <= weight <= performance_tracker.weight_adjustment['max_weight']
    
    def test_calculate_source_weights_empty_data(self, performance_tracker):
        """Test source weight calculation with empty data."""
        weights = performance_tracker.calculate_source_weights({})
        assert weights == {}
    
    def test_store_performance_metrics(self, performance_tracker):
        """Test storing performance metrics."""
        metrics = {
            'source': 'test_source',
            'window_days': 30,
            'count': 10,
            'mae': 1.5,
            'rmse': 2.0,
            'accuracy_3f': 0.8,
            'performance_category': 'good'
        }
        
        # Store metrics
        performance_tracker.store_performance_metrics(metrics)
        
        # Check that file was created and contains data
        assert performance_tracker.performance_file.exists()
        
        stored_data = pd.read_parquet(performance_tracker.performance_file)
        assert len(stored_data) == 1
        assert stored_data.iloc[0]['source'] == 'test_source'
        assert stored_data.iloc[0]['mae'] == 1.5
    
    def test_get_performance_history(self, performance_tracker):
        """Test retrieving performance history."""
        # Store some test metrics
        for i in range(5):
            metrics = {
                'source': 'test_source',
                'window_days': 30,
                'count': 10,
                'mae': 1.5 + i * 0.1,
                'accuracy_3f': 0.8 - i * 0.02
            }
            performance_tracker.store_performance_metrics(metrics)
        
        # Get history
        history = performance_tracker.get_performance_history('test_source', days=30)
        
        assert len(history) == 5
        assert all(history['source'] == 'test_source')
        assert history['mae'].iloc[0] == 1.5  # First stored metric
    
    def test_get_performance_history_no_file(self, performance_tracker):
        """Test getting performance history when no file exists."""
        history = performance_tracker.get_performance_history('test_source')
        assert history.empty
    
    def test_detect_performance_degradation(self, performance_tracker):
        """Test performance degradation detection."""
        # Store historical data with declining performance
        timestamps = [datetime.now() - timedelta(days=i) for i in range(20, 0, -1)]
        
        for i, timestamp in enumerate(timestamps):
            # Create declining accuracy over time
            accuracy = 0.9 - (i * 0.02)  # Decline from 0.9 to 0.5
            
            metrics_data = pd.DataFrame([{
                'timestamp': timestamp,
                'source': 'test_source',
                'accuracy_3f': accuracy,
                'window_days': 30
            }])
            
            performance_tracker.data_manager.save_source_data('model_performance', metrics_data, append=True)
        
        # Test degradation detection
        degradation = performance_tracker.detect_performance_degradation('test_source', threshold_drop=0.1)
        assert degradation  # Should detect degradation
        
        # Test with small threshold
        no_degradation = performance_tracker.detect_performance_degradation('test_source', threshold_drop=0.5)
        assert not no_degradation  # Should not detect with high threshold
    
    def test_get_current_source_weights_no_history(self, performance_tracker):
        """Test getting current weights when no history exists."""
        weights = performance_tracker.get_current_source_weights()
        
        # Should return equal weights
        expected_sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        assert set(weights.keys()) == set(expected_sources)
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # All weights should be approximately equal
        expected_weight = 1.0 / len(expected_sources)
        for weight in weights.values():
            assert abs(weight - expected_weight) < 0.01
    
    def test_generate_performance_report(self, performance_tracker, sample_weather_data, sample_actual_temps):
        """Test generating comprehensive performance report."""
        # Store sample data for multiple sources
        sources = ['nws', 'openweather', 'tomorrow']
        for source in sources:
            # Add some variation to each source
            data = sample_weather_data.copy()
            data['predicted_high'] += np.random.normal(0, 1, len(data))
            performance_tracker.data_manager.save_source_data(source, data)
        
        performance_tracker.data_manager.save_source_data('actual_temperatures', sample_actual_temps)
        
        # Generate report
        report = performance_tracker.generate_performance_report(window_days=30)
        
        # Check report structure
        assert 'timestamp' in report
        assert 'window_days' in report
        assert 'performance_data' in report
        assert 'current_weights' in report
        assert 'degradation_alerts' in report
        assert 'summary' in report
        
        # Check that performance data contains expected sources
        perf_data = report['performance_data']
        for source in sources:
            assert source in perf_data
            if 'error' not in perf_data[source]:
                assert 'accuracy_3f' in perf_data[source]
    
    def test_run_daily_performance_tracking(self, performance_tracker, sample_weather_data, sample_actual_temps):
        """Test running daily performance tracking."""
        # Store sample data
        performance_tracker.data_manager.save_source_data('nws', sample_weather_data)
        performance_tracker.data_manager.save_source_data('actual_temperatures', sample_actual_temps)
        
        # Run daily tracking
        results = performance_tracker.run_daily_performance_tracking()
        
        # Check results structure
        assert 'short_term' in results
        assert 'medium_term' in results
        assert 'long_term' in results
        assert 'seasonal' in results
        assert 'updated_weights' in results
        assert 'performance_report' in results
        
        # Check that performance file was created and contains data
        assert performance_tracker.performance_file.exists()
        stored_data = pd.read_parquet(performance_tracker.performance_file)
        assert len(stored_data) > 0


if __name__ == "__main__":
    pytest.main([__file__])