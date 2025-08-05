"""Tests for the automated model retraining system."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import json
import pickle

from .model_retrainer import ModelRetrainer
from .performance_tracker import PerformanceTracker
from ..utils.data_manager import DataManager


class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.is_fitted = False
        self.params = {}
    
    def fit(self, X, y):
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        # Return simple predictions based on first feature
        return X.iloc[:, 0] + np.random.normal(0, 1, len(X))
    
    def set_params(self, **params):
        self.params.update(params)
        return self


class TestModelRetrainer:
    """Test cases for ModelRetrainer."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_data_manager(self, temp_data_dir):
        """Create mock DataManager with test data."""
        data_manager = DataManager()
        data_manager.data_dir = temp_data_dir
        data_manager.models_dir = temp_data_dir / 'models'
        data_manager.models_dir.mkdir(exist_ok=True)
        
        # Update file paths to use temp directory
        for key in data_manager.data_files:
            data_manager.data_files[key] = temp_data_dir / f"{key}_data.parquet"
        
        return data_manager
    
    @pytest.fixture
    def mock_performance_tracker(self, mock_data_manager):
        """Create mock PerformanceTracker."""
        return PerformanceTracker(mock_data_manager)
    
    @pytest.fixture
    def model_retrainer(self, mock_data_manager, mock_performance_tracker, temp_data_dir):
        """Create ModelRetrainer instance with mocks."""
        retrainer = ModelRetrainer(mock_data_manager, mock_performance_tracker)
        retrainer.models_dir = temp_data_dir / 'models'
        retrainer.models_dir.mkdir(exist_ok=True)
        retrainer.performance_history_file = temp_data_dir / 'performance_history.json'
        
        # Mock available models
        retrainer.available_models = {
            'mock_model': lambda: MockModel
        }
        
        return retrainer
    
    @pytest.fixture
    def sample_training_data(self, mock_data_manager):
        """Create sample training data."""
        # Create weather data for multiple sources
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sources = ['nws', 'openweather', 'tomorrow']
        for source in sources:
            data = []
            for i, date_val in enumerate(dates):
                base_temp = 75 + 5 * np.sin(i * 0.2)
                noise = np.random.normal(0, 2)
                
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
            
            source_df = pd.DataFrame(data)
            mock_data_manager.save_source_data(source, source_df)
        
        # Create actual temperature data
        actual_data = []
        for i, date_val in enumerate(dates):
            base_temp = 75 + 5 * np.sin(i * 0.2)
            actual_noise = np.random.normal(0, 1)
            
            actual_data.append({
                'date': date_val,
                'actual_high': base_temp + actual_noise,
                'actual_low': base_temp - 15 + actual_noise,
                'source': 'NOAA'
            })
        
        actual_df = pd.DataFrame(actual_data)
        mock_data_manager.save_source_data('actual_temperatures', actual_df)
        
        return len(dates)
    
    def test_check_retraining_needed_no_history(self, model_retrainer, sample_training_data):
        """Test retraining check when no history exists."""
        result = model_retrainer.check_retraining_needed('test_model')
        
        assert result['model_name'] == 'test_model'
        assert result['should_retrain'] == True  # Should retrain when no history
        assert 'No previous retraining record found' in result['reasons']
        assert 'check_timestamp' in result
    
    def test_check_retraining_needed_with_history(self, model_retrainer, sample_training_data):
        """Test retraining check with existing history."""
        # Create mock history
        history = {
            'test_model': {
                'last_retraining': (datetime.now() - timedelta(days=3)).isoformat(),
                'retraining_info': {'validation_metrics': {'accuracy_3f': 0.85}}
            }
        }
        
        with open(model_retrainer.performance_history_file, 'w') as f:
            json.dump(history, f)
        
        result = model_retrainer.check_retraining_needed('test_model')
        
        assert result['model_name'] == 'test_model'
        assert 'last_retraining' in result
        assert result['last_retraining'] is not None
    
    def test_check_retraining_needed_performance_degradation(self, model_retrainer, sample_training_data):
        """Test retraining check with performance degradation."""
        # Mock performance tracker to return degraded performance
        def mock_track_performance(window_days):
            if window_days == 7:  # Recent performance (poor)
                return {
                    'nws': {'accuracy_3f': 0.60, 'mae': 3.0},
                    'openweather': {'accuracy_3f': 0.55, 'mae': 3.5}
                }
            else:  # Historical performance (good)
                return {
                    'nws': {'accuracy_3f': 0.85, 'mae': 1.5},
                    'openweather': {'accuracy_3f': 0.80, 'mae': 2.0}
                }
        
        model_retrainer.performance_tracker.track_all_sources_performance = mock_track_performance
        
        result = model_retrainer.check_retraining_needed('test_model')
        
        assert result['should_retrain'] == True
        assert any('accuracy dropped' in reason for reason in result['reasons'])
    
    def test_prepare_training_data(self, model_retrainer, sample_training_data):
        """Test training data preparation."""
        features, targets = model_retrainer.prepare_training_data(max_days=30)
        
        assert isinstance(features, pd.DataFrame)
        assert isinstance(targets, pd.Series)
        assert len(features) == len(targets)
        assert len(features) > 0
        assert len(features.columns) > 0
    
    def test_prepare_training_data_no_actual_temps(self, model_retrainer):
        """Test training data preparation with no actual temperatures."""
        with pytest.raises(ValueError, match="No actual temperature data available"):
            model_retrainer.prepare_training_data()
    
    def test_retrain_model_success(self, model_retrainer, sample_training_data):
        """Test successful model retraining."""
        result = model_retrainer.retrain_model('mock_model', hyperparameter_optimization=False)
        
        assert 'error' not in result
        assert result['model_type'] == 'mock_model'
        assert 'model_path' in result
        assert 'train_metrics' in result
        assert 'validation_metrics' in result
        assert result['training_samples'] > 0
        assert result['validation_samples'] > 0
        
        # Check that model file was created
        model_path = Path(result['model_path'])
        assert model_path.exists()
    
    def test_retrain_model_invalid_type(self, model_retrainer, sample_training_data):
        """Test retraining with invalid model type."""
        result = model_retrainer.retrain_model('invalid_model')
        
        assert 'error' in result
        assert 'not available' in result['error']
    
    def test_retrain_model_insufficient_data(self, model_retrainer):
        """Test retraining with insufficient data."""
        # Set high minimum data requirement
        model_retrainer.retraining_config['min_data_points'] = 1000
        
        result = model_retrainer.retrain_model('mock_model')
        
        assert 'error' in result
        assert 'Insufficient training data' in result['error']
    
    def test_optimize_hyperparameters(self, model_retrainer, sample_training_data):
        """Test hyperparameter optimization."""
        features, targets = model_retrainer.prepare_training_data(max_days=30)
        
        # Split data
        split_idx = int(len(features) * 0.8)
        train_features = features.iloc[:split_idx]
        train_targets = targets.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_targets = targets.iloc[split_idx:]
        
        model = MockModel()
        
        # Mock parameter grid for testing
        original_param_grids = {
            'mock': {
                'param1': [1, 2],
                'param2': [0.1, 0.2]
            }
        }
        
        # This would normally be called internally
        best_params = model_retrainer._optimize_hyperparameters(
            model, train_features, train_targets, val_features, val_targets
        )
        
        # Since we don't have a real parameter grid for MockModel, this might return None
        # That's expected behavior
        assert best_params is None or isinstance(best_params, dict)
    
    def test_get_last_retraining_date_no_file(self, model_retrainer):
        """Test getting last retraining date when no file exists."""
        result = model_retrainer._get_last_retraining_date('test_model')
        assert result is None
    
    def test_get_last_retraining_date_with_history(self, model_retrainer):
        """Test getting last retraining date with existing history."""
        test_date = datetime.now() - timedelta(days=5)
        history = {
            'test_model': {
                'last_retraining': test_date.isoformat()
            }
        }
        
        with open(model_retrainer.performance_history_file, 'w') as f:
            json.dump(history, f)
        
        result = model_retrainer._get_last_retraining_date('test_model')
        assert result == test_date.date()
    
    def test_update_retraining_history(self, model_retrainer):
        """Test updating retraining history."""
        retraining_info = {
            'model_type': 'test_model',
            'validation_metrics': {'accuracy_3f': 0.85}
        }
        
        model_retrainer._update_retraining_history('test_model', retraining_info)
        
        # Check that file was created and contains correct data
        assert model_retrainer.performance_history_file.exists()
        
        with open(model_retrainer.performance_history_file, 'r') as f:
            history = json.load(f)
        
        assert 'test_model' in history
        assert 'last_retraining' in history['test_model']
        assert history['test_model']['retraining_info'] == retraining_info
    
    def test_detect_model_switching_need_no_models(self, model_retrainer):
        """Test model switching detection with no models."""
        result = model_retrainer.detect_model_switching_need()
        
        assert result['should_switch'] == False
        assert 'No model performance data available' in result['reason']
    
    def test_detect_model_switching_need_with_models(self, model_retrainer):
        """Test model switching detection with existing models."""
        # Create mock model files
        model_file1 = model_retrainer.models_dir / 'mock_model_20250101_120000.pkl'
        model_file2 = model_retrainer.models_dir / 'mock_model_20250102_120000.pkl'
        
        with open(model_file1, 'wb') as f:
            pickle.dump(MockModel(), f)
        with open(model_file2, 'wb') as f:
            pickle.dump(MockModel(), f)
        
        # Create performance history
        history = {
            'mock_model': {
                'last_retraining': datetime.now().isoformat(),
                'retraining_info': {
                    'validation_metrics': {'accuracy_3f': 0.85, 'mae': 1.5}
                }
            }
        }
        
        with open(model_retrainer.performance_history_file, 'w') as f:
            json.dump(history, f)
        
        result = model_retrainer.detect_model_switching_need()
        
        assert 'should_switch' in result
        assert 'current_model' in result
        assert 'recommended_model' in result
        assert 'performance_comparison' in result
    
    def test_run_automated_retraining(self, model_retrainer, sample_training_data):
        """Test the complete automated retraining process."""
        result = model_retrainer.run_automated_retraining()
        
        assert 'timestamp' in result
        assert 'retraining_checks' in result
        assert 'retraining_results' in result
        assert 'model_switching' in result
        assert 'summary' in result
        
        # Check summary structure
        summary = result['summary']
        assert 'models_checked' in summary
        assert 'models_needing_retraining' in summary
        assert 'models_retrained' in summary
        assert 'successful_retraining' in summary
        assert 'model_switching_recommended' in summary
    
    def test_retraining_config_validation(self, model_retrainer):
        """Test that retraining configuration is properly set."""
        config = model_retrainer.retraining_config
        
        assert 'min_data_points' in config
        assert 'performance_threshold' in config
        assert 'degradation_threshold' in config
        assert 'retraining_frequency_days' in config
        assert 'max_training_data_days' in config
        assert 'validation_split' in config
        
        # Check reasonable values
        assert config['min_data_points'] > 0
        assert 0 < config['performance_threshold'] < 1
        assert 0 < config['degradation_threshold'] < 1
        assert 0 < config['validation_split'] < 1


if __name__ == "__main__":
    pytest.main([__file__])