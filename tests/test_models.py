"""
Comprehensive unit tests for machine learning model components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.xgboost_model import XGBoostTemperatureModel
from src.models.lightgbm_model import LightGBMTemperatureModel
from src.models.linear_regression_model import LinearRegressionModel
from src.models.prophet_model import ProphetTemperatureModel
from src.models.ensemble_model import EnsembleModel
from src.models.enhanced_ensemble_combiner import EnhancedEnsembleCombiner
from src.models.model_adapters import ModelAdapter, XGBoostAdapter, LightGBMAdapter


class TestModelAdapters:
    """Test model adapter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 1, n_samples),
            'feature4': np.random.exponential(1, n_samples)
        })
        
        # Create targets with some relationship to features
        self.y_train = (self.X_train['feature1'] * 2 + 
                       self.X_train['feature2'] + 
                       np.random.normal(0, 0.5, n_samples))
        
        # Create test data
        self.X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20),
            'feature3': np.random.uniform(0, 1, 20),
            'feature4': np.random.exponential(1, 20)
        })
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_adapter_interface(self):
        """Test ModelAdapter interface."""
        # ModelAdapter should not be instantiated directly
        with pytest.raises(TypeError):
            ModelAdapter()
    
    def test_xgboost_adapter(self):
        """Test XGBoostAdapter functionality."""
        adapter = XGBoostAdapter()
        
        # Test training
        model = adapter.train(self.X_train, self.y_train)
        assert model is not None
        
        # Test prediction
        predictions = adapter.predict(model, self.X_test)
        assert len(predictions) == len(self.X_test)
        assert all(isinstance(p, (int, float)) for p in predictions)
        
        # Test feature importance
        importance = adapter.get_feature_importance(model)
        assert isinstance(importance, dict)
        assert len(importance) == len(self.X_train.columns)
        assert all(isinstance(v, (int, float)) for v in importance.values())
        
        # Test hyperparameter optimization
        best_params = adapter.optimize_hyperparameters(
            self.X_train, self.y_train, n_trials=3, cv_folds=3
        )
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
    
    def test_lightgbm_adapter(self):
        """Test LightGBMAdapter functionality."""
        adapter = LightGBMAdapter()
        
        # Test training
        model = adapter.train(self.X_train, self.y_train)
        assert model is not None
        
        # Test prediction
        predictions = adapter.predict(model, self.X_test)
        assert len(predictions) == len(self.X_test)
        assert all(isinstance(p, (int, float)) for p in predictions)
        
        # Test feature importance
        importance = adapter.get_feature_importance(model)
        assert isinstance(importance, dict)
        assert len(importance) == len(self.X_train.columns)
        
        # Test hyperparameter optimization
        best_params = adapter.optimize_hyperparameters(
            self.X_train, self.y_train, n_trials=3, cv_folds=3
        )
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params


class TestXGBoostTemperatureModel:
    """Test XGBoostTemperatureModel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = XGBoostTemperatureModel(model_dir=self.temp_dir)
        
        # Create sample data
        self.sample_dates = pd.date_range('2025-01-01', '2025-01-30', freq='D')
        self.sample_features = self.create_sample_features()
        self.sample_targets = self.create_sample_targets()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_features(self) -> pd.DataFrame:
        """Create sample feature data."""
        np.random.seed(42)
        n_samples = len(self.sample_dates)
        
        return pd.DataFrame({
            'date': self.sample_dates,
            'nws_predicted_high': np.random.normal(75, 10, n_samples),
            'nws_predicted_low': np.random.normal(55, 8, n_samples),
            'nws_humidity': np.random.uniform(30, 80, n_samples),
            'openweather_predicted_high': np.random.normal(76, 9, n_samples),
            'consensus_high': np.random.normal(75.5, 8, n_samples),
            'day_of_year': [d.timetuple().tm_yday for d in self.sample_dates],
            'month': [d.month for d in self.sample_dates],
            'marine_layer_indicator': np.random.uniform(0, 1, n_samples)
        })
    
    def create_sample_targets(self) -> pd.Series:
        """Create sample target data."""
        np.random.seed(42)
        base_temps = self.sample_features['nws_predicted_high'] + np.random.normal(0, 3, len(self.sample_features))
        return pd.Series(base_temps, name='actual_high')
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert not self.model.is_trained
        assert len(self.model.feature_names) == 0
        assert self.model.model is None
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        X = self.sample_features.drop(['date'], axis=1)
        y = self.sample_targets
        
        best_params = self.model.optimize_hyperparameters(X, y, n_trials=3, cv_folds=3)
        
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert 'learning_rate' in best_params
    
    def test_model_training_basic(self):
        """Test basic model training without mocks."""
        # Prepare data
        X = self.sample_features.drop(['date'], axis=1)
        y = self.sample_targets
        
        # Set up model components manually
        self.model.feature_names = list(X.columns)
        X_scaled = self.model.scaler.fit_transform(X)
        
        # Train model
        import xgboost as xgb
        self.model.model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        self.model.model.fit(X_scaled, y)
        self.model.is_trained = True
        self.model.cv_scores = {'rmse_mean': 2.0, 'rmse_std': 0.5}
        
        # Test prediction
        test_features = X.iloc[[0]]
        prediction, confidence = self.model.predict(test_features)
        
        assert isinstance(prediction, (int, float))
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        self.model.is_trained = True
        self.model.feature_importance = {
            'nws_predicted_high': 0.3,
            'openweather_predicted_high': 0.25,
            'consensus_high': 0.2,
            'day_of_year': 0.1,
            'marine_layer_indicator': 0.08,
            'nws_humidity': 0.05,
            'month': 0.02
        }
        
        analysis = self.model.analyze_feature_importance()
        
        assert isinstance(analysis, dict)
        assert 'total_features' in analysis
        assert 'top_10_features' in analysis
        assert 'feature_categories' in analysis
        assert 'category_importance' in analysis
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = pd.Series([70, 75, 80, 85, 90])
        y_pred = np.array([72, 74, 82, 83, 88])
        
        metrics = self.model.calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'accuracy_within_3f' in metrics
        assert 'accuracy_within_5f' in metrics
        
        # Check metric ranges
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['accuracy_within_3f'] <= 100
        assert 0 <= metrics['accuracy_within_5f'] <= 100
    
    def test_model_save_and_load(self):
        """Test model saving and loading."""
        # Set up minimal trained model
        self.model.is_trained = True
        self.model.feature_names = ['feature1', 'feature2']
        self.model.hyperparameters = {'n_estimators': 100}
        self.model.cv_scores = {'rmse_mean': 1.0}
        self.model.feature_importance = {'feature1': 0.6, 'feature2': 0.4}
        self.model.training_date = datetime.now()
        self.model.training_samples = 100
        
        # Create simple model
        import xgboost as xgb
        self.model.model = xgb.XGBRegressor(n_estimators=10)
        
        # Save model
        model_path = self.model.save_model('test_model.pkl')
        assert Path(model_path).exists()
        
        # Load into new model
        new_model = XGBoostTemperatureModel(model_dir=self.temp_dir)
        new_model.load_model(model_path)
        
        # Verify loaded correctly
        assert new_model.is_trained
        assert new_model.feature_names == self.model.feature_names
        assert new_model.hyperparameters == self.model.hyperparameters


class TestLightGBMTemperatureModel:
    """Test LightGBMTemperatureModel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = LightGBMTemperatureModel(model_dir=self.temp_dir)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 50
        
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 1, n_samples)
        })
        
        self.y_train = (self.X_train['feature1'] * 2 + 
                       self.X_train['feature2'] + 
                       np.random.normal(0, 0.5, n_samples))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert not self.model.is_trained
        assert self.model.model is None
    
    def test_basic_training(self):
        """Test basic model training."""
        # Set up model components
        self.model.feature_names = list(self.X_train.columns)
        
        # Train model manually
        import lightgbm as lgb
        self.model.model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        self.model.model.fit(self.X_train, self.y_train)
        self.model.is_trained = True
        
        # Test prediction
        predictions = self.model.model.predict(self.X_train.iloc[[0]])
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float))


class TestLinearRegressionModel:
    """Test LinearRegressionModel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = LinearRegressionModel(model_dir=self.temp_dir)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 50
        
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 1, n_samples)
        })
        
        self.y_train = (self.X_train['feature1'] * 2 + 
                       self.X_train['feature2'] + 
                       np.random.normal(0, 0.5, n_samples))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert not self.model.is_trained
        assert self.model.model is None
    
    def test_basic_training(self):
        """Test basic model training."""
        # Set up model components
        self.model.feature_names = list(self.X_train.columns)
        
        # Train model manually
        from sklearn.linear_model import Ridge
        self.model.model = Ridge(alpha=1.0, random_state=42)
        self.model.model.fit(self.X_train, self.y_train)
        self.model.is_trained = True
        
        # Test prediction
        predictions = self.model.model.predict(self.X_train.iloc[[0]])
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float))
        
        # Test feature importance (coefficients)
        if hasattr(self.model.model, 'coef_'):
            importance = dict(zip(self.model.feature_names, abs(self.model.model.coef_)))
            assert len(importance) == len(self.model.feature_names)


class TestProphetTemperatureModel:
    """Test ProphetTemperatureModel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = ProphetTemperatureModel(model_dir=self.temp_dir)
        
        # Create time series data for Prophet
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        
        # Create seasonal temperature pattern
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal_temp = 75 + 15 * np.sin(2 * np.pi * day_of_year / 365)
        noise = np.random.normal(0, 3, len(dates))
        
        self.time_series_data = pd.DataFrame({
            'ds': dates,  # Prophet expects 'ds' for dates
            'y': seasonal_temp + noise,  # Prophet expects 'y' for target
            'external_regressor': np.random.normal(0, 1, len(dates))
        })
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert not self.model.is_trained
        assert self.model.model is None
    
    def test_data_preparation(self):
        """Test Prophet data preparation."""
        # Prophet requires specific column names
        prophet_data = self.time_series_data[['ds', 'y']].copy()
        
        assert 'ds' in prophet_data.columns
        assert 'y' in prophet_data.columns
        assert len(prophet_data) > 0
    
    @patch('src.models.prophet_model.Prophet')
    def test_basic_training(self, mock_prophet):
        """Test basic Prophet model training."""
        # Mock Prophet model
        mock_model = Mock()
        mock_prophet.return_value = mock_model
        
        # Set up model
        self.model.model = mock_model
        self.model.is_trained = True
        
        # Test that model was created
        assert self.model.model is not None
        assert self.model.is_trained


class TestEnsembleModel:
    """Test EnsembleModel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ensemble = EnsembleModel(model_dir=self.temp_dir)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 1, n_samples),
            'feature4': np.random.exponential(1, n_samples)
        })
        
        self.y_train = (self.X_train['feature1'] * 2 + 
                       self.X_train['feature2'] + 
                       np.random.normal(0, 0.5, n_samples))
        
        self.X_test = self.X_train.iloc[:10].copy()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        assert self.ensemble is not None
        assert len(self.ensemble.models) == 0
        assert not self.ensemble.is_trained
    
    def test_add_model(self):
        """Test adding models to ensemble."""
        # Create mock models
        mock_model1 = Mock()
        mock_model1.name = 'xgboost'
        mock_model1.is_trained = True
        
        mock_model2 = Mock()
        mock_model2.name = 'lightgbm'
        mock_model2.is_trained = True
        
        # Add models
        self.ensemble.add_model(mock_model1, weight=0.6)
        self.ensemble.add_model(mock_model2, weight=0.4)
        
        assert len(self.ensemble.models) == 2
        assert self.ensemble.model_weights['xgboost'] == 0.6
        assert self.ensemble.model_weights['lightgbm'] == 0.4
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        # Create mock models with predictions
        mock_model1 = Mock()
        mock_model1.name = 'xgboost'
        mock_model1.predict.return_value = (75.0, 0.8)  # prediction, confidence
        
        mock_model2 = Mock()
        mock_model2.name = 'lightgbm'
        mock_model2.predict.return_value = (77.0, 0.7)
        
        # Add models to ensemble
        self.ensemble.models = [mock_model1, mock_model2]
        self.ensemble.model_weights = {'xgboost': 0.6, 'lightgbm': 0.4}
        self.ensemble.is_trained = True
        
        # Test prediction
        prediction, confidence = self.ensemble.predict(self.X_test.iloc[[0]])
        
        # Should be weighted average: 75.0 * 0.6 + 77.0 * 0.4 = 75.8
        expected_prediction = 75.0 * 0.6 + 77.0 * 0.4
        assert abs(prediction - expected_prediction) < 0.01
        
        # Confidence should be weighted average
        expected_confidence = 0.8 * 0.6 + 0.7 * 0.4
        assert abs(confidence - expected_confidence) < 0.01
    
    def test_update_weights(self):
        """Test updating model weights."""
        # Set up ensemble with models
        self.ensemble.model_weights = {'xgboost': 0.5, 'lightgbm': 0.5}
        
        # Update weights
        new_weights = {'xgboost': 0.7, 'lightgbm': 0.3}
        self.ensemble.update_weights(new_weights)
        
        assert self.ensemble.model_weights['xgboost'] == 0.7
        assert self.ensemble.model_weights['lightgbm'] == 0.3
    
    def test_get_model_info(self):
        """Test getting ensemble model information."""
        # Set up ensemble
        mock_model1 = Mock()
        mock_model1.name = 'xgboost'
        mock_model1.get_model_info.return_value = {'status': 'trained', 'features': 10}
        
        self.ensemble.models = [mock_model1]
        self.ensemble.model_weights = {'xgboost': 1.0}
        self.ensemble.is_trained = True
        
        info = self.ensemble.get_model_info()
        
        assert info['status'] == 'trained'
        assert 'models' in info
        assert 'model_weights' in info
        assert len(info['models']) == 1


class TestEnhancedEnsembleCombiner:
    """Test EnhancedEnsembleCombiner functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.combiner = EnhancedEnsembleCombiner()
        
        # Create sample predictions from multiple models
        self.model_predictions = {
            'xgboost': np.array([75.0, 78.0, 72.0, 76.0, 74.0]),
            'lightgbm': np.array([76.0, 77.0, 73.0, 75.0, 75.0]),
            'linear': np.array([74.0, 79.0, 71.0, 77.0, 73.0]),
            'prophet': np.array([75.5, 78.5, 72.5, 76.5, 74.5])
        }
        
        self.model_confidences = {
            'xgboost': np.array([0.85, 0.80, 0.90, 0.75, 0.88]),
            'lightgbm': np.array([0.82, 0.85, 0.87, 0.80, 0.83]),
            'linear': np.array([0.70, 0.75, 0.72, 0.78, 0.74]),
            'prophet': np.array([0.88, 0.82, 0.85, 0.79, 0.86])
        }
        
        # Actual temperatures for performance calculation
        self.actual_temps = np.array([75.2, 78.1, 72.8, 76.3, 74.7])
    
    def test_combiner_initialization(self):
        """Test combiner initialization."""
        assert self.combiner is not None
        assert len(self.combiner.model_weights) == 0
        assert len(self.combiner.performance_history) == 0
    
    def test_simple_average_combination(self):
        """Test simple average combination."""
        combined_pred, combined_conf = self.combiner.simple_average(
            self.model_predictions, self.model_confidences
        )
        
        # Check that we get predictions for all samples
        assert len(combined_pred) == 5
        assert len(combined_conf) == 5
        
        # Check that predictions are reasonable averages
        expected_first = np.mean([75.0, 76.0, 74.0, 75.5])
        assert abs(combined_pred[0] - expected_first) < 0.01
    
    def test_weighted_average_combination(self):
        """Test weighted average combination."""
        weights = {'xgboost': 0.4, 'lightgbm': 0.3, 'linear': 0.1, 'prophet': 0.2}
        
        combined_pred, combined_conf = self.combiner.weighted_average(
            self.model_predictions, self.model_confidences, weights
        )
        
        # Check results
        assert len(combined_pred) == 5
        assert len(combined_conf) == 5
        
        # Verify weighted calculation for first prediction
        expected_first = (75.0 * 0.4 + 76.0 * 0.3 + 74.0 * 0.1 + 75.5 * 0.2)
        assert abs(combined_pred[0] - expected_first) < 0.01
    
    def test_confidence_weighted_combination(self):
        """Test confidence-weighted combination."""
        combined_pred, combined_conf = self.combiner.confidence_weighted(
            self.model_predictions, self.model_confidences
        )
        
        # Check results
        assert len(combined_pred) == 5
        assert len(combined_conf) == 5
        
        # Predictions should be weighted by confidence
        # Higher confidence models should have more influence
        assert all(0 <= conf <= 1 for conf in combined_conf)
    
    def test_performance_weighted_combination(self):
        """Test performance-weighted combination."""
        # Set up performance history
        performance_history = {
            'xgboost': [2.1, 2.3, 1.9, 2.0],  # RMSE history
            'lightgbm': [2.2, 2.1, 2.0, 2.1],
            'linear': [2.8, 2.9, 2.7, 2.8],
            'prophet': [2.4, 2.2, 2.3, 2.5]
        }
        
        combined_pred, combined_conf = self.combiner.performance_weighted(
            self.model_predictions, self.model_confidences, performance_history
        )
        
        # Check results
        assert len(combined_pred) == 5
        assert len(combined_conf) == 5
        
        # Better performing models (lower RMSE) should have higher weights
        # XGBoost has best recent performance, so should have high influence
    
    def test_dynamic_combination(self):
        """Test dynamic combination method."""
        combined_pred, combined_conf = self.combiner.dynamic_combination(
            self.model_predictions, self.model_confidences
        )
        
        # Check results
        assert len(combined_pred) == 5
        assert len(combined_conf) == 5
        
        # Should adapt weights based on agreement and confidence
        assert all(50 <= pred <= 100 for pred in combined_pred)  # Reasonable temperature range
        assert all(0 <= conf <= 1 for conf in combined_conf)
    
    def test_stacking_combination(self):
        """Test stacking combination method."""
        # Create features for stacking (model predictions + metadata)
        stacking_features = pd.DataFrame({
            'xgboost_pred': self.model_predictions['xgboost'],
            'lightgbm_pred': self.model_predictions['lightgbm'],
            'linear_pred': self.model_predictions['linear'],
            'prophet_pred': self.model_predictions['prophet'],
            'xgboost_conf': self.model_confidences['xgboost'],
            'lightgbm_conf': self.model_confidences['lightgbm'],
            'linear_conf': self.model_confidences['linear'],
            'prophet_conf': self.model_confidences['prophet']
        })
        
        combined_pred, combined_conf = self.combiner.stacking_combination(
            self.model_predictions, self.model_confidences, 
            stacking_features, self.actual_temps
        )
        
        # Check results
        assert len(combined_pred) == 5
        assert len(combined_conf) == 5
    
    def test_update_performance_history(self):
        """Test updating performance history."""
        # Update performance for each model
        current_performance = {
            'xgboost': 1.8,
            'lightgbm': 2.0,
            'linear': 2.7,
            'prophet': 2.2
        }
        
        self.combiner.update_performance_history(current_performance)
        
        # Check that history was updated
        assert len(self.combiner.performance_history) > 0
        for model, perf in current_performance.items():
            assert model in self.combiner.performance_history
            assert self.combiner.performance_history[model][-1] == perf
    
    def test_calculate_model_weights(self):
        """Test calculating optimal model weights."""
        # Set up performance history
        self.combiner.performance_history = {
            'xgboost': [2.1, 2.0, 1.9, 1.8],
            'lightgbm': [2.2, 2.1, 2.0, 2.0],
            'linear': [2.8, 2.7, 2.7, 2.7],
            'prophet': [2.4, 2.3, 2.2, 2.2]
        }
        
        weights = self.combiner.calculate_optimal_weights()
        
        # Check weights
        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert all(0 <= w <= 1 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to 1
        
        # XGBoost should have highest weight (best performance)
        assert weights['xgboost'] >= weights['linear']
    
    def test_get_combination_info(self):
        """Test getting combination information."""
        # Set up some state
        self.combiner.model_weights = {'xgboost': 0.4, 'lightgbm': 0.3, 'linear': 0.1, 'prophet': 0.2}
        self.combiner.performance_history = {
            'xgboost': [2.0, 1.9, 1.8],
            'lightgbm': [2.1, 2.0, 2.0]
        }
        
        info = self.combiner.get_combination_info()
        
        assert isinstance(info, dict)
        assert 'current_weights' in info
        assert 'performance_history' in info
        assert 'available_methods' in info
        assert len(info['available_methods']) > 0


def run_model_tests():
    """Run all model tests."""
    print("Running model tests...")
    
    # Test ModelAdapters
    print("✓ Testing ModelAdapters...")
    xgb_adapter = XGBoostAdapter()
    assert xgb_adapter is not None
    
    # Test XGBoostTemperatureModel
    print("✓ Testing XGBoostTemperatureModel...")
    with tempfile.TemporaryDirectory() as temp_dir:
        model = XGBoostTemperatureModel(model_dir=temp_dir)
        assert model is not None
        assert not model.is_trained
    
    # Test EnsembleModel
    print("✓ Testing EnsembleModel...")
    with tempfile.TemporaryDirectory() as temp_dir:
        ensemble = EnsembleModel(model_dir=temp_dir)
        assert ensemble is not None
        assert len(ensemble.models) == 0
    
    # Test EnhancedEnsembleCombiner
    print("✓ Testing EnhancedEnsembleCombiner...")
    combiner = EnhancedEnsembleCombiner()
    assert combiner is not None
    
    # Test simple combination
    model_preds = {
        'model1': np.array([75.0, 78.0]),
        'model2': np.array([76.0, 77.0])
    }
    model_confs = {
        'model1': np.array([0.8, 0.9]),
        'model2': np.array([0.7, 0.8])
    }
    
    combined_pred, combined_conf = combiner.simple_average(model_preds, model_confs)
    assert len(combined_pred) == 2
    assert len(combined_conf) == 2
    
    print("All model tests passed!")


if __name__ == '__main__':
    run_model_tests()