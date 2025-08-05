"""Tests for XGBoost temperature prediction model."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.models.xgboost_model import XGBoostTemperatureModel


class TestXGBoostTemperatureModel(unittest.TestCase):
    """Test cases for XGBoost temperature prediction model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for model files
        self.temp_dir = tempfile.mkdtemp()
        self.model = XGBoostTemperatureModel(model_dir=self.temp_dir)
        
        # Create sample data
        self.sample_dates = pd.date_range('2025-01-01', '2025-01-30', freq='D')
        self.sample_features = self.create_sample_features()
        self.sample_targets = self.create_sample_targets()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_features(self) -> pd.DataFrame:
        """Create sample feature data for testing."""
        np.random.seed(42)
        n_samples = len(self.sample_dates)
        
        features = pd.DataFrame({
            'date': self.sample_dates,
            'nws_predicted_high': np.random.normal(75, 10, n_samples),
            'nws_predicted_low': np.random.normal(55, 8, n_samples),
            'nws_humidity': np.random.uniform(30, 80, n_samples),
            'nws_pressure': np.random.normal(1013, 10, n_samples),
            'nws_wind_speed': np.random.uniform(0, 20, n_samples),
            'openweather_predicted_high': np.random.normal(76, 9, n_samples),
            'openweather_predicted_low': np.random.normal(56, 7, n_samples),
            'openweather_humidity': np.random.uniform(35, 75, n_samples),
            'day_of_year': [d.timetuple().tm_yday for d in self.sample_dates],
            'month': [d.month for d in self.sample_dates],
            'is_weekend': [(d.weekday() >= 5) for d in self.sample_dates],
            'consensus_high': np.random.normal(75.5, 8, n_samples),
            'temp_range': np.random.uniform(15, 25, n_samples),
            'marine_layer_indicator': np.random.uniform(0, 1, n_samples),
            'santa_ana_indicator': np.random.uniform(0, 1, n_samples)
        })
        
        return features
    
    def create_sample_targets(self) -> pd.Series:
        """Create sample target data for testing."""
        np.random.seed(42)
        # Create targets that are somewhat correlated with features
        base_temps = self.sample_features['nws_predicted_high'] + np.random.normal(0, 3, len(self.sample_features))
        return pd.Series(base_temps, name='actual_high')
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertFalse(self.model.is_trained)
        self.assertEqual(len(self.model.feature_names), 0)
        self.assertIsNone(self.model.model)
        self.assertTrue(Path(self.temp_dir).exists())
    
    @patch('src.models.xgboost_model.FeaturePipeline')
    @patch('src.models.xgboost_model.DataManager')
    def test_prepare_training_data(self, mock_data_manager, mock_feature_pipeline):
        """Test training data preparation."""
        # Mock feature pipeline
        mock_pipeline = Mock()
        mock_pipeline.create_complete_features.return_value = self.sample_features
        mock_feature_pipeline.return_value = mock_pipeline
        
        # Mock data manager
        mock_dm = Mock()
        actual_temps = pd.DataFrame({
            'date': self.sample_dates,
            'actual_high': self.sample_targets
        })
        mock_dm.load_source_data.return_value = actual_temps
        mock_data_manager.return_value = mock_dm
        
        # Create new model instance to use mocks
        model = XGBoostTemperatureModel(model_dir=self.temp_dir)
        
        # Test data preparation
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 30)
        
        X, y = model.prepare_training_data(start_date, end_date)
        
        self.assertFalse(X.empty)
        self.assertFalse(y.empty)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 0)
        
        # Check that feature names are stored
        self.assertGreater(len(model.feature_names), 0)
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        # Create simple test data
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] + np.random.normal(0, 0.1, 100))
        
        # Test with minimal trials for speed
        best_params = self.model.optimize_hyperparameters(X, y, n_trials=5, cv_folds=3)
        
        self.assertIsInstance(best_params, dict)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
        self.assertIn('learning_rate', best_params)
    
    @patch('src.models.xgboost_model.FeaturePipeline')
    @patch('src.models.xgboost_model.DataManager')
    def test_model_training(self, mock_data_manager, mock_feature_pipeline):
        """Test model training."""
        # Mock feature pipeline
        mock_pipeline = Mock()
        mock_pipeline.create_complete_features.return_value = self.sample_features
        mock_feature_pipeline.return_value = mock_pipeline
        
        # Mock data manager
        mock_dm = Mock()
        actual_temps = pd.DataFrame({
            'date': self.sample_dates,
            'actual_high': self.sample_targets
        })
        mock_dm.load_source_data.return_value = actual_temps
        mock_data_manager.return_value = mock_dm
        
        # Create new model instance to use mocks
        model = XGBoostTemperatureModel(model_dir=self.temp_dir)
        
        # Train model (without hyperparameter optimization for speed)
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 30)
        
        results = model.train(
            start_date=start_date,
            end_date=end_date,
            optimize_hyperparams=False,
            include_ensemble=False,
            include_la_patterns=False
        )
        
        # Check training results
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
        self.assertGreater(len(model.feature_names), 0)
        self.assertIsInstance(results, dict)
        self.assertIn('training_samples', results)
        self.assertIn('cv_scores', results)
        self.assertIn('training_metrics', results)
        self.assertIn('feature_importance_top10', results)
        
        # Check that metrics are reasonable
        self.assertGreater(results['training_samples'], 0)
        self.assertIn('rmse_mean', results['cv_scores'])
        self.assertGreater(results['cv_scores']['rmse_mean'], 0)
    
    def test_prediction(self):
        """Test model prediction."""
        # First train a simple model
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] + np.random.normal(0, 0.1, 100))
        
        # Manually set up model components for testing
        self.model.feature_names = ['feature1', 'feature2']
        X_scaled = self.model.scaler.fit_transform(X)
        
        import xgboost as xgb
        self.model.model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        self.model.model.fit(X_scaled, y)
        self.model.is_trained = True
        self.model.cv_scores = {'rmse_mean': 1.0, 'rmse_std': 0.1}
        
        # Test prediction
        test_features = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [0.5]
        })
        
        prediction, confidence = self.model.predict(test_features)
        
        self.assertIsInstance(prediction, (int, float))
        self.assertIsInstance(confidence, (int, float))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_prediction_without_training(self):
        """Test that prediction fails when model is not trained."""
        test_features = pd.DataFrame({'feature1': [1.0]})
        
        with self.assertRaises(ValueError):
            self.model.predict(test_features)
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        # Set up a trained model
        self.model.is_trained = True
        self.model.feature_importance = {
            'nws_predicted_high': 0.3,
            'openweather_predicted_high': 0.25,
            'consensus_high': 0.2,
            'day_of_year': 0.1,
            'marine_layer_indicator': 0.08,
            'santa_ana_indicator': 0.05,
            'humidity': 0.02
        }
        
        analysis = self.model.analyze_feature_importance()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('total_features', analysis)
        self.assertIn('top_10_features', analysis)
        self.assertIn('feature_categories', analysis)
        self.assertIn('category_importance', analysis)
        
        # Check that categories are properly identified
        self.assertIn('temperature', analysis['feature_categories'])
        self.assertIn('atmospheric', analysis['feature_categories'])
        self.assertIn('la_patterns', analysis['feature_categories'])
    
    def test_get_feature_importance(self):
        """Test getting top feature importance."""
        # Set up a trained model
        self.model.is_trained = True
        self.model.feature_importance = {
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2
        }
        
        top_features = self.model.get_feature_importance(top_n=2)
        
        self.assertEqual(len(top_features), 2)
        self.assertIn('feature1', top_features)
        self.assertIn('feature2', top_features)
        self.assertEqual(top_features['feature1'], 0.5)
    
    def test_get_feature_importance_without_training(self):
        """Test that feature importance fails when model is not trained."""
        with self.assertRaises(ValueError):
            self.model.get_feature_importance()
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = pd.Series([70, 75, 80, 85, 90])
        y_pred = np.array([72, 74, 82, 83, 88])
        
        metrics = self.model.calculate_metrics(y_true, y_pred)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        self.assertIn('accuracy_within_3f', metrics)
        self.assertIn('accuracy_within_5f', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreaterEqual(metrics['accuracy_within_3f'], 0)
        self.assertLessEqual(metrics['accuracy_within_3f'], 100)
    
    def test_calculate_prediction_confidence(self):
        """Test prediction confidence calculation."""
        # Set up model with CV scores
        self.model.cv_scores = {'rmse_mean': 2.0, 'rmse_std': 0.5}
        
        # Test with complete features
        complete_features = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0]
        })
        
        confidence = self.model.calculate_prediction_confidence(complete_features)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with missing features
        incomplete_features = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [np.nan],
            'feature3': [3.0]
        })
        
        confidence_incomplete = self.model.calculate_prediction_confidence(incomplete_features)
        self.assertLess(confidence_incomplete, confidence)  # Should be lower with missing data
    
    def test_model_save_and_load(self):
        """Test model saving and loading."""
        # Set up a minimal trained model
        self.model.is_trained = True
        self.model.feature_names = ['feature1', 'feature2']
        self.model.hyperparameters = {'n_estimators': 100}
        self.model.cv_scores = {'rmse_mean': 1.0}
        self.model.feature_importance = {'feature1': 0.6, 'feature2': 0.4}
        self.model.training_date = datetime.now()
        self.model.training_samples = 100
        
        # Create a simple model for testing
        import xgboost as xgb
        self.model.model = xgb.XGBRegressor(n_estimators=10)
        
        # Save model
        model_path = self.model.save_model('test_model.pkl')
        self.assertTrue(Path(model_path).exists())
        
        # Create new model and load
        new_model = XGBoostTemperatureModel(model_dir=self.temp_dir)
        self.assertFalse(new_model.is_trained)
        
        new_model.load_model(model_path)
        
        # Check that model was loaded correctly
        self.assertTrue(new_model.is_trained)
        self.assertEqual(new_model.feature_names, self.model.feature_names)
        self.assertEqual(new_model.hyperparameters, self.model.hyperparameters)
        self.assertEqual(new_model.training_samples, self.model.training_samples)
    
    def test_get_model_info(self):
        """Test getting model information."""
        # Test untrained model
        info = self.model.get_model_info()
        self.assertEqual(info['status'], 'not_trained')
        
        # Test trained model
        self.model.is_trained = True
        self.model.training_date = datetime.now()
        self.model.training_samples = 100
        self.model.feature_names = ['feature1', 'feature2']
        self.model.hyperparameters = {'n_estimators': 100}
        self.model.cv_scores = {'rmse_mean': 1.0}
        self.model.feature_importance = {'feature1': 0.6, 'feature2': 0.4}
        
        info = self.model.get_model_info()
        
        self.assertEqual(info['status'], 'trained')
        self.assertIn('training_date', info)
        self.assertEqual(info['training_samples'], 100)
        self.assertEqual(info['features_count'], 2)
        self.assertIn('hyperparameters', info)
        self.assertIn('cv_performance', info)
        self.assertIn('top_features', info)


class TestXGBoostModelIntegration(unittest.TestCase):
    """Integration tests for XGBoost model with real-like data."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_realistic_weather_data(self, n_days: int = 60) -> tuple:
        """Create realistic weather data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=n_days, freq='D')
        
        # Create realistic LA weather patterns
        base_temp = 75 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Seasonal variation
        
        features = pd.DataFrame({
            'date': dates,
            'nws_predicted_high': base_temp + np.random.normal(0, 3, n_days),
            'nws_predicted_low': base_temp - 20 + np.random.normal(0, 2, n_days),
            'nws_humidity': np.random.uniform(40, 80, n_days),
            'nws_pressure': np.random.normal(1013, 5, n_days),
            'nws_wind_speed': np.random.exponential(5, n_days),
            'openweather_predicted_high': base_temp + np.random.normal(1, 2, n_days),
            'openweather_predicted_low': base_temp - 19 + np.random.normal(0, 2, n_days),
            'openweather_humidity': np.random.uniform(35, 85, n_days),
            'day_of_year': [d.timetuple().tm_yday for d in dates],
            'month': [d.month for d in dates],
            'is_weekend': [d.weekday() >= 5 for d in dates],
            'consensus_high': base_temp + np.random.normal(0.5, 1, n_days),
            'temp_range': np.random.uniform(15, 25, n_days)
        })
        
        # Create targets with realistic relationship to features
        actual_temps = (features['consensus_high'] + 
                       np.random.normal(0, 2, n_days) +  # Random error
                       (features['humidity'] - 60) * -0.05)  # Humidity effect
        
        return features, pd.Series(actual_temps, name='actual_high')
    
    @patch('src.models.xgboost_model.FeaturePipeline')
    @patch('src.models.xgboost_model.DataManager')
    def test_full_training_pipeline(self, mock_data_manager, mock_feature_pipeline):
        """Test the complete training pipeline with realistic data."""
        # Create realistic data
        features, targets = self.create_realistic_weather_data(60)
        
        # Mock feature pipeline
        mock_pipeline = Mock()
        mock_pipeline.create_complete_features.return_value = features
        mock_feature_pipeline.return_value = mock_pipeline
        
        # Mock data manager
        mock_dm = Mock()
        actual_temps = pd.DataFrame({
            'date': features['date'],
            'actual_high': targets
        })
        mock_dm.load_source_data.return_value = actual_temps
        mock_data_manager.return_value = mock_dm
        
        # Create and train model
        model = XGBoostTemperatureModel(model_dir=self.temp_dir)
        
        results = model.train(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 1),
            optimize_hyperparams=False,  # Skip for speed
            include_ensemble=False,
            include_la_patterns=False
        )
        
        # Verify training results
        self.assertTrue(model.is_trained)
        self.assertGreater(results['training_samples'], 0)
        self.assertLess(results['cv_scores']['rmse_mean'], 10)  # Should be reasonable for temperature
        self.assertGreater(results['training_metrics']['accuracy_within_3f'], 50)  # At least 50% within 3°F
        
        # Test prediction
        test_features = features.iloc[[0]]  # Use first row for prediction
        prediction, confidence = model.predict(test_features)
        
        self.assertIsInstance(prediction, (int, float))
        self.assertGreater(prediction, 50)  # Reasonable temperature for LA
        self.assertLess(prediction, 120)
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)
        
        # Test feature importance
        importance = model.get_feature_importance(top_n=5)
        self.assertGreater(len(importance), 0)
        
        # Temperature features should be important
        temp_features = [f for f in importance.keys() if 'temp' in f.lower() or 'high' in f.lower()]
        self.assertGreater(len(temp_features), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add unit tests
    test_suite.addTest(unittest.makeSuite(TestXGBoostTemperatureModel))
    test_suite.addTest(unittest.makeSuite(TestXGBoostModelIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)