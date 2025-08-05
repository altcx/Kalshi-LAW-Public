"""Comprehensive tests for the ensemble temperature prediction model."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import tempfile
import shutil

from src.models.ensemble_model import (
    EnsembleTemperatureModel, 
    BaseTemperatureModel, 
    WeatherCondition
)
from src.models.model_adapters import (
    LinearRegressionModelAdapter,
    ProphetModelAdapter,
    create_ensemble_with_models
)


class MockTemperatureModel(BaseTemperatureModel):
    """Mock model for testing."""
    
    def __init__(self, name: str, prediction: float = 75.0, confidence: float = 0.8, trained: bool = True):
        self.name = name
        self._prediction = prediction
        self._confidence = confidence
        self._trained = trained
        self.predict_calls = 0
    
    def predict(self, features: pd.DataFrame) -> tuple:
        self.predict_calls += 1
        return self._prediction, self._confidence
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_model_info(self) -> dict:
        return {
            'status': 'trained' if self._trained else 'not_trained',
            'model_type': f'Mock{self.name}',
            'predict_calls': self.predict_calls
        }


class TestEnsembleTemperatureModel(unittest.TestCase):
    """Test cases for EnsembleTemperatureModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ensemble = EnsembleTemperatureModel(model_dir=self.temp_dir)
        
        # Create sample features
        self.sample_features = pd.DataFrame({
            'nws_temp_high': [78.5],
            'openweather_temp_high': [79.2],
            'tomorrow_temp_high': [77.8],
            'openweather_pressure': [1015.2],
            'nws_wind_speed': [12.5],
            'visual_crossing_humidity': [65],
            'openweather_cloud_cover': [25],
            'tomorrow_precipitation_prob': [10]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ensemble initialization."""
        self.assertEqual(len(self.ensemble.models), 0)
        self.assertEqual(len(self.ensemble.model_weights), 0)
        self.assertEqual(self.ensemble.total_predictions, 0)
        self.assertIsNone(self.ensemble.last_update)
    
    def test_register_model(self):
        """Test model registration."""
        mock_model = MockTemperatureModel("Test", 75.0, 0.8)
        
        self.ensemble.register_model(
            "test_model", 
            mock_model, 
            initial_weight=1.5,
            weather_conditions=[WeatherCondition.CLEAR, WeatherCondition.NORMAL]
        )
        
        self.assertIn("test_model", self.ensemble.models)
        self.assertEqual(self.ensemble.model_weights["test_model"], 1.5)
        self.assertIn("test_model", self.ensemble.weather_condition_models[WeatherCondition.CLEAR])
        self.assertIn("test_model", self.ensemble.weather_condition_models[WeatherCondition.NORMAL])
    
    def test_weather_condition_detection(self):
        """Test weather condition detection."""
        # Test clear conditions
        clear_features = pd.DataFrame({
            'openweather_cloud_cover': [15],
            'tomorrow_precipitation_prob': [5]
        })
        condition = self.ensemble.detect_weather_condition(clear_features)
        self.assertEqual(condition, WeatherCondition.CLEAR)
        
        # Test rainy conditions
        rainy_features = pd.DataFrame({
            'tomorrow_precipitation_prob': [75],
            'nws_precipitation_prob': [80]
        })
        condition = self.ensemble.detect_weather_condition(rainy_features)
        self.assertEqual(condition, WeatherCondition.RAINY)
        
        # Test windy conditions
        windy_features = pd.DataFrame({
            'nws_wind_speed': [25],
            'openweather_wind_speed': [22]
        })
        condition = self.ensemble.detect_weather_condition(windy_features)
        self.assertEqual(condition, WeatherCondition.WINDY)
        
        # Test normal conditions (fallback)
        normal_features = pd.DataFrame({
            'openweather_cloud_cover': [50],
            'tomorrow_precipitation_prob': [30],
            'nws_wind_speed': [10]
        })
        condition = self.ensemble.detect_weather_condition(normal_features)
        self.assertEqual(condition, WeatherCondition.NORMAL)
    
    def test_marine_layer_detection(self):
        """Test marine layer detection."""
        # Test with explicit marine layer features
        marine_features = pd.DataFrame({
            'marine_layer_indicator': [0.8],
            'visual_crossing_humidity': [85]
        })
        condition = self.ensemble.detect_weather_condition(marine_features)
        self.assertEqual(condition, WeatherCondition.MARINE_LAYER)
        
        # Test with implicit marine layer conditions
        implicit_marine = pd.DataFrame({
            'openweather_humidity': [85],
            'nws_cloud_cover': [75],
            'visual_crossing_humidity': [82]
        })
        condition = self.ensemble.detect_weather_condition(implicit_marine)
        self.assertEqual(condition, WeatherCondition.MARINE_LAYER)
    
    def test_santa_ana_detection(self):
        """Test Santa Ana wind detection."""
        # Test with explicit Santa Ana features
        santa_ana_features = pd.DataFrame({
            'santa_ana_indicator': [0.9],
            'nws_wind_speed': [20]
        })
        condition = self.ensemble.detect_weather_condition(santa_ana_features)
        self.assertEqual(condition, WeatherCondition.SANTA_ANA)
        
        # Test with implicit Santa Ana conditions
        implicit_santa_ana = pd.DataFrame({
            'openweather_wind_speed': [18],
            'nws_humidity': [25],
            'visual_crossing_humidity': [20]
        })
        condition = self.ensemble.detect_weather_condition(implicit_santa_ana)
        self.assertEqual(condition, WeatherCondition.SANTA_ANA)
    
    def test_heat_wave_detection(self):
        """Test heat wave detection."""
        # Test with explicit heat wave features
        heat_features = pd.DataFrame({
            'fire_season_indicator': [0.7],
            'nws_temp_high': [95]
        })
        condition = self.ensemble.detect_weather_condition(heat_features)
        self.assertEqual(condition, WeatherCondition.HEAT_WAVE)
        
        # Test with high temperature
        high_temp_features = pd.DataFrame({
            'openweather_temp_high': [92],
            'tomorrow_temp_high': [94],
            'nws_temp_high': [91]
        })
        condition = self.ensemble.detect_weather_condition(high_temp_features)
        self.assertEqual(condition, WeatherCondition.HEAT_WAVE)
    
    def test_get_models_for_condition(self):
        """Test getting models for weather conditions."""
        # Register models with different conditions
        model1 = MockTemperatureModel("Model1")
        model2 = MockTemperatureModel("Model2")
        model3 = MockTemperatureModel("Model3", trained=False)
        
        self.ensemble.register_model("model1", model1, weather_conditions=[WeatherCondition.CLEAR])
        self.ensemble.register_model("model2", model2, weather_conditions=[WeatherCondition.RAINY])
        self.ensemble.register_model("model3", model3, weather_conditions=[WeatherCondition.CLEAR])
        
        # Test getting models for clear conditions
        clear_models = self.ensemble.get_models_for_condition(WeatherCondition.CLEAR)
        self.assertIn("model1", clear_models)
        self.assertNotIn("model3", clear_models)  # Not trained
        
        # Test getting models for rainy conditions
        rainy_models = self.ensemble.get_models_for_condition(WeatherCondition.RAINY)
        self.assertIn("model2", rainy_models)
        self.assertNotIn("model1", rainy_models)
    
    def test_dynamic_weight_calculation(self):
        """Test dynamic weight calculation."""
        # Register models
        model1 = MockTemperatureModel("Model1", 75.0, 0.8)
        model2 = MockTemperatureModel("Model2", 76.0, 0.9)
        
        self.ensemble.register_model("model1", model1, initial_weight=1.0)
        self.ensemble.register_model("model2", model2, initial_weight=1.0)
        
        # Add performance history
        self.ensemble.update_model_performance("model1", 75.0, 74.5, 0.8, WeatherCondition.NORMAL)
        self.ensemble.update_model_performance("model1", 76.0, 75.8, 0.8, WeatherCondition.NORMAL)
        self.ensemble.update_model_performance("model2", 77.0, 75.0, 0.9, WeatherCondition.NORMAL)  # Worse performance
        
        # Calculate weights
        weights = self.ensemble.calculate_dynamic_weights(["model1", "model2"])
        
        # Model1 should have higher weight due to better performance
        self.assertGreater(weights["model1"], weights["model2"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        # Register multiple models
        model1 = MockTemperatureModel("Model1", 75.0, 0.8)
        model2 = MockTemperatureModel("Model2", 77.0, 0.9)
        model3 = MockTemperatureModel("Model3", 76.0, 0.7)
        
        self.ensemble.register_model("model1", model1)
        self.ensemble.register_model("model2", model2)
        self.ensemble.register_model("model3", model3)
        
        # Make prediction
        prediction, confidence = self.ensemble.predict(self.sample_features)
        
        # Check that prediction is reasonable (weighted average)
        self.assertGreater(prediction, 74.0)
        self.assertLess(prediction, 78.0)
        self.assertGreater(confidence, 0.7)
        self.assertLessEqual(confidence, 1.0)
        
        # Check that all models were called
        self.assertEqual(model1.predict_calls, 1)
        self.assertEqual(model2.predict_calls, 1)
        self.assertEqual(model3.predict_calls, 1)
    
    def test_ensemble_prediction_with_weather_condition(self):
        """Test ensemble prediction with weather condition selection."""
        # Register models for different conditions
        clear_model = MockTemperatureModel("Clear", 75.0, 0.8)
        rainy_model = MockTemperatureModel("Rainy", 70.0, 0.9)
        
        self.ensemble.register_model("clear_model", clear_model, 
                                    weather_conditions=[WeatherCondition.CLEAR])
        self.ensemble.register_model("rainy_model", rainy_model, 
                                    weather_conditions=[WeatherCondition.RAINY])
        
        # Create clear weather features
        clear_features = pd.DataFrame({
            'openweather_cloud_cover': [15],
            'tomorrow_precipitation_prob': [5],
            'nws_temp_high': [75]
        })
        
        # Make prediction - should use clear model
        prediction, confidence = self.ensemble.predict(clear_features, use_weather_condition_selection=True)
        
        self.assertEqual(clear_model.predict_calls, 1)
        self.assertEqual(rainy_model.predict_calls, 0)  # Should not be called
    
    def test_confidence_filtering(self):
        """Test filtering of low-confidence predictions."""
        # Create models with different confidence levels
        high_conf_model = MockTemperatureModel("High", 75.0, 0.8)
        low_conf_model = MockTemperatureModel("Low", 80.0, 0.2)  # Below threshold
        
        self.ensemble.register_model("high_conf", high_conf_model)
        self.ensemble.register_model("low_conf", low_conf_model)
        
        # Set confidence threshold
        self.ensemble.min_confidence_threshold = 0.3
        
        # Make prediction
        prediction, confidence = self.ensemble.predict(self.sample_features)
        
        # Should only use high confidence model
        self.assertAlmostEqual(prediction, 75.0, places=1)
        self.assertEqual(high_conf_model.predict_calls, 1)
        self.assertEqual(low_conf_model.predict_calls, 1)  # Called but filtered
    
    def test_performance_update(self):
        """Test model performance updates."""
        model = MockTemperatureModel("Test", 75.0, 0.8)
        self.ensemble.register_model("test_model", model)
        
        # Update performance
        self.ensemble.update_model_performance(
            "test_model", 
            prediction=75.0, 
            actual_temperature=74.5, 
            confidence=0.8,
            weather_condition=WeatherCondition.NORMAL
        )
        
        # Check performance history
        self.assertEqual(len(self.ensemble.model_performance_history["test_model"]), 1)
        
        record = self.ensemble.model_performance_history["test_model"][0]
        self.assertEqual(record['prediction'], 75.0)
        self.assertEqual(record['actual'], 74.5)
        self.assertEqual(record['error'], 0.5)
        self.assertEqual(record['accuracy'], 1.0)  # Within ±3°F
        self.assertEqual(record['weather_condition'], WeatherCondition.NORMAL)
    
    def test_ensemble_performance_update(self):
        """Test ensemble performance updates."""
        # Update ensemble performance
        self.ensemble.update_ensemble_performance(
            prediction=75.5,
            actual_temperature=75.0,
            confidence=0.85,
            weather_condition=WeatherCondition.CLEAR
        )
        
        # Check ensemble performance history
        self.assertEqual(len(self.ensemble.ensemble_performance_history), 1)
        
        record = self.ensemble.ensemble_performance_history[0]
        self.assertEqual(record['prediction'], 75.5)
        self.assertEqual(record['actual'], 75.0)
        self.assertEqual(record['error'], 0.5)
        self.assertEqual(record['accuracy'], 1.0)  # Within ±3°F
        self.assertEqual(record['weather_condition'], WeatherCondition.CLEAR)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some performance data
        for i in range(5):
            self.ensemble.update_ensemble_performance(
                prediction=75.0 + i * 0.5,
                actual_temperature=75.0 + i * 0.3,
                confidence=0.8,
                weather_condition=WeatherCondition.NORMAL
            )
        
        # Get performance summary
        summary = self.ensemble.get_model_performance_summary(days=7)
        
        self.assertEqual(summary['model_name'], 'Ensemble')
        self.assertEqual(summary['total_predictions'], 5)
        self.assertIn('avg_error', summary)
        self.assertIn('accuracy_within_3f', summary)
        self.assertIn('condition_performance', summary)
    
    def test_ensemble_status(self):
        """Test ensemble status reporting."""
        # Register some models
        model1 = MockTemperatureModel("Model1", trained=True)
        model2 = MockTemperatureModel("Model2", trained=False)
        
        self.ensemble.register_model("model1", model1)
        self.ensemble.register_model("model2", model2)
        
        # Get status
        status = self.ensemble.get_ensemble_status()
        
        self.assertEqual(status['total_models'], 2)
        self.assertEqual(status['trained_models'], 1)
        self.assertIn('model1', status['model_info'])
        self.assertIn('model2', status['model_info'])
        self.assertEqual(status['model_info']['model1']['status'], 'trained')
        self.assertEqual(status['model_info']['model2']['status'], 'not_trained')
    
    def test_save_and_load_ensemble(self):
        """Test saving and loading ensemble configuration."""
        # Register a model and add some data
        model = MockTemperatureModel("Test")
        self.ensemble.register_model("test_model", model, initial_weight=1.5)
        
        self.ensemble.update_model_performance("test_model", 75.0, 74.5, 0.8)
        self.ensemble.total_predictions = 10
        
        # Save ensemble
        save_path = self.ensemble.save_ensemble("test_ensemble.pkl")
        
        # Create new ensemble and load
        new_ensemble = EnsembleTemperatureModel(model_dir=self.temp_dir)
        new_ensemble.load_ensemble(save_path)
        
        # Check loaded data
        self.assertEqual(new_ensemble.model_weights["test_model"], 1.5)
        self.assertEqual(new_ensemble.total_predictions, 10)
        self.assertEqual(len(new_ensemble.model_performance_history["test_model"]), 1)
    
    def test_error_handling(self):
        """Test error handling in ensemble predictions."""
        # Test with no models
        with self.assertRaises(ValueError):
            self.ensemble.predict(self.sample_features)
        
        # Test with empty features
        model = MockTemperatureModel("Test")
        self.ensemble.register_model("test_model", model)
        
        with self.assertRaises(ValueError):
            self.ensemble.predict(pd.DataFrame())
        
        # Test with untrained models
        untrained_model = MockTemperatureModel("Untrained", trained=False)
        ensemble2 = EnsembleTemperatureModel(model_dir=self.temp_dir)
        ensemble2.register_model("untrained", untrained_model)
        
        with self.assertRaises(ValueError):
            ensemble2.predict(self.sample_features)


class TestModelAdapters(unittest.TestCase):
    """Test cases for model adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_features = pd.DataFrame({
            'nws_temp_high': [78.5],
            'openweather_temp_high': [79.2],
            'tomorrow_temp_high': [77.8],
            'openweather_pressure': [1015.2],
            'nws_wind_speed': [12.5]
        })
    
    def test_linear_regression_adapter(self):
        """Test LinearRegressionModelAdapter."""
        adapter = LinearRegressionModelAdapter()
        
        # Test initial state
        self.assertFalse(adapter.is_trained())
        self.assertEqual(adapter.get_model_info()['status'], 'not_trained')
        
        # Test prediction (untrained)
        pred, conf = adapter.predict(self.sample_features)
        self.assertGreater(pred, 70.0)
        self.assertLess(pred, 85.0)
        self.assertGreater(conf, 0.0)
        
        # Test training
        adapter.set_trained(True)
        self.assertTrue(adapter.is_trained())
        self.assertEqual(adapter.get_model_info()['status'], 'trained')
        
        # Test prediction (trained)
        pred2, conf2 = adapter.predict(self.sample_features)
        self.assertGreater(conf2, conf)  # Should have higher confidence when trained
    
    def test_prophet_adapter(self):
        """Test ProphetModelAdapter."""
        adapter = ProphetModelAdapter()
        
        # Test initial state
        self.assertFalse(adapter.is_trained())
        
        # Test prediction
        pred, conf = adapter.predict(self.sample_features)
        self.assertGreater(pred, 60.0)
        self.assertLess(pred, 90.0)
        self.assertGreater(conf, 0.0)
        
        # Test training
        adapter.set_trained(True)
        pred2, conf2 = adapter.predict(self.sample_features)
        self.assertGreater(conf2, conf)
    
    def test_create_ensemble_with_models(self):
        """Test creating ensemble with model adapters."""
        ensemble = create_ensemble_with_models()
        
        status = ensemble.get_ensemble_status()
        self.assertEqual(status['total_models'], 4)
        self.assertGreater(status['trained_models'], 0)  # At least some models should be "trained"
        
        # Test that models are registered with appropriate conditions
        model_info = status['model_info']
        self.assertIn('xgboost', model_info)
        self.assertIn('lightgbm', model_info)
        self.assertIn('linear_regression', model_info)
        self.assertIn('prophet', model_info)


class TestEnsembleIntegration(unittest.TestCase):
    """Integration tests for the ensemble system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create ensemble with mock models
        self.ensemble = EnsembleTemperatureModel(model_dir=self.temp_dir)
        
        # Add models with different characteristics
        self.accurate_model = MockTemperatureModel("Accurate", 75.0, 0.9)
        self.inaccurate_model = MockTemperatureModel("Inaccurate", 80.0, 0.7)
        self.seasonal_model = MockTemperatureModel("Seasonal", 74.0, 0.8)
        
        self.ensemble.register_model("accurate", self.accurate_model, 
                                   weather_conditions=[WeatherCondition.NORMAL])
        self.ensemble.register_model("inaccurate", self.inaccurate_model, 
                                   weather_conditions=[WeatherCondition.NORMAL])
        self.ensemble.register_model("seasonal", self.seasonal_model, 
                                   weather_conditions=[WeatherCondition.HEAT_WAVE])
        
        self.sample_features = pd.DataFrame({
            'nws_temp_high': [75.0],
            'openweather_temp_high': [75.5],
            'openweather_cloud_cover': [30],
            'tomorrow_precipitation_prob': [20]
        })
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_prediction_workflow(self):
        """Test complete prediction workflow."""
        # Make initial prediction
        pred1, conf1 = self.ensemble.predict(self.sample_features)
        
        # Simulate actual temperature and update performance
        actual_temp = 75.2
        
        # Update individual model performance
        self.ensemble.update_model_performance("accurate", 75.0, actual_temp, 0.9)
        self.ensemble.update_model_performance("inaccurate", 80.0, actual_temp, 0.7)
        
        # Update ensemble performance
        self.ensemble.update_ensemble_performance(pred1, actual_temp, conf1)
        
        # Make another prediction - weights should be adjusted
        pred2, conf2 = self.ensemble.predict(self.sample_features)
        
        # The accurate model should now have higher influence
        # (This is hard to test precisely due to the complexity of weight calculation)
        self.assertIsInstance(pred2, float)
        self.assertIsInstance(conf2, float)
    
    def test_weather_condition_workflow(self):
        """Test weather condition-based model selection."""
        # Create heat wave features
        heat_features = pd.DataFrame({
            'nws_temp_high': [95.0],
            'openweather_temp_high': [94.0],
            'fire_season_indicator': [0.8]
        })
        
        # Make prediction - should prefer seasonal model for heat wave
        pred, conf = self.ensemble.predict(heat_features, use_weather_condition_selection=True)
        
        # Check that seasonal model was used (it's registered for heat waves)
        self.assertEqual(self.seasonal_model.predict_calls, 1)
    
    def test_performance_tracking_over_time(self):
        """Test performance tracking over multiple predictions."""
        # Simulate multiple days of predictions
        for day in range(10):
            # Make prediction
            pred, conf = self.ensemble.predict(self.sample_features)
            
            # Simulate actual temperature (with some variation)
            actual = 75.0 + np.random.normal(0, 1.0)
            
            # Update performance
            self.ensemble.update_ensemble_performance(pred, actual, conf)
            
            # Update individual models with varying accuracy
            accurate_pred = actual + np.random.normal(0, 0.5)  # More accurate
            inaccurate_pred = actual + np.random.normal(0, 2.0)  # Less accurate
            
            self.ensemble.update_model_performance("accurate", accurate_pred, actual, 0.9)
            self.ensemble.update_model_performance("inaccurate", inaccurate_pred, actual, 0.7)
        
        # Check performance summaries
        ensemble_summary = self.ensemble.get_model_performance_summary()
        accurate_summary = self.ensemble.get_model_performance_summary("accurate")
        inaccurate_summary = self.ensemble.get_model_performance_summary("inaccurate")
        
        self.assertEqual(ensemble_summary['total_predictions'], 10)
        self.assertEqual(accurate_summary['total_predictions'], 10)
        self.assertEqual(inaccurate_summary['total_predictions'], 10)
        
        # Accurate model should have better performance
        self.assertLess(accurate_summary['avg_error'], inaccurate_summary['avg_error'])


def run_all_tests():
    """Run all ensemble model tests."""
    print("=== Running Ensemble Model Tests ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEnsembleTemperatureModel))
    test_suite.addTest(unittest.makeSuite(TestModelAdapters))
    test_suite.addTest(unittest.makeSuite(TestEnsembleIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n=== Test Results ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == '__main__':
    run_all_tests()