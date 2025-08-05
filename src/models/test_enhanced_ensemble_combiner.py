"""Tests for the enhanced ensemble combiner system."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.models.enhanced_ensemble_combiner import (
    EnhancedEnsembleCombiner, ModelPrediction, EnsemblePrediction,
    ModelPerformanceMetric
)
from src.models.ensemble_model import EnsembleTemperatureModel, WeatherCondition, BaseTemperatureModel


class MockTemperatureModel(BaseTemperatureModel):
    """Mock temperature model for testing."""
    
    def __init__(self, name: str, prediction: float, confidence: float, trained: bool = True):
        self.name = name
        self.prediction = prediction
        self.confidence = confidence
        self.trained = trained
    
    def predict(self, features: pd.DataFrame) -> tuple:
        return self.prediction, self.confidence
    
    def is_trained(self) -> bool:
        return self.trained
    
    def get_model_info(self) -> dict:
        return {
            'status': 'trained' if self.trained else 'not_trained',
            'model_type': f'Mock_{self.name}'
        }


class TestEnhancedEnsembleCombiner(unittest.TestCase):
    """Test cases for EnhancedEnsembleCombiner."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        
        # Create base ensemble
        self.base_ensemble = EnsembleTemperatureModel(model_dir=self.temp_dir)
        
        # Create mock models
        self.model1 = MockTemperatureModel("Model1", 75.0, 0.8, True)
        self.model2 = MockTemperatureModel("Model2", 76.0, 0.9, True)
        self.model3 = MockTemperatureModel("Model3", 77.0, 0.7, True)
        self.model4 = MockTemperatureModel("Model4", 74.0, 0.6, True)
        
        # Register models with different weather conditions
        self.base_ensemble.register_model("model1", self.model1, 1.0, [WeatherCondition.NORMAL])
        self.base_ensemble.register_model("model2", self.model2, 1.0, [WeatherCondition.CLEAR])
        self.base_ensemble.register_model("model3", self.model3, 1.0, [WeatherCondition.CLOUDY])
        self.base_ensemble.register_model("model4", self.model4, 1.0, [WeatherCondition.RAINY])
        
        # Create enhanced combiner
        self.combiner = EnhancedEnsembleCombiner(self.base_ensemble)
        
        # Sample features
        self.sample_features = pd.DataFrame({
            'nws_temp_high': [78.5],
            'openweather_temp_high': [79.2],
            'tomorrow_temp_high': [77.8],
            'openweather_pressure': [1015.2],
            'nws_wind_speed': [12.5],
            'visual_crossing_humidity': [65]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test enhanced combiner initialization."""
        self.assertIsInstance(self.combiner.base_ensemble, EnsembleTemperatureModel)
        self.assertEqual(self.combiner.default_weighting_strategy, 'adaptive')
        self.assertEqual(self.combiner.default_selection_strategy, 'adaptive_selection')
        self.assertEqual(self.combiner.min_models_for_ensemble, 2)
        self.assertEqual(self.combiner.max_models_for_ensemble, 4)
        self.assertGreater(self.combiner.confidence_boost_factor, 0)
    
    def test_model_selection_by_weather_condition(self):
        """Test model selection by weather condition."""
        # Test normal condition
        selected = self.combiner._select_by_weather_condition(
            self.sample_features, WeatherCondition.NORMAL
        )
        self.assertIn("model1", selected)
        
        # Test clear condition
        selected = self.combiner._select_by_weather_condition(
            self.sample_features, WeatherCondition.CLEAR
        )
        self.assertIn("model2", selected)
    
    def test_model_selection_by_performance(self):
        """Test model selection by performance."""
        # Add some performance history
        self.base_ensemble.update_model_performance("model1", 75.0, 74.5, 0.8)
        self.base_ensemble.update_model_performance("model2", 76.0, 78.0, 0.9)  # Worse performance
        
        selected = self.combiner._select_by_performance(
            self.sample_features, WeatherCondition.NORMAL
        )
        
        # Should prefer model1 due to better performance
        self.assertIn("model1", selected)
        self.assertTrue(len(selected) <= self.combiner.max_models_for_ensemble)
    
    def test_model_selection_for_diversity(self):
        """Test model selection for diversity."""
        # Create models with diverse predictions
        diverse_model1 = MockTemperatureModel("Diverse1", 70.0, 0.8, True)
        diverse_model2 = MockTemperatureModel("Diverse2", 80.0, 0.8, True)
        diverse_model3 = MockTemperatureModel("Diverse3", 75.0, 0.8, True)
        
        self.base_ensemble.register_model("diverse1", diverse_model1)
        self.base_ensemble.register_model("diverse2", diverse_model2)
        self.base_ensemble.register_model("diverse3", diverse_model3)
        
        selected = self.combiner._select_for_diversity(
            self.sample_features, WeatherCondition.NORMAL
        )
        
        self.assertTrue(len(selected) <= self.combiner.max_models_for_ensemble)
        self.assertGreater(len(selected), 0)
    
    def test_adaptive_model_selection(self):
        """Test adaptive model selection."""
        # Add performance history
        self.base_ensemble.update_model_performance("model1", 75.0, 74.8, 0.8)
        self.base_ensemble.update_model_performance("model2", 76.0, 75.5, 0.9)
        
        selected = self.combiner._adaptive_model_selection(
            self.sample_features, WeatherCondition.NORMAL
        )
        
        self.assertTrue(len(selected) <= self.combiner.max_models_for_ensemble)
        self.assertGreater(len(selected), 0)
    
    def test_get_model_predictions(self):
        """Test getting model predictions."""
        model_names = ["model1", "model2"]
        predictions = self.combiner._get_model_predictions(
            model_names, self.sample_features, WeatherCondition.NORMAL
        )
        
        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertIsInstance(pred, ModelPrediction)
            self.assertIn(pred.model_name, model_names)
            self.assertGreater(pred.confidence, 0)
    
    def test_performance_based_weighting(self):
        """Test performance-based weighting."""
        # Create model predictions
        predictions = [
            ModelPrediction("model1", 75.0, 0.8, 0.0, WeatherCondition.NORMAL),
            ModelPrediction("model2", 76.0, 0.9, 0.0, WeatherCondition.NORMAL)
        ]
        
        # Add performance history (model1 better)
        self.base_ensemble.update_model_performance("model1", 75.0, 74.8, 0.8)
        self.base_ensemble.update_model_performance("model2", 76.0, 78.0, 0.9)
        
        weights = self.combiner._calculate_performance_weights(
            predictions, WeatherCondition.NORMAL
        )
        
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertGreater(weights["model1"], weights["model2"])  # Better performance = higher weight
    
    def test_confidence_based_weighting(self):
        """Test confidence-based weighting."""
        predictions = [
            ModelPrediction("model1", 75.0, 0.9, 0.0, WeatherCondition.NORMAL),
            ModelPrediction("model2", 76.0, 0.7, 0.0, WeatherCondition.NORMAL)
        ]
        
        weights = self.combiner._calculate_confidence_weights(
            predictions, WeatherCondition.NORMAL
        )
        
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertGreater(weights["model1"], weights["model2"])  # Higher confidence = higher weight
    
    def test_hybrid_weighting(self):
        """Test hybrid weighting strategy."""
        predictions = [
            ModelPrediction("model1", 75.0, 0.8, 0.0, WeatherCondition.NORMAL),
            ModelPrediction("model2", 76.0, 0.9, 0.0, WeatherCondition.NORMAL)
        ]
        
        weights = self.combiner._calculate_hybrid_weights(
            predictions, WeatherCondition.NORMAL
        )
        
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertEqual(len(weights), 2)
    
    def test_adaptive_weighting(self):
        """Test adaptive weighting strategy."""
        predictions = [
            ModelPrediction("model1", 75.0, 0.8, 0.0, WeatherCondition.NORMAL),
            ModelPrediction("model2", 76.0, 0.9, 0.0, WeatherCondition.NORMAL)
        ]
        
        weights = self.combiner._calculate_adaptive_weights(
            predictions, WeatherCondition.NORMAL
        )
        
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertEqual(len(weights), 2)
    
    def test_combine_predictions(self):
        """Test prediction combination."""
        predictions = [
            ModelPrediction("model1", 75.0, 0.8, 0.6, WeatherCondition.NORMAL),
            ModelPrediction("model2", 77.0, 0.9, 0.4, WeatherCondition.NORMAL)
        ]
        
        ensemble_pred, ensemble_conf = self.combiner._combine_predictions(predictions)
        
        # Should be weighted average
        expected_pred = (75.0 * 0.6 * 0.8 + 77.0 * 0.4 * 0.9) / (0.6 * 0.8 + 0.4 * 0.9)
        self.assertAlmostEqual(ensemble_pred, expected_pred, places=2)
        self.assertGreater(ensemble_conf, 0)
        self.assertLessEqual(ensemble_conf, 1.0)
    
    def test_confidence_boost_calculation(self):
        """Test confidence boost calculation."""
        # Test with single model (no boost)
        single_pred = [ModelPrediction("model1", 75.0, 0.8, 0.5, WeatherCondition.NORMAL)]
        boost = self.combiner._calculate_confidence_boost(single_pred)
        self.assertEqual(boost, 0.0)
        
        # Test with multiple models
        multi_preds = [
            ModelPrediction("model1", 75.0, 0.8, 0.5, WeatherCondition.NORMAL),
            ModelPrediction("model2", 75.5, 0.9, 0.5, WeatherCondition.NORMAL)
        ]
        boost = self.combiner._calculate_confidence_boost(multi_preds)
        self.assertGreater(boost, 0.0)
        self.assertLessEqual(boost, self.combiner.confidence_boost_factor)
    
    def test_full_prediction_workflow(self):
        """Test complete prediction workflow."""
        # Add some performance history
        self.base_ensemble.update_model_performance("model1", 75.0, 74.8, 0.8)
        self.base_ensemble.update_model_performance("model2", 76.0, 75.9, 0.9)
        
        ensemble_pred = self.combiner.predict(self.sample_features)
        
        self.assertIsInstance(ensemble_pred, EnsemblePrediction)
        self.assertGreater(ensemble_pred.prediction, 70.0)
        self.assertLess(ensemble_pred.prediction, 80.0)
        self.assertGreater(ensemble_pred.confidence, 0.0)
        self.assertLessEqual(ensemble_pred.confidence, 1.0)
        self.assertGreater(ensemble_pred.total_models_used, 0)
        self.assertGreaterEqual(ensemble_pred.confidence_boost, 0.0)
    
    def test_prediction_with_different_strategies(self):
        """Test predictions with different weighting and selection strategies."""
        strategies = [
            ('performance_based', 'weather_condition'),
            ('confidence_weighted', 'performance_threshold'),
            ('hybrid', 'diversity_based'),
            ('adaptive', 'adaptive_selection')
        ]
        
        for weight_strategy, select_strategy in strategies:
            with self.subTest(weight=weight_strategy, select=select_strategy):
                try:
                    ensemble_pred = self.combiner.predict(
                        self.sample_features,
                        weighting_strategy=weight_strategy,
                        selection_strategy=select_strategy
                    )
                    
                    self.assertIsInstance(ensemble_pred, EnsemblePrediction)
                    self.assertGreater(ensemble_pred.prediction, 0)
                    self.assertGreater(ensemble_pred.confidence, 0)
                    
                except Exception as e:
                    self.fail(f"Strategy combination failed: {e}")
    
    def test_ensemble_analysis(self):
        """Test ensemble prediction analysis."""
        ensemble_pred = self.combiner.predict(self.sample_features)
        analysis = self.combiner.get_ensemble_analysis(ensemble_pred)
        
        # Check required fields
        required_fields = [
            'ensemble_prediction', 'ensemble_confidence', 'weather_condition',
            'ensemble_method', 'models_used', 'confidence_boost',
            'prediction_stats', 'weight_stats', 'confidence_stats',
            'model_contributions'
        ]
        
        for field in required_fields:
            self.assertIn(field, analysis)
        
        # Check statistics
        self.assertIn('mean', analysis['prediction_stats'])
        self.assertIn('std', analysis['prediction_stats'])
        self.assertIn('mean', analysis['weight_stats'])
        self.assertIn('weight_entropy', analysis['weight_stats'])
        
        # Check model contributions
        self.assertGreater(len(analysis['model_contributions']), 0)
        for contrib in analysis['model_contributions']:
            self.assertIn('model', contrib)
            self.assertIn('prediction', contrib)
            self.assertIn('confidence', contrib)
            self.assertIn('weight', contrib)
    
    def test_configuration_update(self):
        """Test configuration updates."""
        original_min = self.combiner.min_models_for_ensemble
        original_max = self.combiner.max_models_for_ensemble
        
        # Update configuration
        self.combiner.update_configuration(
            min_models_for_ensemble=3,
            max_models_for_ensemble=5,
            confidence_boost_factor=0.2
        )
        
        self.assertEqual(self.combiner.min_models_for_ensemble, 3)
        self.assertEqual(self.combiner.max_models_for_ensemble, 5)
        self.assertEqual(self.combiner.confidence_boost_factor, 0.2)
        
        # Test invalid parameter
        self.combiner.update_configuration(invalid_param="test")
        # Should not raise error, just log warning
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with no trained models
        empty_ensemble = EnsembleTemperatureModel(model_dir=self.temp_dir)
        empty_combiner = EnhancedEnsembleCombiner(empty_ensemble)
        
        with self.assertRaises(ValueError):
            empty_combiner.predict(self.sample_features)
        
        # Test with empty features
        with self.assertRaises(ValueError):
            self.combiner.predict(pd.DataFrame())
        
        # Test combine predictions with empty list
        with self.assertRaises(ValueError):
            self.combiner._combine_predictions([])
    
    def test_weather_condition_specific_predictions(self):
        """Test predictions for different weather conditions."""
        # Create features for different weather conditions
        weather_features = {
            WeatherCondition.CLEAR: pd.DataFrame({
                'openweather_cloud_cover': [10],
                'tomorrow_precipitation_prob': [5]
            }),
            WeatherCondition.RAINY: pd.DataFrame({
                'openweather_cloud_cover': [85],
                'tomorrow_precipitation_prob': [80]
            }),
            WeatherCondition.MARINE_LAYER: pd.DataFrame({
                'visual_crossing_humidity': [88],
                'marine_layer_indicator': [0.9]
            })
        }
        
        for condition, features in weather_features.items():
            with self.subTest(condition=condition):
                try:
                    ensemble_pred = self.combiner.predict(features)
                    self.assertIsInstance(ensemble_pred, EnsemblePrediction)
                    # Weather condition might not be detected perfectly with limited features
                    # but prediction should still work
                except Exception as e:
                    self.fail(f"Prediction failed for {condition}: {e}")
    
    def test_model_filtering_by_confidence(self):
        """Test that low-confidence models are filtered out."""
        # Create a model with very low confidence
        low_conf_model = MockTemperatureModel("LowConf", 75.0, 0.1, True)
        self.base_ensemble.register_model("low_conf", low_conf_model)
        
        # Set high confidence threshold
        self.base_ensemble.min_confidence_threshold = 0.5
        
        predictions = self.combiner._get_model_predictions(
            ["model1", "low_conf"], self.sample_features, WeatherCondition.NORMAL
        )
        
        # Low confidence model should be filtered out
        model_names = [pred.model_name for pred in predictions]
        self.assertIn("model1", model_names)
        self.assertNotIn("low_conf", model_names)


class TestModelPrediction(unittest.TestCase):
    """Test ModelPrediction dataclass."""
    
    def test_model_prediction_creation(self):
        """Test ModelPrediction creation."""
        pred = ModelPrediction(
            model_name="test_model",
            prediction=75.5,
            confidence=0.8,
            weight=0.6,
            weather_condition=WeatherCondition.NORMAL
        )
        
        self.assertEqual(pred.model_name, "test_model")
        self.assertEqual(pred.prediction, 75.5)
        self.assertEqual(pred.confidence, 0.8)
        self.assertEqual(pred.weight, 0.6)
        self.assertEqual(pred.weather_condition, WeatherCondition.NORMAL)
        self.assertIsNone(pred.feature_importance)


class TestEnsemblePrediction(unittest.TestCase):
    """Test EnsemblePrediction dataclass."""
    
    def test_ensemble_prediction_creation(self):
        """Test EnsemblePrediction creation."""
        model_preds = [
            ModelPrediction("model1", 75.0, 0.8, 0.6, WeatherCondition.NORMAL),
            ModelPrediction("model2", 76.0, 0.9, 0.4, WeatherCondition.NORMAL)
        ]
        
        ensemble_pred = EnsemblePrediction(
            prediction=75.4,
            confidence=0.85,
            weather_condition=WeatherCondition.NORMAL,
            model_predictions=model_preds,
            ensemble_method="adaptive+adaptive_selection",
            total_models_used=2,
            confidence_boost=0.05
        )
        
        self.assertEqual(ensemble_pred.prediction, 75.4)
        self.assertEqual(ensemble_pred.confidence, 0.85)
        self.assertEqual(ensemble_pred.weather_condition, WeatherCondition.NORMAL)
        self.assertEqual(len(ensemble_pred.model_predictions), 2)
        self.assertEqual(ensemble_pred.ensemble_method, "adaptive+adaptive_selection")
        self.assertEqual(ensemble_pred.total_models_used, 2)
        self.assertEqual(ensemble_pred.confidence_boost, 0.05)


if __name__ == '__main__':
    unittest.main()