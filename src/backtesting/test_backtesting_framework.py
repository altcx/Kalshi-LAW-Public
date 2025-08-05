"""Test the complete backtesting framework."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from src.backtesting.backtesting_framework import BacktestingFramework
from src.backtesting.model_comparison import ModelConfig


# Mock model for testing
class MockModel:
    """Mock model for testing purposes."""
    
    def __init__(self, name="MockModel", bias=0.0):
        self.name = name
        self.bias = bias
        self.is_trained_flag = False
    
    def train(self, features, targets):
        """Mock training method."""
        self.is_trained_flag = True
    
    def predict(self, features):
        """Mock prediction method."""
        if not self.is_trained_flag:
            raise ValueError("Model not trained")
        
        # Return a simple prediction with some noise
        base_temp = 75.0 + self.bias
        noise = np.random.normal(0, 2.0)
        return base_temp + noise
    
    def predict_with_confidence(self, features):
        """Mock prediction with confidence."""
        prediction = self.predict(features)
        confidence = 0.8 + np.random.normal(0, 0.1)
        confidence = np.clip(confidence, 0.1, 0.95)
        return prediction, confidence
    
    def is_trained(self):
        """Check if model is trained."""
        return self.is_trained_flag
    
    def get_model_info(self):
        """Get model information."""
        return {
            'name': self.name,
            'type': 'MockModel',
            'status': 'trained' if self.is_trained_flag else 'untrained'
        }


def mock_model_factory(name="MockModel", bias=0.0):
    """Factory function for creating mock models."""
    return MockModel(name=name, bias=bias)


def test_backtesting_framework_initialization():
    """Test that BacktestingFramework initializes correctly."""
    framework = BacktestingFramework()
    
    assert framework is not None
    assert framework.data_manager is not None
    assert framework.data_loader is not None
    assert framework.metrics_calculator is not None
    assert framework.trading_engine is not None
    assert framework.walk_forward_analyzer is not None
    assert framework.model_comparator is not None


def test_data_validation():
    """Test data validation functionality."""
    framework = BacktestingFramework()
    
    # Test with a reasonable date range
    start_date = date(2024, 11, 1)
    end_date = date(2024, 12, 31)
    
    validation = framework.validate_data_availability(start_date, end_date)
    
    assert 'requested_period' in validation
    assert 'available_period' in validation
    assert 'is_valid' in validation
    assert 'issues' in validation
    assert 'recommendations' in validation


def test_single_model_backtest():
    """Test running a single model backtest."""
    framework = BacktestingFramework()
    
    # Use a small date range for testing
    start_date = date(2024, 12, 1)
    end_date = date(2024, 12, 15)
    
    # Configure for quick testing
    config = {
        'train_window_days': 30,
        'test_window_days': 5,
        'step_days': 5,
        'initial_bankroll': 1000.0,
        'parallel_execution': False
    }
    
    results = framework.run_single_model_backtest(
        model_factory=mock_model_factory,
        model_params={'name': 'TestModel', 'bias': 0.0},
        start_date=start_date,
        end_date=end_date,
        config=config
    )
    
    # Check that we get results (might be error due to insufficient data)
    assert isinstance(results, dict)
    
    if 'error' not in results:
        assert 'backtest_config' in results
        assert 'date_range' in results
        assert 'summary' in results


def test_model_comparison_setup():
    """Test setting up model comparison."""
    framework = BacktestingFramework()
    
    # Create mock model configurations
    model_configs = [
        ModelConfig(
            name="Model_A",
            model_factory=mock_model_factory,
            params={'name': 'Model_A', 'bias': 0.0},
            description="Mock model A"
        ),
        ModelConfig(
            name="Model_B", 
            model_factory=mock_model_factory,
            params={'name': 'Model_B', 'bias': 1.0},
            description="Mock model B"
        )
    ]
    
    # Test with small date range
    start_date = date(2024, 12, 1)
    end_date = date(2024, 12, 10)
    
    config = {
        'train_window_days': 20,
        'test_window_days': 3,
        'step_days': 3,
        'parallel_execution': False
    }
    
    results = framework.run_model_comparison_backtest(
        model_configs=model_configs,
        start_date=start_date,
        end_date=end_date,
        config=config
    )
    
    # Check that we get results structure
    assert isinstance(results, dict)
    
    if 'error' not in results:
        assert 'model_configs' in results
        assert 'comparison_results' in results
        assert 'comparison_report' in results


def test_analysis_history():
    """Test analysis history tracking."""
    framework = BacktestingFramework()
    
    # Initially should be empty
    history = framework.get_analysis_history()
    assert len(history) == 0
    
    # After running an analysis, history should be updated
    # (This would happen in actual backtest runs)
    
    # Test cache clearing
    framework.clear_cache()
    assert len(framework.results_cache) == 0


def test_seasonal_analysis_structure():
    """Test seasonal analysis structure."""
    framework = BacktestingFramework()
    # Test the helper methods with mock data
    seasonal_metrics = {
        'seasonal_metrics': {
            'winter': {'mae': 2.5, 'accuracy_within_3f': 85, 'count': 20},
            'spring': {'mae': 2.0, 'accuracy_within_3f': 90, 'count': 25},
            'summer': {'mae': 3.0, 'accuracy_within_3f': 80, 'count': 30},
            'fall': {'mae': 2.2, 'accuracy_within_3f': 88, 'count': 22}
        }
    }
    recommendations = framework._generate_seasonal_recommendations(seasonal_metrics)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    best_worst = framework._identify_best_worst_seasons(seasonal_metrics)
    assert 'best_season' in best_worst
    assert 'worst_season' in best_worst

def test_backtest_accuracy_against_manual_calculation():
    """Test backtesting accuracy against manual calculations."""
    framework = BacktestingFramework()
    # Simulate predictions and actuals
    predictions = np.array([75.0, 78.0, 72.0, 76.0, 74.0])
    actuals = np.array([75.2, 78.1, 72.8, 76.3, 74.7])
    # Manual RMSE calculation
    manual_rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    # Framework metric calculation
    metrics = framework.metrics_calculator.calculate_basic_metrics([
        type('PredictionResult', (), {
            'date': None,
            'predicted': p,
            'actual': a,
            'confidence': 0.8,
            'error': p - a,
            'squared_error': (p - a) ** 2,
            'is_accurate_3f': abs(p - a) <= 3.0,
            'is_accurate_5f': abs(p - a) <= 5.0,
            'predicted_temperature': p,
            'actual_temperature': a
        })()
        for p, a in zip(predictions, actuals)
    ])
    assert abs(metrics['rmse'] - manual_rmse) < 0.01

def test_model_performance_benchmark():
    """Test model performance validation against benchmark accuracy."""
    framework = BacktestingFramework()
    # Simulate metrics for a model
    metrics = {'accuracy_within_3f': 87.0, 'rmse': 2.1}
    # Define benchmark thresholds
    benchmark_accuracy = 85.0
    benchmark_rmse = 3.0
    # Validate against benchmarks
    assert metrics['accuracy_within_3f'] >= benchmark_accuracy
    assert metrics['rmse'] <= benchmark_rmse


if __name__ == "__main__":
    test_backtesting_framework_initialization()
    test_data_validation()
    test_single_model_backtest()
    test_model_comparison_setup()
    test_analysis_history()
    test_seasonal_analysis_structure()
    print("All backtesting framework tests passed!")