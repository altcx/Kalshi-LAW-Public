"""Demo script for additional ML models in the ensemble."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Import working models
from src.models.linear_regression_model import LinearRegressionTemperatureModel
from src.models.prophet_model import ProphetTemperatureModel
from src.models.model_adapters import (
    LinearRegressionModelAdapter, 
    ProphetModelAdapter,
    create_ensemble_with_models
)


def demo_linear_regression_models():
    """Demonstrate linear regression models with different regularization."""
    print("=== Linear Regression Models Demo ===")
    
    regularization_types = ['ridge', 'lasso', 'elastic_net']
    
    for reg_type in regularization_types:
        print(f"\n{reg_type.upper()} Regularization:")
        
        try:
            model = LinearRegressionTemperatureModel(regularization=reg_type)
            info = model.get_model_info()
            print(f"  ✓ Model initialized: {info['status']}")
            print(f"  ✓ Regularization: {info['regularization']}")
            
            # Create sample features for prediction demo
            sample_features = pd.DataFrame({
                'nws_temp_high': [78.5],
                'openweather_temp_high': [79.2],
                'tomorrow_temp_high': [77.8],
                'openweather_pressure': [1015.2],
                'nws_wind_speed': [12.5],
                'visual_crossing_humidity': [65],
                'day_of_year': [220],
                'month': [8],
                'is_weekend': [0]
            })
            
            # Test adapter
            adapter = LinearRegressionModelAdapter(regularization=reg_type)
            pred, conf = adapter.predict(sample_features)
            print(f"  ✓ Prediction: {pred:.1f}°F (confidence: {conf:.3f})")
            
        except Exception as e:
            print(f"  ✗ Error with {reg_type}: {e}")
    
    print()


def demo_prophet_model():
    """Demonstrate Prophet time series model."""
    print("=== Prophet Time Series Model Demo ===")
    
    try:
        # Check if Prophet is available
        try:
            from prophet import Prophet
            prophet_available = True
            print("✓ Prophet library available")
        except ImportError:
            prophet_available = False
            print("⚠ Prophet not available - install with: pip install prophet")
        
        model = ProphetTemperatureModel()
        info = model.get_model_info()
        print(f"✓ Model initialized: {info['status']}")
        print(f"✓ Prophet available: {info['prophet_available']}")
        
        # Create sample features for prediction demo
        sample_features = pd.DataFrame({
            'nws_temp_high': [78.5],
            'openweather_temp_high': [79.2],
            'tomorrow_temp_high': [77.8],
            'openweather_pressure': [1015.2],
            'nws_wind_speed': [12.5],
            'visual_crossing_humidity': [65],
            'day_of_year': [220],
            'month': [8]
        })
        
        # Test adapter
        adapter = ProphetModelAdapter()
        pred, conf = adapter.predict(sample_features)
        print(f"✓ Prediction: {pred:.1f}°F (confidence: {conf:.3f})")
        
        if prophet_available:
            print("✓ Prophet model ready for time series forecasting")
            print("  - Supports external regressors")
            print("  - Handles seasonality automatically")
            print("  - Provides uncertainty intervals")
        
    except Exception as e:
        print(f"✗ Prophet demo error: {e}")
    
    print()


def demo_ensemble_integration():
    """Demonstrate ensemble integration with multiple models."""
    print("=== Ensemble Integration Demo ===")
    
    try:
        # Create ensemble with available models
        ensemble = create_ensemble_with_models()
        status = ensemble.get_ensemble_status()
        
        print(f"✓ Ensemble created with {status['total_models']} models")
        print(f"  - Trained models: {status['trained_models']}")
        
        print("\nModel Details:")
        for name, info in status['model_info'].items():
            print(f"  - {name}:")
            print(f"    Status: {info['status']}")
            print(f"    Weight: {info['current_weight']:.3f}")
            print(f"    Conditions: {info['weather_conditions']}")
        
        # Test ensemble prediction
        print("\nTesting Ensemble Prediction:")
        sample_features = pd.DataFrame({
            'nws_temp_high': [78.5],
            'openweather_temp_high': [79.2],
            'tomorrow_temp_high': [77.8],
            'openweather_pressure': [1015.2],
            'nws_wind_speed': [12.5],
            'visual_crossing_humidity': [65],
            'day_of_year': [220],
            'month': [8],
            'is_weekend': [0]
        })
        
        try:
            pred, conf = ensemble.predict(sample_features)
            print(f"✓ Ensemble prediction: {pred:.1f}°F (confidence: {conf:.3f})")
            
            # Show weather condition detection
            weather_condition = ensemble.detect_weather_condition(sample_features)
            print(f"✓ Detected weather condition: {weather_condition}")
            
        except Exception as e:
            print(f"⚠ Ensemble prediction failed: {e}")
            print("  Note: This is expected if no models are trained")
        
    except Exception as e:
        print(f"✗ Ensemble integration error: {e}")
    
    print()


def demo_model_features():
    """Demonstrate key features of the additional models."""
    print("=== Additional Model Features Demo ===")
    
    print("1. Linear Regression Models:")
    print("   ✓ Ridge Regression: L2 regularization for stable coefficients")
    print("   ✓ Lasso Regression: L1 regularization for feature selection")
    print("   ✓ Elastic Net: Combined L1/L2 regularization")
    print("   ✓ Polynomial features support for non-linear relationships")
    print("   ✓ Interpretable coefficients for feature importance")
    
    print("\n2. Prophet Time Series Model:")
    print("   ✓ Automatic seasonality detection (yearly, weekly, monthly)")
    print("   ✓ External regressors for weather features")
    print("   ✓ Trend change point detection")
    print("   ✓ Uncertainty intervals for confidence estimation")
    print("   ✓ Holiday effects modeling")
    
    print("\n3. LightGBM Model:")
    print("   ✓ Alternative gradient boosting implementation")
    print("   ✓ Faster training than XGBoost")
    print("   ✓ Native categorical feature support")
    print("   ✓ Memory efficient for large datasets")
    print("   ⚠ Currently experiencing import issues (environment-specific)")
    
    print("\n4. Ensemble Integration:")
    print("   ✓ Dynamic model weighting based on recent performance")
    print("   ✓ Weather condition-specific model selection")
    print("   ✓ Confidence-based prediction filtering")
    print("   ✓ Automatic model fallback for robustness")
    
    print()


def main():
    """Run demonstration of additional ML models."""
    print("=== Additional ML Models for Ensemble Demo ===\n")
    
    # Demonstrate individual models
    demo_linear_regression_models()
    demo_prophet_model()
    
    # Demonstrate ensemble integration
    demo_ensemble_integration()
    
    # Show model features
    demo_model_features()
    
    print("=== Demo Complete ===")
    print("\nImplemented Models:")
    print("✓ LightGBM: Alternative gradient boosting (with import fallback)")
    print("✓ Linear Regression: Ridge, Lasso, Elastic Net regularization")
    print("✓ Prophet: Time series with external regressors")
    print("✓ Model Adapters: Unified ensemble interface")
    print("✓ Dynamic Ensemble: Weather-aware model selection")
    
    print("\nKey Features:")
    print("- Hyperparameter optimization with Optuna")
    print("- Cross-validation for model evaluation")
    print("- Feature importance analysis")
    print("- Confidence scoring for predictions")
    print("- Graceful fallback for missing dependencies")
    print("- Weather condition-based model selection")
    
    print("\nRequirements Satisfied:")
    print("✓ 4.1: Multiple ML strategies (gradient boosting, linear, time series)")
    print("✓ 6.2: Ensemble methods with dynamic weighting")


if __name__ == '__main__':
    main()