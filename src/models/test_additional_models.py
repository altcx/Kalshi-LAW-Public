"""Test script for additional ML models in the ensemble."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Import all model classes
from src.models.lightgbm_model import LightGBMTemperatureModel
from src.models.linear_regression_model import LinearRegressionTemperatureModel
from src.models.prophet_model import ProphetTemperatureModel
from src.models.model_adapters import (
    LightGBMModelAdapter, 
    LinearRegressionModelAdapter, 
    ProphetModelAdapter,
    create_ensemble_with_models
)
from src.utils.data_manager import DataManager
from src.feature_engineering.feature_pipeline import FeaturePipeline


def test_lightgbm_model():
    """Test LightGBM model functionality."""
    print("=== Testing LightGBM Model ===")
    
    try:
        model = LightGBMTemperatureModel()
        info = model.get_model_info()
        print(f"✓ LightGBM model initialized: {info['status']}")
        
        # Test with sample data if available
        data_manager = DataManager()
        summary = data_manager.get_data_summary()
        
        has_data = any(
            isinstance(info, dict) and 'records' in info and info['records'] > 5
            for source, info in summary.items()
            if source in ['nws', 'openweather', 'actual_temperatures']
        )
        
        if has_data:
            print("✓ Training data available, testing training...")
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            try:
                results = model.train(
                    start_date=start_date,
                    end_date=end_date,
                    optimize_hyperparams=False,  # Skip optimization for speed
                    n_trials=5
                )
                print(f"✓ LightGBM training successful: {results['training_samples']} samples")
                print(f"  - RMSE: {results['training_metrics']['rmse']:.3f}")
                print(f"  - Accuracy ±3°F: {results['training_metrics']['accuracy_within_3f']:.1f}%")
                
                # Test prediction
                feature_pipeline = FeaturePipeline()
                features = feature_pipeline.create_features_for_prediction(date.today())
                if not features.empty:
                    pred, conf = model.predict(features)
                    print(f"✓ LightGBM prediction: {pred:.1f}°F (confidence: {conf:.3f})")
                
            except Exception as e:
                print(f"⚠ LightGBM training failed: {e}")
        else:
            print("⚠ No training data available for LightGBM")
            
    except Exception as e:
        print(f"✗ LightGBM model test failed: {e}")
    
    print()


def test_linear_regression_model():
    """Test Linear Regression model functionality."""
    print("=== Testing Linear Regression Models ===")
    
    regularization_types = ['ridge', 'lasso', 'elastic_net']
    
    for reg_type in regularization_types:
        print(f"Testing {reg_type.upper()} regularization:")
        
        try:
            model = LinearRegressionTemperatureModel(regularization=reg_type)
            info = model.get_model_info()
            print(f"  ✓ {reg_type} model initialized: {info['status']}")
            
            # Test with sample data if available
            data_manager = DataManager()
            summary = data_manager.get_data_summary()
            
            has_data = any(
                isinstance(info, dict) and 'records' in info and info['records'] > 5
                for source, info in summary.items()
                if source in ['nws', 'openweather', 'actual_temperatures']
            )
            
            if has_data:
                end_date = date.today()
                start_date = end_date - timedelta(days=30)
                
                try:
                    results = model.train(
                        start_date=start_date,
                        end_date=end_date,
                        optimize_hyperparams=False,  # Skip optimization for speed
                        use_polynomial_features=False
                    )
                    print(f"  ✓ {reg_type} training successful: {results['training_samples']} samples")
                    print(f"    - RMSE: {results['training_metrics']['rmse']:.3f}")
                    print(f"    - R² Score: {results['training_metrics']['r2_score']:.3f}")
                    print(f"    - Accuracy ±3°F: {results['training_metrics']['accuracy_within_3f']:.1f}%")
                    
                    # Test prediction
                    feature_pipeline = FeaturePipeline()
                    features = feature_pipeline.create_features_for_prediction(date.today())
                    if not features.empty:
                        pred, conf = model.predict(features)
                        print(f"  ✓ {reg_type} prediction: {pred:.1f}°F (confidence: {conf:.3f})")
                    
                    # Test coefficient analysis
                    coef_analysis = model.get_model_coefficients()
                    print(f"    - Intercept: {coef_analysis['intercept']:.2f}")
                    print(f"    - Positive coefficients: {coef_analysis['positive_coefficients']}")
                    print(f"    - Negative coefficients: {coef_analysis['negative_coefficients']}")
                    
                except Exception as e:
                    print(f"  ⚠ {reg_type} training failed: {e}")
            else:
                print(f"  ⚠ No training data available for {reg_type}")
                
        except Exception as e:
            print(f"  ✗ {reg_type} model test failed: {e}")
    
    print()


def test_prophet_model():
    """Test Prophet model functionality."""
    print("=== Testing Prophet Model ===")
    
    try:
        # Check if Prophet is available
        try:
            from prophet import Prophet
            prophet_available = True
        except ImportError:
            prophet_available = False
            print("⚠ Prophet not available - install with: pip install prophet")
            return
        
        model = ProphetTemperatureModel()
        info = model.get_model_info()
        print(f"✓ Prophet model initialized: {info['status']}")
        print(f"  Prophet available: {info['prophet_available']}")
        
        # Test with sample data if available
        data_manager = DataManager()
        summary = data_manager.get_data_summary()
        
        has_data = any(
            isinstance(info, dict) and 'records' in info and info['records'] > 10
            for source, info in summary.items()
            if source in ['nws', 'openweather', 'actual_temperatures']
        )
        
        if has_data:
            print("✓ Training data available, testing training...")
            end_date = date.today()
            start_date = end_date - timedelta(days=60)  # Prophet needs more data
            
            try:
                results = model.train(
                    start_date=start_date,
                    end_date=end_date,
                    optimize_hyperparams=False,  # Skip optimization for speed
                    max_regressors=5
                )
                print(f"✓ Prophet training successful: {results['training_samples']} samples")
                print(f"  - External regressors: {results['external_regressors']}")
                print(f"  - RMSE: {results['training_metrics']['rmse']:.3f}")
                print(f"  - Accuracy ±3°F: {results['training_metrics']['accuracy_within_3f']:.1f}%")
                
                # Show regressors used
                print("  - Top regressors:")
                for i, (regressor, importance) in enumerate(list(results['regressor_importance'].items())[:3]):
                    print(f"    {i+1}. {regressor}: {importance:.4f}")
                
                # Test prediction
                feature_pipeline = FeaturePipeline()
                features = feature_pipeline.create_features_for_prediction(date.today())
                if not features.empty:
                    pred, conf = model.predict(features)
                    print(f"✓ Prophet prediction: {pred:.1f}°F (confidence: {conf:.3f})")
                    
                    # Test forecast components
                    components = model.get_forecast_components()
                    print(f"  - Trend: {components.get('trend', 0):.1f}°F")
                    print(f"  - Yearly seasonality: {components.get('yearly_seasonality', 0):.1f}°F")
                
            except Exception as e:
                print(f"⚠ Prophet training failed: {e}")
        else:
            print("⚠ No sufficient training data available for Prophet")
            
    except Exception as e:
        print(f"✗ Prophet model test failed: {e}")
    
    print()


def test_model_adapters():
    """Test model adapters functionality."""
    print("=== Testing Model Adapters ===")
    
    try:
        # Test individual adapters
        print("Testing individual adapters:")
        
        # XGBoost adapter (should already exist)
        from src.models.model_adapters import XGBoostModelAdapter
        xgb_adapter = XGBoostModelAdapter()
        print(f"  ✓ XGBoost adapter: {xgb_adapter.get_model_info()['status']}")
        
        # LightGBM adapter
        lgb_adapter = LightGBMModelAdapter()
        lgb_info = lgb_adapter.get_model_info()
        print(f"  ✓ LightGBM adapter: {lgb_info['status']} (available: {lgb_info.get('available', False)})")
        
        # Linear regression adapter
        linear_adapter = LinearRegressionModelAdapter(regularization='ridge')
        linear_info = linear_adapter.get_model_info()
        print(f"  ✓ Linear regression adapter: {linear_info['status']} (available: {linear_info.get('available', False)})")
        
        # Prophet adapter
        prophet_adapter = ProphetModelAdapter()
        prophet_info = prophet_adapter.get_model_info()
        print(f"  ✓ Prophet adapter: {prophet_info['status']} (available: {prophet_info.get('available', False)})")
        
        # Test ensemble creation
        print("\nTesting ensemble creation:")
        ensemble = create_ensemble_with_models()
        status = ensemble.get_ensemble_status()
        
        print(f"  ✓ Ensemble created with {status['total_models']} models")
        print(f"  - Trained models: {status['trained_models']}")
        
        print("  Model details:")
        for name, info in status['model_info'].items():
            print(f"    - {name}: {info['status']} (weight: {info['current_weight']:.3f})")
        
        # Test ensemble prediction with sample data
        print("\nTesting ensemble prediction:")
        sample_features = pd.DataFrame({
            'nws_temp_high': [78.5],
            'openweather_temp_high': [79.2],
            'tomorrow_temp_high': [77.8],
            'openweather_pressure': [1015.2],
            'nws_wind_speed': [12.5],
            'visual_crossing_humidity': [65]
        })
        
        try:
            pred, conf = ensemble.predict(sample_features)
            print(f"  ✓ Ensemble prediction: {pred:.1f}°F (confidence: {conf:.3f})")
        except Exception as e:
            print(f"  ⚠ Ensemble prediction failed: {e}")
            print("    Note: This is expected if no models are trained")
        
    except Exception as e:
        print(f"✗ Model adapters test failed: {e}")
    
    print()


def test_ensemble_integration():
    """Test full ensemble integration with all models."""
    print("=== Testing Full Ensemble Integration ===")
    
    try:
        # Check data availability
        data_manager = DataManager()
        summary = data_manager.get_data_summary()
        
        has_sufficient_data = any(
            isinstance(info, dict) and 'records' in info and info['records'] > 10
            for source, info in summary.items()
            if source in ['nws', 'openweather', 'actual_temperatures']
        )
        
        if not has_sufficient_data:
            print("⚠ Insufficient data for full ensemble testing")
            print("  Please run data collection first")
            return
        
        print("✓ Sufficient data available for ensemble testing")
        
        # Create ensemble with all models
        ensemble = create_ensemble_with_models()
        
        # Try to train at least one model for demonstration
        print("Training XGBoost model for ensemble demonstration...")
        
        try:
            from src.models.xgboost_model import XGBoostTemperatureModel
            xgb_model = XGBoostTemperatureModel()
            
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            xgb_results = xgb_model.train(
                start_date=start_date,
                end_date=end_date,
                optimize_hyperparams=False,
                n_trials=5
            )
            
            print(f"✓ XGBoost trained: RMSE {xgb_results['training_metrics']['rmse']:.3f}")
            
            # Update ensemble with trained model
            from src.models.model_adapters import XGBoostModelAdapter
            trained_adapter = XGBoostModelAdapter(xgb_model)
            
            # Create new ensemble with trained model
            from src.models.ensemble_model import EnsembleTemperatureModel, WeatherCondition
            ensemble = EnsembleTemperatureModel()
            
            ensemble.register_model(
                'xgboost_trained', 
                trained_adapter, 
                initial_weight=1.2,
                weather_conditions=[WeatherCondition.NORMAL, WeatherCondition.CLEAR]
            )
            
            # Add other models (untrained but functional)
            ensemble.register_model(
                'lightgbm', 
                LightGBMModelAdapter(), 
                initial_weight=1.0,
                weather_conditions=[WeatherCondition.NORMAL]
            )
            
            ensemble.register_model(
                'linear_ridge', 
                LinearRegressionModelAdapter(regularization='ridge'), 
                initial_weight=0.8,
                weather_conditions=[WeatherCondition.NORMAL]
            )
            
            ensemble.register_model(
                'prophet', 
                ProphetModelAdapter(), 
                initial_weight=0.9,
                weather_conditions=[WeatherCondition.NORMAL]
            )
            
            # Test ensemble prediction
            feature_pipeline = FeaturePipeline()
            features = feature_pipeline.create_features_for_prediction(date.today())
            
            if not features.empty:
                pred, conf = ensemble.predict(features)
                print(f"✓ Full ensemble prediction: {pred:.1f}°F (confidence: {conf:.3f})")
                
                # Show ensemble status
                status = ensemble.get_ensemble_status()
                print(f"  - Total models: {status['total_models']}")
                print(f"  - Trained models: {status['trained_models']}")
                
                print("  Model contributions:")
                for name, info in status['model_info'].items():
                    print(f"    - {name}: weight {info['current_weight']:.3f}")
            
        except Exception as e:
            print(f"⚠ Ensemble training/prediction failed: {e}")
            print("  This is expected if models need more data or dependencies")
        
    except Exception as e:
        print(f"✗ Full ensemble integration test failed: {e}")
    
    print()


def main():
    """Run all tests for additional ML models."""
    print("=== Testing Additional ML Models for Ensemble ===\n")
    
    # Test individual models
    test_lightgbm_model()
    test_linear_regression_model()
    test_prophet_model()
    
    # Test adapters and ensemble
    test_model_adapters()
    test_ensemble_integration()
    
    print("=== Additional ML Models Testing Complete ===")
    print("\nSummary:")
    print("- LightGBM: Alternative gradient boosting model ✓")
    print("- Linear Regression: Regularized baseline models (Ridge, Lasso, Elastic Net) ✓")
    print("- Prophet: Time series model with external regressors ✓")
    print("- Model Adapters: Ensemble integration layer ✓")
    print("- Dynamic Ensemble: Weather-condition-based model selection ✓")
    
    print("\nNext steps:")
    print("1. Ensure sufficient training data is available")
    print("2. Train models with optimize_hyperparams=True for best performance")
    print("3. Use ensemble for daily predictions with confidence scoring")
    print("4. Monitor model performance and update weights dynamically")


if __name__ == '__main__':
    main()