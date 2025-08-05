"""Demonstration of the ensemble temperature prediction model."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from loguru import logger

from src.models.ensemble_model import EnsembleTemperatureModel, WeatherCondition
from src.models.model_adapters import create_ensemble_with_models
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.utils.data_manager import DataManager


def create_sample_features() -> pd.DataFrame:
    """Create sample weather features for demonstration."""
    return pd.DataFrame({
        # Temperature predictions from different APIs
        'nws_temp_high': [78.5],
        'openweather_temp_high': [79.2],
        'tomorrow_temp_high': [77.8],
        'visual_crossing_temp_high': [78.0],
        'weatherbit_temp_high': [78.8],
        
        # Atmospheric conditions
        'openweather_pressure': [1015.2],
        'nws_humidity': [65],
        'tomorrow_wind_speed': [12.5],
        'visual_crossing_cloud_cover': [30],
        'weatherbit_precipitation_prob': [15],
        
        # Ensemble features
        'temp_consensus_mean': [78.46],
        'temp_consensus_std': [0.52],
        'api_agreement_score': [0.85],
        'temp_rolling_3d_avg': [77.2],
        'temp_trend_7d': [1.2],
        
        # LA-specific patterns
        'marine_layer_indicator': [0.3],
        'santa_ana_indicator': [0.1],
        'heat_island_effect': [2.1],
        'fire_season_indicator': [0.4],
        
        # Quality metrics
        'data_quality_score': [0.92],
        'missing_data_ratio': [0.05]
    })


def create_weather_condition_features() -> dict:
    """Create features for different weather conditions."""
    return {
        WeatherCondition.CLEAR: pd.DataFrame({
            'nws_temp_high': [80.0],
            'openweather_temp_high': [79.5],
            'openweather_cloud_cover': [10],
            'tomorrow_precipitation_prob': [5],
            'nws_wind_speed': [8],
            'visual_crossing_humidity': [55]
        }),
        
        WeatherCondition.RAINY: pd.DataFrame({
            'nws_temp_high': [72.0],
            'openweather_temp_high': [71.5],
            'openweather_cloud_cover': [85],
            'tomorrow_precipitation_prob': [80],
            'nws_wind_speed': [15],
            'visual_crossing_humidity': [85]
        }),
        
        WeatherCondition.MARINE_LAYER: pd.DataFrame({
            'nws_temp_high': [75.0],
            'openweather_temp_high': [74.5],
            'openweather_cloud_cover': [90],
            'visual_crossing_humidity': [88],
            'marine_layer_indicator': [0.9],
            'nws_wind_speed': [6]
        }),
        
        WeatherCondition.SANTA_ANA: pd.DataFrame({
            'nws_temp_high': [85.0],
            'openweather_temp_high': [86.0],
            'nws_wind_speed': [25],
            'visual_crossing_humidity': [20],
            'santa_ana_indicator': [0.8],
            'openweather_cloud_cover': [15]
        }),
        
        WeatherCondition.HEAT_WAVE: pd.DataFrame({
            'nws_temp_high': [95.0],
            'openweather_temp_high': [94.0],
            'tomorrow_temp_high': [96.0],
            'fire_season_indicator': [0.9],
            'heat_island_effect': [4.5],
            'visual_crossing_humidity': [35]
        })
    }


def demonstrate_basic_ensemble():
    """Demonstrate basic ensemble functionality."""
    print("=== Basic Ensemble Functionality ===\n")
    
    # Create ensemble with model adapters
    ensemble = create_ensemble_with_models()
    
    print("1. Ensemble Status:")
    status = ensemble.get_ensemble_status()
    print(f"   Total models: {status['total_models']}")
    print(f"   Trained models: {status['trained_models']}")
    print(f"   Performance window: {status['performance_window_days']} days")
    print(f"   Min confidence threshold: {status['min_confidence_threshold']}")
    
    print("\n2. Registered Models:")
    for name, info in status['model_info'].items():
        print(f"   {name}:")
        print(f"      Status: {info['status']}")
        print(f"      Type: {info.get('model_type', 'unknown')}")
        print(f"      Current weight: {info['current_weight']:.3f}")
        print(f"      Weather conditions: {info['weather_conditions']}")
    
    return ensemble


def demonstrate_weather_condition_detection(ensemble):
    """Demonstrate weather condition detection and model selection."""
    print("\n=== Weather Condition Detection ===\n")
    
    condition_features = create_weather_condition_features()
    
    for condition, features in condition_features.items():
        detected = ensemble.detect_weather_condition(features)
        suitable_models = ensemble.get_models_for_condition(condition)
        
        print(f"Condition: {condition}")
        print(f"   Detected: {detected}")
        print(f"   Suitable models: {suitable_models}")
        print(f"   Match: {'✓' if detected == condition else '✗'}")
        print()


def demonstrate_ensemble_predictions(ensemble):
    """Demonstrate ensemble predictions with different weather conditions."""
    print("=== Ensemble Predictions ===\n")
    
    condition_features = create_weather_condition_features()
    
    for condition, features in condition_features.items():
        try:
            # Make prediction with weather condition selection
            pred_with_condition, conf_with_condition = ensemble.predict(
                features, use_weather_condition_selection=True
            )
            
            # Make prediction without weather condition selection
            pred_without_condition, conf_without_condition = ensemble.predict(
                features, use_weather_condition_selection=False
            )
            
            print(f"{condition}:")
            print(f"   With condition selection: {pred_with_condition:.1f}°F (conf: {conf_with_condition:.3f})")
            print(f"   Without condition selection: {pred_without_condition:.1f}°F (conf: {conf_without_condition:.3f})")
            print(f"   Difference: {abs(pred_with_condition - pred_without_condition):.1f}°F")
            print()
            
        except Exception as e:
            print(f"{condition}: Error - {e}")
            print()


def demonstrate_performance_tracking(ensemble):
    """Demonstrate performance tracking and dynamic weighting."""
    print("=== Performance Tracking and Dynamic Weighting ===\n")
    
    # Simulate multiple predictions with performance updates
    sample_features = create_sample_features()
    
    print("1. Simulating 10 days of predictions with performance updates...")
    
    for day in range(10):
        # Make ensemble prediction
        pred, conf = ensemble.predict(sample_features, use_weather_condition_selection=False)
        
        # Simulate actual temperature (with some realistic variation)
        base_temp = 78.0
        actual_temp = base_temp + np.random.normal(0, 1.5)  # ±1.5°F variation
        
        # Update ensemble performance
        ensemble.update_ensemble_performance(
            prediction=pred,
            actual_temperature=actual_temp,
            confidence=conf,
            weather_condition=WeatherCondition.NORMAL,
            prediction_date=date.today() - timedelta(days=9-day)
        )
        
        # Simulate individual model performance with different accuracy levels
        models = ['linear_regression', 'prophet']  # Only "trained" models
        
        for model_name in models:
            if model_name == 'linear_regression':
                # More accurate model
                model_pred = actual_temp + np.random.normal(0, 1.0)
                model_conf = 0.7
            else:  # prophet
                # Less accurate model
                model_pred = actual_temp + np.random.normal(0, 2.0)
                model_conf = 0.6
            
            ensemble.update_model_performance(
                model_name=model_name,
                prediction=model_pred,
                actual_temperature=actual_temp,
                confidence=model_conf,
                weather_condition=WeatherCondition.NORMAL,
                prediction_date=date.today() - timedelta(days=9-day)
            )
        
        if day % 3 == 0:  # Print every 3rd day
            print(f"   Day {day+1}: Predicted {pred:.1f}°F, Actual {actual_temp:.1f}°F, Error {abs(pred-actual_temp):.1f}°F")
    
    print("\n2. Performance Summaries:")
    
    # Ensemble performance
    ensemble_summary = ensemble.get_model_performance_summary()
    print(f"   Ensemble Performance (last {ensemble_summary['period_days']} days):")
    print(f"      Total predictions: {ensemble_summary['total_predictions']}")
    print(f"      Average error: {ensemble_summary['avg_error']:.2f}°F")
    print(f"      RMSE: {ensemble_summary['rmse']:.2f}°F")
    print(f"      Accuracy within ±3°F: {ensemble_summary['accuracy_within_3f']:.1f}%")
    print(f"      Average confidence: {ensemble_summary['avg_confidence']:.3f}")
    
    # Individual model performance
    for model_name in ['linear_regression', 'prophet']:
        model_summary = ensemble.get_model_performance_summary(model_name)
        if 'error' not in model_summary:
            print(f"   {model_name} Performance:")
            print(f"      Average error: {model_summary['avg_error']:.2f}°F")
            print(f"      Accuracy within ±3°F: {model_summary['accuracy_within_3f']:.1f}%")
    
    print("\n3. Updated Model Weights:")
    current_status = ensemble.get_ensemble_status()
    for name, info in current_status['model_info'].items():
        if info['status'] == 'trained':
            print(f"   {name}: {info['current_weight']:.3f}")


def demonstrate_ensemble_with_real_data():
    """Demonstrate ensemble with real data if available."""
    print("\n=== Real Data Integration ===\n")
    
    try:
        # Check if we have real data
        data_manager = DataManager()
        summary = data_manager.get_data_summary()
        
        has_weather_data = any(
            isinstance(info, dict) and 'records' in info and info['records'] > 0
            for source, info in summary.items()
            if source in ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        )
        
        if has_weather_data:
            print("1. Real weather data available - creating features...")
            
            feature_pipeline = FeaturePipeline()
            today_features = feature_pipeline.create_features_for_prediction(date.today())
            
            if not today_features.empty:
                print(f"   Created {len(today_features.columns)} features for today")
                
                # Create ensemble and make prediction
                ensemble = create_ensemble_with_models()
                
                try:
                    prediction, confidence = ensemble.predict(today_features)
                    print(f"   Real data prediction: {prediction:.1f}°F (confidence: {confidence:.3f})")
                    
                    # Detect weather condition
                    condition = ensemble.detect_weather_condition(today_features)
                    print(f"   Detected weather condition: {condition}")
                    
                except Exception as e:
                    print(f"   Error making prediction with real data: {e}")
                    print("   Note: XGBoost and LightGBM models need to be trained first")
            else:
                print("   No features available for today")
        else:
            print("1. No real weather data available")
            print("   Run data collection first to see real data integration")
    
    except Exception as e:
        print(f"1. Error accessing real data: {e}")


def demonstrate_ensemble_configuration():
    """Demonstrate ensemble configuration and customization."""
    print("\n=== Ensemble Configuration ===\n")
    
    # Create custom ensemble
    ensemble = EnsembleTemperatureModel()
    
    print("1. Custom Configuration:")
    print(f"   Performance window: {ensemble.performance_window} days")
    print(f"   Min confidence threshold: {ensemble.min_confidence_threshold}")
    print(f"   Max models per prediction: {ensemble.max_models_per_prediction}")
    
    # Modify configuration
    ensemble.performance_window = 21  # 3 weeks
    ensemble.min_confidence_threshold = 0.4  # Higher threshold
    ensemble.max_models_per_prediction = 3  # Limit to top 3 models
    
    print("\n2. Modified Configuration:")
    print(f"   Performance window: {ensemble.performance_window} days")
    print(f"   Min confidence threshold: {ensemble.min_confidence_threshold}")
    print(f"   Max models per prediction: {ensemble.max_models_per_prediction}")
    
    print("\n3. Weather Condition Model Mapping:")
    for condition, models in ensemble.weather_condition_models.items():
        print(f"   {condition}: {models if models else 'No specific models'}")


def demonstrate_save_load():
    """Demonstrate saving and loading ensemble configuration."""
    print("\n=== Save/Load Functionality ===\n")
    
    # Create and configure ensemble
    ensemble = create_ensemble_with_models()
    
    # Add some performance data
    sample_features = create_sample_features()
    pred, conf = ensemble.predict(sample_features, use_weather_condition_selection=False)
    ensemble.update_ensemble_performance(pred, 78.2, conf)
    
    print("1. Saving ensemble configuration...")
    save_path = ensemble.save_ensemble("demo_ensemble.pkl")
    print(f"   Saved to: {save_path}")
    
    print("\n2. Loading ensemble configuration...")
    new_ensemble = EnsembleTemperatureModel()
    new_ensemble.load_ensemble(save_path)
    
    print("   Loaded ensemble status:")
    status = new_ensemble.get_ensemble_status()
    print(f"   Total predictions: {status['total_predictions']}")
    print(f"   Performance records: {status['ensemble_performance_records']}")
    
    print("   ✓ Save/load functionality working")


def main():
    """Run the complete ensemble model demonstration."""
    print("=== Ensemble Temperature Model Demonstration ===\n")
    
    try:
        # Basic functionality
        ensemble = demonstrate_basic_ensemble()
        
        # Weather condition detection
        demonstrate_weather_condition_detection(ensemble)
        
        # Ensemble predictions
        demonstrate_ensemble_predictions(ensemble)
        
        # Performance tracking
        demonstrate_performance_tracking(ensemble)
        
        # Real data integration
        demonstrate_ensemble_with_real_data()
        
        # Configuration
        demonstrate_ensemble_configuration()
        
        # Save/load
        demonstrate_save_load()
        
        print("\n=== Key Ensemble Features Demonstrated ===")
        print("✓ Dynamic weighting based on recent model performance")
        print("✓ Weather condition-specific model selection")
        print("✓ Ensemble prediction with confidence scoring")
        print("✓ Performance tracking and history management")
        print("✓ Configurable ensemble parameters")
        print("✓ Model adapter system for integration")
        print("✓ Save/load functionality for persistence")
        print("✓ Error handling and robustness")
        
        print("\n=== Requirements Compliance ===")
        print("✓ Requirement 4.1: Dynamic weighting system implemented")
        print("✓ Requirement 4.6: Model selection logic for weather conditions")
        print("✓ Requirement 6.2: Ensemble prediction with confidence scoring")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"Demo error: {e}")
    
    print("\n=== Ensemble Model Demonstration Complete ===")


if __name__ == '__main__':
    main()