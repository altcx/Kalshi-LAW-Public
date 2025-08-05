"""Demo script for XGBoost temperature prediction model."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.models.xgboost_model import XGBoostTemperatureModel
from src.utils.data_manager import DataManager


def create_sample_data_for_demo():
    """Create sample weather and actual temperature data for demonstration."""
    print("Creating sample data for demonstration...")
    
    data_manager = DataManager()
    
    # Create 90 days of sample weather data
    dates = pd.date_range('2024-11-01', '2025-01-29', freq='D')
    n_days = len(dates)
    
    # Create realistic LA weather patterns
    np.random.seed(42)
    
    # Seasonal temperature pattern for LA
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal_temp = 75 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
    
    # Create weather source data
    sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
    
    for source in sources:
        # Add some source-specific bias and noise
        source_bias = np.random.normal(0, 1)  # Each source has slight bias
        source_noise = 0.5 + np.random.uniform(0, 1)  # Different noise levels
        
        weather_data = pd.DataFrame({
            'date': dates,
            'forecast_date': dates - timedelta(days=1),  # Forecast made day before
            'predicted_high': seasonal_temp + source_bias + np.random.normal(0, source_noise, n_days),
            'predicted_low': seasonal_temp - 20 + source_bias + np.random.normal(0, source_noise * 0.8, n_days),
            'humidity': np.random.uniform(30, 85, n_days),
            'pressure': np.random.normal(1013, 8, n_days),
            'wind_speed': np.random.exponential(6, n_days),
            'wind_direction': np.random.uniform(0, 360, n_days),
            'cloud_cover': np.random.uniform(0, 100, n_days),
            'precipitation_prob': np.random.exponential(15, n_days),
            'data_quality_score': np.random.uniform(0.8, 1.0, n_days)
        })
        
        # Ensure reasonable ranges
        weather_data['predicted_high'] = np.clip(weather_data['predicted_high'], 50, 110)
        weather_data['predicted_low'] = np.clip(weather_data['predicted_low'], 35, 85)
        weather_data['precipitation_prob'] = np.clip(weather_data['precipitation_prob'], 0, 100)
        weather_data['wind_speed'] = np.clip(weather_data['wind_speed'], 0, 50)
        
        # Save to data manager
        data_manager.save_source_data(source, weather_data, append=False)
        print(f"  Created {len(weather_data)} records for {source}")
    
    # Create actual temperature data (ground truth)
    # Make it somewhat correlated with forecasts but with realistic error
    consensus_forecast = np.mean([
        seasonal_temp + np.random.normal(0, 0.5, n_days) for _ in sources
    ], axis=0)
    
    actual_highs = (consensus_forecast + 
                   np.random.normal(0, 2.5, n_days) +  # Forecast error
                   np.random.normal(0, 1, n_days) * (day_of_year % 30 < 5))  # Occasional weather surprises
    
    actual_temps = pd.DataFrame({
        'date': dates,
        'actual_high': actual_highs,
        'actual_low': actual_highs - 20 + np.random.normal(0, 3, n_days),
        'source': 'NOAA'
    })
    
    # Ensure reasonable ranges
    actual_temps['actual_high'] = np.clip(actual_temps['actual_high'], 45, 115)
    actual_temps['actual_low'] = np.clip(actual_temps['actual_low'], 30, 90)
    
    data_manager.save_source_data('actual_temperatures', actual_temps, append=False)
    print(f"  Created {len(actual_temps)} actual temperature records")
    
    print("Sample data creation complete!\n")
    return True


def demonstrate_model_training():
    """Demonstrate XGBoost model training with hyperparameter optimization."""
    print("=== XGBoost Model Training Demo ===\n")
    
    # Initialize model
    model = XGBoostTemperatureModel()
    
    # Check data availability
    data_manager = DataManager()
    summary = data_manager.get_data_summary()
    
    print("1. Data Availability Check:")
    for source, info in summary.items():
        if isinstance(info, dict) and 'records' in info:
            print(f"   {source}: {info['records']} records")
            if 'date_range' in info:
                print(f"     Date range: {info['date_range']['start']} to {info['date_range']['end']}")
        else:
            print(f"   {source}: {info}")
    print()
    
    # Define training period
    end_date = date(2025, 1, 20)  # Use data up to Jan 20
    start_date = end_date - timedelta(days=60)  # 60 days of training data
    
    print(f"2. Training Model ({start_date} to {end_date}):")
    print("   This includes hyperparameter optimization and may take a few minutes...")
    
    try:
        # Train model with hyperparameter optimization
        training_results = model.train(
            start_date=start_date,
            end_date=end_date,
            optimize_hyperparams=True,
            n_trials=30,  # Reasonable number for demo
            include_ensemble=True,
            include_la_patterns=True
        )
        
        print("   Training Results:")
        print(f"   ✓ Training samples: {training_results['training_samples']}")
        print(f"   ✓ Features used: {training_results['features_used']}")
        print(f"   ✓ Cross-validation RMSE: {training_results['cv_scores']['rmse_mean']:.3f} ± {training_results['cv_scores']['rmse_std']:.3f}")
        print(f"   ✓ Training RMSE: {training_results['training_metrics']['rmse']:.3f}")
        print(f"   ✓ Training MAE: {training_results['training_metrics']['mae']:.3f}")
        print(f"   ✓ R² Score: {training_results['training_metrics']['r2_score']:.3f}")
        print(f"   ✓ Accuracy within ±3°F: {training_results['training_metrics']['accuracy_within_3f']:.1f}%")
        print(f"   ✓ Accuracy within ±5°F: {training_results['training_metrics']['accuracy_within_5f']:.1f}%")
        
        print("\n   Best Hyperparameters:")
        for param, value in training_results['hyperparameters'].items():
            if param != 'random_state' and param != 'n_jobs':
                print(f"   - {param}: {value}")
        
        return model, training_results
        
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return None, None


def demonstrate_feature_importance(model):
    """Demonstrate feature importance analysis."""
    print("\n=== Feature Importance Analysis ===\n")
    
    if not model or not model.is_trained:
        print("Model not available for feature importance analysis")
        return
    
    # Get detailed feature importance analysis
    importance_analysis = model.analyze_feature_importance()
    
    print("1. Feature Overview:")
    print(f"   Total features: {importance_analysis['total_features']}")
    print(f"   Feature importance range: {importance_analysis['importance_distribution']['min']:.4f} to {importance_analysis['importance_distribution']['max']:.4f}")
    print(f"   Mean importance: {importance_analysis['importance_distribution']['mean']:.4f}")
    
    print("\n2. Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(importance_analysis['top_10_features'].items())[:10]):
        print(f"   {i+1:2d}. {feature:<30} {importance:.4f}")
    
    print("\n3. Feature Importance by Category:")
    for category, importance in sorted(importance_analysis['category_importance'].items(), 
                                     key=lambda x: x[1], reverse=True):
        if importance > 0:
            percentage = (importance / sum(importance_analysis['category_importance'].values())) * 100
            print(f"   {category:<15} {importance:.4f} ({percentage:.1f}%)")
    
    print("\n4. Top Features by Category:")
    for category, features in importance_analysis['feature_categories'].items():
        if features and importance_analysis['category_importance'][category] > 0:
            print(f"\n   {category.upper()}:")
            for feature, importance in features[:3]:  # Top 3 in each category
                print(f"     - {feature}: {importance:.4f}")


def demonstrate_predictions(model):
    """Demonstrate making predictions with the trained model."""
    print("\n=== Prediction Demonstration ===\n")
    
    if not model or not model.is_trained:
        print("Model not available for predictions")
        return
    
    from src.feature_engineering.feature_pipeline import FeaturePipeline
    feature_pipeline = FeaturePipeline()
    
    # Make predictions for recent dates
    prediction_dates = [
        date(2025, 1, 21),
        date(2025, 1, 22),
        date(2025, 1, 23),
        date(2025, 1, 24),
        date(2025, 1, 25)
    ]
    
    print("1. Recent Predictions:")
    predictions = []
    
    for pred_date in prediction_dates:
        try:
            # Get features for the date
            features = feature_pipeline.create_features_for_prediction(pred_date)
            
            if not features.empty:
                prediction, confidence = model.predict(features)
                predictions.append((pred_date, prediction, confidence))
                
                print(f"   {pred_date}: {prediction:.1f}°F (confidence: {confidence:.3f})")
            else:
                print(f"   {pred_date}: No features available")
                
        except Exception as e:
            print(f"   {pred_date}: Error - {e}")
    
    # Compare with actual temperatures if available
    if predictions:
        print("\n2. Prediction vs Actual Comparison:")
        data_manager = DataManager()
        
        for pred_date, prediction, confidence in predictions:
            actual_temp = data_manager.get_actual_temperature(pred_date)
            if actual_temp is not None:
                error = abs(prediction - actual_temp)
                print(f"   {pred_date}: Predicted {prediction:.1f}°F, Actual {actual_temp:.1f}°F, Error {error:.1f}°F")
            else:
                print(f"   {pred_date}: Predicted {prediction:.1f}°F, Actual not available")


def demonstrate_model_validation(model):
    """Demonstrate model validation on test data."""
    print("\n=== Model Validation ===\n")
    
    if not model or not model.is_trained:
        print("Model not available for validation")
        return
    
    # Use recent data for validation (after training period)
    test_start = date(2025, 1, 21)
    test_end = date(2025, 1, 29)
    
    print(f"1. Validating on test period: {test_start} to {test_end}")
    
    try:
        validation_results = model.validate_model_performance(test_start, test_end)
        
        if 'error' in validation_results:
            print(f"   Validation error: {validation_results['error']}")
            return
        
        print("   Validation Results:")
        print(f"   ✓ Test samples: {validation_results['test_samples']}")
        print(f"   ✓ Test RMSE: {validation_results['test_metrics']['rmse']:.3f}")
        print(f"   ✓ Test MAE: {validation_results['test_metrics']['mae']:.3f}")
        print(f"   ✓ Test R² Score: {validation_results['test_metrics']['r2_score']:.3f}")
        print(f"   ✓ Test accuracy within ±3°F: {validation_results['test_metrics']['accuracy_within_3f']:.1f}%")
        print(f"   ✓ Test accuracy within ±5°F: {validation_results['test_metrics']['accuracy_within_5f']:.1f}%")
        
        print("\n   Error Analysis:")
        error_analysis = validation_results['error_analysis']
        print(f"   - Mean absolute error: {error_analysis['mean_error']:.2f}°F")
        print(f"   - Median absolute error: {error_analysis['median_error']:.2f}°F")
        print(f"   - Maximum error: {error_analysis['max_error']:.2f}°F")
        print(f"   - Days within ±1°F: {error_analysis['days_within_1f']}/{error_analysis['total_test_days']}")
        print(f"   - Days within ±2°F: {error_analysis['days_within_2f']}/{error_analysis['total_test_days']}")
        print(f"   - Days within ±3°F: {error_analysis['days_within_3f']}/{error_analysis['total_test_days']}")
        
    except Exception as e:
        print(f"   Validation failed: {e}")


def demonstrate_model_persistence(model):
    """Demonstrate saving and loading the model."""
    print("\n=== Model Persistence ===\n")
    
    if not model or not model.is_trained:
        print("Model not available for persistence demo")
        return
    
    print("1. Saving Model:")
    try:
        model_path = model.save_model()
        print(f"   ✓ Model saved to: {model_path}")
        
        # Get model info before loading
        original_info = model.get_model_info()
        
        print("\n2. Loading Model:")
        new_model = XGBoostTemperatureModel()
        new_model.load_model(model_path)
        
        loaded_info = new_model.get_model_info()
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Training samples: {loaded_info['training_samples']}")
        print(f"   ✓ Features count: {loaded_info['features_count']}")
        print(f"   ✓ Training date: {loaded_info['training_date']}")
        
        # Verify model works
        print("\n3. Testing Loaded Model:")
        from src.feature_engineering.feature_pipeline import FeaturePipeline
        feature_pipeline = FeaturePipeline()
        
        test_features = feature_pipeline.create_features_for_prediction(date(2025, 1, 25))
        if not test_features.empty:
            prediction, confidence = new_model.predict(test_features)
            print(f"   ✓ Test prediction: {prediction:.1f}°F (confidence: {confidence:.3f})")
        else:
            print("   - No test features available")
            
    except Exception as e:
        print(f"   ✗ Persistence demo failed: {e}")


def main():
    """Run the complete XGBoost model demonstration."""
    print("🌡️  XGBoost Temperature Prediction Model Demo")
    print("=" * 50)
    
    # Check if we have data, create sample data if needed
    data_manager = DataManager()
    summary = data_manager.get_data_summary()
    
    has_weather_data = any(
        isinstance(info, dict) and 'records' in info and info['records'] > 0
        for source, info in summary.items()
        if source in ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
    )
    
    has_actual_temps = (
        'actual_temperatures' in summary and 
        isinstance(summary['actual_temperatures'], dict) and 
        'records' in summary['actual_temperatures'] and 
        summary['actual_temperatures']['records'] > 0
    )
    
    if not has_weather_data or not has_actual_temps:
        print("\nNo sufficient data found. Creating sample data for demonstration...")
        create_sample_data_for_demo()
    
    # Demonstrate model training
    model, training_results = demonstrate_model_training()
    
    if model and training_results:
        # Demonstrate feature importance
        demonstrate_feature_importance(model)
        
        # Demonstrate predictions
        demonstrate_predictions(model)
        
        # Demonstrate validation
        demonstrate_model_validation(model)
        
        # Demonstrate model persistence
        demonstrate_model_persistence(model)
        
        print("\n" + "=" * 50)
        print("🎉 XGBoost Model Demo Complete!")
        print("\nKey Achievements:")
        print(f"✓ Trained model with {training_results['features_used']} features")
        print(f"✓ Achieved {training_results['training_metrics']['accuracy_within_3f']:.1f}% accuracy within ±3°F")
        print(f"✓ Cross-validation RMSE: {training_results['cv_scores']['rmse_mean']:.3f}°F")
        print("✓ Comprehensive feature importance analysis")
        print("✓ Model persistence and loading capabilities")
        print("✓ Prediction confidence scoring")
        
        print("\nThe XGBoost model is ready for:")
        print("• Daily temperature predictions")
        print("• Feature importance analysis")
        print("• Model interpretability")
        print("• Integration with trading recommendations")
        
    else:
        print("\n❌ Demo could not complete due to training issues")
        print("Please check data availability and try again")


if __name__ == '__main__':
    main()