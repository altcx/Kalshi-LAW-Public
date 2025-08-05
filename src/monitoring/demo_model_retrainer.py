"""Demo script for the automated model retraining system."""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import json

from .model_retrainer import ModelRetrainer
from .performance_tracker import PerformanceTracker
from ..utils.data_manager import DataManager


class MockXGBoostModel:
    """Mock XGBoost model for demonstration."""
    
    def __init__(self):
        self.is_fitted = False
        self.params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        }
    
    def fit(self, X, y):
        """Fit the model."""
        self.is_fitted = True
        self.feature_names = list(X.columns)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Simple prediction based on first few features
        predictions = []
        for _, row in X.iterrows():
            # Use first feature as base prediction with some noise
            base_pred = row.iloc[0] if len(row) > 0 else 75.0
            noise = np.random.normal(0, 1.5)
            predictions.append(base_pred + noise)
        
        return np.array(predictions)
    
    def set_params(self, **params):
        """Set model parameters."""
        self.params.update(params)
        return self


def create_sample_training_data():
    """Create sample training data for demonstration."""
    print("Creating sample training data...")
    
    data_manager = DataManager()
    
    # Create date range for the last 60 days
    end_date = date.today()
    start_date = end_date - timedelta(days=60)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create weather data for multiple sources
    sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
    
    for source in sources:
        data = []
        for i, date_val in enumerate(dates):
            # Create realistic temperature predictions with source-specific characteristics
            base_temp = 75 + 10 * np.sin(i * 0.1) + 5 * np.cos(i * 0.05)  # Seasonal variation
            
            # Add source-specific bias and accuracy
            source_bias = {
                'nws': 0.5,           # Slight warm bias
                'openweather': -0.3,  # Slight cool bias
                'tomorrow': 0.8,      # Warm bias
                'weatherbit': -0.5,   # Cool bias
                'visual_crossing': 0.2 # Minimal bias
            }
            
            source_accuracy = {
                'nws': 1.5,           # High accuracy (low noise)
                'openweather': 2.0,   # Good accuracy
                'tomorrow': 2.5,      # Moderate accuracy
                'weatherbit': 3.0,    # Lower accuracy
                'visual_crossing': 1.8 # Good accuracy
            }
            
            bias = source_bias.get(source, 0)
            noise_std = source_accuracy.get(source, 2.0)
            noise = np.random.normal(bias, noise_std)
            
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
                'data_quality_score': 0.85 + np.random.normal(0, 0.1)
            })
        
        source_df = pd.DataFrame(data)
        data_manager.save_source_data(source, source_df)
        print(f"  - Created {len(data)} records for {source}")
    
    # Create actual temperature data (ground truth)
    actual_data = []
    for i, date_val in enumerate(dates):
        base_temp = 75 + 10 * np.sin(i * 0.1) + 5 * np.cos(i * 0.05)
        actual_noise = np.random.normal(0, 1.0)  # Less noise than predictions
        
        actual_data.append({
            'date': date_val,
            'actual_high': base_temp + actual_noise,
            'actual_low': base_temp - 15 + actual_noise,
            'source': 'NOAA'
        })
    
    actual_df = pd.DataFrame(actual_data)
    data_manager.save_source_data('actual_temperatures', actual_df)
    print(f"  - Created {len(actual_data)} actual temperature records")
    
    return data_manager, len(dates)


def demo_model_retraining():
    """Demonstrate the automated model retraining system."""
    print("=== Automated Model Retraining System Demo ===\n")
    
    # Create sample training data
    data_manager, data_points = create_sample_training_data()
    
    # Initialize components
    performance_tracker = PerformanceTracker(data_manager)
    model_retrainer = ModelRetrainer(data_manager, performance_tracker)
    
    # Mock the available models for demonstration
    model_retrainer.available_models = {
        'mock_xgboost': lambda: MockXGBoostModel
    }
    
    print(f"Created {data_points} days of training data\n")
    
    # Check if retraining is needed
    print("=== Checking Retraining Need ===")
    retraining_check = model_retrainer.check_retraining_needed('mock_xgboost')
    
    print(f"Model: {retraining_check['model_name']}")
    print(f"Should retrain: {retraining_check['should_retrain']}")
    print(f"Reasons: {'; '.join(retraining_check['reasons'])}")
    print(f"Total data points: {retraining_check['total_data_points']}")
    
    if retraining_check.get('last_retraining'):
        print(f"Last retraining: {retraining_check['last_retraining']}")
    else:
        print("Last retraining: Never")
    
    # Prepare training data
    print("\n=== Preparing Training Data ===")
    try:
        features, targets = model_retrainer.prepare_training_data(max_days=60)
        print(f"Training features shape: {features.shape}")
        print(f"Training targets shape: {targets.shape}")
        print(f"Feature columns: {list(features.columns)[:5]}...")  # Show first 5 columns
        
        # Show some statistics
        print(f"Target temperature range: {targets.min():.1f}°F to {targets.max():.1f}°F")
        print(f"Target temperature mean: {targets.mean():.1f}°F")
        
    except Exception as e:
        print(f"Error preparing training data: {e}")
        features, targets = None, None
    
    # Demonstrate model retraining
    if features is not None and len(features) >= model_retrainer.retraining_config['min_data_points']:
        print("\n=== Model Retraining ===")
        retraining_result = model_retrainer.retrain_model('mock_xgboost', hyperparameter_optimization=False)
        
        if 'error' not in retraining_result:
            print(f"Retraining successful!")
            print(f"Model type: {retraining_result['model_type']}")
            print(f"Training samples: {retraining_result['training_samples']}")
            print(f"Validation samples: {retraining_result['validation_samples']}")
            print(f"Features count: {retraining_result['features_count']}")
            
            # Show performance metrics
            train_metrics = retraining_result['train_metrics']
            val_metrics = retraining_result['validation_metrics']
            
            print(f"\nTraining Performance:")
            print(f"  - MAE: {train_metrics.get('mae', 0):.2f}°F")
            print(f"  - RMSE: {train_metrics.get('rmse', 0):.2f}°F")
            print(f"  - Accuracy (±3°F): {train_metrics.get('accuracy_3f', 0):.1%}")
            
            print(f"\nValidation Performance:")
            print(f"  - MAE: {val_metrics.get('mae', 0):.2f}°F")
            print(f"  - RMSE: {val_metrics.get('rmse', 0):.2f}°F")
            print(f"  - Accuracy (±3°F): {val_metrics.get('accuracy_3f', 0):.1%}")
            
            print(f"\nModel saved to: {retraining_result['model_path']}")
            
        else:
            print(f"Retraining failed: {retraining_result['error']}")
    
    else:
        print(f"\n=== Skipping Model Retraining ===")
        print("Insufficient training data or data preparation failed")
    
    # Check retraining history
    print("\n=== Retraining History ===")
    last_retraining = model_retrainer._get_last_retraining_date('mock_xgboost')
    if last_retraining:
        print(f"Last retraining date: {last_retraining}")
    else:
        print("No retraining history found")
    
    # Check if model switching is needed
    print("\n=== Model Switching Analysis ===")
    switching_result = model_retrainer.detect_model_switching_need()
    
    print(f"Should switch models: {switching_result['should_switch']}")
    if switching_result.get('current_model'):
        print(f"Current model: {switching_result['current_model']}")
    if switching_result.get('recommended_model'):
        print(f"Recommended model: {switching_result['recommended_model']}")
    
    if switching_result.get('performance_comparison'):
        print("\nModel Performance Comparison:")
        for model, perf in switching_result['performance_comparison'].items():
            print(f"  - {model}: {perf.get('accuracy_3f', 0):.1%} accuracy, {perf.get('mae', 0):.2f}°F MAE")
    
    # Run complete automated retraining process
    print("\n=== Automated Retraining Process ===")
    automation_result = model_retrainer.run_automated_retraining()
    
    if 'error' not in automation_result:
        summary = automation_result['summary']
        print(f"Models checked: {summary['models_checked']}")
        print(f"Models needing retraining: {summary['models_needing_retraining']}")
        print(f"Models retrained: {summary['models_retrained']}")
        print(f"Successful retraining: {summary['successful_retraining']}")
        print(f"Model switching recommended: {summary['model_switching_recommended']}")
        
        if summary.get('recommended_model'):
            print(f"Recommended model: {summary['recommended_model']}")
    else:
        print(f"Automation failed: {automation_result['error']}")
    
    # Show configuration
    print("\n=== Retraining Configuration ===")
    config = model_retrainer.retraining_config
    print(f"Minimum data points: {config['min_data_points']}")
    print(f"Performance threshold: {config['performance_threshold']:.1%}")
    print(f"Degradation threshold: {config['degradation_threshold']:.1%}")
    print(f"Retraining frequency: {config['retraining_frequency_days']} days")
    print(f"Max training data: {config['max_training_data_days']} days")
    print(f"Validation split: {config['validation_split']:.1%}")
    
    print("\n=== Demo Complete ===")
    print("Automated model retraining system successfully demonstrated!")
    print("Check the models/ directory for saved models and performance history.")


if __name__ == "__main__":
    demo_model_retraining()