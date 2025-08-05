"""LightGBM regression model for daily high temperature prediction."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, Optional, Tuple, List, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import joblib
from pathlib import Path

# LightGBM and ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna

from src.utils.data_manager import DataManager
from src.feature_engineering.feature_pipeline import FeaturePipeline


class LightGBMTemperatureModel:
    """LightGBM regression model for predicting daily high temperatures."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the LightGBM model.
        
        Args:
            model_dir: Directory to save/load models (default: models/)
        """
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM is not available. Please install with: pip install lightgbm")
            # Don't raise exception during import, just mark as unavailable
            self._available = False
            return
        
        self._available = True
        
        self.model_dir = Path(model_dir) if model_dir else Path('models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Model metadata
        self.training_date = None
        self.training_samples = 0
        self.cv_scores = {}
        self.feature_importance = {}
        self.hyperparameters = {}
        
        # Data components
        self.data_manager = DataManager()
        self.feature_pipeline = FeaturePipeline()
        
        logger.info(f"LightGBMTemperatureModel initialized with model directory: {self.model_dir}")
    
    def prepare_training_data(self, start_date: date, end_date: date, 
                            include_ensemble: bool = True, 
                            include_la_patterns: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with features and targets.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            include_ensemble: Whether to include ensemble features
            include_la_patterns: Whether to include LA-specific features
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        logger.info(f"Preparing training data from {start_date} to {end_date}")
        
        # Create complete feature set
        features = self.feature_pipeline.create_complete_features(
            start_date, end_date, 
            include_ensemble=include_ensemble,
            include_la_patterns=include_la_patterns
        )
        
        if features.empty:
            logger.error("No features available for training")
            return pd.DataFrame(), pd.Series()
        
        # Load actual temperatures for targets
        actual_temps = self.data_manager.load_source_data('actual_temperatures', start_date, end_date)
        
        if actual_temps.empty:
            logger.error("No actual temperature data available for training")
            return pd.DataFrame(), pd.Series()
        
        # Merge features with targets
        features['date'] = pd.to_datetime(features['date'])
        actual_temps['date'] = pd.to_datetime(actual_temps['date'])
        
        merged = features.merge(actual_temps[['date', 'actual_high']], on='date', how='inner')
        
        if merged.empty:
            logger.error("No matching data between features and actual temperatures")
            return pd.DataFrame(), pd.Series()
        
        # Separate features and targets
        feature_cols = [col for col in merged.columns if col not in ['date', 'actual_high']]
        X = merged[feature_cols]
        y = merged['actual_high']
        
        logger.info(f"Prepared training data: {len(X)} samples, {len(feature_cols)} features")
        return X, y
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                n_trials: int = 100, cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target values
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        def objective(trial):
            # Define hyperparameter search space for LightGBM
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'random_state': 42,
                'verbose': -1
            }
            
            # Create model with trial parameters
            model = lgb.LGBMRegressor(**params)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            
            # Return negative RMSE (Optuna minimizes)
            return np.sqrt(-scores.mean())
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Hyperparameter optimization complete. Best RMSE: {best_score:.3f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def train(self, start_date: date, end_date: date, 
              optimize_hyperparams: bool = True, 
              n_trials: int = 50,
              include_ensemble: bool = True,
              include_la_patterns: bool = True) -> Dict[str, Any]:
        """Train the LightGBM model.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of hyperparameter optimization trials
            include_ensemble: Whether to include ensemble features
            include_la_patterns: Whether to include LA-specific features
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training LightGBM model from {start_date} to {end_date}")
        
        # Prepare training data
        X, y = self.prepare_training_data(start_date, end_date, include_ensemble, include_la_patterns)
        
        if X.empty or y.empty:
            raise ValueError("No training data available")
        
        # Handle categorical features
        X_processed = X.copy()
        
        # Identify categorical columns
        categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns
        
        # LightGBM can handle categorical features natively, but we'll encode for consistency
        if len(categorical_columns) > 0:
            logger.info(f"One-hot encoding {len(categorical_columns)} categorical features: {list(categorical_columns)}")
            X_processed = pd.get_dummies(X_processed, columns=categorical_columns, drop_first=True)
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Store processed feature names for later use
        self.feature_names = list(X_processed.columns)
        
        # Scale features (optional for tree-based models, but helps with consistency)
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            best_params = self.optimize_hyperparameters(X_scaled, y, n_trials=n_trials)
            self.hyperparameters = best_params
        else:
            # Use default parameters optimized for regression
            self.hyperparameters = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_estimators': 500,
                'random_state': 42,
                'verbose': -1
            }
        
        # Train final model
        self.model = lgb.LGBMRegressor(**self.hyperparameters)
        self.model.fit(X_scaled, y)
        
        # Calculate cross-validation scores
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
        self.cv_scores = {
            'rmse_mean': np.sqrt(-cv_scores.mean()),
            'rmse_std': np.sqrt(cv_scores.std()),
            'scores': cv_scores.tolist()
        }
        
        # Calculate feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Update training metadata
        self.training_date = datetime.now()
        self.training_samples = len(X)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        training_metrics = self.calculate_metrics(y, y_pred)
        
        training_results = {
            'training_samples': self.training_samples,
            'features_used': len(self.feature_names),
            'hyperparameters': self.hyperparameters,
            'cv_scores': self.cv_scores,
            'training_metrics': training_metrics,
            'feature_importance_top10': dict(sorted(self.feature_importance.items(), 
                                                  key=lambda x: x[1], reverse=True)[:10])
        }
        
        logger.info(f"Model training complete. CV RMSE: {self.cv_scores['rmse_mean']:.3f} ± {self.cv_scores['rmse_std']:.3f}")
        return training_results
    
    def predict(self, features: pd.DataFrame) -> Tuple[float, float]:
        """Make a temperature prediction.
        
        Args:
            features: DataFrame with features for prediction
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if features.empty:
            raise ValueError("No features provided for prediction")
        
        # Handle categorical features (same as training)
        features_processed = features.copy()
        
        # Identify categorical columns
        categorical_columns = features_processed.select_dtypes(include=['object', 'category']).columns
        
        # One-hot encode categorical features
        if len(categorical_columns) > 0:
            features_processed = pd.get_dummies(features_processed, columns=categorical_columns, drop_first=True)
        
        # Ensure features match training features
        missing_features = set(self.feature_names) - set(features_processed.columns)
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            # Fill missing features with default values
            for feature in missing_features:
                features_processed[feature] = 0  # Default value
        
        # Remove extra features not in training
        extra_features = set(features_processed.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features not in training: {extra_features}")
            features_processed = features_processed.drop(columns=extra_features)
        
        # Select and order features to match training
        features_ordered = features_processed[self.feature_names].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features_ordered)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on feature similarity to training data
        # and cross-validation performance
        confidence = self.calculate_prediction_confidence(features_ordered)
        
        return prediction, confidence
    
    def calculate_prediction_confidence(self, features: pd.DataFrame) -> float:
        """Calculate confidence score for a prediction.
        
        Args:
            features: Features used for prediction
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from cross-validation performance
        base_confidence = max(0.1, 1.0 - (self.cv_scores['rmse_mean'] / 20.0))  # Normalize by typical temp range
        
        # Adjust based on feature completeness
        feature_completeness = 1.0 - (features.isnull().sum().sum() / len(features.columns))
        
        # Combine factors
        confidence = base_confidence * feature_completeness
        
        return min(1.0, max(0.0, confidence))
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Temperature-specific metrics
        within_3f = np.mean(np.abs(y_true - y_pred) <= 3.0) * 100  # Percentage within ±3°F
        within_5f = np.mean(np.abs(y_true - y_pred) <= 5.0) * 100  # Percentage within ±5°F
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'accuracy_within_3f': within_3f,
            'accuracy_within_5f': within_5f
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Sort by importance and return top N
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:top_n])
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """Save the trained model to disk.
        
        Args:
            filename: Optional filename (default: lightgbm_model_YYYYMMDD.pkl)
            
        Returns:
            Path to saved model file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lightgbm_model_{timestamp}.pkl"
        
        model_path = self.model_dir / filename
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance,
            'training_date': self.training_date,
            'training_samples': self.training_samples
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.hyperparameters = model_data['hyperparameters']
            self.cv_scores = model_data['cv_scores']
            self.feature_importance = model_data['feature_importance']
            self.training_date = model_data['training_date']
            self.training_samples = model_data['training_samples']
            self.is_trained = True
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'training_samples': self.training_samples,
            'features_count': len(self.feature_names),
            'hyperparameters': self.hyperparameters,
            'cv_performance': self.cv_scores,
            'top_features': dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        }


def main():
    """Demonstrate the LightGBM model."""
    print("=== LightGBM Temperature Model Demo ===\n")
    
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM is not available. Please install with: pip install lightgbm")
        return
    
    # Initialize model
    model = LightGBMTemperatureModel()
    
    print("1. Model Information:")
    info = model.get_model_info()
    print(f"   Status: {info['status']}")
    
    # Check if we have data for training
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
    
    if not has_weather_data:
        print("\n   No weather forecast data available for training")
        print("   Please run data collection first")
        return
    
    if not has_actual_temps:
        print("\n   No actual temperature data available for training")
        print("   Please run actual temperature collection first")
        return
    
    # If we have data, demonstrate training
    print("\n2. Training Model (this may take a few minutes)...")
    
    try:
        # Use recent data for training
        end_date = date.today()
        start_date = end_date - timedelta(days=60)  # Use 60 days of data
        
        training_results = model.train(
            start_date=start_date,
            end_date=end_date,
            optimize_hyperparams=True,
            n_trials=20  # Reduced for demo
        )
        
        print("   Training Results:")
        print(f"   - Training samples: {training_results['training_samples']}")
        print(f"   - Features used: {training_results['features_used']}")
        print(f"   - CV RMSE: {training_results['cv_scores']['rmse_mean']:.3f} ± {training_results['cv_scores']['rmse_std']:.3f}")
        print(f"   - Training RMSE: {training_results['training_metrics']['rmse']:.3f}")
        print(f"   - Accuracy within ±3°F: {training_results['training_metrics']['accuracy_within_3f']:.1f}%")
        
        print("\n3. Feature Importance Analysis:")
        print("   Top 5 most important features:")
        for i, (feature, importance) in enumerate(list(training_results['feature_importance_top10'].items())[:5]):
            print(f"   {i+1}. {feature}: {importance:.4f}")
        
        # Save the model
        print("\n4. Saving Model...")
        model_path = model.save_model()
        print(f"   Model saved to: {model_path}")
        
        # Demonstrate prediction
        print("\n5. Making Sample Prediction...")
        feature_pipeline = FeaturePipeline()
        
        # Get features for today
        today_features = feature_pipeline.create_features_for_prediction(date.today())
        
        if not today_features.empty:
            prediction, confidence = model.predict(today_features)
            print(f"   Prediction for today: {prediction:.1f}°F")
            print(f"   Confidence: {confidence:.3f}")
        else:
            print("   No features available for today's prediction")
        
    except Exception as e:
        print(f"   Error during training: {e}")
        logger.error(f"Training error: {e}")
    
    print("\n=== LightGBM Model Demo Complete ===")


if __name__ == '__main__':
    main()