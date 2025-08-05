"""Facebook Prophet time series model for daily high temperature prediction with external regressors."""

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
import warnings

# Prophet and related libraries
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.utils.data_manager import DataManager
from src.feature_engineering.feature_pipeline import FeaturePipeline


class ProphetTemperatureModel:
    """Facebook Prophet time series model for predicting daily high temperatures with external regressors."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the Prophet model.
        
        Args:
            model_dir: Directory to save/load models (default: models/)
        """
        self.model_dir = Path(model_dir) if model_dir else Path('models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check Prophet availability
        if not PROPHET_AVAILABLE:
            logger.error("Prophet is not available. Please install with: pip install prophet")
            raise ImportError("Prophet is required for this model")
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()  # For external regressors
        self.external_regressors = []
        self.is_trained = False
        
        # Model metadata
        self.training_date = None
        self.training_samples = 0
        self.cv_scores = {}
        self.hyperparameters = {}
        self.regressor_importance = {}
        
        # Data components
        self.data_manager = DataManager()
        self.feature_pipeline = FeaturePipeline()
        
        logger.info(f"ProphetTemperatureModel initialized with model directory: {self.model_dir}")
    
    def prepare_training_data(self, start_date: date, end_date: date, 
                            include_ensemble: bool = True, 
                            include_la_patterns: bool = True,
                            max_regressors: int = 10) -> pd.DataFrame:
        """Prepare training data in Prophet format with external regressors.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            include_ensemble: Whether to include ensemble features
            include_la_patterns: Whether to include LA-specific features
            max_regressors: Maximum number of external regressors to use
            
        Returns:
            DataFrame in Prophet format (ds, y, regressor1, regressor2, ...)
        """
        logger.info(f"Preparing Prophet training data from {start_date} to {end_date}")
        
        # Create complete feature set
        features = self.feature_pipeline.create_complete_features(
            start_date, end_date, 
            include_ensemble=include_ensemble,
            include_la_patterns=include_la_patterns
        )
        
        if features.empty:
            logger.error("No features available for training")
            return pd.DataFrame()
        
        # Load actual temperatures for targets
        actual_temps = self.data_manager.load_source_data('actual_temperatures', start_date, end_date)
        
        if actual_temps.empty:
            logger.error("No actual temperature data available for training")
            return pd.DataFrame()
        
        # Merge features with targets
        features['date'] = pd.to_datetime(features['date'])
        actual_temps['date'] = pd.to_datetime(actual_temps['date'])
        
        merged = features.merge(actual_temps[['date', 'actual_high']], on='date', how='inner')
        
        if merged.empty:
            logger.error("No matching data between features and actual temperatures")
            return pd.DataFrame()
        
        # Create Prophet format DataFrame
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = merged['date']  # Prophet requires 'ds' for dates
        prophet_df['y'] = merged['actual_high']  # Prophet requires 'y' for target
        
        # Select best external regressors
        feature_cols = [col for col in merged.columns if col not in ['date', 'actual_high']]
        
        # Filter to numeric columns only
        numeric_features = merged[feature_cols].select_dtypes(include=[np.number])
        
        # Select most important features based on correlation with target
        correlations = numeric_features.corrwith(merged['actual_high']).abs().sort_values(ascending=False)
        
        # Remove features with very low correlation or too many missing values
        valid_correlations = correlations.dropna()
        valid_correlations = valid_correlations[valid_correlations > 0.05]  # Minimum correlation threshold
        
        # Select top regressors
        selected_regressors = valid_correlations.head(max_regressors).index.tolist()
        
        # Add external regressors to Prophet DataFrame
        for regressor in selected_regressors:
            regressor_data = merged[regressor].fillna(merged[regressor].median())
            prophet_df[regressor] = regressor_data
        
        self.external_regressors = selected_regressors
        
        logger.info(f"Prepared Prophet data: {len(prophet_df)} samples with {len(selected_regressors)} external regressors")
        logger.info(f"Selected regressors: {selected_regressors}")
        
        return prophet_df
    
    def optimize_hyperparameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize Prophet hyperparameters.
        
        Args:
            df: Prophet format DataFrame
            
        Returns:
            Dictionary with best hyperparameters
        """
        logger.info("Optimizing Prophet hyperparameters")
        
        # Define hyperparameter combinations to test
        param_combinations = [
            # Conservative parameters
            {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.8
            },
            # Moderate parameters
            {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.8
            },
            # More flexible parameters
            {
                'changepoint_prior_scale': 0.2,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.9
            },
            # Multiplicative seasonality
            {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'multiplicative',
                'changepoint_range': 0.8
            }
        ]
        
        best_params = None
        best_score = float('inf')
        
        for params in param_combinations:
            try:
                # Create model with parameters
                model = Prophet(**params)
                
                # Add external regressors
                for regressor in self.external_regressors:
                    model.add_regressor(regressor)
                
                # Fit model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(df)
                
                # Perform cross-validation
                cv_results = cross_validation(
                    model, 
                    initial='30 days', 
                    period='7 days', 
                    horizon='7 days',
                    disable_tqdm=True
                )
                
                # Calculate performance metrics
                perf_metrics = performance_metrics(cv_results)
                avg_rmse = perf_metrics['rmse'].mean()
                
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_params = params
                
                logger.info(f"Tested params: RMSE = {avg_rmse:.3f}")
                
            except Exception as e:
                logger.warning(f"Error testing parameters {params}: {e}")
                continue
        
        if best_params is None:
            # Fallback to default parameters
            best_params = {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.8
            }
            logger.warning("Using default parameters due to optimization failures")
        
        logger.info(f"Best parameters found: {best_params} (RMSE: {best_score:.3f})")
        return best_params
    
    def train(self, start_date: date, end_date: date, 
              optimize_hyperparams: bool = True,
              include_ensemble: bool = True,
              include_la_patterns: bool = True,
              max_regressors: int = 10) -> Dict[str, Any]:
        """Train the Prophet model.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            optimize_hyperparams: Whether to optimize hyperparameters
            include_ensemble: Whether to include ensemble features
            include_la_patterns: Whether to include LA-specific features
            max_regressors: Maximum number of external regressors
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training Prophet model from {start_date} to {end_date}")
        
        # Prepare training data
        df = self.prepare_training_data(
            start_date, end_date, 
            include_ensemble, include_la_patterns, 
            max_regressors
        )
        
        if df.empty:
            raise ValueError("No training data available")
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            self.hyperparameters = self.optimize_hyperparameters(df)
        else:
            # Use default parameters
            self.hyperparameters = {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.8
            }
        
        # Create and configure model
        self.model = Prophet(**self.hyperparameters)
        
        # Add external regressors
        for regressor in self.external_regressors:
            self.model.add_regressor(regressor)
        
        # Add custom seasonalities for LA weather patterns
        self.model.add_seasonality(
            name='monthly', 
            period=30.5, 
            fourier_order=5
        )
        
        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)
        
        # Calculate cross-validation scores
        try:
            cv_results = cross_validation(
                self.model, 
                initial='30 days', 
                period='7 days', 
                horizon='7 days',
                disable_tqdm=True
            )
            
            perf_metrics = performance_metrics(cv_results)
            self.cv_scores = {
                'rmse_mean': perf_metrics['rmse'].mean(),
                'rmse_std': perf_metrics['rmse'].std(),
                'mae_mean': perf_metrics['mae'].mean(),
                'mape_mean': perf_metrics['mape'].mean()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            self.cv_scores = {'rmse_mean': 0.0, 'rmse_std': 0.0, 'mae_mean': 0.0, 'mape_mean': 0.0}
        
        # Calculate regressor importance (coefficients)
        if hasattr(self.model, 'params') and 'beta' in self.model.params:
            regressor_coefs = self.model.params['beta'].mean(axis=0)
            self.regressor_importance = dict(zip(self.external_regressors, np.abs(regressor_coefs)))
        else:
            self.regressor_importance = {reg: 1.0 for reg in self.external_regressors}
        
        # Update training metadata
        self.training_date = datetime.now()
        self.training_samples = len(df)
        self.is_trained = True
        
        # Calculate training metrics
        future = self.model.make_future_dataframe(periods=0)
        for regressor in self.external_regressors:
            future[regressor] = df[regressor]
        
        forecast = self.model.predict(future)
        y_pred = forecast['yhat'].values
        y_true = df['y'].values
        
        training_metrics = self.calculate_metrics(y_true, y_pred)
        
        training_results = {
            'training_samples': self.training_samples,
            'external_regressors': len(self.external_regressors),
            'regressors_used': self.external_regressors,
            'hyperparameters': self.hyperparameters,
            'cv_scores': self.cv_scores,
            'training_metrics': training_metrics,
            'regressor_importance': dict(sorted(self.regressor_importance.items(), 
                                              key=lambda x: x[1], reverse=True)),
            'seasonality_components': {
                'yearly': True,
                'weekly': True,
                'monthly': True,
                'daily': False
            }
        }
        
        logger.info(f"Prophet model training complete. CV RMSE: {self.cv_scores['rmse_mean']:.3f}")
        return training_results
    
    def predict(self, features: pd.DataFrame, target_date: Optional[date] = None) -> Tuple[float, float]:
        """Make a temperature prediction.
        
        Args:
            features: DataFrame with features for prediction
            target_date: Date to predict for (default: tomorrow)
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if features.empty:
            raise ValueError("No features provided for prediction")
        
        # Set target date
        if target_date is None:
            target_date = date.today() + timedelta(days=1)
        
        # Create future dataframe for prediction
        future = pd.DataFrame()
        future['ds'] = [pd.to_datetime(target_date)]
        
        # Add external regressors
        for regressor in self.external_regressors:
            if regressor in features.columns:
                regressor_value = features[regressor].iloc[0] if not features[regressor].isna().all() else 0
            else:
                # Use median value from training if regressor not available
                regressor_value = 0
                logger.warning(f"Regressor {regressor} not found in features, using default value")
            
            future[regressor] = [regressor_value]
        
        # Make prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)
        
        prediction = forecast['yhat'].iloc[0]
        
        # Calculate confidence based on prediction interval
        yhat_lower = forecast['yhat_lower'].iloc[0]
        yhat_upper = forecast['yhat_upper'].iloc[0]
        
        # Confidence based on prediction interval width (narrower = higher confidence)
        interval_width = yhat_upper - yhat_lower
        max_reasonable_interval = 20.0  # 20°F is a reasonable max interval
        confidence = max(0.1, 1.0 - (interval_width / max_reasonable_interval))
        
        # Adjust confidence based on cross-validation performance
        if self.cv_scores.get('rmse_mean', 0) > 0:
            cv_confidence = max(0.1, 1.0 - (self.cv_scores['rmse_mean'] / 20.0))
            confidence = (confidence + cv_confidence) / 2
        
        confidence = min(1.0, max(0.0, confidence))
        
        return prediction, confidence
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
        within_3f = np.mean(np.abs(y_true - y_pred) <= 3.0) * 100
        within_5f = np.mean(np.abs(y_true - y_pred) <= 5.0) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'accuracy_within_3f': within_3f,
            'accuracy_within_5f': within_5f
        }
    
    def get_forecast_components(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """Get forecast components breakdown.
        
        Args:
            target_date: Date to analyze (default: tomorrow)
            
        Returns:
            Dictionary with forecast components
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get forecast components")
        
        if target_date is None:
            target_date = date.today() + timedelta(days=1)
        
        # Create future dataframe
        future = pd.DataFrame()
        future['ds'] = [pd.to_datetime(target_date)]
        
        # Add external regressors with default values
        for regressor in self.external_regressors:
            future[regressor] = [0]  # Default value
        
        # Get forecast with components
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)
        
        components = {}
        
        # Main prediction
        components['prediction'] = forecast['yhat'].iloc[0]
        components['prediction_lower'] = forecast['yhat_lower'].iloc[0]
        components['prediction_upper'] = forecast['yhat_upper'].iloc[0]
        
        # Trend component
        if 'trend' in forecast.columns:
            components['trend'] = forecast['trend'].iloc[0]
        
        # Seasonal components
        if 'yearly' in forecast.columns:
            components['yearly_seasonality'] = forecast['yearly'].iloc[0]
        
        if 'weekly' in forecast.columns:
            components['weekly_seasonality'] = forecast['weekly'].iloc[0]
        
        if 'monthly' in forecast.columns:
            components['monthly_seasonality'] = forecast['monthly'].iloc[0]
        
        # External regressor contributions
        regressor_contributions = {}
        for regressor in self.external_regressors:
            if regressor in forecast.columns:
                regressor_contributions[regressor] = forecast[regressor].iloc[0]
        
        components['regressor_contributions'] = regressor_contributions
        
        return components
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """Save the trained model to disk.
        
        Args:
            filename: Optional filename (default: prophet_model_YYYYMMDD.pkl)
            
        Returns:
            Path to saved model file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prophet_model_{timestamp}.pkl"
        
        model_path = self.model_dir / filename
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'external_regressors': self.external_regressors,
            'hyperparameters': self.hyperparameters,
            'cv_scores': self.cv_scores,
            'regressor_importance': self.regressor_importance,
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
            self.external_regressors = model_data['external_regressors']
            self.hyperparameters = model_data['hyperparameters']
            self.cv_scores = model_data['cv_scores']
            self.regressor_importance = model_data['regressor_importance']
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
            return {'status': 'not_trained', 'prophet_available': PROPHET_AVAILABLE}
        
        return {
            'status': 'trained',
            'prophet_available': PROPHET_AVAILABLE,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'training_samples': self.training_samples,
            'external_regressors_count': len(self.external_regressors),
            'external_regressors': self.external_regressors,
            'hyperparameters': self.hyperparameters,
            'cv_performance': self.cv_scores,
            'top_regressors': dict(sorted(self.regressor_importance.items(), key=lambda x: x[1], reverse=True)[:5]),
            'model_type': 'Facebook Prophet with External Regressors'
        }


def main():
    """Demonstrate the Prophet model."""
    print("=== Facebook Prophet Temperature Model Demo ===\n")
    
    if not PROPHET_AVAILABLE:
        print("Prophet is not available. Please install with: pip install prophet")
        return
    
    # Initialize model
    model = ProphetTemperatureModel()
    
    print("1. Model Information:")
    info = model.get_model_info()
    print(f"   Status: {info['status']}")
    print(f"   Prophet Available: {info['prophet_available']}")
    
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
    print("\n2. Training Model (this may take several minutes)...")
    
    try:
        # Use recent data for training
        end_date = date.today()
        start_date = end_date - timedelta(days=90)  # Prophet works better with more data
        
        training_results = model.train(
            start_date=start_date,
            end_date=end_date,
            optimize_hyperparams=True,
            max_regressors=8
        )
        
        print("   Training Results:")
        print(f"   - Training samples: {training_results['training_samples']}")
        print(f"   - External regressors: {training_results['external_regressors']}")
        print(f"   - CV RMSE: {training_results['cv_scores']['rmse_mean']:.3f}")
        print(f"   - Training RMSE: {training_results['training_metrics']['rmse']:.3f}")
        print(f"   - Accuracy within ±3°F: {training_results['training_metrics']['accuracy_within_3f']:.1f}%")
        
        print("\n3. External Regressors Used:")
        for i, regressor in enumerate(training_results['regressors_used'][:5]):
            importance = training_results['regressor_importance'].get(regressor, 0)
            print(f"   {i+1}. {regressor}: {importance:.4f}")
        
        print("\n4. Seasonality Components:")
        for component, enabled in training_results['seasonality_components'].items():
            print(f"   - {component}: {'Enabled' if enabled else 'Disabled'}")
        
        # Save the model
        print("\n5. Saving Model...")
        model_path = model.save_model()
        print(f"   Model saved to: {model_path}")
        
        # Demonstrate prediction
        print("\n6. Making Sample Prediction...")
        feature_pipeline = FeaturePipeline()
        
        # Get features for today
        today_features = feature_pipeline.create_features_for_prediction(date.today())
        
        if not today_features.empty:
            prediction, confidence = model.predict(today_features)
            print(f"   Prediction for tomorrow: {prediction:.1f}°F")
            print(f"   Confidence: {confidence:.3f}")
            
            # Show forecast components
            components = model.get_forecast_components()
            print(f"   Forecast breakdown:")
            print(f"   - Trend: {components.get('trend', 0):.1f}°F")
            print(f"   - Yearly seasonality: {components.get('yearly_seasonality', 0):.1f}°F")
            print(f"   - Weekly seasonality: {components.get('weekly_seasonality', 0):.1f}°F")
            print(f"   - Monthly seasonality: {components.get('monthly_seasonality', 0):.1f}°F")
            
        else:
            print("   No features available for today's prediction")
        
    except Exception as e:
        print(f"   Error during training: {e}")
        logger.error(f"Training error: {e}")
    
    print("\n=== Prophet Model Demo Complete ===")


if __name__ == '__main__':
    main()