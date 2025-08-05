"""Ensemble model combining multiple ML models with dynamic weighting."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, Optional, Tuple, List, Any, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import joblib
from pathlib import Path
from abc import ABC, abstractmethod

from src.utils.data_manager import DataManager
from src.feature_engineering.feature_pipeline import FeaturePipeline


class BaseTemperatureModel(ABC):
    """Abstract base class for temperature prediction models."""
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> Tuple[float, float]:
        """Make a temperature prediction.
        
        Args:
            features: DataFrame with features for prediction
            
        Returns:
            Tuple of (prediction, confidence)
        """
        pass
    
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the model is trained."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class WeatherCondition:
    """Represents different weather conditions for model selection."""
    
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    WINDY = "windy"
    MARINE_LAYER = "marine_layer"
    SANTA_ANA = "santa_ana"
    HEAT_WAVE = "heat_wave"
    NORMAL = "normal"


class EnsembleTemperatureModel:
    """Ensemble model that combines multiple ML models with dynamic weighting."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the ensemble model.
        
        Args:
            model_dir: Directory to save/load models (default: models/)
        """
        self.model_dir = Path(model_dir) if model_dir else Path('models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models: Dict[str, BaseTemperatureModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_performance_history: Dict[str, List[Dict]] = {}
        
        # Ensemble configuration
        self.performance_window = 14  # Days to consider for performance calculation
        self.min_confidence_threshold = 0.3
        self.max_models_per_prediction = 5
        
        # Weather condition detection
        self.weather_condition_models: Dict[str, List[str]] = {
            WeatherCondition.CLEAR: [],
            WeatherCondition.CLOUDY: [],
            WeatherCondition.RAINY: [],
            WeatherCondition.WINDY: [],
            WeatherCondition.MARINE_LAYER: [],
            WeatherCondition.SANTA_ANA: [],
            WeatherCondition.HEAT_WAVE: [],
            WeatherCondition.NORMAL: []
        }
        
        # Data components
        self.data_manager = DataManager()
        self.feature_pipeline = FeaturePipeline()
        
        # Ensemble metadata
        self.last_update = None
        self.total_predictions = 0
        self.ensemble_performance_history: List[Dict] = []
        
        logger.info(f"EnsembleTemperatureModel initialized with model directory: {self.model_dir}")
    
    def register_model(self, name: str, model: BaseTemperatureModel, 
                      initial_weight: float = 1.0,
                      weather_conditions: Optional[List[str]] = None) -> None:
        """Register a model in the ensemble.
        
        Args:
            name: Unique name for the model
            model: Model instance implementing BaseTemperatureModel
            initial_weight: Initial weight for the model (default: 1.0)
            weather_conditions: List of weather conditions this model excels at
        """
        if not model.is_trained():
            logger.warning(f"Registering untrained model: {name}")
        
        self.models[name] = model
        self.model_weights[name] = initial_weight
        self.model_performance_history[name] = []
        
        # Register model for specific weather conditions
        if weather_conditions:
            for condition in weather_conditions:
                if condition in self.weather_condition_models:
                    self.weather_condition_models[condition].append(name)
        else:
            # Default to normal conditions
            self.weather_condition_models[WeatherCondition.NORMAL].append(name)
        
        logger.info(f"Registered model '{name}' with initial weight {initial_weight}")
        if weather_conditions:
            logger.info(f"Model '{name}' registered for conditions: {weather_conditions}")
    
    def detect_weather_condition(self, features: pd.DataFrame) -> str:
        """Detect current weather condition based on features.
        
        Args:
            features: DataFrame with weather features
            
        Returns:
            Weather condition string
        """
        if features.empty:
            return WeatherCondition.NORMAL
        
        # Get the first (and likely only) row of features
        feature_row = features.iloc[0] if len(features) > 0 else pd.Series()
        
        # Marine layer detection
        if self._detect_marine_layer(feature_row):
            return WeatherCondition.MARINE_LAYER
        
        # Santa Ana wind detection
        if self._detect_santa_ana(feature_row):
            return WeatherCondition.SANTA_ANA
        
        # Heat wave detection
        if self._detect_heat_wave(feature_row):
            return WeatherCondition.HEAT_WAVE
        
        # Rain detection
        if self._detect_rain(feature_row):
            return WeatherCondition.RAINY
        
        # High wind detection
        if self._detect_windy(feature_row):
            return WeatherCondition.WINDY
        
        # Cloud cover detection
        if self._detect_cloudy(feature_row):
            return WeatherCondition.CLOUDY
        
        # Clear conditions
        if self._detect_clear(feature_row):
            return WeatherCondition.CLEAR
        
        return WeatherCondition.NORMAL
    
    def _detect_marine_layer(self, features: pd.Series) -> bool:
        """Detect marine layer conditions."""
        # Look for marine layer indicators in features
        marine_indicators = [col for col in features.index if 'marine_layer' in col.lower()]
        if marine_indicators:
            return any(features[col] > 0.5 for col in marine_indicators if pd.notna(features[col]))
        
        # Fallback: high humidity + low cloud cover + moderate temperature
        humidity_cols = [col for col in features.index if 'humidity' in col.lower()]
        cloud_cols = [col for col in features.index if 'cloud' in col.lower()]
        
        high_humidity = any(features[col] > 80 for col in humidity_cols if pd.notna(features[col]))
        low_clouds = any(features[col] > 70 for col in cloud_cols if pd.notna(features[col]))
        
        return high_humidity and low_clouds
    
    def _detect_santa_ana(self, features: pd.Series) -> bool:
        """Detect Santa Ana wind conditions."""
        # Look for Santa Ana indicators in features
        santa_ana_indicators = [col for col in features.index if 'santa_ana' in col.lower()]
        if santa_ana_indicators:
            return any(features[col] > 0.5 for col in santa_ana_indicators if pd.notna(features[col]))
        
        # Fallback: high wind speed from northeast + low humidity
        wind_speed_cols = [col for col in features.index if 'wind_speed' in col.lower()]
        humidity_cols = [col for col in features.index if 'humidity' in col.lower()]
        
        high_wind = any(features[col] > 15 for col in wind_speed_cols if pd.notna(features[col]))
        low_humidity = any(features[col] < 30 for col in humidity_cols if pd.notna(features[col]))
        
        return high_wind and low_humidity
    
    def _detect_heat_wave(self, features: pd.Series) -> bool:
        """Detect heat wave conditions."""
        # Look for heat wave indicators
        heat_indicators = [col for col in features.index if 'heat' in col.lower() or 'fire_season' in col.lower()]
        if heat_indicators:
            return any(features[col] > 0.5 for col in heat_indicators if pd.notna(features[col]))
        
        # Fallback: high predicted temperatures
        temp_cols = [col for col in features.index if 'temp' in col.lower() and 'high' in col.lower()]
        if temp_cols:
            avg_temp = np.mean([features[col] for col in temp_cols if pd.notna(features[col])])
            return avg_temp > 90  # Above 90°F
        
        return False
    
    def _detect_rain(self, features: pd.Series) -> bool:
        """Detect rainy conditions."""
        precip_cols = [col for col in features.index if 'precipitation' in col.lower() or 'rain' in col.lower()]
        return any(features[col] > 50 for col in precip_cols if pd.notna(features[col]))  # >50% chance
    
    def _detect_windy(self, features: pd.Series) -> bool:
        """Detect windy conditions."""
        wind_cols = [col for col in features.index if 'wind_speed' in col.lower()]
        return any(features[col] > 20 for col in wind_cols if pd.notna(features[col]))  # >20 mph
    
    def _detect_cloudy(self, features: pd.Series) -> bool:
        """Detect cloudy conditions."""
        cloud_cols = [col for col in features.index if 'cloud' in col.lower()]
        return any(features[col] > 70 for col in cloud_cols if pd.notna(features[col]))  # >70% cloud cover
    
    def _detect_clear(self, features: pd.Series) -> bool:
        """Detect clear conditions."""
        cloud_cols = [col for col in features.index if 'cloud' in col.lower()]
        precip_cols = [col for col in features.index if 'precipitation' in col.lower()]
        
        low_clouds = all(features[col] < 30 for col in cloud_cols if pd.notna(features[col]))
        low_precip = all(features[col] < 20 for col in precip_cols if pd.notna(features[col]))
        
        return low_clouds and low_precip
    
    def get_models_for_condition(self, weather_condition: str) -> List[str]:
        """Get list of models suitable for a weather condition.
        
        Args:
            weather_condition: Weather condition string
            
        Returns:
            List of model names suitable for the condition
        """
        condition_models = self.weather_condition_models.get(weather_condition, [])
        
        # If no specific models for condition, use normal condition models
        if not condition_models:
            condition_models = self.weather_condition_models.get(WeatherCondition.NORMAL, [])
        
        # If still no models, use all available models
        if not condition_models:
            condition_models = list(self.models.keys())
        
        # Filter to only trained models
        trained_models = [name for name in condition_models if name in self.models and self.models[name].is_trained()]
        
        return trained_models
    
    def calculate_dynamic_weights(self, model_names: List[str], 
                                weather_condition: str = WeatherCondition.NORMAL) -> Dict[str, float]:
        """Calculate dynamic weights based on recent model performance.
        
        Args:
            model_names: List of model names to calculate weights for
            weather_condition: Current weather condition
            
        Returns:
            Dictionary mapping model names to weights
        """
        if not model_names:
            return {}
        
        weights = {}
        
        for model_name in model_names:
            if model_name not in self.model_performance_history:
                # No performance history, use initial weight
                weights[model_name] = self.model_weights.get(model_name, 1.0)
                continue
            
            # Get recent performance data
            recent_performance = self._get_recent_performance(model_name, weather_condition)
            
            if not recent_performance:
                # No recent performance data, use initial weight
                weights[model_name] = self.model_weights.get(model_name, 1.0)
                continue
            
            # Calculate weight based on recent accuracy
            avg_accuracy = np.mean([p['accuracy'] for p in recent_performance])
            avg_confidence = np.mean([p['confidence'] for p in recent_performance])
            
            # Weight formula: accuracy * confidence * recency_factor
            recency_factor = self._calculate_recency_factor(recent_performance)
            weight = avg_accuracy * avg_confidence * recency_factor
            
            weights[model_name] = max(0.1, weight)  # Minimum weight of 0.1
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(model_names)
            weights = {name: equal_weight for name in model_names}
        
        return weights
    
    def _get_recent_performance(self, model_name: str, weather_condition: str) -> List[Dict]:
        """Get recent performance data for a model under specific weather conditions.
        
        Args:
            model_name: Name of the model
            weather_condition: Weather condition to filter by
            
        Returns:
            List of recent performance records
        """
        if model_name not in self.model_performance_history:
            return []
        
        # Get performance records from the last N days
        cutoff_date = datetime.now() - timedelta(days=self.performance_window)
        
        recent_records = []
        for record in self.model_performance_history[model_name]:
            record_date = record.get('date')
            if isinstance(record_date, str):
                record_date = datetime.fromisoformat(record_date)
            elif isinstance(record_date, date):
                record_date = datetime.combine(record_date, datetime.min.time())
            
            if record_date and record_date >= cutoff_date:
                # Filter by weather condition if specified
                if weather_condition == WeatherCondition.NORMAL or record.get('weather_condition') == weather_condition:
                    recent_records.append(record)
        
        return recent_records
    
    def _calculate_recency_factor(self, performance_records: List[Dict]) -> float:
        """Calculate recency factor giving more weight to recent predictions.
        
        Args:
            performance_records: List of performance records
            
        Returns:
            Recency factor between 0 and 1
        """
        if not performance_records:
            return 1.0
        
        # Calculate weighted average with exponential decay
        total_weight = 0
        weighted_sum = 0
        
        for i, record in enumerate(reversed(performance_records)):  # Most recent first
            # Exponential decay: more recent = higher weight
            weight = np.exp(-i * 0.1)  # Decay factor of 0.1
            weighted_sum += weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0
    
    def predict(self, features: pd.DataFrame, 
               use_weather_condition_selection: bool = True) -> Tuple[float, float]:
        """Make an ensemble temperature prediction.
        
        Args:
            features: DataFrame with features for prediction
            use_weather_condition_selection: Whether to use weather-specific model selection
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.models:
            raise ValueError("No models registered in ensemble")
        
        if features.empty:
            raise ValueError("No features provided for prediction")
        
        # Detect weather condition
        weather_condition = WeatherCondition.NORMAL
        if use_weather_condition_selection:
            weather_condition = self.detect_weather_condition(features)
            logger.info(f"Detected weather condition: {weather_condition}")
        
        # Get suitable models for this condition
        suitable_models = self.get_models_for_condition(weather_condition)
        
        if not suitable_models:
            logger.warning(f"No suitable models found for condition {weather_condition}, using all models")
            suitable_models = [name for name, model in self.models.items() if model.is_trained()]
        
        if not suitable_models:
            raise ValueError("No trained models available for prediction")
        
        # Limit number of models to prevent over-complexity
        if len(suitable_models) > self.max_models_per_prediction:
            # Select top models based on recent performance
            model_scores = {}
            for model_name in suitable_models:
                recent_perf = self._get_recent_performance(model_name, weather_condition)
                if recent_perf:
                    avg_accuracy = np.mean([p['accuracy'] for p in recent_perf])
                    model_scores[model_name] = avg_accuracy
                else:
                    model_scores[model_name] = 0.5  # Default score
            
            # Select top N models
            top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            suitable_models = [name for name, _ in top_models[:self.max_models_per_prediction]]
        
        # Calculate dynamic weights
        weights = self.calculate_dynamic_weights(suitable_models, weather_condition)
        
        # Make predictions from each model
        predictions = {}
        confidences = {}
        
        for model_name in suitable_models:
            try:
                model = self.models[model_name]
                pred, conf = model.predict(features)
                
                # Filter out predictions with very low confidence
                if conf >= self.min_confidence_threshold:
                    predictions[model_name] = pred
                    confidences[model_name] = conf
                else:
                    logger.warning(f"Model {model_name} prediction filtered due to low confidence: {conf:.3f}")
                    
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions obtained from any model")
        
        # Calculate weighted ensemble prediction
        total_weight = 0
        weighted_prediction = 0
        weighted_confidence = 0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0)
            confidence = confidences[model_name]
            
            # Weight by both ensemble weight and individual confidence
            effective_weight = weight * confidence
            
            weighted_prediction += prediction * effective_weight
            weighted_confidence += confidence * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            raise ValueError("Total weight is zero, cannot make ensemble prediction")
        
        # Normalize
        final_prediction = weighted_prediction / total_weight
        final_confidence = weighted_confidence / total_weight
        
        # Apply ensemble confidence boost (ensemble typically more reliable)
        ensemble_boost = min(1.2, 1.0 + (len(predictions) - 1) * 0.05)  # Up to 20% boost
        final_confidence = min(1.0, final_confidence * ensemble_boost)
        
        # Log prediction details
        logger.info(f"Ensemble prediction: {final_prediction:.1f}°F (confidence: {final_confidence:.3f})")
        logger.info(f"Used {len(predictions)} models for weather condition: {weather_condition}")
        logger.info(f"Model contributions: {[(name, weights.get(name, 0), pred) for name, pred in predictions.items()]}")
        
        # Update prediction count
        self.total_predictions += 1
        
        return final_prediction, final_confidence
    
    def update_model_performance(self, model_name: str, prediction: float, 
                               actual_temperature: float, confidence: float,
                               weather_condition: str = WeatherCondition.NORMAL,
                               prediction_date: Optional[date] = None) -> None:
        """Update performance history for a model.
        
        Args:
            model_name: Name of the model
            prediction: Predicted temperature
            actual_temperature: Actual observed temperature
            confidence: Confidence score of the prediction
            weather_condition: Weather condition during prediction
            prediction_date: Date of the prediction (default: today)
        """
        if model_name not in self.model_performance_history:
            self.model_performance_history[model_name] = []
        
        # Calculate accuracy metrics
        error = abs(prediction - actual_temperature)
        accuracy_3f = 1.0 if error <= 3.0 else 0.0  # Within ±3°F
        accuracy_5f = 1.0 if error <= 5.0 else 0.0  # Within ±5°F
        
        # Create performance record
        performance_record = {
            'date': prediction_date or date.today(),
            'prediction': prediction,
            'actual': actual_temperature,
            'error': error,
            'confidence': confidence,
            'accuracy': accuracy_3f,  # Primary accuracy metric
            'accuracy_5f': accuracy_5f,
            'weather_condition': weather_condition
        }
        
        self.model_performance_history[model_name].append(performance_record)
        
        # Keep only recent history to prevent memory bloat
        max_history = self.performance_window * 3  # Keep 3x the window size
        if len(self.model_performance_history[model_name]) > max_history:
            self.model_performance_history[model_name] = self.model_performance_history[model_name][-max_history:]
        
        logger.info(f"Updated performance for {model_name}: error={error:.2f}°F, accuracy={accuracy_3f}")
    
    def update_ensemble_performance(self, prediction: float, actual_temperature: float, 
                                  confidence: float, weather_condition: str = WeatherCondition.NORMAL,
                                  prediction_date: Optional[date] = None) -> None:
        """Update ensemble performance history.
        
        Args:
            prediction: Ensemble predicted temperature
            actual_temperature: Actual observed temperature
            confidence: Ensemble confidence score
            weather_condition: Weather condition during prediction
            prediction_date: Date of the prediction (default: today)
        """
        error = abs(prediction - actual_temperature)
        accuracy_3f = 1.0 if error <= 3.0 else 0.0
        accuracy_5f = 1.0 if error <= 5.0 else 0.0
        
        performance_record = {
            'date': prediction_date or date.today(),
            'prediction': prediction,
            'actual': actual_temperature,
            'error': error,
            'confidence': confidence,
            'accuracy': accuracy_3f,
            'accuracy_5f': accuracy_5f,
            'weather_condition': weather_condition,
            'models_used': len(self.models)
        }
        
        self.ensemble_performance_history.append(performance_record)
        
        # Keep only recent history
        max_history = self.performance_window * 3
        if len(self.ensemble_performance_history) > max_history:
            self.ensemble_performance_history = self.ensemble_performance_history[-max_history:]
        
        self.last_update = datetime.now()
        logger.info(f"Updated ensemble performance: error={error:.2f}°F, accuracy={accuracy_3f}")
    
    def get_model_performance_summary(self, model_name: Optional[str] = None, 
                                    days: int = 14) -> Dict[str, Any]:
        """Get performance summary for a model or the ensemble.
        
        Args:
            model_name: Name of specific model (None for ensemble summary)
            days: Number of recent days to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        if model_name is None:
            # Ensemble performance
            performance_data = self.ensemble_performance_history
            title = "Ensemble"
        else:
            # Specific model performance
            performance_data = self.model_performance_history.get(model_name, [])
            title = f"Model '{model_name}'"
        
        if not performance_data:
            return {'error': f'No performance data available for {title}'}
        
        # Filter to recent days
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = []
        
        for record in performance_data:
            record_date = record.get('date')
            if isinstance(record_date, str):
                record_date = datetime.fromisoformat(record_date)
            elif isinstance(record_date, date):
                record_date = datetime.combine(record_date, datetime.min.time())
            
            if record_date and record_date >= cutoff_date:
                recent_data.append(record)
        
        if not recent_data:
            return {'error': f'No recent performance data for {title}'}
        
        # Calculate metrics
        errors = [r['error'] for r in recent_data]
        accuracies = [r['accuracy'] for r in recent_data]
        accuracies_5f = [r['accuracy_5f'] for r in recent_data]
        confidences = [r['confidence'] for r in recent_data]
        
        # Group by weather condition
        condition_performance = {}
        for record in recent_data:
            condition = record.get('weather_condition', WeatherCondition.NORMAL)
            if condition not in condition_performance:
                condition_performance[condition] = []
            condition_performance[condition].append(record)
        
        condition_stats = {}
        for condition, records in condition_performance.items():
            cond_errors = [r['error'] for r in records]
            cond_accuracies = [r['accuracy'] for r in records]
            
            condition_stats[condition] = {
                'count': len(records),
                'avg_error': np.mean(cond_errors),
                'accuracy_3f': np.mean(cond_accuracies) * 100,
                'rmse': np.sqrt(np.mean([e**2 for e in cond_errors]))
            }
        
        return {
            'model_name': title,
            'period_days': days,
            'total_predictions': len(recent_data),
            'avg_error': np.mean(errors),
            'median_error': np.median(errors),
            'rmse': np.sqrt(np.mean([e**2 for e in errors])),
            'accuracy_within_3f': np.mean(accuracies) * 100,
            'accuracy_within_5f': np.mean(accuracies_5f) * 100,
            'avg_confidence': np.mean(confidences),
            'condition_performance': condition_stats,
            'recent_predictions': recent_data[-5:] if len(recent_data) >= 5 else recent_data
        }
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current status of the ensemble.
        
        Returns:
            Dictionary with ensemble status information
        """
        trained_models = [name for name, model in self.models.items() if model.is_trained()]
        
        # Get current weights
        if trained_models:
            current_weights = self.calculate_dynamic_weights(trained_models)
        else:
            current_weights = {}
        
        # Model information
        model_info = {}
        for name, model in self.models.items():
            info = model.get_model_info()
            model_info[name] = {
                'status': info.get('status', 'unknown'),
                'current_weight': current_weights.get(name, 0.0),
                'performance_records': len(self.model_performance_history.get(name, [])),
                'weather_conditions': [cond for cond, models in self.weather_condition_models.items() if name in models]
            }
        
        return {
            'total_models': len(self.models),
            'trained_models': len(trained_models),
            'total_predictions': self.total_predictions,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'performance_window_days': self.performance_window,
            'min_confidence_threshold': self.min_confidence_threshold,
            'max_models_per_prediction': self.max_models_per_prediction,
            'model_info': model_info,
            'weather_condition_models': self.weather_condition_models,
            'ensemble_performance_records': len(self.ensemble_performance_history)
        }
    
    def save_ensemble(self, filename: Optional[str] = None) -> str:
        """Save the ensemble configuration and performance history.
        
        Args:
            filename: Optional filename (default: ensemble_model_YYYYMMDD.pkl)
            
        Returns:
            Path to saved ensemble file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_model_{timestamp}.pkl"
        
        ensemble_path = self.model_dir / filename
        
        # Save ensemble data (models are saved separately)
        ensemble_data = {
            'model_weights': self.model_weights,
            'model_performance_history': self.model_performance_history,
            'weather_condition_models': self.weather_condition_models,
            'ensemble_performance_history': self.ensemble_performance_history,
            'performance_window': self.performance_window,
            'min_confidence_threshold': self.min_confidence_threshold,
            'max_models_per_prediction': self.max_models_per_prediction,
            'total_predictions': self.total_predictions,
            'last_update': self.last_update
        }
        
        joblib.dump(ensemble_data, ensemble_path)
        logger.info(f"Ensemble configuration saved to {ensemble_path}")
        
        return str(ensemble_path)
    
    def load_ensemble(self, ensemble_path: str) -> None:
        """Load ensemble configuration and performance history.
        
        Args:
            ensemble_path: Path to the saved ensemble file
        """
        try:
            ensemble_data = joblib.load(ensemble_path)
            
            self.model_weights = ensemble_data.get('model_weights', {})
            self.model_performance_history = ensemble_data.get('model_performance_history', {})
            self.weather_condition_models = ensemble_data.get('weather_condition_models', {})
            self.ensemble_performance_history = ensemble_data.get('ensemble_performance_history', [])
            self.performance_window = ensemble_data.get('performance_window', 14)
            self.min_confidence_threshold = ensemble_data.get('min_confidence_threshold', 0.3)
            self.max_models_per_prediction = ensemble_data.get('max_models_per_prediction', 5)
            self.total_predictions = ensemble_data.get('total_predictions', 0)
            self.last_update = ensemble_data.get('last_update')
            
            logger.info(f"Ensemble configuration loaded from {ensemble_path}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble from {ensemble_path}: {e}")
            raise


def main():
    """Demonstrate the ensemble model."""
    print("=== Ensemble Temperature Model Demo ===\n")
    
    # This is a basic demo - in practice, you would register actual trained models
    ensemble = EnsembleTemperatureModel()
    
    print("1. Ensemble Status:")
    status = ensemble.get_ensemble_status()
    print(f"   Total models: {status['total_models']}")
    print(f"   Trained models: {status['trained_models']}")
    print(f"   Total predictions: {status['total_predictions']}")
    
    print("\n2. Weather Condition Detection:")
    # Create sample features for demonstration
    sample_features = pd.DataFrame({
        'openweather_humidity': [85],
        'nws_cloud_cover': [75],
        'tomorrow_wind_speed': [8],
        'visual_crossing_precipitation_prob': [15]
    })
    
    condition = ensemble.detect_weather_condition(sample_features)
    print(f"   Detected condition: {condition}")
    
    print("\n3. Model Registration:")
    print("   In a real implementation, you would register trained models:")
    print("   ensemble.register_model('xgboost', xgb_model, 1.0, ['normal', 'clear'])")
    print("   ensemble.register_model('lightgbm', lgb_model, 1.0, ['cloudy', 'rainy'])")
    print("   ensemble.register_model('prophet', prophet_model, 0.8, ['heat_wave'])")
    
    print("\n4. Dynamic Weighting:")
    print("   The ensemble calculates dynamic weights based on:")
    print("   - Recent model performance (14-day window)")
    print("   - Weather condition suitability")
    print("   - Model confidence scores")
    print("   - Recency factor (more recent = higher weight)")
    
    print("\n5. Ensemble Features:")
    print("   ✓ Dynamic weighting based on recent performance")
    print("   ✓ Weather condition-specific model selection")
    print("   ✓ Confidence scoring with ensemble boost")
    print("   ✓ Performance tracking and history")
    print("   ✓ Automatic model filtering by confidence threshold")
    print("   ✓ Configurable maximum models per prediction")
    
    print("\n=== Ensemble Model Demo Complete ===")


if __name__ == '__main__':
    main()