"""Enhanced ensemble combination system with improved dynamic weighting and model selection."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from src.models.ensemble_model import EnsembleTemperatureModel, WeatherCondition, BaseTemperatureModel


class ModelPerformanceMetric(Enum):
    """Metrics for evaluating model performance."""
    ACCURACY_3F = "accuracy_3f"  # Within ±3°F
    ACCURACY_5F = "accuracy_5f"  # Within ±5°F
    RMSE = "rmse"
    MAE = "mae"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class ModelPrediction:
    """Container for model prediction with metadata."""
    model_name: str
    prediction: float
    confidence: float
    weight: float
    weather_condition: str
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction with detailed breakdown."""
    prediction: float
    confidence: float
    weather_condition: str
    model_predictions: List[ModelPrediction]
    ensemble_method: str
    total_models_used: int
    confidence_boost: float


class EnhancedEnsembleCombiner:
    """Enhanced ensemble combination system with sophisticated weighting and selection."""
    
    def __init__(self, base_ensemble: EnsembleTemperatureModel):
        """Initialize the enhanced combiner.
        
        Args:
            base_ensemble: Base ensemble model to enhance
        """
        self.base_ensemble = base_ensemble
        
        # Enhanced configuration
        self.performance_metrics = [
            ModelPerformanceMetric.ACCURACY_3F,
            ModelPerformanceMetric.RMSE,
            ModelPerformanceMetric.CONFIDENCE_CALIBRATION
        ]
        
        # Weighting strategies
        self.weighting_strategies = {
            'performance_based': self._calculate_performance_weights,
            'confidence_weighted': self._calculate_confidence_weights,
            'hybrid': self._calculate_hybrid_weights,
            'adaptive': self._calculate_adaptive_weights
        }
        
        # Model selection strategies
        self.selection_strategies = {
            'weather_condition': self._select_by_weather_condition,
            'performance_threshold': self._select_by_performance,
            'diversity_based': self._select_for_diversity,
            'adaptive_selection': self._adaptive_model_selection
        }
        
        # Configuration
        self.default_weighting_strategy = 'adaptive'
        self.default_selection_strategy = 'adaptive_selection'
        self.min_models_for_ensemble = 2
        self.max_models_for_ensemble = 4
        self.confidence_boost_factor = 0.15  # 15% boost for ensemble
        self.diversity_threshold = 2.0  # Minimum prediction spread for diversity
        
        logger.info("EnhancedEnsembleCombiner initialized")
    
    def predict(self, features: pd.DataFrame, 
               weighting_strategy: str = None,
               selection_strategy: str = None,
               target_date: Optional[date] = None) -> EnsemblePrediction:
        """Make an enhanced ensemble prediction.
        
        Args:
            features: DataFrame with features for prediction
            weighting_strategy: Strategy for calculating model weights
            selection_strategy: Strategy for selecting models
            target_date: Target date for prediction (for performance lookup)
            
        Returns:
            EnsemblePrediction with detailed breakdown
        """
        if features.empty:
            raise ValueError("No features provided for prediction")
        
        weighting_strategy = weighting_strategy or self.default_weighting_strategy
        selection_strategy = selection_strategy or self.default_selection_strategy
        
        # Detect weather condition
        weather_condition = self.base_ensemble.detect_weather_condition(features)
        logger.info(f"Detected weather condition: {weather_condition}")
        
        # Select models using chosen strategy
        selected_models = self._select_models(features, weather_condition, selection_strategy)
        
        if len(selected_models) < self.min_models_for_ensemble:
            logger.warning(f"Only {len(selected_models)} models available, using all trained models")
            selected_models = [name for name, model in self.base_ensemble.models.items() 
                             if model.is_trained()]
        
        if not selected_models:
            raise ValueError("No trained models available for prediction")
        
        # Get predictions from selected models
        model_predictions = self._get_model_predictions(selected_models, features, weather_condition)
        
        if not model_predictions:
            raise ValueError("No valid predictions obtained from selected models")
        
        # Calculate weights using chosen strategy
        weights = self._calculate_weights(model_predictions, weighting_strategy, weather_condition)
        
        # Update model predictions with calculated weights
        for pred in model_predictions:
            pred.weight = weights.get(pred.model_name, 0.0)
        
        # Combine predictions
        ensemble_pred, ensemble_conf = self._combine_predictions(model_predictions)
        
        # Apply confidence boost
        confidence_boost = self._calculate_confidence_boost(model_predictions)
        final_confidence = min(1.0, ensemble_conf * (1.0 + confidence_boost))
        
        # Create ensemble prediction object
        ensemble_prediction = EnsemblePrediction(
            prediction=ensemble_pred,
            confidence=final_confidence,
            weather_condition=weather_condition,
            model_predictions=model_predictions,
            ensemble_method=f"{selection_strategy}+{weighting_strategy}",
            total_models_used=len(model_predictions),
            confidence_boost=confidence_boost
        )
        
        logger.info(f"Enhanced ensemble prediction: {ensemble_pred:.1f}°F "
                   f"(confidence: {final_confidence:.3f}, boost: {confidence_boost:.3f})")
        
        return ensemble_prediction
    
    def _select_models(self, features: pd.DataFrame, weather_condition: str, 
                      strategy: str) -> List[str]:
        """Select models using the specified strategy.
        
        Args:
            features: Input features
            weather_condition: Detected weather condition
            strategy: Selection strategy name
            
        Returns:
            List of selected model names
        """
        if strategy not in self.selection_strategies:
            logger.warning(f"Unknown selection strategy: {strategy}, using weather_condition")
            strategy = 'weather_condition'
        
        return self.selection_strategies[strategy](features, weather_condition)
    
    def _select_by_weather_condition(self, features: pd.DataFrame, 
                                   weather_condition: str) -> List[str]:
        """Select models based on weather condition suitability."""
        suitable_models = self.base_ensemble.get_models_for_condition(weather_condition)
        
        # Filter to trained models only
        trained_models = [name for name in suitable_models 
                         if name in self.base_ensemble.models and 
                         self.base_ensemble.models[name].is_trained()]
        
        return trained_models[:self.max_models_for_ensemble]
    
    def _select_by_performance(self, features: pd.DataFrame, 
                             weather_condition: str) -> List[str]:
        """Select models based on recent performance."""
        all_models = [name for name, model in self.base_ensemble.models.items() 
                     if model.is_trained()]
        
        # Calculate recent performance scores
        model_scores = {}
        for model_name in all_models:
            recent_perf = self.base_ensemble._get_recent_performance(model_name, weather_condition)
            if recent_perf:
                # Use accuracy as primary metric
                avg_accuracy = np.mean([p['accuracy'] for p in recent_perf])
                model_scores[model_name] = avg_accuracy
            else:
                model_scores[model_name] = 0.5  # Default score
        
        # Select top performing models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in sorted_models[:self.max_models_for_ensemble]]
        
        return selected
    
    def _select_for_diversity(self, features: pd.DataFrame, 
                            weather_condition: str) -> List[str]:
        """Select models to maximize prediction diversity."""
        all_models = [name for name, model in self.base_ensemble.models.items() 
                     if model.is_trained()]
        
        if len(all_models) <= self.max_models_for_ensemble:
            return all_models
        
        # Get predictions from all models
        predictions = {}
        for model_name in all_models:
            try:
                model = self.base_ensemble.models[model_name]
                pred, conf = model.predict(features)
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Error getting prediction from {model_name}: {e}")
                continue
        
        if len(predictions) <= self.max_models_for_ensemble:
            return list(predictions.keys())
        
        # Select diverse models using greedy selection
        selected = []
        remaining = list(predictions.keys())
        
        # Start with the model closest to median prediction
        pred_values = list(predictions.values())
        median_pred = np.median(pred_values)
        closest_to_median = min(remaining, key=lambda x: abs(predictions[x] - median_pred))
        selected.append(closest_to_median)
        remaining.remove(closest_to_median)
        
        # Add models that maximize diversity
        while len(selected) < self.max_models_for_ensemble and remaining:
            best_candidate = None
            best_diversity = 0
            
            for candidate in remaining:
                # Calculate diversity as minimum distance to selected models
                min_distance = min(abs(predictions[candidate] - predictions[sel]) 
                                 for sel in selected)
                if min_distance > best_diversity:
                    best_diversity = min_distance
                    best_candidate = candidate
            
            if best_candidate and best_diversity >= self.diversity_threshold:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # If no diverse candidate found, add best performing remaining model
                if remaining:
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
        
        return selected
    
    def _adaptive_model_selection(self, features: pd.DataFrame, 
                                weather_condition: str) -> List[str]:
        """Adaptive model selection combining multiple strategies."""
        # Start with weather condition suitable models
        weather_models = self._select_by_weather_condition(features, weather_condition)
        
        # Add top performing models not already selected
        performance_models = self._select_by_performance(features, weather_condition)
        
        # Combine and deduplicate
        combined = list(set(weather_models + performance_models))
        
        # If we have too many, prioritize by recent performance
        if len(combined) > self.max_models_for_ensemble:
            model_scores = {}
            for model_name in combined:
                recent_perf = self.base_ensemble._get_recent_performance(model_name, weather_condition)
                if recent_perf:
                    avg_accuracy = np.mean([p['accuracy'] for p in recent_perf])
                    model_scores[model_name] = avg_accuracy
                else:
                    model_scores[model_name] = 0.5
            
            # Select top performers
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            combined = [name for name, _ in sorted_models[:self.max_models_for_ensemble]]
        
        return combined
    
    def _get_model_predictions(self, model_names: List[str], features: pd.DataFrame,
                             weather_condition: str) -> List[ModelPrediction]:
        """Get predictions from specified models.
        
        Args:
            model_names: List of model names to get predictions from
            features: Input features
            weather_condition: Current weather condition
            
        Returns:
            List of ModelPrediction objects
        """
        predictions = []
        
        for model_name in model_names:
            try:
                model = self.base_ensemble.models[model_name]
                pred, conf = model.predict(features)
                
                # Filter by confidence threshold
                if conf >= self.base_ensemble.min_confidence_threshold:
                    model_pred = ModelPrediction(
                        model_name=model_name,
                        prediction=pred,
                        confidence=conf,
                        weight=0.0,  # Will be set later
                        weather_condition=weather_condition
                    )
                    predictions.append(model_pred)
                else:
                    logger.warning(f"Model {model_name} prediction filtered due to low confidence: {conf:.3f}")
                    
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_name}: {e}")
                continue
        
        return predictions
    
    def _calculate_weights(self, model_predictions: List[ModelPrediction], 
                         strategy: str, weather_condition: str) -> Dict[str, float]:
        """Calculate model weights using the specified strategy.
        
        Args:
            model_predictions: List of model predictions
            strategy: Weighting strategy name
            weather_condition: Current weather condition
            
        Returns:
            Dictionary mapping model names to weights
        """
        if strategy not in self.weighting_strategies:
            logger.warning(f"Unknown weighting strategy: {strategy}, using performance_based")
            strategy = 'performance_based'
        
        return self.weighting_strategies[strategy](model_predictions, weather_condition)
    
    def _calculate_performance_weights(self, model_predictions: List[ModelPrediction],
                                     weather_condition: str) -> Dict[str, float]:
        """Calculate weights based on recent model performance."""
        weights = {}
        
        for pred in model_predictions:
            recent_perf = self.base_ensemble._get_recent_performance(pred.model_name, weather_condition)
            
            if recent_perf:
                # Calculate composite performance score
                accuracies = [p['accuracy'] for p in recent_perf]
                errors = [p['error'] for p in recent_perf]
                
                avg_accuracy = np.mean(accuracies)
                avg_error = np.mean(errors)
                
                # Weight formula: accuracy / (1 + error)
                performance_score = avg_accuracy / (1 + avg_error)
                weights[pred.model_name] = performance_score
            else:
                # No performance history, use initial weight
                weights[pred.model_name] = self.base_ensemble.model_weights.get(pred.model_name, 1.0)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Equal weights fallback
            equal_weight = 1.0 / len(model_predictions)
            weights = {pred.model_name: equal_weight for pred in model_predictions}
        
        return weights
    
    def _calculate_confidence_weights(self, model_predictions: List[ModelPrediction],
                                    weather_condition: str) -> Dict[str, float]:
        """Calculate weights based on model confidence scores."""
        weights = {}
        
        for pred in model_predictions:
            # Use confidence as base weight, adjusted by recent performance
            base_weight = pred.confidence
            
            # Adjust by recent performance if available
            recent_perf = self.base_ensemble._get_recent_performance(pred.model_name, weather_condition)
            if recent_perf:
                avg_accuracy = np.mean([p['accuracy'] for p in recent_perf])
                performance_adjustment = avg_accuracy  # 0-1 multiplier
                adjusted_weight = base_weight * performance_adjustment
            else:
                adjusted_weight = base_weight * 0.8  # Slight penalty for no history
            
            weights[pred.model_name] = adjusted_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            equal_weight = 1.0 / len(model_predictions)
            weights = {pred.model_name: equal_weight for pred in model_predictions}
        
        return weights
    
    def _calculate_hybrid_weights(self, model_predictions: List[ModelPrediction],
                                weather_condition: str) -> Dict[str, float]:
        """Calculate weights using hybrid approach combining performance and confidence."""
        performance_weights = self._calculate_performance_weights(model_predictions, weather_condition)
        confidence_weights = self._calculate_confidence_weights(model_predictions, weather_condition)
        
        # Combine with 70% performance, 30% confidence
        hybrid_weights = {}
        for pred in model_predictions:
            name = pred.model_name
            perf_weight = performance_weights.get(name, 0.0)
            conf_weight = confidence_weights.get(name, 0.0)
            hybrid_weights[name] = 0.7 * perf_weight + 0.3 * conf_weight
        
        # Normalize
        total_weight = sum(hybrid_weights.values())
        if total_weight > 0:
            hybrid_weights = {name: weight / total_weight for name, weight in hybrid_weights.items()}
        
        return hybrid_weights
    
    def _calculate_adaptive_weights(self, model_predictions: List[ModelPrediction],
                                  weather_condition: str) -> Dict[str, float]:
        """Calculate adaptive weights that adjust based on prediction agreement."""
        # Start with hybrid weights
        base_weights = self._calculate_hybrid_weights(model_predictions, weather_condition)
        
        # Calculate prediction spread (diversity)
        predictions = [pred.prediction for pred in model_predictions]
        pred_std = np.std(predictions)
        
        # If predictions are very similar (low diversity), weight by confidence more
        # If predictions are diverse, weight by performance more
        diversity_threshold = 2.0  # degrees
        
        if pred_std < diversity_threshold:
            # Low diversity - trust confidence more
            confidence_weights = self._calculate_confidence_weights(model_predictions, weather_condition)
            adaptive_weights = {}
            for pred in model_predictions:
                name = pred.model_name
                base_w = base_weights.get(name, 0.0)
                conf_w = confidence_weights.get(name, 0.0)
                # Blend towards confidence weighting
                adaptive_weights[name] = 0.4 * base_w + 0.6 * conf_w
        else:
            # High diversity - trust performance more
            performance_weights = self._calculate_performance_weights(model_predictions, weather_condition)
            adaptive_weights = {}
            for pred in model_predictions:
                name = pred.model_name
                base_w = base_weights.get(name, 0.0)
                perf_w = performance_weights.get(name, 0.0)
                # Blend towards performance weighting
                adaptive_weights[name] = 0.3 * base_w + 0.7 * perf_w
        
        # Normalize
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            adaptive_weights = {name: weight / total_weight for name, weight in adaptive_weights.items()}
        
        return adaptive_weights
    
    def _combine_predictions(self, model_predictions: List[ModelPrediction]) -> Tuple[float, float]:
        """Combine model predictions using weighted average.
        
        Args:
            model_predictions: List of model predictions with weights
            
        Returns:
            Tuple of (ensemble_prediction, ensemble_confidence)
        """
        if not model_predictions:
            raise ValueError("No model predictions to combine")
        
        # Calculate weighted prediction
        total_weight = 0
        weighted_prediction = 0
        weighted_confidence = 0
        
        for pred in model_predictions:
            effective_weight = pred.weight * pred.confidence
            weighted_prediction += pred.prediction * effective_weight
            weighted_confidence += pred.confidence * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            raise ValueError("Total weight is zero, cannot combine predictions")
        
        ensemble_prediction = weighted_prediction / total_weight
        ensemble_confidence = weighted_confidence / total_weight
        
        return ensemble_prediction, ensemble_confidence
    
    def _calculate_confidence_boost(self, model_predictions: List[ModelPrediction]) -> float:
        """Calculate confidence boost based on ensemble characteristics.
        
        Args:
            model_predictions: List of model predictions
            
        Returns:
            Confidence boost factor (0.0 to confidence_boost_factor)
        """
        if len(model_predictions) <= 1:
            return 0.0
        
        # Base boost increases with number of models
        num_models_boost = min(0.05 * (len(model_predictions) - 1), 0.1)  # Up to 10%
        
        # Agreement boost - higher when predictions are similar
        predictions = [pred.prediction for pred in model_predictions]
        pred_std = np.std(predictions)
        
        # Lower standard deviation = higher agreement = higher boost
        max_std_for_boost = 3.0  # degrees
        agreement_boost = max(0, (max_std_for_boost - pred_std) / max_std_for_boost * 0.05)  # Up to 5%
        
        # Confidence boost - higher when individual confidences are high
        avg_confidence = np.mean([pred.confidence for pred in model_predictions])
        confidence_boost = (avg_confidence - 0.5) * 0.1 if avg_confidence > 0.5 else 0  # Up to 5%
        
        total_boost = num_models_boost + agreement_boost + confidence_boost
        return min(total_boost, self.confidence_boost_factor)
    
    def get_ensemble_analysis(self, ensemble_prediction: EnsemblePrediction) -> Dict[str, Any]:
        """Get detailed analysis of an ensemble prediction.
        
        Args:
            ensemble_prediction: EnsemblePrediction object to analyze
            
        Returns:
            Dictionary with detailed analysis
        """
        model_preds = ensemble_prediction.model_predictions
        
        # Basic statistics
        predictions = [pred.prediction for pred in model_preds]
        weights = [pred.weight for pred in model_preds]
        confidences = [pred.confidence for pred in model_preds]
        
        analysis = {
            'ensemble_prediction': ensemble_prediction.prediction,
            'ensemble_confidence': ensemble_prediction.confidence,
            'weather_condition': ensemble_prediction.weather_condition,
            'ensemble_method': ensemble_prediction.ensemble_method,
            'models_used': ensemble_prediction.total_models_used,
            'confidence_boost': ensemble_prediction.confidence_boost,
            
            # Prediction statistics
            'prediction_stats': {
                'mean': np.mean(predictions),
                'median': np.median(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'range': np.max(predictions) - np.min(predictions)
            },
            
            # Weight statistics
            'weight_stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max_weight': np.max(weights),
                'min_weight': np.min(weights),
                'weight_entropy': -np.sum([w * np.log(w + 1e-10) for w in weights])
            },
            
            # Confidence statistics
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            
            # Model contributions
            'model_contributions': [
                {
                    'model': pred.model_name,
                    'prediction': pred.prediction,
                    'confidence': pred.confidence,
                    'weight': pred.weight,
                    'contribution': pred.prediction * pred.weight
                }
                for pred in model_preds
            ]
        }
        
        return analysis
    
    def update_configuration(self, **kwargs) -> None:
        """Update ensemble combiner configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        valid_params = {
            'min_models_for_ensemble', 'max_models_for_ensemble',
            'confidence_boost_factor', 'diversity_threshold',
            'default_weighting_strategy', 'default_selection_strategy'
        }
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
                logger.info(f"Updated {param} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {param}")


def main():
    """Demonstrate the enhanced ensemble combiner."""
    print("=== Enhanced Ensemble Combiner Demo ===\n")
    
    # This would normally use a real ensemble with trained models
    from src.models.model_adapters import create_ensemble_with_models
    
    # Create base ensemble
    base_ensemble = create_ensemble_with_models()
    
    # Create enhanced combiner
    combiner = EnhancedEnsembleCombiner(base_ensemble)
    
    print("1. Enhanced Combiner Configuration:")
    print(f"   Default weighting strategy: {combiner.default_weighting_strategy}")
    print(f"   Default selection strategy: {combiner.default_selection_strategy}")
    print(f"   Model range: {combiner.min_models_for_ensemble}-{combiner.max_models_for_ensemble}")
    print(f"   Confidence boost factor: {combiner.confidence_boost_factor}")
    
    print("\n2. Available Strategies:")
    print(f"   Weighting: {list(combiner.weighting_strategies.keys())}")
    print(f"   Selection: {list(combiner.selection_strategies.keys())}")
    
    # Create sample features
    sample_features = pd.DataFrame({
        'nws_temp_high': [78.5],
        'openweather_temp_high': [79.2],
        'tomorrow_temp_high': [77.8],
        'openweather_pressure': [1015.2],
        'nws_wind_speed': [12.5],
        'visual_crossing_humidity': [65]
    })
    
    print("\n3. Making Enhanced Predictions:")
    
    strategies_to_test = [
        ('performance_based', 'weather_condition'),
        ('confidence_weighted', 'performance_threshold'),
        ('hybrid', 'diversity_based'),
        ('adaptive', 'adaptive_selection')
    ]
    
    for weight_strategy, select_strategy in strategies_to_test:
        try:
            ensemble_pred = combiner.predict(
                sample_features,
                weighting_strategy=weight_strategy,
                selection_strategy=select_strategy
            )
            
            print(f"   {weight_strategy} + {select_strategy}:")
            print(f"      Prediction: {ensemble_pred.prediction:.1f}°F")
            print(f"      Confidence: {ensemble_pred.confidence:.3f}")
            print(f"      Models used: {ensemble_pred.total_models_used}")
            print(f"      Confidence boost: {ensemble_pred.confidence_boost:.3f}")
            
        except Exception as e:
            print(f"   {weight_strategy} + {select_strategy}: Error - {e}")
    
    print("\n4. Detailed Analysis:")
    try:
        # Make prediction with default strategy
        ensemble_pred = combiner.predict(sample_features)
        analysis = combiner.get_ensemble_analysis(ensemble_pred)
        
        print(f"   Prediction range: {analysis['prediction_stats']['min']:.1f}°F - {analysis['prediction_stats']['max']:.1f}°F")
        print(f"   Prediction std: {analysis['prediction_stats']['std']:.2f}°F")
        print(f"   Weight entropy: {analysis['weight_stats']['weight_entropy']:.3f}")
        
        print("\n   Model Contributions:")
        for contrib in analysis['model_contributions']:
            print(f"      {contrib['model']}: {contrib['prediction']:.1f}°F "
                  f"(weight: {contrib['weight']:.3f}, conf: {contrib['confidence']:.3f})")
            
    except Exception as e:
        print(f"   Analysis error: {e}")
    
    print("\n=== Enhanced Ensemble Combiner Demo Complete ===")


if __name__ == '__main__':
    main()