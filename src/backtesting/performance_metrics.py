"""Performance metrics calculation for backtesting."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import date, datetime
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Types of performance metrics."""
    ACCURACY = "accuracy"
    ERROR = "error"
    CONFIDENCE = "confidence"
    TRADING = "trading"
    SEASONAL = "seasonal"


@dataclass
class PredictionResult:
    """Container for a single prediction result."""
    date: date
    predicted_temperature: float
    actual_temperature: float
    confidence: float
    model_name: Optional[str] = None
    weather_condition: Optional[str] = None
    
    @property
    def error(self) -> float:
        """Calculate absolute error."""
        return abs(self.predicted_temperature - self.actual_temperature)
    
    @property
    def squared_error(self) -> float:
        """Calculate squared error."""
        return (self.predicted_temperature - self.actual_temperature) ** 2
    
    @property
    def is_accurate_3f(self) -> bool:
        """Check if prediction is within ±3°F."""
        return self.error <= 3.0
    
    @property
    def is_accurate_5f(self) -> bool:
        """Check if prediction is within ±5°F."""
        return self.error <= 5.0


class PerformanceMetricsCalculator:
    """Calculates various performance metrics for temperature predictions."""
    
    def __init__(self):
        """Initialize the performance metrics calculator."""
        self.logger = logger
        
        # Thresholds for different accuracy levels
        self.accuracy_thresholds = {
            'excellent': 2.0,    # Within ±2°F
            'good': 3.0,         # Within ±3°F
            'acceptable': 5.0,   # Within ±5°F
            'poor': 10.0         # Within ±10°F
        }
        
        # Seasonal definitions (month-based)
        self.seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11]
        }
        
        logger.info("PerformanceMetricsCalculator initialized")
    
    def calculate_basic_metrics(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """Calculate basic accuracy and error metrics.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary with basic metrics
        """
        if not predictions:
            return {'error': 'No predictions provided'}
        
        errors = [pred.error for pred in predictions]
        squared_errors = [pred.squared_error for pred in predictions]
        
        # Basic error metrics
        mae = np.mean(errors)  # Mean Absolute Error
        rmse = np.sqrt(np.mean(squared_errors))  # Root Mean Square Error
        median_error = np.median(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        std_error = np.std(errors)
        
        # Accuracy metrics
        total_predictions = len(predictions)
        accurate_3f = sum(1 for pred in predictions if pred.is_accurate_3f)
        accurate_5f = sum(1 for pred in predictions if pred.is_accurate_5f)
        
        accuracy_3f = (accurate_3f / total_predictions) * 100
        accuracy_5f = (accurate_5f / total_predictions) * 100
        
        # Accuracy by threshold
        accuracy_by_threshold = {}
        for threshold_name, threshold_value in self.accuracy_thresholds.items():
            accurate_count = sum(1 for pred in predictions if pred.error <= threshold_value)
            accuracy_by_threshold[f'accuracy_{threshold_name}'] = (accurate_count / total_predictions) * 100
        
        # Bias (systematic over/under prediction)
        signed_errors = [pred.predicted_temperature - pred.actual_temperature for pred in predictions]
        bias = np.mean(signed_errors)
        
        metrics = {
            'total_predictions': total_predictions,
            'mae': mae,
            'rmse': rmse,
            'median_error': median_error,
            'max_error': max_error,
            'min_error': min_error,
            'std_error': std_error,
            'accuracy_within_3f': accuracy_3f,
            'accuracy_within_5f': accuracy_5f,
            'bias': bias,
            **accuracy_by_threshold
        }
        
        logger.info(f"Calculated basic metrics for {total_predictions} predictions: "
                   f"MAE={mae:.2f}°F, RMSE={rmse:.2f}°F, Accuracy(±3°F)={accuracy_3f:.1f}%")
        
        return metrics
    
    def calculate_confidence_metrics(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """Calculate confidence-related metrics.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary with confidence metrics
        """
        if not predictions:
            return {'error': 'No predictions provided'}
        
        confidences = [pred.confidence for pred in predictions]
        errors = [pred.error for pred in predictions]
        
        # Basic confidence statistics
        avg_confidence = np.mean(confidences)
        median_confidence = np.median(confidences)
        std_confidence = np.std(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        # Confidence calibration analysis
        # Group predictions by confidence bins and check if accuracy matches confidence
        confidence_bins = np.arange(0, 1.1, 0.1)  # 0.0, 0.1, 0.2, ..., 1.0
        calibration_data = []
        
        for i in range(len(confidence_bins) - 1):
            bin_min = confidence_bins[i]
            bin_max = confidence_bins[i + 1]
            
            # Find predictions in this confidence bin
            bin_predictions = [
                pred for pred in predictions 
                if bin_min <= pred.confidence < bin_max
            ]
            
            if bin_predictions:
                bin_accuracy = np.mean([pred.is_accurate_3f for pred in bin_predictions]) * 100
                bin_avg_confidence = np.mean([pred.confidence for pred in bin_predictions]) * 100
                bin_count = len(bin_predictions)
                
                calibration_data.append({
                    'confidence_bin': f"{bin_min:.1f}-{bin_max:.1f}",
                    'avg_confidence': bin_avg_confidence,
                    'actual_accuracy': bin_accuracy,
                    'count': bin_count,
                    'calibration_error': abs(bin_avg_confidence - bin_accuracy)
                })
        
        # Overall calibration error (Expected Calibration Error)
        if calibration_data:
            total_predictions = len(predictions)
            ece = sum(
                (data['count'] / total_predictions) * data['calibration_error']
                for data in calibration_data
            )
        else:
            ece = 0.0
        
        # Confidence-accuracy correlation
        try:
            # Convert boolean accuracy to numeric for correlation
            accuracy_numeric = [1.0 if pred.is_accurate_3f else 0.0 for pred in predictions]
            confidence_accuracy_corr = np.corrcoef(confidences, accuracy_numeric)[0, 1]
            if np.isnan(confidence_accuracy_corr):
                confidence_accuracy_corr = 0.0
        except:
            confidence_accuracy_corr = 0.0
        
        # High confidence performance
        high_conf_threshold = 0.8
        high_conf_predictions = [pred for pred in predictions if pred.confidence >= high_conf_threshold]
        
        if high_conf_predictions:
            high_conf_accuracy = np.mean([pred.is_accurate_3f for pred in high_conf_predictions]) * 100
            high_conf_mae = np.mean([pred.error for pred in high_conf_predictions])
        else:
            high_conf_accuracy = 0.0
            high_conf_mae = 0.0
        
        metrics = {
            'avg_confidence': avg_confidence,
            'median_confidence': median_confidence,
            'std_confidence': std_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'expected_calibration_error': ece,
            'confidence_accuracy_correlation': confidence_accuracy_corr,
            'high_confidence_count': len(high_conf_predictions),
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_mae': high_conf_mae,
            'calibration_bins': calibration_data
        }
        
        logger.info(f"Calculated confidence metrics: avg_confidence={avg_confidence:.3f}, "
                   f"calibration_error={ece:.2f}, high_conf_accuracy={high_conf_accuracy:.1f}%")
        
        return metrics
    
    def calculate_seasonal_metrics(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Calculate performance metrics by season.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary with seasonal performance metrics
        """
        if not predictions:
            return {'error': 'No predictions provided'}
        
        # Group predictions by season
        seasonal_predictions = {season: [] for season in self.seasons.keys()}
        
        for pred in predictions:
            month = pred.date.month
            for season, months in self.seasons.items():
                if month in months:
                    seasonal_predictions[season].append(pred)
                    break
        
        # Calculate metrics for each season
        seasonal_metrics = {}
        
        for season, season_preds in seasonal_predictions.items():
            if not season_preds:
                seasonal_metrics[season] = {'count': 0, 'note': 'No predictions available'}
                continue
            
            # Basic metrics for this season
            basic_metrics = self.calculate_basic_metrics(season_preds)
            confidence_metrics = self.calculate_confidence_metrics(season_preds)
            
            # Season-specific analysis
            temps = [pred.actual_temperature for pred in season_preds]
            avg_temp = np.mean(temps)
            temp_range = np.max(temps) - np.min(temps)
            
            seasonal_metrics[season] = {
                'count': len(season_preds),
                'avg_actual_temperature': avg_temp,
                'temperature_range': temp_range,
                'mae': basic_metrics['mae'],
                'rmse': basic_metrics['rmse'],
                'accuracy_within_3f': basic_metrics['accuracy_within_3f'],
                'accuracy_within_5f': basic_metrics['accuracy_within_5f'],
                'bias': basic_metrics['bias'],
                'avg_confidence': confidence_metrics['avg_confidence']
            }
        
        # Find best and worst performing seasons
        seasons_with_data = {
            season: metrics for season, metrics in seasonal_metrics.items() 
            if isinstance(metrics, dict) and metrics.get('count', 0) > 0
        }
        
        if seasons_with_data:
            best_season = min(seasons_with_data.keys(), 
                            key=lambda s: seasons_with_data[s]['mae'])
            worst_season = max(seasons_with_data.keys(), 
                             key=lambda s: seasons_with_data[s]['mae'])
            
            seasonal_summary = {
                'best_season': best_season,
                'worst_season': worst_season,
                'best_season_mae': seasons_with_data[best_season]['mae'],
                'worst_season_mae': seasons_with_data[worst_season]['mae'],
                'seasonal_variation': seasons_with_data[worst_season]['mae'] - seasons_with_data[best_season]['mae']
            }
        else:
            seasonal_summary = {'note': 'No seasonal data available'}
        
        result = {
            'seasonal_metrics': seasonal_metrics,
            'seasonal_summary': seasonal_summary
        }
        
        logger.info(f"Calculated seasonal metrics for {len(seasons_with_data)} seasons")
        
        return result
    
    def calculate_trend_metrics(self, predictions: List[PredictionResult], 
                              window_days: int = 30) -> Dict[str, Any]:
        """Calculate performance trends over time.
        
        Args:
            predictions: List of prediction results (should be sorted by date)
            window_days: Window size for rolling metrics
            
        Returns:
            Dictionary with trend metrics
        """
        if not predictions:
            return {'error': 'No predictions provided'}
        
        # Sort predictions by date
        sorted_predictions = sorted(predictions, key=lambda p: p.date)
        
        # Calculate rolling metrics
        rolling_metrics = []
        
        for i in range(len(sorted_predictions)):
            # Define window around current prediction
            start_idx = max(0, i - window_days // 2)
            end_idx = min(len(sorted_predictions), i + window_days // 2 + 1)
            
            window_predictions = sorted_predictions[start_idx:end_idx]
            
            if len(window_predictions) >= 5:  # Minimum window size
                window_mae = np.mean([pred.error for pred in window_predictions])
                window_accuracy = np.mean([pred.is_accurate_3f for pred in window_predictions]) * 100
                window_confidence = np.mean([pred.confidence for pred in window_predictions])
                
                rolling_metrics.append({
                    'date': sorted_predictions[i].date,
                    'mae': window_mae,
                    'accuracy_3f': window_accuracy,
                    'avg_confidence': window_confidence,
                    'window_size': len(window_predictions)
                })
        
        if not rolling_metrics:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trend statistics
        mae_values = [m['mae'] for m in rolling_metrics]
        accuracy_values = [m['accuracy_3f'] for m in rolling_metrics]
        confidence_values = [m['avg_confidence'] for m in rolling_metrics]
        
        # Linear trend analysis (simple slope calculation)
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
        
        mae_trend = calculate_trend(mae_values)
        accuracy_trend = calculate_trend(accuracy_values)
        confidence_trend = calculate_trend(confidence_values)
        
        # Performance stability (coefficient of variation)
        mae_stability = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0
        accuracy_stability = np.std(accuracy_values) / np.mean(accuracy_values) if np.mean(accuracy_values) > 0 else 0
        
        # Recent vs. early performance comparison
        split_point = len(rolling_metrics) // 2
        if split_point > 0:
            early_mae = np.mean(mae_values[:split_point])
            recent_mae = np.mean(mae_values[split_point:])
            mae_improvement = early_mae - recent_mae  # Positive = improvement
            
            early_accuracy = np.mean(accuracy_values[:split_point])
            recent_accuracy = np.mean(accuracy_values[split_point:])
            accuracy_improvement = recent_accuracy - early_accuracy  # Positive = improvement
        else:
            mae_improvement = 0.0
            accuracy_improvement = 0.0
        
        trend_metrics = {
            'rolling_window_days': window_days,
            'total_windows': len(rolling_metrics),
            'mae_trend_slope': mae_trend,
            'accuracy_trend_slope': accuracy_trend,
            'confidence_trend_slope': confidence_trend,
            'mae_stability': mae_stability,
            'accuracy_stability': accuracy_stability,
            'mae_improvement': mae_improvement,
            'accuracy_improvement': accuracy_improvement,
            'rolling_metrics': rolling_metrics[-10:] if len(rolling_metrics) > 10 else rolling_metrics  # Last 10 windows
        }
        
        logger.info(f"Calculated trend metrics over {len(rolling_metrics)} windows: "
                   f"MAE trend={mae_trend:.4f}, accuracy trend={accuracy_trend:.4f}")
        
        return trend_metrics
    
    def calculate_comprehensive_metrics(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Calculate a comprehensive set of performance metrics.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary with all calculated metrics
        """
        if not predictions:
            return {'error': 'No predictions provided'}
        
        logger.info(f"Calculating comprehensive metrics for {len(predictions)} predictions")
        
        # Calculate all metric types
        basic_metrics = self.calculate_basic_metrics(predictions)
        confidence_metrics = self.calculate_confidence_metrics(predictions)
        seasonal_metrics = self.calculate_seasonal_metrics(predictions)
        trend_metrics = self.calculate_trend_metrics(predictions)
        
        # Additional summary statistics
        date_range = {
            'start_date': min(pred.date for pred in predictions).isoformat(),
            'end_date': max(pred.date for pred in predictions).isoformat(),
            'total_days': (max(pred.date for pred in predictions) - min(pred.date for pred in predictions)).days + 1
        }
        
        # Temperature range analysis
        actual_temps = [pred.actual_temperature for pred in predictions]
        predicted_temps = [pred.predicted_temperature for pred in predictions]
        
        temperature_analysis = {
            'actual_temp_range': {
                'min': np.min(actual_temps),
                'max': np.max(actual_temps),
                'mean': np.mean(actual_temps),
                'std': np.std(actual_temps)
            },
            'predicted_temp_range': {
                'min': np.min(predicted_temps),
                'max': np.max(predicted_temps),
                'mean': np.mean(predicted_temps),
                'std': np.std(predicted_temps)
            }
        }
        
        # Overall performance grade
        accuracy_3f = basic_metrics.get('accuracy_within_3f', 0)
        mae = basic_metrics.get('mae', float('inf'))
        
        if accuracy_3f >= 85 and mae <= 2.5:
            performance_grade = 'Excellent'
        elif accuracy_3f >= 75 and mae <= 3.5:
            performance_grade = 'Good'
        elif accuracy_3f >= 65 and mae <= 5.0:
            performance_grade = 'Acceptable'
        else:
            performance_grade = 'Needs Improvement'
        
        comprehensive_metrics = {
            'summary': {
                'total_predictions': len(predictions),
                'date_range': date_range,
                'performance_grade': performance_grade,
                'key_metrics': {
                    'mae': basic_metrics.get('mae', 0),
                    'rmse': basic_metrics.get('rmse', 0),
                    'accuracy_within_3f': basic_metrics.get('accuracy_within_3f', 0),
                    'avg_confidence': confidence_metrics.get('avg_confidence', 0)
                }
            },
            'basic_metrics': basic_metrics,
            'confidence_metrics': confidence_metrics,
            'seasonal_metrics': seasonal_metrics,
            'trend_metrics': trend_metrics,
            'temperature_analysis': temperature_analysis,
            'calculation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Comprehensive metrics calculated: Grade={performance_grade}, "
                   f"MAE={mae:.2f}°F, Accuracy={accuracy_3f:.1f}%")
        
        return comprehensive_metrics
    
    def compare_models(self, model_predictions: Dict[str, List[PredictionResult]]) -> Dict[str, Any]:
        """Compare performance metrics across multiple models.
        
        Args:
            model_predictions: Dictionary mapping model names to their predictions
            
        Returns:
            Dictionary with model comparison results
        """
        if not model_predictions:
            return {'error': 'No model predictions provided'}
        
        model_metrics = {}
        
        # Calculate metrics for each model
        for model_name, predictions in model_predictions.items():
            if predictions:
                model_metrics[model_name] = self.calculate_basic_metrics(predictions)
            else:
                model_metrics[model_name] = {'error': 'No predictions available'}
        
        # Find best performing models
        valid_models = {
            name: metrics for name, metrics in model_metrics.items()
            if isinstance(metrics, dict) and 'mae' in metrics
        }
        
        if not valid_models:
            return {'error': 'No valid model metrics available'}
        
        # Rankings
        best_mae = min(valid_models.keys(), key=lambda m: valid_models[m]['mae'])
        best_accuracy = max(valid_models.keys(), key=lambda m: valid_models[m]['accuracy_within_3f'])
        best_rmse = min(valid_models.keys(), key=lambda m: valid_models[m]['rmse'])
        
        # Statistical significance testing (simplified)
        model_names = list(valid_models.keys())
        pairwise_comparisons = []
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                pred1 = model_predictions[model1]
                pred2 = model_predictions[model2]
                
                # Find common dates for fair comparison
                dates1 = {pred.date for pred in pred1}
                dates2 = {pred.date for pred in pred2}
                common_dates = dates1.intersection(dates2)
                
                if len(common_dates) >= 10:  # Minimum for comparison
                    errors1 = [pred.error for pred in pred1 if pred.date in common_dates]
                    errors2 = [pred.error for pred in pred2 if pred.date in common_dates]
                    
                    # Simple t-test approximation
                    mean_diff = np.mean(errors1) - np.mean(errors2)
                    
                    pairwise_comparisons.append({
                        'model1': model1,
                        'model2': model2,
                        'common_predictions': len(common_dates),
                        'mae_difference': mean_diff,
                        'better_model': model2 if mean_diff > 0 else model1
                    })
        
        comparison_results = {
            'model_metrics': model_metrics,
            'rankings': {
                'best_mae': best_mae,
                'best_accuracy': best_accuracy,
                'best_rmse': best_rmse
            },
            'pairwise_comparisons': pairwise_comparisons,
            'summary': {
                'total_models': len(model_predictions),
                'valid_models': len(valid_models),
                'best_overall_mae': valid_models[best_mae]['mae'],
                'best_overall_accuracy': valid_models[best_accuracy]['accuracy_within_3f']
            }
        }
        
        logger.info(f"Model comparison completed: {len(valid_models)} models, "
                   f"best MAE: {best_mae} ({valid_models[best_mae]['mae']:.2f}°F)")
        
        return comparison_results