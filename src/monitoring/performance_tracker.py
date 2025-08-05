"""Daily performance tracking system for weather prediction models."""

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from ..utils.data_manager import DataManager
from ..utils.config import config


class PerformanceTracker:
    """Tracks prediction accuracy vs actual temperatures and manages source weights."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize performance tracker.
        
        Args:
            data_manager: DataManager instance (creates new one if None)
        """
        self.data_manager = data_manager or DataManager()
        self.performance_file = self.data_manager.data_files['model_performance']
        
        # Performance tracking windows (in days)
        self.tracking_windows = {
            'short_term': 7,
            'medium_term': 14,
            'long_term': 30,
            'seasonal': 90
        }
        
        # Accuracy thresholds for alerts
        self.accuracy_thresholds = {
            'excellent': 0.90,
            'good': 0.80,
            'acceptable': 0.70,
            'poor': 0.60
        }
        
        # Source weight adjustment parameters
        self.weight_adjustment = {
            'learning_rate': 0.1,
            'min_weight': 0.05,
            'max_weight': 0.50,
            'decay_factor': 0.95
        }
        
        logger.info("PerformanceTracker initialized")
    
    def _calculate_forecast_horizon(self, merged_data: pd.DataFrame) -> Optional[float]:
        """Calculate average forecast horizon in days.
        
        Args:
            merged_data: DataFrame with date and forecast_date columns
            
        Returns:
            Average forecast horizon in days or None if calculation fails
        """
        try:
            if 'forecast_date' not in merged_data.columns or merged_data.empty:
                return None
            
            # Ensure both columns are date objects
            dates = merged_data['date'].copy()
            forecast_dates = merged_data['forecast_date'].copy()
            
            # Convert to date objects if needed
            if not pd.api.types.is_object_dtype(dates):
                dates = pd.to_datetime(dates).dt.date
            if not pd.api.types.is_object_dtype(forecast_dates):
                forecast_dates = pd.to_datetime(forecast_dates).dt.date
            
            # Calculate differences
            horizons = []
            for date_val, forecast_date_val in zip(dates, forecast_dates):
                if pd.notna(date_val) and pd.notna(forecast_date_val):
                    if isinstance(date_val, str):
                        date_val = pd.to_datetime(date_val).date()
                    if isinstance(forecast_date_val, str):
                        forecast_date_val = pd.to_datetime(forecast_date_val).date()
                    
                    horizon = (date_val - forecast_date_val).days
                    horizons.append(horizon)
            
            return np.mean(horizons) if horizons else None
            
        except Exception as e:
            logger.warning(f"Error calculating forecast horizon: {e}")
            return None
    
    def calculate_accuracy_metrics(self, predictions: pd.Series, actuals: pd.Series) -> Dict[str, float]:
        """Calculate various accuracy metrics for temperature predictions.
        
        Args:
            predictions: Series of predicted temperatures
            actuals: Series of actual temperatures
            
        Returns:
            Dictionary of accuracy metrics
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return {}
        
        # Ensure same length and remove NaN values
        valid_mask = ~(pd.isna(predictions) | pd.isna(actuals))
        pred_clean = predictions[valid_mask]
        actual_clean = actuals[valid_mask]
        
        if len(pred_clean) == 0:
            return {}
        
        # Calculate error metrics
        errors = pred_clean - actual_clean
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        metrics = {
            'count': len(pred_clean),
            'mae': abs_errors.mean(),  # Mean Absolute Error
            'rmse': np.sqrt(squared_errors.mean()),  # Root Mean Square Error
            'mape': (abs_errors / np.abs(actual_clean)).mean() * 100,  # Mean Absolute Percentage Error
            'bias': errors.mean(),  # Average bias (positive = overestimate)
            'std_error': errors.std(),  # Standard deviation of errors
            'accuracy_1f': (abs_errors <= 1.0).mean(),  # Accuracy within ±1°F
            'accuracy_2f': (abs_errors <= 2.0).mean(),  # Accuracy within ±2°F
            'accuracy_3f': (abs_errors <= 3.0).mean(),  # Accuracy within ±3°F
            'accuracy_5f': (abs_errors <= 5.0).mean(),  # Accuracy within ±5°F
            'max_error': abs_errors.max(),
            'min_error': abs_errors.min(),
            'median_error': abs_errors.median()
        }
        
        return metrics
    
    def track_source_performance(self, source: str, window_days: int = 30) -> Dict[str, Any]:
        """Track performance for a specific data source.
        
        Args:
            source: Data source name
            window_days: Number of days to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Load source data and actual temperatures
            end_date = date.today()
            start_date = end_date - timedelta(days=window_days)
            
            source_data = self.data_manager.load_source_data(source)
            actual_temps = self.data_manager.load_source_data('actual_temperatures')
            
            # Filter by date range after loading
            if not source_data.empty and 'date' in source_data.columns:
                if not pd.api.types.is_object_dtype(source_data['date']):
                    source_data['date'] = pd.to_datetime(source_data['date']).dt.date
                source_data = source_data[
                    (source_data['date'] >= start_date) & 
                    (source_data['date'] <= end_date)
                ]
            
            if not actual_temps.empty and 'date' in actual_temps.columns:
                if not pd.api.types.is_object_dtype(actual_temps['date']):
                    actual_temps['date'] = pd.to_datetime(actual_temps['date']).dt.date
                actual_temps = actual_temps[
                    (actual_temps['date'] >= start_date) & 
                    (actual_temps['date'] <= end_date)
                ]
            
            if source_data.empty or actual_temps.empty:
                logger.warning(f"No data available for {source} performance tracking")
                return {'error': 'No data available'}
            
            # Merge source predictions with actual temperatures
            # Ensure date columns are in the correct format for merging
            if 'date' in source_data.columns:
                source_data = source_data.copy()
                if not pd.api.types.is_object_dtype(source_data['date']):
                    source_data['date'] = pd.to_datetime(source_data['date']).dt.date
            if 'date' in actual_temps.columns:
                actual_temps = actual_temps.copy()
                if not pd.api.types.is_object_dtype(actual_temps['date']):
                    actual_temps['date'] = pd.to_datetime(actual_temps['date']).dt.date
            
            merged = pd.merge(source_data, actual_temps, on='date', how='inner')
            
            if merged.empty:
                return {'error': 'No matching dates between predictions and actuals'}
            
            # Calculate accuracy metrics
            predictions = merged['predicted_high']
            actuals = merged['actual_high']
            metrics = self.calculate_accuracy_metrics(predictions, actuals)
            
            if not metrics:
                return {'error': 'Could not calculate metrics'}
            
            # Add source-specific information
            metrics.update({
                'source': source,
                'window_days': window_days,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'data_quality_avg': merged['data_quality_score'].mean() if 'data_quality_score' in merged.columns else None,
                'forecast_horizon_avg': self._calculate_forecast_horizon(merged) if 'forecast_date' in merged.columns else None
            })
            
            # Determine performance category
            accuracy_3f = metrics.get('accuracy_3f', 0)
            if accuracy_3f >= self.accuracy_thresholds['excellent']:
                metrics['performance_category'] = 'excellent'
            elif accuracy_3f >= self.accuracy_thresholds['good']:
                metrics['performance_category'] = 'good'
            elif accuracy_3f >= self.accuracy_thresholds['acceptable']:
                metrics['performance_category'] = 'acceptable'
            else:
                metrics['performance_category'] = 'poor'
            
            logger.info(f"Performance tracking for {source}: {metrics['performance_category']} "
                       f"(±3°F accuracy: {accuracy_3f:.3f}, MAE: {metrics['mae']:.2f}°F)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking performance for {source}: {e}")
            return {'error': str(e)}
    
    def track_all_sources_performance(self, window_days: int = 30) -> Dict[str, Dict[str, Any]]:
        """Track performance for all weather data sources.
        
        Args:
            window_days: Number of days to analyze
            
        Returns:
            Dictionary mapping source names to performance metrics
        """
        weather_sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        performance_data = {}
        
        for source in weather_sources:
            performance_data[source] = self.track_source_performance(source, window_days)
        
        # Calculate overall ensemble performance if predictions exist
        try:
            predictions_data = self.data_manager.load_source_data('predictions')
            if not predictions_data.empty:
                end_date = date.today()
                start_date = end_date - timedelta(days=window_days)
                
                predictions_data['date'] = pd.to_datetime(predictions_data['date']).dt.date
                recent_predictions = predictions_data[
                    (predictions_data['date'] >= start_date) & 
                    (predictions_data['date'] <= end_date)
                ]
                
                if not recent_predictions.empty and 'actual_temperature' in recent_predictions.columns:
                    valid_predictions = recent_predictions.dropna(subset=['actual_temperature'])
                    if not valid_predictions.empty:
                        ensemble_metrics = self.calculate_accuracy_metrics(
                            valid_predictions['predicted_high'],
                            valid_predictions['actual_temperature']
                        )
                        ensemble_metrics.update({
                            'source': 'ensemble',
                            'window_days': window_days,
                            'start_date': start_date.isoformat(),
                            'end_date': end_date.isoformat()
                        })
                        performance_data['ensemble'] = ensemble_metrics
        
        except Exception as e:
            logger.error(f"Error calculating ensemble performance: {e}")
        
        return performance_data
    
    def store_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store performance metrics to the performance tracking file.
        
        Args:
            metrics: Dictionary of performance metrics to store
        """
        try:
            # Convert metrics to DataFrame row
            metrics_row = pd.DataFrame([{
                'timestamp': datetime.now(),
                'source': metrics.get('source', 'unknown'),
                'window_days': metrics.get('window_days', 30),
                'count': metrics.get('count', 0),
                'mae': metrics.get('mae', None),
                'rmse': metrics.get('rmse', None),
                'mape': metrics.get('mape', None),
                'bias': metrics.get('bias', None),
                'std_error': metrics.get('std_error', None),
                'accuracy_1f': metrics.get('accuracy_1f', None),
                'accuracy_2f': metrics.get('accuracy_2f', None),
                'accuracy_3f': metrics.get('accuracy_3f', None),
                'accuracy_5f': metrics.get('accuracy_5f', None),
                'max_error': metrics.get('max_error', None),
                'min_error': metrics.get('min_error', None),
                'median_error': metrics.get('median_error', None),
                'performance_category': metrics.get('performance_category', 'unknown'),
                'data_quality_avg': metrics.get('data_quality_avg', None),
                'forecast_horizon_avg': metrics.get('forecast_horizon_avg', None),
                'error_message': metrics.get('error_message', None)
            }])
            
            # Append to performance file
            self.data_manager.save_source_data('model_performance', metrics_row, append=True)
            logger.info(f"Stored performance metrics for {metrics.get('source', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    def get_performance_history(self, source: str, days: int = 90) -> pd.DataFrame:
        """Get historical performance data for a source.
        
        Args:
            source: Data source name
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical performance metrics
        """
        try:
            if not self.performance_file.exists():
                logger.warning("No performance history file found")
                return pd.DataFrame()
            
            performance_df = pd.read_parquet(self.performance_file)
            
            # Filter by source and date range
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_df = performance_df[
                (performance_df['source'] == source) &
                (performance_df['timestamp'] >= cutoff_date)
            ].copy()
            
            # Sort by timestamp
            filtered_df = filtered_df.sort_values('timestamp')
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error getting performance history for {source}: {e}")
            return pd.DataFrame()
    
    def calculate_source_weights(self, performance_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate dynamic weights for data sources based on recent performance.
        
        Args:
            performance_data: Dictionary of performance metrics for each source
            
        Returns:
            Dictionary mapping source names to weights (sum to 1.0)
        """
        try:
            # Extract accuracy scores (using ±3°F accuracy as primary metric)
            source_scores = {}
            for source, metrics in performance_data.items():
                if source == 'ensemble' or 'error' in metrics:
                    continue
                
                accuracy_3f = metrics.get('accuracy_3f', 0)
                data_quality = metrics.get('data_quality_avg', 0.5)
                count = metrics.get('count', 0)
                
                # Combine accuracy with data quality and sample size
                # Give more weight to sources with more data points (up to a limit)
                count_factor = min(count / 20.0, 1.0)  # Cap at 20 data points
                combined_score = accuracy_3f * 0.7 + data_quality * 0.2 + count_factor * 0.1
                
                source_scores[source] = max(combined_score, 0.01)  # Minimum score to avoid zero weights
            
            if not source_scores:
                logger.warning("No valid source scores for weight calculation")
                return {}
            
            # Convert scores to weights using softmax-like approach
            # Apply temperature parameter to control weight distribution
            temperature = 2.0  # Higher = more uniform, lower = more concentrated
            exp_scores = {source: np.exp(score / temperature) for source, score in source_scores.items()}
            total_exp = sum(exp_scores.values())
            
            weights = {source: exp_score / total_exp for source, exp_score in exp_scores.items()}
            
            # Apply min/max weight constraints
            for source in weights:
                weights[source] = max(self.weight_adjustment['min_weight'], 
                                    min(self.weight_adjustment['max_weight'], weights[source]))
            
            # Renormalize after applying constraints
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {source: weight / total_weight for source, weight in weights.items()}
            
            logger.info(f"Calculated source weights: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating source weights: {e}")
            return {}
    
    def update_source_weights(self, window_days: int = 14) -> Dict[str, float]:
        """Update source weights based on recent performance.
        
        Args:
            window_days: Number of days to use for performance calculation
            
        Returns:
            Dictionary of updated source weights
        """
        try:
            # Get recent performance for all sources
            performance_data = self.track_all_sources_performance(window_days)
            
            # Calculate new weights
            new_weights = self.calculate_source_weights(performance_data)
            
            if new_weights:
                # Store the weight update
                weight_update = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'source': 'weight_update',
                    'window_days': window_days,
                    'weights': new_weights,
                    'performance_data': performance_data
                }])
                
                self.data_manager.save_source_data('model_performance', weight_update, append=True)
                logger.info(f"Updated source weights based on {window_days}-day performance")
            
            return new_weights
            
        except Exception as e:
            logger.error(f"Error updating source weights: {e}")
            return {}
    
    def get_current_source_weights(self) -> Dict[str, float]:
        """Get the most recent source weights.
        
        Returns:
            Dictionary of current source weights
        """
        try:
            if not self.performance_file.exists():
                # Return equal weights if no history
                sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
                equal_weight = 1.0 / len(sources)
                return {source: equal_weight for source in sources}
            
            performance_df = pd.read_parquet(self.performance_file)
            
            # Find most recent weight update
            weight_updates = performance_df[performance_df['source'] == 'weight_update']
            if weight_updates.empty:
                # Return equal weights if no updates found
                sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
                equal_weight = 1.0 / len(sources)
                return {source: equal_weight for source in sources}
            
            latest_update = weight_updates.sort_values('timestamp').iloc[-1]
            return latest_update['weights']
            
        except Exception as e:
            logger.error(f"Error getting current source weights: {e}")
            # Return equal weights as fallback
            sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
            equal_weight = 1.0 / len(sources)
            return {source: equal_weight for source in sources}
    
    def detect_performance_degradation(self, source: str, threshold_drop: float = 0.1) -> bool:
        """Detect if a source's performance has degraded significantly.
        
        Args:
            source: Data source name
            threshold_drop: Minimum drop in accuracy to trigger alert
            
        Returns:
            True if performance degradation detected
        """
        try:
            # Get recent performance history
            history = self.get_performance_history(source, days=60)
            if len(history) < 2:
                return False
            
            # Compare recent performance to historical average
            recent_performance = history.tail(7)['accuracy_3f'].mean()
            historical_performance = history.head(-7)['accuracy_3f'].mean() if len(history) > 7 else recent_performance
            
            performance_drop = historical_performance - recent_performance
            
            if performance_drop > threshold_drop:
                logger.warning(f"Performance degradation detected for {source}: "
                             f"dropped {performance_drop:.3f} (recent: {recent_performance:.3f}, "
                             f"historical: {historical_performance:.3f})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting performance degradation for {source}: {e}")
            return False
    
    def get_performance_summary(self, start_date: date, end_date: date) -> Optional[Dict[str, Any]]:
        """Get a comprehensive performance summary for the dashboard.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with performance summary data or None if no data available
        """
        try:
            # Get performance for all sources
            window_days = (end_date - start_date).days
            all_performance = self.track_all_sources_performance(window_days)
            
            if not all_performance:
                return None
            
            # Build summary structure
            summary = {
                'daily_accuracy': {},
                'source_performance': {},
                'error_distribution': [],
                'confidence_calibration': {}
            }
            
            # Extract source performance
            for source, metrics in all_performance.items():
                if 'error' not in metrics:
                    summary['source_performance'][source] = {
                        'accuracy_7d': metrics.get('accuracy_3f', 0),
                        'mae': metrics.get('mae', 0),
                        'rmse': metrics.get('rmse', 0)
                    }
            
            # Get daily accuracy from predictions if available
            try:
                predictions_df = self.data_manager.load_predictions(start_date, end_date)
                if not predictions_df.empty and 'actual_temperature' in predictions_df.columns:
                    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
                    valid_predictions = predictions_df.dropna(subset=['actual_temperature'])
                    
                    # Calculate daily accuracy
                    for _, row in valid_predictions.iterrows():
                        error = abs(row['predicted_high'] - row['actual_temperature'])
                        accuracy = 1.0 if error <= 3.0 else 0.0
                        summary['daily_accuracy'][row['date']] = accuracy
                        summary['error_distribution'].append(row['predicted_high'] - row['actual_temperature'])
                    
                    # Calculate confidence calibration
                    if 'confidence' in valid_predictions.columns:
                        confidence_bins = [70, 80, 90]
                        for bin_threshold in confidence_bins:
                            high_conf_preds = valid_predictions[valid_predictions['confidence'] >= bin_threshold]
                            if not high_conf_preds.empty:
                                errors = abs(high_conf_preds['predicted_high'] - high_conf_preds['actual_temperature'])
                                accuracy = (errors <= 3.0).mean() * 100
                                summary['confidence_calibration'][bin_threshold] = accuracy
            
            except Exception as e:
                logger.warning(f"Could not calculate daily accuracy: {e}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return None
    
    def generate_performance_report(self, window_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report for all sources.
        
        Args:
            window_days: Number of days to analyze
            
        Returns:
            Dictionary with comprehensive performance report
        """
        try:
            # Get performance data for all sources
            performance_data = self.track_all_sources_performance(window_days)
            
            # Get current weights
            current_weights = self.get_current_source_weights()
            
            # Detect any performance issues
            degradation_alerts = {}
            for source in ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']:
                degradation_alerts[source] = self.detect_performance_degradation(source)
            
            # Calculate summary statistics
            valid_sources = [source for source, metrics in performance_data.items() 
                           if source != 'ensemble' and 'error' not in metrics]
            
            if valid_sources:
                accuracy_scores = [performance_data[source].get('accuracy_3f', 0) for source in valid_sources]
                mae_scores = [performance_data[source].get('mae', float('inf')) for source in valid_sources]
                
                summary = {
                    'best_source': valid_sources[np.argmax(accuracy_scores)],
                    'worst_source': valid_sources[np.argmin(accuracy_scores)],
                    'avg_accuracy_3f': np.mean(accuracy_scores),
                    'avg_mae': np.mean([mae for mae in mae_scores if mae != float('inf')]),
                    'sources_with_issues': sum(degradation_alerts.values())
                }
            else:
                summary = {'error': 'No valid source data available'}
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'window_days': window_days,
                'performance_data': performance_data,
                'current_weights': current_weights,
                'degradation_alerts': degradation_alerts,
                'summary': summary
            }
            
            logger.info(f"Generated performance report: {summary}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def run_daily_performance_tracking(self) -> Dict[str, Any]:
        """Run daily performance tracking and weight updates.
        
        Returns:
            Dictionary with tracking results
        """
        try:
            logger.info("Starting daily performance tracking")
            
            # Track performance for all time windows
            results = {}
            for window_name, window_days in self.tracking_windows.items():
                logger.info(f"Tracking {window_name} performance ({window_days} days)")
                performance_data = self.track_all_sources_performance(window_days)
                
                # Store metrics for each source
                for source, metrics in performance_data.items():
                    if 'error' not in metrics:
                        self.store_performance_metrics(metrics)
                    else:
                        # Store error information for tracking
                        error_metrics = {
                            'source': source,
                            'window_days': window_days,
                            'count': 0,
                            'error_message': metrics.get('error', 'Unknown error')
                        }
                        self.store_performance_metrics(error_metrics)
                
                results[window_name] = performance_data
            
            # Update source weights based on medium-term performance
            new_weights = self.update_source_weights(self.tracking_windows['medium_term'])
            results['updated_weights'] = new_weights
            
            # Generate comprehensive report
            report = self.generate_performance_report(self.tracking_windows['long_term'])
            results['performance_report'] = report
            
            logger.info("Daily performance tracking completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in daily performance tracking: {e}")
            return {'error': str(e)}