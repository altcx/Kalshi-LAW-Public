"""Automated model retraining system for weather prediction models."""

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import pickle
import json

from ..utils.data_manager import DataManager
from ..utils.config import config
from .performance_tracker import PerformanceTracker


class ModelRetrainer:
    """Handles automated model retraining and performance degradation detection."""
    
    def __init__(self, data_manager: Optional[DataManager] = None, 
                 performance_tracker: Optional[PerformanceTracker] = None):
        """Initialize model retrainer.
        
        Args:
            data_manager: DataManager instance (creates new one if None)
            performance_tracker: PerformanceTracker instance (creates new one if None)
        """
        self.data_manager = data_manager or DataManager()
        self.performance_tracker = performance_tracker or PerformanceTracker(self.data_manager)
        
        # Model storage directory
        self.models_dir = Path('models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Retraining configuration
        self.retraining_config = {
            'min_data_points': 30,          # Minimum data points needed for retraining
            'performance_threshold': 0.70,  # Minimum ±3°F accuracy to avoid retraining
            'degradation_threshold': 0.10,  # Drop in accuracy that triggers retraining
            'retraining_frequency_days': 7, # How often to check for retraining
            'max_training_data_days': 365,  # Maximum days of data to use for training
            'validation_split': 0.2,        # Fraction of data to use for validation
            'hyperparameter_trials': 50,    # Number of hyperparameter optimization trials
            'early_stopping_patience': 10   # Early stopping patience for training
        }
        
        # Model performance history for degradation detection
        self.performance_history_file = self.models_dir / 'performance_history.json'
        
        # Available model types for retraining
        self.available_models = {
            'xgboost': self._get_xgboost_model,
            'lightgbm': self._get_lightgbm_model,
            'linear_regression': self._get_linear_regression_model,
            'prophet': self._get_prophet_model
        }
        
        logger.info("ModelRetrainer initialized")
    
    def _get_xgboost_model(self):
        """Get XGBoost model class."""
        try:
            from ..models.xgboost_model import XGBoostModel
            return XGBoostModel
        except ImportError:
            logger.error("XGBoost model not available")
            return None
    
    def _get_lightgbm_model(self):
        """Get LightGBM model class."""
        try:
            from ..models.lightgbm_model import LightGBMModel
            return LightGBMModel
        except ImportError:
            logger.error("LightGBM model not available")
            return None
    
    def _get_linear_regression_model(self):
        """Get Linear Regression model class."""
        try:
            from ..models.linear_regression_model import LinearRegressionModel
            return LinearRegressionModel
        except ImportError:
            logger.error("Linear Regression model not available")
            return None
    
    def _get_prophet_model(self):
        """Get Prophet model class."""
        try:
            from ..models.prophet_model import ProphetModel
            return ProphetModel
        except ImportError:
            logger.error("Prophet model not available")
            return None
    
    def check_retraining_needed(self, model_name: str = 'ensemble') -> Dict[str, Any]:
        """Check if model retraining is needed based on performance degradation.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Dictionary with retraining recommendation and reasons
        """
        try:
            logger.info(f"Checking if retraining is needed for {model_name}")
            
            # Get recent performance data
            recent_performance = self.performance_tracker.track_all_sources_performance(
                window_days=self.retraining_config['retraining_frequency_days']
            )
            
            # Get historical performance for comparison
            historical_performance = self.performance_tracker.track_all_sources_performance(
                window_days=30
            )
            
            reasons = []
            should_retrain = False
            
            # Check for performance degradation
            for source, recent_metrics in recent_performance.items():
                if source == 'ensemble' or 'error' in recent_metrics:
                    continue
                
                historical_metrics = historical_performance.get(source, {})
                if 'error' in historical_metrics:
                    continue
                
                recent_accuracy = recent_metrics.get('accuracy_3f', 0)
                historical_accuracy = historical_metrics.get('accuracy_3f', 0)
                
                # Check if performance dropped significantly
                performance_drop = historical_accuracy - recent_accuracy
                if performance_drop > self.retraining_config['degradation_threshold']:
                    reasons.append(f"{source}: accuracy dropped by {performance_drop:.3f}")
                    should_retrain = True
                
                # Check if performance is below threshold
                if recent_accuracy < self.retraining_config['performance_threshold']:
                    reasons.append(f"{source}: accuracy {recent_accuracy:.3f} below threshold {self.retraining_config['performance_threshold']}")
                    should_retrain = True
            
            # Check if we have enough new data for retraining
            data_summary = self.data_manager.get_data_summary()
            total_data_points = 0
            for source, info in data_summary.items():
                if source not in ['model_performance', 'predictions'] and 'records' in info:
                    total_data_points += info['records']
            
            if total_data_points < self.retraining_config['min_data_points']:
                reasons.append(f"Insufficient data: {total_data_points} < {self.retraining_config['min_data_points']}")
                should_retrain = False
            
            # Check time since last retraining
            last_retraining = self._get_last_retraining_date(model_name)
            if last_retraining:
                days_since_retraining = (date.today() - last_retraining).days
                if days_since_retraining >= self.retraining_config['retraining_frequency_days']:
                    reasons.append(f"Scheduled retraining: {days_since_retraining} days since last retraining")
                    should_retrain = True
            else:
                reasons.append("No previous retraining record found")
                should_retrain = True
            
            result = {
                'model_name': model_name,
                'should_retrain': should_retrain,
                'reasons': reasons,
                'recent_performance': recent_performance,
                'historical_performance': historical_performance,
                'total_data_points': total_data_points,
                'last_retraining': last_retraining.isoformat() if last_retraining else None,
                'check_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Retraining check for {model_name}: {'NEEDED' if should_retrain else 'NOT NEEDED'}")
            if reasons:
                logger.info(f"Reasons: {'; '.join(reasons)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking retraining need: {e}")
            return {
                'model_name': model_name,
                'should_retrain': False,
                'error': str(e),
                'check_timestamp': datetime.now().isoformat()
            }
    
    def _get_last_retraining_date(self, model_name: str) -> Optional[date]:
        """Get the date of the last retraining for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Date of last retraining or None if not found
        """
        try:
            if not self.performance_history_file.exists():
                return None
            
            with open(self.performance_history_file, 'r') as f:
                history = json.load(f)
            
            model_history = history.get(model_name, {})
            last_retraining_str = model_history.get('last_retraining')
            
            if last_retraining_str:
                return datetime.fromisoformat(last_retraining_str).date()
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting last retraining date: {e}")
            return None
    
    def _update_retraining_history(self, model_name: str, retraining_info: Dict[str, Any]) -> None:
        """Update the retraining history for a model.
        
        Args:
            model_name: Name of the model
            retraining_info: Information about the retraining
        """
        try:
            # Load existing history
            history = {}
            if self.performance_history_file.exists():
                with open(self.performance_history_file, 'r') as f:
                    history = json.load(f)
            
            # Update history for this model
            if model_name not in history:
                history[model_name] = {}
            
            history[model_name].update({
                'last_retraining': datetime.now().isoformat(),
                'retraining_info': retraining_info
            })
            
            # Save updated history
            with open(self.performance_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Updated retraining history for {model_name}")
            
        except Exception as e:
            logger.error(f"Error updating retraining history: {e}")
    
    def prepare_training_data(self, max_days: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from all available sources.
        
        Args:
            max_days: Maximum number of days of data to use (None for all)
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        try:
            max_days = max_days or self.retraining_config['max_training_data_days']
            
            # Load data from all sources
            end_date = date.today()
            start_date = end_date - timedelta(days=max_days)
            
            all_source_data = self.data_manager.load_all_sources(start_date, end_date)
            actual_temps = self.data_manager.load_source_data('actual_temperatures', start_date, end_date)
            
            if actual_temps.empty:
                raise ValueError("No actual temperature data available for training")
            
            # Prepare features using feature engineering pipeline
            try:
                from ..feature_engineering.feature_pipeline import FeaturePipeline
                feature_pipeline = FeaturePipeline()
                
                # Create features for each date
                features_list = []
                targets_list = []
                
                # Convert actual temps date column
                actual_temps['date'] = pd.to_datetime(actual_temps['date']).dt.date
                
                for _, actual_row in actual_temps.iterrows():
                    target_date = actual_row['date']
                    actual_temp = actual_row['actual_high']
                    
                    # Get source data for this date
                    date_source_data = {}
                    for source, source_df in all_source_data.items():
                        if not source_df.empty:
                            source_df['date'] = pd.to_datetime(source_df['date']).dt.date
                            date_data = source_df[source_df['date'] == target_date]
                            if not date_data.empty:
                                date_source_data[source] = date_data.iloc[0].to_dict()
                    
                    if date_source_data:
                        # Create features for this date
                        features = feature_pipeline.create_features(date_source_data, target_date)
                        if features is not None and not features.empty:
                            features_list.append(features)
                            targets_list.append(actual_temp)
                
                if not features_list:
                    raise ValueError("No valid features could be created from the data")
                
                # Combine all features
                features_df = pd.concat(features_list, ignore_index=True)
                targets_series = pd.Series(targets_list)
                
                logger.info(f"Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features")
                return features_df, targets_series
                
            except ImportError:
                logger.warning("Feature pipeline not available, using basic features")
                # Fallback to basic feature creation
                return self._create_basic_features(all_source_data, actual_temps)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def _create_basic_features(self, all_source_data: Dict[str, pd.DataFrame], 
                              actual_temps: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create basic features when feature pipeline is not available.
        
        Args:
            all_source_data: Dictionary of source data
            actual_temps: Actual temperature data
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        features_list = []
        targets_list = []
        
        # Convert actual temps date column
        actual_temps['date'] = pd.to_datetime(actual_temps['date']).dt.date
        
        for _, actual_row in actual_temps.iterrows():
            target_date = actual_row['date']
            actual_temp = actual_row['actual_high']
            
            # Create basic features from source predictions
            feature_row = {}
            
            for source, source_df in all_source_data.items():
                if not source_df.empty:
                    source_df['date'] = pd.to_datetime(source_df['date']).dt.date
                    date_data = source_df[source_df['date'] == target_date]
                    
                    if not date_data.empty:
                        row = date_data.iloc[0]
                        feature_row[f'{source}_predicted_high'] = row.get('predicted_high', np.nan)
                        feature_row[f'{source}_humidity'] = row.get('humidity', np.nan)
                        feature_row[f'{source}_pressure'] = row.get('pressure', np.nan)
                        feature_row[f'{source}_wind_speed'] = row.get('wind_speed', np.nan)
                        feature_row[f'{source}_quality'] = row.get('data_quality_score', np.nan)
            
            # Add date-based features
            feature_row['day_of_year'] = target_date.timetuple().tm_yday
            feature_row['month'] = target_date.month
            feature_row['day_of_week'] = target_date.weekday()
            
            if feature_row:
                features_list.append(feature_row)
                targets_list.append(actual_temp)
        
        if not features_list:
            raise ValueError("No features could be created")
        
        features_df = pd.DataFrame(features_list)
        targets_series = pd.Series(targets_list)
        
        # Fill NaN values with median
        features_df = features_df.fillna(features_df.median())
        
        return features_df, targets_series
    
    def retrain_model(self, model_type: str = 'xgboost', 
                     hyperparameter_optimization: bool = True) -> Dict[str, Any]:
        """Retrain a specific model with new data.
        
        Args:
            model_type: Type of model to retrain ('xgboost', 'lightgbm', etc.)
            hyperparameter_optimization: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with retraining results
        """
        try:
            logger.info(f"Starting retraining for {model_type} model")
            
            # Check if model type is available
            if model_type not in self.available_models:
                raise ValueError(f"Model type {model_type} not available")
            
            model_class = self.available_models[model_type]()
            if model_class is None:
                raise ValueError(f"Could not load {model_type} model class")
            
            # Prepare training data
            features, targets = self.prepare_training_data()
            
            if len(features) < self.retraining_config['min_data_points']:
                raise ValueError(f"Insufficient training data: {len(features)} < {self.retraining_config['min_data_points']}")
            
            # Split data for validation
            split_idx = int(len(features) * (1 - self.retraining_config['validation_split']))
            
            train_features = features.iloc[:split_idx]
            train_targets = targets.iloc[:split_idx]
            val_features = features.iloc[split_idx:]
            val_targets = targets.iloc[split_idx:]
            
            # Initialize model
            model = model_class
            
            # Perform hyperparameter optimization if requested
            best_params = None
            if hyperparameter_optimization:
                logger.info("Performing hyperparameter optimization")
                best_params = self._optimize_hyperparameters(
                    model, train_features, train_targets, val_features, val_targets
                )
                if best_params:
                    # Update model with best parameters
                    if hasattr(model, 'set_params'):
                        model.set_params(**best_params)
            
            # Train the model
            logger.info("Training model with full dataset")
            model.fit(train_features, train_targets)
            
            # Evaluate model performance
            train_predictions = model.predict(train_features)
            val_predictions = model.predict(val_features)
            
            # Calculate metrics
            train_metrics = self.performance_tracker.calculate_accuracy_metrics(
                pd.Series(train_predictions), train_targets
            )
            val_metrics = self.performance_tracker.calculate_accuracy_metrics(
                pd.Series(val_predictions), val_targets
            )
            
            # Save the retrained model
            model_filename = f"{model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = self.models_dir / model_filename
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create retraining results
            retraining_results = {
                'model_type': model_type,
                'model_path': str(model_path),
                'training_samples': len(train_features),
                'validation_samples': len(val_features),
                'features_count': len(features.columns),
                'best_hyperparameters': best_params,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'retraining_timestamp': datetime.now().isoformat(),
                'data_date_range': {
                    'start': features.index.min() if hasattr(features.index, 'min') else 'unknown',
                    'end': features.index.max() if hasattr(features.index, 'max') else 'unknown'
                }
            }
            
            # Update retraining history
            self._update_retraining_history(model_type, retraining_results)
            
            logger.info(f"Model retraining completed successfully for {model_type}")
            logger.info(f"Validation accuracy (±3°F): {val_metrics.get('accuracy_3f', 0):.3f}")
            logger.info(f"Validation MAE: {val_metrics.get('mae', 0):.2f}°F")
            
            return retraining_results
            
        except Exception as e:
            logger.error(f"Error retraining {model_type} model: {e}")
            return {
                'model_type': model_type,
                'error': str(e),
                'retraining_timestamp': datetime.now().isoformat()
            }
    
    def _optimize_hyperparameters(self, model, train_features: pd.DataFrame, 
                                 train_targets: pd.Series, val_features: pd.DataFrame, 
                                 val_targets: pd.Series) -> Optional[Dict[str, Any]]:
        """Optimize hyperparameters for a model.
        
        Args:
            model: Model instance to optimize
            train_features: Training features
            train_targets: Training targets
            val_features: Validation features
            val_targets: Validation targets
            
        Returns:
            Dictionary of best hyperparameters or None if optimization failed
        """
        try:
            # This is a simplified hyperparameter optimization
            # In a real implementation, you might use optuna, hyperopt, or sklearn's GridSearchCV
            
            # Define parameter grids for different model types
            param_grids = {
                'xgboost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'lightgbm': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'linear_regression': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
            
            model_type = type(model).__name__.lower()
            param_grid = None
            
            for key in param_grids:
                if key in model_type:
                    param_grid = param_grids[key]
                    break
            
            if param_grid is None:
                logger.warning(f"No parameter grid defined for {model_type}")
                return None
            
            # Simple grid search (limited to avoid long training times)
            best_score = float('inf')
            best_params = None
            
            # Limit the number of combinations to try
            max_trials = min(self.retraining_config['hyperparameter_trials'], 20)
            trial_count = 0
            
            # Generate parameter combinations
            import itertools
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            for param_combination in itertools.product(*param_values):
                if trial_count >= max_trials:
                    break
                
                params = dict(zip(param_names, param_combination))
                
                try:
                    # Create model with these parameters
                    test_model = type(model)()
                    if hasattr(test_model, 'set_params'):
                        test_model.set_params(**params)
                    
                    # Train and evaluate
                    test_model.fit(train_features, train_targets)
                    val_predictions = test_model.predict(val_features)
                    
                    # Calculate validation error (MAE)
                    val_mae = np.mean(np.abs(val_predictions - val_targets))
                    
                    if val_mae < best_score:
                        best_score = val_mae
                        best_params = params.copy()
                    
                    trial_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error in hyperparameter trial {params}: {e}")
                    continue
            
            if best_params:
                logger.info(f"Best hyperparameters found: {best_params} (MAE: {best_score:.3f})")
            else:
                logger.warning("No valid hyperparameters found")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return None
    
    def detect_model_switching_need(self) -> Dict[str, Any]:
        """Detect if we should switch to a different model type based on performance.
        
        Returns:
            Dictionary with model switching recommendation
        """
        try:
            logger.info("Checking if model switching is needed")
            
            # Get current performance for all available models
            model_performances = {}
            
            for model_type in self.available_models.keys():
                try:
                    # Check if we have a recent model of this type
                    model_files = list(self.models_dir.glob(f"{model_type}_model_*.pkl"))
                    if not model_files:
                        continue
                    
                    # Get the most recent model
                    latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
                    
                    # Load performance history for this model
                    history = {}
                    if self.performance_history_file.exists():
                        with open(self.performance_history_file, 'r') as f:
                            history = json.load(f)
                    
                    model_history = history.get(model_type, {})
                    retraining_info = model_history.get('retraining_info', {})
                    val_metrics = retraining_info.get('validation_metrics', {})
                    
                    if val_metrics:
                        model_performances[model_type] = {
                            'accuracy_3f': val_metrics.get('accuracy_3f', 0),
                            'mae': val_metrics.get('mae', float('inf')),
                            'model_file': str(latest_model_file),
                            'last_retraining': model_history.get('last_retraining')
                        }
                
                except Exception as e:
                    logger.warning(f"Error checking performance for {model_type}: {e}")
                    continue
            
            if not model_performances:
                return {
                    'should_switch': False,
                    'reason': 'No model performance data available',
                    'current_best': None,
                    'recommendation': 'Train initial models'
                }
            
            # Find the best performing model
            best_model = max(model_performances.items(), 
                           key=lambda x: x[1]['accuracy_3f'])
            
            best_model_type = best_model[0]
            best_accuracy = best_model[1]['accuracy_3f']
            
            # Check if current best model is significantly better than others
            accuracy_threshold = 0.05  # 5% improvement threshold
            should_switch = False
            current_model = None
            
            # Determine current model (most recently retrained)
            most_recent_retraining = None
            for model_type, perf in model_performances.items():
                last_retraining = perf.get('last_retraining')
                if last_retraining:
                    retraining_date = datetime.fromisoformat(last_retraining)
                    if most_recent_retraining is None or retraining_date > most_recent_retraining:
                        most_recent_retraining = retraining_date
                        current_model = model_type
            
            if current_model and current_model != best_model_type:
                current_accuracy = model_performances[current_model]['accuracy_3f']
                improvement = best_accuracy - current_accuracy
                
                if improvement > accuracy_threshold:
                    should_switch = True
            
            result = {
                'should_switch': should_switch,
                'current_model': current_model,
                'recommended_model': best_model_type,
                'performance_comparison': model_performances,
                'improvement': best_accuracy - model_performances.get(current_model, {}).get('accuracy_3f', 0) if current_model else 0,
                'check_timestamp': datetime.now().isoformat()
            }
            
            if should_switch:
                logger.info(f"Model switching recommended: {current_model} -> {best_model_type}")
                logger.info(f"Expected improvement: {result['improvement']:.3f}")
            else:
                logger.info("No model switching needed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting model switching need: {e}")
            return {
                'should_switch': False,
                'error': str(e),
                'check_timestamp': datetime.now().isoformat()
            }
    
    def run_automated_retraining(self) -> Dict[str, Any]:
        """Run the complete automated retraining process.
        
        Returns:
            Dictionary with retraining results
        """
        try:
            logger.info("Starting automated retraining process")
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'retraining_checks': {},
                'retraining_results': {},
                'model_switching': {},
                'summary': {}
            }
            
            # Check which models need retraining
            models_to_check = ['ensemble'] + list(self.available_models.keys())
            
            for model_name in models_to_check:
                logger.info(f"Checking retraining need for {model_name}")
                check_result = self.check_retraining_needed(model_name)
                results['retraining_checks'][model_name] = check_result
                
                # If retraining is needed and it's not the ensemble, retrain the model
                if (check_result.get('should_retrain', False) and 
                    model_name != 'ensemble' and 
                    model_name in self.available_models):
                    
                    logger.info(f"Retraining {model_name} model")
                    retraining_result = self.retrain_model(model_name, hyperparameter_optimization=True)
                    results['retraining_results'][model_name] = retraining_result
            
            # Check if model switching is needed
            switching_result = self.detect_model_switching_need()
            results['model_switching'] = switching_result
            
            # Generate summary
            retrained_models = list(results['retraining_results'].keys())
            successful_retraining = [
                model for model, result in results['retraining_results'].items()
                if 'error' not in result
            ]
            
            results['summary'] = {
                'models_checked': len(results['retraining_checks']),
                'models_needing_retraining': len([
                    check for check in results['retraining_checks'].values()
                    if check.get('should_retrain', False)
                ]),
                'models_retrained': len(retrained_models),
                'successful_retraining': len(successful_retraining),
                'model_switching_recommended': switching_result.get('should_switch', False),
                'recommended_model': switching_result.get('recommended_model')
            }
            
            logger.info("Automated retraining process completed")
            logger.info(f"Summary: {results['summary']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in automated retraining process: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'summary': {'error': 'Automated retraining failed'}
            }