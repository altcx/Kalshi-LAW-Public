"""Walk-forward analysis for backtesting temperature prediction models."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
import concurrent.futures
from pathlib import Path

from .historical_data_loader import HistoricalDataLoader
from .performance_metrics import PerformanceMetricsCalculator, PredictionResult
from src.utils.data_manager import DataManager


@dataclass
class WalkForwardSplit:
    """Container for a walk-forward analysis split."""
    split_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_days: int
    test_days: int


@dataclass
class WalkForwardResult:
    """Container for walk-forward analysis results."""
    split: WalkForwardSplit
    predictions: List[PredictionResult]
    training_time: float
    prediction_time: float
    data_quality_issues: List[str]
    model_info: Dict[str, Any]


class WalkForwardAnalyzer:
    """Performs walk-forward analysis for backtesting."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize the walk-forward analyzer.
        
        Args:
            data_manager: DataManager instance (creates new if None)
        """
        self.data_manager = data_manager or DataManager()
        self.data_loader = HistoricalDataLoader(self.data_manager)
        self.metrics_calculator = PerformanceMetricsCalculator()
        
        # Analysis configuration
        self.default_train_window = 365  # Days
        self.default_test_window = 30    # Days
        self.default_step_size = 7       # Days
        self.min_training_data_days = 90 # Minimum training data required
        self.max_parallel_splits = 4     # Maximum parallel processing
        
        # Results storage
        self.results_cache: Dict[str, List[WalkForwardResult]] = {}
        
        logger.info("WalkForwardAnalyzer initialized")
    
    def create_splits(self, start_date: date, end_date: date,
                     train_window_days: Optional[int] = None,
                     test_window_days: Optional[int] = None,
                     step_days: Optional[int] = None) -> List[WalkForwardSplit]:
        """Create walk-forward splits for the given date range.
        
        Args:
            start_date: Overall start date for analysis
            end_date: Overall end date for analysis
            train_window_days: Training window size (default: 365)
            test_window_days: Test window size (default: 30)
            step_days: Step size between splits (default: 7)
            
        Returns:
            List of WalkForwardSplit objects
        """
        train_window = train_window_days or self.default_train_window
        test_window = test_window_days or self.default_test_window
        step_size = step_days or self.default_step_size
        
        splits = []
        current_date = start_date + timedelta(days=train_window)
        
        while current_date + timedelta(days=test_window) <= end_date:
            train_start = current_date - timedelta(days=train_window)
            train_end = current_date - timedelta(days=1)
            test_start = current_date
            test_end = current_date + timedelta(days=test_window - 1)
            
            split = WalkForwardSplit(
                split_id=len(splits) + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_days=train_window,
                test_days=test_window
            )
            
            splits.append(split)
            current_date += timedelta(days=step_size)
        
        logger.info(f"Created {len(splits)} walk-forward splits from {start_date} to {end_date}")
        return splits
    
    def validate_split_data(self, split: WalkForwardSplit) -> Tuple[bool, List[str]]:
        """Validate that sufficient data exists for a split.
        
        Args:
            split: WalkForwardSplit to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check training data availability
        train_data = self.data_loader.load_historical_data(
            split.train_start, split.train_end
        )
        
        if not train_data:
            issues.append("No training data available")
        else:
            # Check each source
            for source, data in train_data.items():
                if data.empty:
                    issues.append(f"No training data for source: {source}")
                elif len(data) < self.min_training_data_days:
                    issues.append(f"Insufficient training data for {source}: {len(data)} days")
        
        # Check test data availability
        test_data = self.data_loader.load_historical_data(
            split.test_start, split.test_end
        )
        
        if not test_data:
            issues.append("No test data available")
        
        # Check actual temperatures for validation
        actual_temps = self.data_loader.load_actual_temperatures(
            split.test_start, split.test_end
        )
        
        if actual_temps.empty:
            issues.append("No actual temperatures available for validation")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Split {split.split_id} validation failed: {issues}")
        
        return is_valid, issues
    
    def run_single_split(self, split: WalkForwardSplit, 
                        model_factory: Callable,
                        model_params: Optional[Dict] = None) -> WalkForwardResult:
        """Run analysis for a single walk-forward split.
        
        Args:
            split: WalkForwardSplit to analyze
            model_factory: Function that creates and returns a model instance
            model_params: Parameters to pass to model factory
            
        Returns:
            WalkForwardResult with analysis results
        """
        logger.info(f"Running split {split.split_id}: train {split.train_start} to {split.train_end}, "
                   f"test {split.test_start} to {split.test_end}")
        
        start_time = datetime.now()
        
        # Validate split data
        is_valid, data_issues = self.validate_split_data(split)
        if not is_valid:
            return WalkForwardResult(
                split=split,
                predictions=[],
                training_time=0.0,
                prediction_time=0.0,
                data_quality_issues=data_issues,
                model_info={'status': 'failed', 'reason': 'data_validation_failed'}
            )
        
        try:
            # Load training data
            train_data = self.data_loader.load_historical_data(
                split.train_start, split.train_end
            )
            
            # Create and train model
            model_params = model_params or {}
            model = model_factory(**model_params)
            
            training_start = datetime.now()
            
            # Prepare training features and targets
            train_features_list = []
            train_targets = []
            
            # Get actual temperatures for training period
            train_actuals = self.data_loader.load_actual_temperatures(
                split.train_start, split.train_end
            )
            
            if train_actuals.empty:
                raise ValueError("No actual temperatures available for training")
            
            train_actuals['date'] = pd.to_datetime(train_actuals['date']).dt.date
            actual_temps_dict = dict(zip(train_actuals['date'], train_actuals['actual_high']))
            
            # Create features for each training day
            current_date = split.train_start
            while current_date <= split.train_end:
                if current_date in actual_temps_dict:
                    # Simulate real-time data availability
                    sim_data = self.data_loader.simulate_real_time_prediction(current_date)
                    
                    if sim_data['available_sources']:
                        features = self.data_loader.prepare_features_for_date(
                            current_date, sim_data['data']
                        )
                        
                        if not features.empty:
                            train_features_list.append(features)
                            train_targets.append(actual_temps_dict[current_date])
                
                current_date += timedelta(days=1)
            
            if not train_features_list:
                raise ValueError("No training features could be prepared")
            
            # Combine all training features
            train_features = pd.concat(train_features_list, ignore_index=True)
            train_targets = pd.Series(train_targets)
            
            # Train the model
            if hasattr(model, 'train'):
                model.train(train_features, train_targets)
            elif hasattr(model, 'fit'):
                model.fit(train_features, train_targets)
            else:
                raise ValueError("Model does not have train() or fit() method")
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Make predictions on test period
            prediction_start = datetime.now()
            predictions = []
            
            # Load actual temperatures for test period
            test_actuals = self.data_loader.load_actual_temperatures(
                split.test_start, split.test_end
            )
            
            if test_actuals.empty:
                raise ValueError("No actual temperatures available for test period")
            
            test_actuals['date'] = pd.to_datetime(test_actuals['date']).dt.date
            test_actual_temps = dict(zip(test_actuals['date'], test_actuals['actual_high']))
            
            # Make predictions for each test day
            current_date = split.test_start
            while current_date <= split.test_end:
                if current_date in test_actual_temps:
                    try:
                        # Simulate real-time prediction
                        sim_data = self.data_loader.simulate_real_time_prediction(current_date)
                        
                        if sim_data['available_sources']:
                            features = self.data_loader.prepare_features_for_date(
                                current_date, sim_data['data']
                            )
                            
                            if not features.empty:
                                # Make prediction
                                if hasattr(model, 'predict'):
                                    if hasattr(model, 'predict_with_confidence'):
                                        pred_temp, confidence = model.predict_with_confidence(features)
                                    else:
                                        pred_temp = model.predict(features)
                                        if isinstance(pred_temp, (list, np.ndarray)):
                                            pred_temp = pred_temp[0]
                                        confidence = 0.8  # Default confidence
                                else:
                                    raise ValueError("Model does not have predict() method")
                                
                                # Create prediction result
                                prediction = PredictionResult(
                                    date=current_date,
                                    predicted_temperature=float(pred_temp),
                                    actual_temperature=test_actual_temps[current_date],
                                    confidence=float(confidence),
                                    model_name=getattr(model, 'name', 'unknown')
                                )
                                
                                predictions.append(prediction)
                                
                    except Exception as e:
                        logger.warning(f"Error making prediction for {current_date}: {e}")
                        continue
                
                current_date += timedelta(days=1)
            
            prediction_time = (datetime.now() - prediction_start).total_seconds()
            
            # Get model info
            model_info = {
                'status': 'success',
                'model_type': type(model).__name__,
                'training_samples': len(train_features),
                'test_predictions': len(predictions)
            }
            
            if hasattr(model, 'get_model_info'):
                model_info.update(model.get_model_info())
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Split {split.split_id} completed: {len(predictions)} predictions, "
                       f"training_time={training_time:.1f}s, prediction_time={prediction_time:.1f}s")
            
            return WalkForwardResult(
                split=split,
                predictions=predictions,
                training_time=training_time,
                prediction_time=prediction_time,
                data_quality_issues=[],
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Error in split {split.split_id}: {e}")
            return WalkForwardResult(
                split=split,
                predictions=[],
                training_time=0.0,
                prediction_time=0.0,
                data_quality_issues=[str(e)],
                model_info={'status': 'failed', 'reason': str(e)}
            )
    
    def run_analysis(self, start_date: date, end_date: date,
                    model_factory: Callable,
                    model_params: Optional[Dict] = None,
                    train_window_days: Optional[int] = None,
                    test_window_days: Optional[int] = None,
                    step_days: Optional[int] = None,
                    parallel: bool = True) -> List[WalkForwardResult]:
        """Run complete walk-forward analysis.
        
        Args:
            start_date: Overall start date for analysis
            end_date: Overall end date for analysis
            model_factory: Function that creates and returns a model instance
            model_params: Parameters to pass to model factory
            train_window_days: Training window size (default: 365)
            test_window_days: Test window size (default: 30)
            step_days: Step size between splits (default: 7)
            parallel: Whether to run splits in parallel
            
        Returns:
            List of WalkForwardResult objects
        """
        logger.info(f"Starting walk-forward analysis from {start_date} to {end_date}")
        
        # Create splits
        splits = self.create_splits(
            start_date, end_date, train_window_days, test_window_days, step_days
        )
        
        if not splits:
            logger.error("No splits created for analysis")
            return []
        
        # Run analysis
        results = []
        
        if parallel and len(splits) > 1:
            # Parallel execution
            max_workers = min(self.max_parallel_splits, len(splits))
            logger.info(f"Running {len(splits)} splits in parallel with {max_workers} workers")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all splits
                future_to_split = {
                    executor.submit(self.run_single_split, split, model_factory, model_params): split
                    for split in splits
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_split):
                    split = future_to_split[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Split {split.split_id} failed with exception: {e}")
                        # Create failed result
                        failed_result = WalkForwardResult(
                            split=split,
                            predictions=[],
                            training_time=0.0,
                            prediction_time=0.0,
                            data_quality_issues=[str(e)],
                            model_info={'status': 'failed', 'reason': str(e)}
                        )
                        results.append(failed_result)
        else:
            # Sequential execution
            logger.info(f"Running {len(splits)} splits sequentially")
            for split in splits:
                result = self.run_single_split(split, model_factory, model_params)
                results.append(result)
        
        # Sort results by split_id
        results.sort(key=lambda r: r.split.split_id)
        
        # Log summary
        successful_splits = [r for r in results if r.model_info.get('status') == 'success']
        total_predictions = sum(len(r.predictions) for r in successful_splits)
        
        logger.info(f"Walk-forward analysis completed: {len(successful_splits)}/{len(splits)} "
                   f"successful splits, {total_predictions} total predictions")
        
        return results
    
    def analyze_results(self, results: List[WalkForwardResult]) -> Dict[str, Any]:
        """Analyze walk-forward results and calculate comprehensive metrics.
        
        Args:
            results: List of WalkForwardResult objects
            
        Returns:
            Dictionary with analysis results
        """
        if not results:
            return {'error': 'No results provided'}
        
        successful_results = [r for r in results if r.model_info.get('status') == 'success']
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        # Combine all predictions
        all_predictions = []
        for result in successful_results:
            all_predictions.extend(result.predictions)
        
        if not all_predictions:
            return {'error': 'No predictions found in results'}
        
        # Calculate comprehensive metrics
        comprehensive_metrics = self.metrics_calculator.calculate_comprehensive_metrics(all_predictions)
        
        # Split-by-split analysis
        split_metrics = []
        for result in successful_results:
            if result.predictions:
                split_basic_metrics = self.metrics_calculator.calculate_basic_metrics(result.predictions)
                split_metrics.append({
                    'split_id': result.split.split_id,
                    'test_start': result.split.test_start.isoformat(),
                    'test_end': result.split.test_end.isoformat(),
                    'predictions_count': len(result.predictions),
                    'mae': split_basic_metrics.get('mae', 0),
                    'rmse': split_basic_metrics.get('rmse', 0),
                    'accuracy_within_3f': split_basic_metrics.get('accuracy_within_3f', 0),
                    'training_time': result.training_time,
                    'prediction_time': result.prediction_time
                })
        
        # Performance stability analysis
        if len(split_metrics) > 1:
            mae_values = [s['mae'] for s in split_metrics]
            accuracy_values = [s['accuracy_within_3f'] for s in split_metrics]
            
            stability_metrics = {
                'mae_std': np.std(mae_values),
                'mae_cv': np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0,
                'accuracy_std': np.std(accuracy_values),
                'accuracy_cv': np.std(accuracy_values) / np.mean(accuracy_values) if np.mean(accuracy_values) > 0 else 0,
                'best_split_mae': min(mae_values),
                'worst_split_mae': max(mae_values),
                'best_split_accuracy': max(accuracy_values),
                'worst_split_accuracy': min(accuracy_values)
            }
        else:
            stability_metrics = {'note': 'Insufficient splits for stability analysis'}
        
        # Timing analysis
        total_training_time = sum(r.training_time for r in successful_results)
        total_prediction_time = sum(r.prediction_time for r in successful_results)
        
        timing_metrics = {
            'total_training_time': total_training_time,
            'total_prediction_time': total_prediction_time,
            'avg_training_time_per_split': total_training_time / len(successful_results),
            'avg_prediction_time_per_split': total_prediction_time / len(successful_results),
            'total_analysis_time': total_training_time + total_prediction_time
        }
        
        # Data quality summary
        all_data_issues = []
        for result in results:
            all_data_issues.extend(result.data_quality_issues)
        
        data_quality_summary = {
            'total_splits': len(results),
            'successful_splits': len(successful_results),
            'failed_splits': len(results) - len(successful_results),
            'success_rate': (len(successful_results) / len(results)) * 100,
            'common_issues': list(set(all_data_issues))
        }
        
        analysis_results = {
            'summary': {
                'total_splits': len(results),
                'successful_splits': len(successful_results),
                'total_predictions': len(all_predictions),
                'overall_mae': comprehensive_metrics['basic_metrics'].get('mae', 0),
                'overall_accuracy_3f': comprehensive_metrics['basic_metrics'].get('accuracy_within_3f', 0),
                'analysis_date': datetime.now().isoformat()
            },
            'comprehensive_metrics': comprehensive_metrics,
            'split_metrics': split_metrics,
            'stability_metrics': stability_metrics,
            'timing_metrics': timing_metrics,
            'data_quality_summary': data_quality_summary
        }
        
        logger.info(f"Walk-forward analysis complete: {len(successful_results)} successful splits, "
                   f"overall MAE={comprehensive_metrics['basic_metrics'].get('mae', 0):.2f}°F, "
                   f"accuracy={comprehensive_metrics['basic_metrics'].get('accuracy_within_3f', 0):.1f}%")
        
        return analysis_results
    
    def save_results(self, results: List[WalkForwardResult], 
                    analysis_name: str, output_dir: Optional[str] = None) -> str:
        """Save walk-forward results to files.
        
        Args:
            results: List of WalkForwardResult objects
            analysis_name: Name for the analysis (used in filenames)
            output_dir: Output directory (default: backtesting_results/)
            
        Returns:
            Path to the saved results directory
        """
        if output_dir is None:
            output_dir = Path('backtesting_results')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = output_dir / f"{analysis_name}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        analysis_results = self.analyze_results(results)
        
        # Save as JSON
        import json
        with open(results_dir / 'analysis_results.json', 'w') as f:
            # Convert dates to strings for JSON serialization
            def date_converter(obj):
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json.dump(analysis_results, f, indent=2, default=date_converter)
        
        # Save predictions as CSV
        all_predictions = []
        for result in results:
            for pred in result.predictions:
                all_predictions.append({
                    'split_id': result.split.split_id,
                    'date': pred.date.isoformat(),
                    'predicted_temperature': pred.predicted_temperature,
                    'actual_temperature': pred.actual_temperature,
                    'error': pred.error,
                    'confidence': pred.confidence,
                    'model_name': pred.model_name,
                    'weather_condition': pred.weather_condition
                })
        
        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            predictions_df.to_csv(results_dir / 'predictions.csv', index=False)
        
        # Save split summary
        split_summary = []
        for result in results:
            split_summary.append({
                'split_id': result.split.split_id,
                'train_start': result.split.train_start.isoformat(),
                'train_end': result.split.train_end.isoformat(),
                'test_start': result.split.test_start.isoformat(),
                'test_end': result.split.test_end.isoformat(),
                'predictions_count': len(result.predictions),
                'training_time': result.training_time,
                'prediction_time': result.prediction_time,
                'status': result.model_info.get('status', 'unknown'),
                'data_issues': '; '.join(result.data_quality_issues)
            })
        
        split_df = pd.DataFrame(split_summary)
        split_df.to_csv(results_dir / 'split_summary.csv', index=False)
        
        logger.info(f"Walk-forward results saved to: {results_dir}")
        return str(results_dir)