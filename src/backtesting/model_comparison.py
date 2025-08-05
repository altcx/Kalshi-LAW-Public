"""Model comparison and optimization for backtesting."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from pathlib import Path
import json
import concurrent.futures
from itertools import product

from .historical_data_loader import HistoricalDataLoader
from .performance_metrics import PerformanceMetricsCalculator, PredictionResult
from .trading_simulation import TradingSimulationEngine
from .walk_forward_analysis import WalkForwardAnalyzer, WalkForwardResult
from src.utils.data_manager import DataManager


@dataclass
class ModelConfig:
    """Configuration for a model to be tested."""
    name: str
    model_factory: Callable
    params: Dict[str, Any]
    description: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing multiple models."""
    model_name: str
    config: ModelConfig
    performance_metrics: Dict[str, Any]
    trading_metrics: Dict[str, Any]
    seasonal_performance: Dict[str, Any]
    walk_forward_results: List[WalkForwardResult]
    optimization_score: float
    rank: int


class ModelComparator:
    """Compares multiple ML models and strategies for temperature prediction."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize the model comparator.
        
        Args:
            data_manager: DataManager instance (creates new if None)
        """
        self.data_manager = data_manager or DataManager()
        self.data_loader = HistoricalDataLoader(self.data_manager)
        self.metrics_calculator = PerformanceMetricsCalculator()
        self.walk_forward_analyzer = WalkForwardAnalyzer(self.data_manager)
        
        # Comparison configuration
        self.scoring_weights = {
            'accuracy_3f': 0.3,      # 30% weight on ±3°F accuracy
            'mae': 0.2,              # 20% weight on MAE (inverted)
            'trading_return': 0.25,   # 25% weight on trading returns
            'sharpe_ratio': 0.15,     # 15% weight on risk-adjusted returns
            'stability': 0.1          # 10% weight on performance stability
        }
        
        # Results storage
        self.comparison_results: List[ComparisonResult] = []
        self.best_model_config: Optional[ModelConfig] = None
        
        logger.info("ModelComparator initialized")
    
    def add_model_config(self, name: str, model_factory: Callable, 
                        params: Dict[str, Any], description: Optional[str] = None) -> None:
        """Add a model configuration for comparison.
        
        Args:
            name: Unique name for the model
            model_factory: Function that creates and returns a model instance
            params: Parameters to pass to model factory
            description: Optional description of the model
        """
        config = ModelConfig(
            name=name,
            model_factory=model_factory,
            params=params,
            description=description
        )
        
        logger.info(f"Added model config: {name}")
    
    def run_single_model_comparison(self, config: ModelConfig, 
                                  start_date: date, end_date: date,
                                  train_window_days: int = 365,
                                  test_window_days: int = 30,
                                  step_days: int = 7) -> ComparisonResult:
        """Run comparison for a single model configuration.
        
        Args:
            config: ModelConfig to test
            start_date: Start date for analysis
            end_date: End date for analysis
            train_window_days: Training window size
            test_window_days: Test window size
            step_days: Step size between splits
            
        Returns:
            ComparisonResult with analysis results
        """
        logger.info(f"Running comparison for model: {config.name}")
        
        try:
            # Run walk-forward analysis
            walk_forward_results = self.walk_forward_analyzer.run_analysis(
                start_date=start_date,
                end_date=end_date,
                model_factory=config.model_factory,
                model_params=config.params,
                train_window_days=train_window_days,
                test_window_days=test_window_days,
                step_days=step_days,
                parallel=False  # Sequential for stability
            )
            
            # Analyze walk-forward results
            wf_analysis = self.walk_forward_analyzer.analyze_results(walk_forward_results)
            
            if 'error' in wf_analysis:
                logger.error(f"Walk-forward analysis failed for {config.name}: {wf_analysis['error']}")
                return self._create_failed_result(config, wf_analysis['error'])
            
            # Extract performance metrics
            performance_metrics = wf_analysis['comprehensive_metrics']
            
            # Prepare data for trading simulation
            predictions_data = []
            for result in walk_forward_results:
                for pred in result.predictions:
                    predictions_data.append({
                        'date': pred.date,
                        'predicted_temp': pred.predicted_temperature,
                        'actual_temp': pred.actual_temperature,
                        'confidence': pred.confidence
                    })
            
            # Run trading simulation
            trading_engine = TradingSimulationEngine(initial_bankroll=10000.0)
            trading_results = trading_engine.run_backtest(predictions_data)
            trading_metrics = trading_results['performance_metrics']
            
            # Calculate seasonal performance
            all_predictions = []
            for result in walk_forward_results:
                all_predictions.extend(result.predictions)
            
            seasonal_performance = self.metrics_calculator.calculate_seasonal_metrics(all_predictions)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                performance_metrics, trading_metrics, wf_analysis
            )
            
            comparison_result = ComparisonResult(
                model_name=config.name,
                config=config,
                performance_metrics=performance_metrics,
                trading_metrics=trading_metrics,
                seasonal_performance=seasonal_performance,
                walk_forward_results=walk_forward_results,
                optimization_score=optimization_score,
                rank=0  # Will be set later during ranking
            )
            
            logger.info(f"Completed comparison for {config.name}: "
                       f"score={optimization_score:.3f}, "
                       f"accuracy={performance_metrics['basic_metrics'].get('accuracy_within_3f', 0):.1f}%")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error in model comparison for {config.name}: {e}")
            return self._create_failed_result(config, str(e))
    
    def _create_failed_result(self, config: ModelConfig, error_message: str) -> ComparisonResult:
        """Create a failed comparison result.
        
        Args:
            config: ModelConfig that failed
            error_message: Error message
            
        Returns:
            ComparisonResult indicating failure
        """
        return ComparisonResult(
            model_name=config.name,
            config=config,
            performance_metrics={'error': error_message},
            trading_metrics={'error': error_message},
            seasonal_performance={'error': error_message},
            walk_forward_results=[],
            optimization_score=0.0,
            rank=999  # Worst rank
        )
    
    def _calculate_optimization_score(self, performance_metrics: Dict, 
                                    trading_metrics: Dict, 
                                    wf_analysis: Dict) -> float:
        """Calculate optimization score for model ranking.
        
        Args:
            performance_metrics: Performance metrics from walk-forward analysis
            trading_metrics: Trading simulation metrics
            wf_analysis: Walk-forward analysis results
            
        Returns:
            Optimization score (higher is better)
        """
        score = 0.0
        
        try:
            basic_metrics = performance_metrics.get('basic_metrics', {})
            
            # Accuracy component (0-100, higher is better)
            accuracy_3f = basic_metrics.get('accuracy_within_3f', 0)
            accuracy_score = accuracy_3f / 100.0
            score += self.scoring_weights['accuracy_3f'] * accuracy_score
            
            # MAE component (lower is better, so invert)
            mae = basic_metrics.get('mae', 10.0)  # Default to high MAE if missing
            mae_score = max(0, 1.0 - (mae / 10.0))  # Normalize assuming 10°F is very bad
            score += self.scoring_weights['mae'] * mae_score
            
            # Trading return component (-100 to +100, normalize to 0-1)
            trading_return = trading_metrics.get('total_return', -100)
            trading_score = max(0, (trading_return + 100) / 200.0)  # Normalize -100 to +100 -> 0 to 1
            score += self.scoring_weights['trading_return'] * trading_score
            
            # Sharpe ratio component (can be negative, normalize)
            sharpe_ratio = trading_metrics.get('sharpe_ratio', -2.0)
            sharpe_score = max(0, (sharpe_ratio + 2.0) / 4.0)  # Normalize -2 to +2 -> 0 to 1
            score += self.scoring_weights['sharpe_ratio'] * sharpe_score
            
            # Stability component (based on performance consistency)
            stability_metrics = wf_analysis.get('stability_metrics', {})
            if isinstance(stability_metrics, dict) and 'mae_cv' in stability_metrics:
                mae_cv = stability_metrics.get('mae_cv', 1.0)
                stability_score = max(0, 1.0 - mae_cv)  # Lower CV is better
            else:
                stability_score = 0.5  # Default middle score
            
            score += self.scoring_weights['stability'] * stability_score
            
        except Exception as e:
            logger.warning(f"Error calculating optimization score: {e}")
            score = 0.0
        
        return score
    
    def run_model_comparison(self, model_configs: List[ModelConfig],
                           start_date: date, end_date: date,
                           train_window_days: int = 365,
                           test_window_days: int = 30,
                           step_days: int = 7,
                           parallel: bool = True) -> List[ComparisonResult]:
        """Run comparison across multiple model configurations.
        
        Args:
            model_configs: List of ModelConfig objects to compare
            start_date: Start date for analysis
            end_date: End date for analysis
            train_window_days: Training window size
            test_window_days: Test window size
            step_days: Step size between splits
            parallel: Whether to run comparisons in parallel
            
        Returns:
            List of ComparisonResult objects, sorted by optimization score
        """
        logger.info(f"Starting model comparison with {len(model_configs)} models")
        
        results = []
        
        if parallel and len(model_configs) > 1:
            # Parallel execution
            max_workers = min(4, len(model_configs))  # Limit workers to avoid resource issues
            logger.info(f"Running {len(model_configs)} model comparisons in parallel with {max_workers} workers")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all model comparisons
                future_to_config = {
                    executor.submit(
                        self.run_single_model_comparison,
                        config, start_date, end_date, train_window_days, test_window_days, step_days
                    ): config
                    for config in model_configs
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Model comparison failed for {config.name}: {e}")
                        failed_result = self._create_failed_result(config, str(e))
                        results.append(failed_result)
        else:
            # Sequential execution
            logger.info(f"Running {len(model_configs)} model comparisons sequentially")
            for config in model_configs:
                result = self.run_single_model_comparison(
                    config, start_date, end_date, train_window_days, test_window_days, step_days
                )
                results.append(result)
        
        # Sort by optimization score (descending)
        results.sort(key=lambda r: r.optimization_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Store results
        self.comparison_results = results
        
        # Identify best model
        if results and results[0].optimization_score > 0:
            self.best_model_config = results[0].config
            logger.info(f"Best model identified: {results[0].model_name} "
                       f"(score: {results[0].optimization_score:.3f})")
        
        logger.info(f"Model comparison completed: {len(results)} models analyzed")
        return results
    
    def analyze_seasonal_performance(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze seasonal performance patterns across models.
        
        Args:
            results: List of ComparisonResult objects
            
        Returns:
            Dictionary with seasonal analysis
        """
        seasonal_analysis = {
            'models': {},
            'season_rankings': {},
            'best_seasonal_models': {},
            'seasonal_consistency': {}
        }
        
        seasons = ['winter', 'spring', 'summer', 'fall']
        
        for result in results:
            if 'error' in result.seasonal_performance:
                continue
            
            model_name = result.model_name
            seasonal_metrics = result.seasonal_performance.get('seasonal_metrics', {})
            
            seasonal_analysis['models'][model_name] = {}
            
            for season in seasons:
                season_data = seasonal_metrics.get(season, {})
                if isinstance(season_data, dict) and 'mae' in season_data:
                    seasonal_analysis['models'][model_name][season] = {
                        'mae': season_data['mae'],
                        'accuracy_3f': season_data.get('accuracy_within_3f', 0),
                        'count': season_data.get('count', 0)
                    }
        
        # Find best model for each season
        for season in seasons:
            season_scores = {}
            for model_name, model_data in seasonal_analysis['models'].items():
                if season in model_data and model_data[season]['count'] > 0:
                    # Score based on accuracy and MAE
                    accuracy = model_data[season]['accuracy_3f']
                    mae = model_data[season]['mae']
                    score = accuracy - (mae * 10)  # Simple scoring
                    season_scores[model_name] = score
            
            if season_scores:
                best_model = max(season_scores.keys(), key=lambda m: season_scores[m])
                seasonal_analysis['best_seasonal_models'][season] = {
                    'model': best_model,
                    'score': season_scores[best_model],
                    'mae': seasonal_analysis['models'][best_model][season]['mae'],
                    'accuracy_3f': seasonal_analysis['models'][best_model][season]['accuracy_3f']
                }
        
        # Calculate seasonal consistency for each model
        for model_name, model_data in seasonal_analysis['models'].items():
            season_maes = [data['mae'] for data in model_data.values() if 'mae' in data]
            if len(season_maes) >= 2:
                consistency = 1.0 / (1.0 + np.std(season_maes))  # Higher is more consistent
                seasonal_analysis['seasonal_consistency'][model_name] = consistency
        
        return seasonal_analysis
    
    def optimize_ensemble_weights(self, results: List[ComparisonResult],
                                top_n_models: int = 5) -> Dict[str, float]:
        """Optimize ensemble weights based on model comparison results.
        
        Args:
            results: List of ComparisonResult objects
            top_n_models: Number of top models to include in ensemble
            
        Returns:
            Dictionary mapping model names to optimal weights
        """
        # Select top N models
        top_models = results[:top_n_models]
        
        if not top_models:
            return {}
        
        # Simple optimization based on optimization scores
        scores = [result.optimization_score for result in top_models]
        total_score = sum(scores)
        
        if total_score == 0:
            # Equal weights if no valid scores
            equal_weight = 1.0 / len(top_models)
            weights = {result.model_name: equal_weight for result in top_models}
        else:
            # Proportional weights based on scores
            weights = {
                result.model_name: result.optimization_score / total_score
                for result in top_models
            }
        
        logger.info(f"Optimized ensemble weights for top {len(weights)} models: {weights}")
        return weights
    
    def generate_comparison_report(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate comprehensive comparison report.
        
        Args:
            results: List of ComparisonResult objects
            
        Returns:
            Dictionary with comprehensive comparison report
        """
        if not results:
            return {'error': 'No comparison results available'}
        
        # Overall summary
        valid_results = [r for r in results if 'error' not in r.performance_metrics]
        
        summary = {
            'total_models': len(results),
            'valid_models': len(valid_results),
            'best_model': results[0].model_name if results else None,
            'best_score': results[0].optimization_score if results else 0,
            'analysis_date': datetime.now().isoformat()
        }
        
        # Model rankings
        model_rankings = []
        for result in results:
            if 'error' not in result.performance_metrics:
                basic_metrics = result.performance_metrics.get('basic_metrics', {})
                model_rankings.append({
                    'rank': result.rank,
                    'model_name': result.model_name,
                    'optimization_score': result.optimization_score,
                    'mae': basic_metrics.get('mae', 0),
                    'accuracy_3f': basic_metrics.get('accuracy_within_3f', 0),
                    'trading_return': result.trading_metrics.get('total_return', 0),
                    'sharpe_ratio': result.trading_metrics.get('sharpe_ratio', 0),
                    'total_predictions': basic_metrics.get('total_predictions', 0)
                })
        
        # Seasonal analysis
        seasonal_analysis = self.analyze_seasonal_performance(results)
        
        # Ensemble optimization
        ensemble_weights = self.optimize_ensemble_weights(results)
        
        # Performance distribution analysis
        if valid_results:
            scores = [r.optimization_score for r in valid_results]
            maes = [r.performance_metrics['basic_metrics'].get('mae', 0) for r in valid_results]
            accuracies = [r.performance_metrics['basic_metrics'].get('accuracy_within_3f', 0) for r in valid_results]
            
            performance_distribution = {
                'optimization_scores': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                },
                'mae_distribution': {
                    'mean': np.mean(maes),
                    'std': np.std(maes),
                    'min': np.min(maes),
                    'max': np.max(maes)
                },
                'accuracy_distribution': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies)
                }
            }
        else:
            performance_distribution = {'note': 'No valid results for distribution analysis'}
        
        # Recommendations
        recommendations = self._generate_recommendations(results)
        
        report = {
            'summary': summary,
            'model_rankings': model_rankings,
            'seasonal_analysis': seasonal_analysis,
            'ensemble_weights': ensemble_weights,
            'performance_distribution': performance_distribution,
            'recommendations': recommendations,
            'scoring_weights': self.scoring_weights
        }
        
        return report
    
    def _generate_recommendations(self, results: List[ComparisonResult]) -> List[str]:
        """Generate recommendations based on comparison results.
        
        Args:
            results: List of ComparisonResult objects
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not results:
            recommendations.append("No models were successfully analyzed.")
            return recommendations
        
        valid_results = [r for r in results if 'error' not in r.performance_metrics]
        
        if not valid_results:
            recommendations.append("All model analyses failed. Check data availability and model configurations.")
            return recommendations
        
        best_result = valid_results[0]
        best_metrics = best_result.performance_metrics.get('basic_metrics', {})
        
        # Overall performance recommendations
        best_accuracy = best_metrics.get('accuracy_within_3f', 0)
        best_mae = best_metrics.get('mae', 0)
        
        if best_accuracy >= 85:
            recommendations.append(f"Excellent performance achieved: {best_result.model_name} reaches {best_accuracy:.1f}% accuracy within ±3°F.")
        elif best_accuracy >= 75:
            recommendations.append(f"Good performance: {best_result.model_name} achieves {best_accuracy:.1f}% accuracy. Consider ensemble methods for improvement.")
        else:
            recommendations.append(f"Performance needs improvement: Best model ({best_result.model_name}) only achieves {best_accuracy:.1f}% accuracy. Consider more sophisticated models or better features.")
        
        # MAE recommendations
        if best_mae <= 2.5:
            recommendations.append(f"Excellent temperature precision: MAE of {best_mae:.2f}°F is very good for weather prediction.")
        elif best_mae <= 4.0:
            recommendations.append(f"Good temperature precision: MAE of {best_mae:.2f}°F is acceptable but could be improved.")
        else:
            recommendations.append(f"Temperature precision needs improvement: MAE of {best_mae:.2f}°F is high for trading applications.")
        
        # Trading performance recommendations
        best_trading_return = best_result.trading_metrics.get('total_return', 0)
        if best_trading_return > 10:
            recommendations.append(f"Strong trading performance: {best_trading_return:.1f}% return suggests profitable strategy.")
        elif best_trading_return > 0:
            recommendations.append(f"Modest trading performance: {best_trading_return:.1f}% return is positive but could be improved.")
        else:
            recommendations.append(f"Trading strategy needs work: {best_trading_return:.1f}% return suggests losses. Review position sizing and contract selection.")
        
        # Ensemble recommendations
        if len(valid_results) >= 3:
            recommendations.append("Consider ensemble methods: Multiple models show different strengths that could be combined.")
        
        # Model-specific recommendations
        if len(valid_results) > 1:
            score_gap = valid_results[0].optimization_score - valid_results[1].optimization_score
            if score_gap < 0.1:
                recommendations.append("Top models are very close in performance. Consider A/B testing or ensemble approaches.")
        
        return recommendations
    
    def save_comparison_results(self, results: List[ComparisonResult], 
                              output_dir: Optional[str] = None) -> str:
        """Save comparison results to files.
        
        Args:
            results: List of ComparisonResult objects
            output_dir: Output directory (default: model_comparison_results/)
            
        Returns:
            Path to the saved results directory
        """
        if output_dir is None:
            output_dir = Path('model_comparison_results')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = output_dir / f"comparison_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save comprehensive report
        report = self.generate_comparison_report(results)
        
        # Save as JSON
        with open(results_dir / 'comparison_report.json', 'w') as f:
            def date_converter(obj):
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json.dump(report, f, indent=2, default=date_converter)
        
        # Save model rankings as CSV
        if report['model_rankings']:
            rankings_df = pd.DataFrame(report['model_rankings'])
            rankings_df.to_csv(results_dir / 'model_rankings.csv', index=False)
        
        # Save detailed results for each model
        for result in results:
            if 'error' not in result.performance_metrics:
                model_dir = results_dir / f"model_{result.model_name}"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model-specific metrics
                model_data = {
                    'model_name': result.model_name,
                    'rank': result.rank,
                    'optimization_score': result.optimization_score,
                    'performance_metrics': result.performance_metrics,
                    'trading_metrics': result.trading_metrics,
                    'seasonal_performance': result.seasonal_performance,
                    'config': {
                        'name': result.config.name,
                        'params': result.config.params,
                        'description': result.config.description
                    }
                }
                
                with open(model_dir / 'model_results.json', 'w') as f:
                    json.dump(model_data, f, indent=2, default=date_converter)
        
        logger.info(f"Model comparison results saved to: {results_dir}")
        return str(results_dir)