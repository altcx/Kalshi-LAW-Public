"""Comprehensive backtesting framework for the Kalshi Weather Predictor."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import json

from .historical_data_loader import HistoricalDataLoader
from .performance_metrics import PerformanceMetricsCalculator, PredictionResult
from .trading_simulation import TradingSimulationEngine
from .walk_forward_analysis import WalkForwardAnalyzer
from .model_comparison import ModelComparator, ModelConfig
from src.utils.data_manager import DataManager


class BacktestingFramework:
    """Comprehensive backtesting framework for temperature prediction models."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize the backtesting framework.
        
        Args:
            data_manager: DataManager instance (creates new if None)
        """
        self.data_manager = data_manager or DataManager()
        
        # Initialize components
        self.data_loader = HistoricalDataLoader(self.data_manager)
        self.metrics_calculator = PerformanceMetricsCalculator()
        self.trading_engine = TradingSimulationEngine()
        self.walk_forward_analyzer = WalkForwardAnalyzer(self.data_manager)
        self.model_comparator = ModelComparator(self.data_manager)
        
        # Framework configuration
        self.default_config = {
            'train_window_days': 365,
            'test_window_days': 30,
            'step_days': 7,
            'initial_bankroll': 10000.0,
            'min_data_quality_threshold': 0.5,
            'parallel_execution': True
        }
        
        # Results storage
        self.results_cache: Dict[str, Any] = {}
        self.analysis_history: List[Dict] = []
        
        logger.info("BacktestingFramework initialized")
    
    def validate_data_availability(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Validate data availability for backtesting period.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating data availability from {start_date} to {end_date}")
        
        # Check available date range
        available_start, available_end = self.data_loader.get_available_date_range()
        
        validation_results = {
            'requested_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'total_days': (end_date - start_date).days + 1
            },
            'available_period': {
                'start': available_start.isoformat() if available_start else None,
                'end': available_end.isoformat() if available_end else None,
                'total_days': (available_end - available_start).days + 1 if available_start and available_end else 0
            },
            'is_valid': False,
            'issues': [],
            'recommendations': []
        }
        
        # Check if requested period is within available data
        if not available_start or not available_end:
            validation_results['issues'].append("No historical data available")
            validation_results['recommendations'].append("Collect historical weather data before running backtests")
            return validation_results
        
        if start_date < available_start:
            validation_results['issues'].append(f"Start date {start_date} is before available data starts {available_start}")
            validation_results['recommendations'].append(f"Use start date >= {available_start}")
        
        if end_date > available_end:
            validation_results['issues'].append(f"End date {end_date} is after available data ends {available_end}")
            validation_results['recommendations'].append(f"Use end date <= {available_end}")
        
        # Check data completeness
        completeness_report = self.data_loader.get_data_completeness_report(start_date, end_date)
        
        if 'overall' in completeness_report:
            avg_completeness = completeness_report['overall'].get('avg_completeness_pct', 0)
            if avg_completeness < 70:
                validation_results['issues'].append(f"Low data completeness: {avg_completeness:.1f}%")
                validation_results['recommendations'].append("Consider collecting more historical data or using a different date range")
        
        # Check for actual temperature data
        actual_temps = self.data_loader.load_actual_temperatures(start_date, end_date)
        if actual_temps.empty:
            validation_results['issues'].append("No actual temperature data available for validation")
            validation_results['recommendations'].append("Collect actual temperature observations for the backtesting period")
        
        validation_results['is_valid'] = len(validation_results['issues']) == 0
        validation_results['data_completeness'] = completeness_report
        
        if validation_results['is_valid']:
            logger.info("Data validation passed - ready for backtesting")
        else:
            logger.warning(f"Data validation failed: {len(validation_results['issues'])} issues found")
        
        return validation_results
    
    def run_single_model_backtest(self, model_factory: Callable, 
                                model_params: Optional[Dict] = None,
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None,
                                config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run backtest for a single model.
        
        Args:
            model_factory: Function that creates and returns a model instance
            model_params: Parameters to pass to model factory
            start_date: Start date for backtesting (auto-detect if None)
            end_date: End date for backtesting (auto-detect if None)
            config: Configuration overrides
            
        Returns:
            Dictionary with backtest results
        """
        # Merge configuration
        backtest_config = {**self.default_config, **(config or {})}
        
        # Auto-detect date range if not provided
        if start_date is None or end_date is None:
            available_start, available_end = self.data_loader.get_available_date_range()
            if start_date is None:
                start_date = available_start + timedelta(days=backtest_config['train_window_days'])
            if end_date is None:
                end_date = available_end
        
        logger.info(f"Running single model backtest from {start_date} to {end_date}")
        
        # Validate data availability
        validation = self.validate_data_availability(start_date, end_date)
        if not validation['is_valid']:
            return {
                'error': 'Data validation failed',
                'validation_results': validation
            }
        
        try:
            # Run walk-forward analysis
            walk_forward_results = self.walk_forward_analyzer.run_analysis(
                start_date=start_date,
                end_date=end_date,
                model_factory=model_factory,
                model_params=model_params,
                train_window_days=backtest_config['train_window_days'],
                test_window_days=backtest_config['test_window_days'],
                step_days=backtest_config['step_days'],
                parallel=backtest_config['parallel_execution']
            )
            
            # Analyze results
            analysis_results = self.walk_forward_analyzer.analyze_results(walk_forward_results)
            
            if 'error' in analysis_results:
                return {
                    'error': 'Walk-forward analysis failed',
                    'details': analysis_results
                }
            
            # Prepare trading simulation data
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
            trading_engine = TradingSimulationEngine(
                initial_bankroll=backtest_config['initial_bankroll']
            )
            trading_results = trading_engine.run_backtest(predictions_data)
            
            # Compile comprehensive results
            backtest_results = {
                'backtest_config': backtest_config,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'total_days': (end_date - start_date).days + 1
                },
                'validation_results': validation,
                'walk_forward_analysis': analysis_results,
                'trading_simulation': trading_results,
                'summary': {
                    'total_predictions': len(predictions_data),
                    'successful_splits': analysis_results['summary']['successful_splits'],
                    'overall_mae': analysis_results['comprehensive_metrics']['basic_metrics'].get('mae', 0),
                    'overall_accuracy_3f': analysis_results['comprehensive_metrics']['basic_metrics'].get('accuracy_within_3f', 0),
                    'trading_return_pct': trading_results['performance_metrics'].get('total_return', 0),
                    'sharpe_ratio': trading_results['performance_metrics'].get('sharpe_ratio', 0),
                    'max_drawdown_pct': trading_results['performance_metrics'].get('max_drawdown', 0)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            cache_key = f"single_model_{start_date}_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.results_cache[cache_key] = backtest_results
            
            # Add to analysis history
            self.analysis_history.append({
                'type': 'single_model_backtest',
                'timestamp': datetime.now().isoformat(),
                'cache_key': cache_key,
                'summary': backtest_results['summary']
            })
            
            logger.info(f"Single model backtest completed: "
                       f"MAE={backtest_results['summary']['overall_mae']:.2f}°F, "
                       f"Accuracy={backtest_results['summary']['overall_accuracy_3f']:.1f}%, "
                       f"Trading Return={backtest_results['summary']['trading_return_pct']:.1f}%")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in single model backtest: {e}")
            return {
                'error': str(e),
                'validation_results': validation
            }
    
    def run_model_comparison_backtest(self, model_configs: List[ModelConfig],
                                    start_date: Optional[date] = None,
                                    end_date: Optional[date] = None,
                                    config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run comprehensive model comparison backtest.
        
        Args:
            model_configs: List of ModelConfig objects to compare
            start_date: Start date for backtesting (auto-detect if None)
            end_date: End date for backtesting (auto-detect if None)
            config: Configuration overrides
            
        Returns:
            Dictionary with comparison results
        """
        # Merge configuration
        backtest_config = {**self.default_config, **(config or {})}
        
        # Auto-detect date range if not provided
        if start_date is None or end_date is None:
            available_start, available_end = self.data_loader.get_available_date_range()
            if start_date is None:
                start_date = available_start + timedelta(days=backtest_config['train_window_days'])
            if end_date is None:
                end_date = available_end
        
        logger.info(f"Running model comparison backtest with {len(model_configs)} models "
                   f"from {start_date} to {end_date}")
        
        # Validate data availability
        validation = self.validate_data_availability(start_date, end_date)
        if not validation['is_valid']:
            return {
                'error': 'Data validation failed',
                'validation_results': validation
            }
        
        try:
            # Run model comparison
            comparison_results = self.model_comparator.run_model_comparison(
                model_configs=model_configs,
                start_date=start_date,
                end_date=end_date,
                train_window_days=backtest_config['train_window_days'],
                test_window_days=backtest_config['test_window_days'],
                step_days=backtest_config['step_days'],
                parallel=backtest_config['parallel_execution']
            )
            
            # Generate comprehensive report
            comparison_report = self.model_comparator.generate_comparison_report(comparison_results)
            
            # Compile results
            backtest_results = {
                'backtest_config': backtest_config,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'total_days': (end_date - start_date).days + 1
                },
                'validation_results': validation,
                'model_configs': [
                    {
                        'name': config.name,
                        'params': config.params,
                        'description': config.description
                    }
                    for config in model_configs
                ],
                'comparison_results': comparison_results,
                'comparison_report': comparison_report,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            cache_key = f"model_comparison_{start_date}_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.results_cache[cache_key] = backtest_results
            
            # Add to analysis history
            self.analysis_history.append({
                'type': 'model_comparison_backtest',
                'timestamp': datetime.now().isoformat(),
                'cache_key': cache_key,
                'summary': {
                    'total_models': len(model_configs),
                    'best_model': comparison_report['summary'].get('best_model'),
                    'best_score': comparison_report['summary'].get('best_score', 0)
                }
            })
            
            logger.info(f"Model comparison backtest completed: "
                       f"Best model: {comparison_report['summary'].get('best_model')} "
                       f"(score: {comparison_report['summary'].get('best_score', 0):.3f})")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in model comparison backtest: {e}")
            return {
                'error': str(e),
                'validation_results': validation
            }
    
    def run_seasonal_analysis(self, model_factory: Callable,
                            model_params: Optional[Dict] = None,
                            start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> Dict[str, Any]:
        """Run seasonal performance analysis for a model.
        
        Args:
            model_factory: Function that creates and returns a model instance
            model_params: Parameters to pass to model factory
            start_date: Start date for analysis (auto-detect if None)
            end_date: End date for analysis (auto-detect if None)
            
        Returns:
            Dictionary with seasonal analysis results
        """
        logger.info("Running seasonal performance analysis")
        
        # Run single model backtest first
        backtest_results = self.run_single_model_backtest(
            model_factory=model_factory,
            model_params=model_params,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in backtest_results:
            return backtest_results
        
        # Extract seasonal metrics
        seasonal_metrics = backtest_results['walk_forward_analysis']['comprehensive_metrics'].get('seasonal_metrics', {})
        
        # Enhanced seasonal analysis
        seasonal_analysis = {
            'seasonal_performance': seasonal_metrics,
            'seasonal_recommendations': self._generate_seasonal_recommendations(seasonal_metrics),
            'best_seasons': self._identify_best_worst_seasons(seasonal_metrics),
            'seasonal_trading_analysis': self._analyze_seasonal_trading_performance(backtest_results)
        }
        
        return {
            'backtest_results': backtest_results,
            'seasonal_analysis': seasonal_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _generate_seasonal_recommendations(self, seasonal_metrics: Dict) -> List[str]:
        """Generate recommendations based on seasonal performance.
        
        Args:
            seasonal_metrics: Seasonal performance metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if 'seasonal_metrics' not in seasonal_metrics:
            recommendations.append("Insufficient data for seasonal analysis")
            return recommendations
        
        season_data = seasonal_metrics['seasonal_metrics']
        
        # Find best and worst performing seasons
        season_maes = {}
        for season, data in season_data.items():
            if isinstance(data, dict) and 'mae' in data and data.get('count', 0) > 0:
                season_maes[season] = data['mae']
        
        if season_maes:
            best_season = min(season_maes.keys(), key=lambda s: season_maes[s])
            worst_season = max(season_maes.keys(), key=lambda s: season_maes[s])
            
            recommendations.append(f"Best performance in {best_season} (MAE: {season_maes[best_season]:.2f}°F)")
            recommendations.append(f"Worst performance in {worst_season} (MAE: {season_maes[worst_season]:.2f}°F)")
            
            mae_range = season_maes[worst_season] - season_maes[best_season]
            if mae_range > 2.0:
                recommendations.append(f"Large seasonal variation ({mae_range:.2f}°F) suggests need for season-specific models")
            else:
                recommendations.append("Consistent performance across seasons")
        
        return recommendations
    
    def _identify_best_worst_seasons(self, seasonal_metrics: Dict) -> Dict[str, Any]:
        """Identify best and worst performing seasons.
        
        Args:
            seasonal_metrics: Seasonal performance metrics
            
        Returns:
            Dictionary with best/worst season analysis
        """
        if 'seasonal_metrics' not in seasonal_metrics:
            return {'error': 'No seasonal metrics available'}
        
        season_data = seasonal_metrics['seasonal_metrics']
        season_performance = {}
        
        for season, data in season_data.items():
            if isinstance(data, dict) and 'mae' in data and data.get('count', 0) > 0:
                season_performance[season] = {
                    'mae': data['mae'],
                    'accuracy_3f': data.get('accuracy_within_3f', 0),
                    'count': data['count'],
                    'score': data.get('accuracy_within_3f', 0) - data['mae']  # Simple composite score
                }
        
        if not season_performance:
            return {'error': 'No valid seasonal performance data'}
        
        # Find best and worst
        best_season = max(season_performance.keys(), key=lambda s: season_performance[s]['score'])
        worst_season = min(season_performance.keys(), key=lambda s: season_performance[s]['score'])
        
        return {
            'best_season': {
                'season': best_season,
                **season_performance[best_season]
            },
            'worst_season': {
                'season': worst_season,
                **season_performance[worst_season]
            },
            'all_seasons': season_performance
        }
    
    def _analyze_seasonal_trading_performance(self, backtest_results: Dict) -> Dict[str, Any]:
        """Analyze trading performance by season.
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Dictionary with seasonal trading analysis
        """
        trading_history = backtest_results.get('trading_simulation', {}).get('trading_history', [])
        
        if not trading_history:
            return {'error': 'No trading history available'}
        
        # Group trades by season
        seasonal_trades = {'winter': [], 'spring': [], 'summer': [], 'fall': []}
        season_months = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11]
        }
        
        for trade in trading_history:
            trade_date = datetime.fromisoformat(trade['date']).date()
            month = trade_date.month
            
            for season, months in season_months.items():
                if month in months:
                    seasonal_trades[season].append(trade)
                    break
        
        # Calculate seasonal trading metrics
        seasonal_trading_metrics = {}
        
        for season, trades in seasonal_trades.items():
            if trades:
                total_pnl = sum(trade.get('pnl', 0) for trade in trades)
                winning_trades = [t for t in trades if t.get('is_winner') is True]
                win_rate = len(winning_trades) / len(trades) * 100
                
                seasonal_trading_metrics[season] = {
                    'total_trades': len(trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl_per_trade': total_pnl / len(trades)
                }
            else:
                seasonal_trading_metrics[season] = {
                    'total_trades': 0,
                    'note': 'No trades in this season'
                }
        
        return seasonal_trading_metrics
    
    def save_backtest_results(self, results: Dict[str, Any], 
                            output_dir: Optional[str] = None,
                            analysis_name: Optional[str] = None) -> str:
        """Save backtest results to files.
        
        Args:
            results: Backtest results dictionary
            output_dir: Output directory (default: backtest_results/)
            analysis_name: Name for the analysis (default: auto-generated)
            
        Returns:
            Path to the saved results directory
        """
        if output_dir is None:
            output_dir = Path('backtest_results')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate analysis name if not provided
        if analysis_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_type = 'comparison' if 'comparison_results' in results else 'single_model'
            analysis_name = f"{analysis_type}_{timestamp}"
        
        results_dir = output_dir / analysis_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        with open(results_dir / 'backtest_results.json', 'w') as f:
            def date_converter(obj):
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json.dump(results, f, indent=2, default=date_converter)
        
        # Save summary as CSV if available
        if 'summary' in results:
            summary_df = pd.DataFrame([results['summary']])
            summary_df.to_csv(results_dir / 'summary.csv', index=False)
        
        # Save trading history if available
        if 'trading_simulation' in results and 'trading_history' in results['trading_simulation']:
            trading_df = pd.DataFrame(results['trading_simulation']['trading_history'])
            if not trading_df.empty:
                trading_df.to_csv(results_dir / 'trading_history.csv', index=False)
        
        logger.info(f"Backtest results saved to: {results_dir}")
        return str(results_dir)
    
    def get_analysis_history(self) -> List[Dict]:
        """Get history of all analyses run by this framework.
        
        Returns:
            List of analysis history records
        """
        return self.analysis_history.copy()
    
    def clear_cache(self) -> None:
        """Clear the results cache."""
        self.results_cache.clear()
        logger.info("Results cache cleared")