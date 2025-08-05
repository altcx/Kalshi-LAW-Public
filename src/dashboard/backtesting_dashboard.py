"""Backtesting and analysis dashboard for model performance evaluation."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from loguru import logger
from pathlib import Path

from src.utils.data_manager import DataManager
from src.utils.config import config
from src.backtesting.backtesting_framework import BacktestingFramework
from src.backtesting.model_comparison import ModelComparator
from src.backtesting.performance_metrics import PerformanceMetricsCalculator


class BacktestingDashboard:
    """Dashboard for backtesting analysis and model comparison."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize backtesting dashboard.
        
        Args:
            data_manager: DataManager instance (creates new one if None)
        """
        self.data_manager = data_manager or DataManager()
        self.backtesting_framework = BacktestingFramework(self.data_manager)
        self.model_comparison = ModelComparator(self.data_manager)
        self.performance_metrics = PerformanceMetricsCalculator()
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Dashboard configuration
        self.figure_size = (15, 10)
        self.dpi = 100
        
        logger.info("BacktestingDashboard initialized")
    
    def run_custom_backtest(self, start_date: date, end_date: date, 
                           parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a custom backtest with specified parameters.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            parameters: Custom backtest parameters
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Set default parameters if none provided
            if parameters is None:
                parameters = {
                    'models': ['ensemble', 'xgboost', 'lightgbm'],
                    'retraining_frequency': 7,  # days
                    'min_training_days': 30,
                    'confidence_threshold': 0.7,
                    'trading_enabled': True,
                    'initial_bankroll': 10000
                }
            
            logger.info(f"Running custom backtest from {start_date} to {end_date}")
            
            # Run the backtest
            backtest_results = self.backtesting_framework.run_backtest(
                start_date=start_date,
                end_date=end_date,
                models=parameters.get('models', ['ensemble']),
                retraining_frequency=parameters.get('retraining_frequency', 7),
                min_training_days=parameters.get('min_training_days', 30)
            )
            
            # Calculate performance metrics
            if backtest_results and 'predictions' in backtest_results:
                predictions_df = pd.DataFrame(backtest_results['predictions'])
                
                if not predictions_df.empty and 'actual_temperature' in predictions_df.columns:
                    # Calculate accuracy metrics
                    valid_predictions = predictions_df.dropna(subset=['actual_temperature'])
                    
                    if not valid_predictions.empty:
                        metrics = self.performance_metrics.calculate_accuracy_metrics(
                            valid_predictions['predicted_high'],
                            valid_predictions['actual_temperature']
                        )
                        
                        backtest_results['performance_metrics'] = metrics
                        
                        # Calculate trading metrics if enabled
                        if parameters.get('trading_enabled', False):
                            trading_metrics = self._calculate_trading_metrics(
                                valid_predictions, 
                                parameters.get('initial_bankroll', 10000)
                            )
                            backtest_results['trading_metrics'] = trading_metrics
            
            backtest_results.update({
                'start_date': start_date,
                'end_date': end_date,
                'parameters': parameters,
                'generated_at': datetime.now()
            })
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running custom backtest: {e}")
            return {
                'error': str(e),
                'start_date': start_date,
                'end_date': end_date,
                'generated_at': datetime.now()
            }
    
    def compare_models(self, start_date: date, end_date: date, 
                      models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare performance of different models.
        
        Args:
            start_date: Start date for comparison
            end_date: End date for comparison
            models: List of models to compare
            
        Returns:
            Dictionary with model comparison results
        """
        try:
            if models is None:
                models = ['ensemble', 'xgboost', 'lightgbm', 'linear_regression', 'prophet']
            
            logger.info(f"Comparing models: {models}")
            
            # Run comparison
            comparison_results = self.model_comparison.compare_models(
                models=models,
                start_date=start_date,
                end_date=end_date
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {'error': str(e)}
    
    def optimize_strategy(self, start_date: date, end_date: date, 
                         optimization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize trading strategy parameters.
        
        Args:
            start_date: Start date for optimization
            end_date: End date for optimization
            optimization_params: Parameters to optimize
            
        Returns:
            Dictionary with optimization results
        """
        try:
            if optimization_params is None:
                optimization_params = {
                    'confidence_thresholds': [0.6, 0.7, 0.8, 0.9],
                    'position_sizes': [0.01, 0.02, 0.05, 0.10],
                    'retraining_frequencies': [3, 7, 14, 30],
                    'objective': 'sharpe_ratio'  # or 'total_return', 'max_drawdown'
                }
            
            logger.info("Running strategy optimization")
            
            best_params = {}
            best_score = float('-inf')
            optimization_results = []
            
            # Grid search over parameter combinations
            confidence_thresholds = optimization_params.get('confidence_thresholds', [0.7])
            position_sizes = optimization_params.get('position_sizes', [0.05])
            retraining_frequencies = optimization_params.get('retraining_frequencies', [7])
            
            for conf_thresh in confidence_thresholds:
                for pos_size in position_sizes:
                    for retrain_freq in retraining_frequencies:
                        # Run backtest with these parameters
                        params = {
                            'confidence_threshold': conf_thresh,
                            'position_size': pos_size,
                            'retraining_frequency': retrain_freq,
                            'trading_enabled': True,
                            'initial_bankroll': 10000
                        }
                        
                        backtest_result = self.run_custom_backtest(start_date, end_date, params)
                        
                        if 'error' not in backtest_result and 'trading_metrics' in backtest_result:
                            trading_metrics = backtest_result['trading_metrics']
                            
                            # Calculate objective score
                            objective = optimization_params.get('objective', 'sharpe_ratio')
                            if objective == 'sharpe_ratio':
                                score = trading_metrics.get('sharpe_ratio', 0)
                            elif objective == 'total_return':
                                score = trading_metrics.get('total_return', 0)
                            elif objective == 'max_drawdown':
                                score = -trading_metrics.get('max_drawdown', 1)  # Negative because we want to minimize
                            else:
                                score = trading_metrics.get('sharpe_ratio', 0)
                            
                            optimization_results.append({
                                'parameters': params,
                                'score': score,
                                'trading_metrics': trading_metrics,
                                'performance_metrics': backtest_result.get('performance_metrics', {})
                            })
                            
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_results': optimization_results,
                'objective': optimization_params.get('objective', 'sharpe_ratio'),
                'start_date': start_date,
                'end_date': end_date,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            return {'error': str(e)}
    
    def run_what_if_analysis(self, base_scenario: Dict[str, Any], 
                            variations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run what-if analysis with different scenarios.
        
        Args:
            base_scenario: Base scenario parameters
            variations: List of parameter variations to test
            
        Returns:
            Dictionary with what-if analysis results
        """
        try:
            logger.info("Running what-if analysis")
            
            results = []
            
            # Run base scenario
            base_result = self.run_custom_backtest(
                base_scenario['start_date'],
                base_scenario['end_date'],
                base_scenario.get('parameters', {})
            )
            
            results.append({
                'scenario': 'Base',
                'parameters': base_scenario.get('parameters', {}),
                'result': base_result
            })
            
            # Run variations
            for i, variation in enumerate(variations):
                # Merge base parameters with variation
                params = base_scenario.get('parameters', {}).copy()
                params.update(variation.get('parameter_changes', {}))
                
                variation_result = self.run_custom_backtest(
                    base_scenario['start_date'],
                    base_scenario['end_date'],
                    params
                )
                
                results.append({
                    'scenario': variation.get('name', f'Variation {i+1}'),
                    'parameters': params,
                    'result': variation_result
                })
            
            return {
                'base_scenario': base_scenario,
                'variations': variations,
                'results': results,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error running what-if analysis: {e}")
            return {'error': str(e)}
    
    def create_backtest_results_plot(self, backtest_results: Dict[str, Any]) -> Figure:
        """Create a plot showing backtest results.
        
        Args:
            backtest_results: Backtest results dictionary
            
        Returns:
            Matplotlib figure with backtest results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        try:
            if 'error' in backtest_results:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, f"Error: {backtest_results['error']}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            predictions = backtest_results.get('predictions', [])
            if not predictions:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No prediction data available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            predictions_df = pd.DataFrame(predictions)
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            
            # Plot 1: Predictions vs Actuals
            if 'actual_temperature' in predictions_df.columns:
                valid_preds = predictions_df.dropna(subset=['actual_temperature'])
                
                if not valid_preds.empty:
                    ax1.plot(valid_preds['date'], valid_preds['predicted_high'], 
                            'o-', label='Predicted', alpha=0.7, markersize=4)
                    ax1.plot(valid_preds['date'], valid_preds['actual_temperature'], 
                            's-', label='Actual', alpha=0.7, markersize=4)
                    
                    ax1.set_title('Predictions vs Actuals', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Temperature (°F)', fontsize=12)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Prediction Errors
            if 'actual_temperature' in predictions_df.columns:
                valid_preds = predictions_df.dropna(subset=['actual_temperature'])
                
                if not valid_preds.empty:
                    errors = valid_preds['predicted_high'] - valid_preds['actual_temperature']
                    ax2.plot(valid_preds['date'], errors, 'o-', alpha=0.7, markersize=4)
                    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                    ax2.axhline(y=3, color='orange', linestyle=':', alpha=0.7, label='±3°F')
                    ax2.axhline(y=-3, color='orange', linestyle=':', alpha=0.7)
                    
                    ax2.set_title('Prediction Errors', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Error (°F)', fontsize=12)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Error Distribution
            if 'actual_temperature' in predictions_df.columns:
                valid_preds = predictions_df.dropna(subset=['actual_temperature'])
                
                if not valid_preds.empty:
                    errors = valid_preds['predicted_high'] - valid_preds['actual_temperature']
                    ax3.hist(errors, bins=20, alpha=0.7, edgecolor='black')
                    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect')
                    ax3.axvline(x=errors.mean(), color='green', linestyle='-', alpha=0.7, 
                               label=f'Mean: {errors.mean():.2f}°F')
                    
                    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
                    ax3.set_xlabel('Error (°F)', fontsize=12)
                    ax3.set_ylabel('Frequency', fontsize=12)
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
            
            # Plot 4: Performance Metrics
            performance_metrics = backtest_results.get('performance_metrics', {})
            if performance_metrics:
                metrics_names = ['MAE', 'RMSE', 'Accuracy ±3°F', 'Accuracy ±5°F']
                metrics_values = [
                    performance_metrics.get('mae', 0),
                    performance_metrics.get('rmse', 0),
                    performance_metrics.get('accuracy_3f', 0),
                    performance_metrics.get('accuracy_5f', 0)
                ]
                
                bars = ax4.bar(metrics_names, metrics_values, alpha=0.7)
                ax4.set_title('Performance Metrics', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Value', fontsize=12)
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, metrics_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating backtest results plot: {e}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        return fig
    
    def create_model_comparison_plot(self, comparison_results: Dict[str, Any]) -> Figure:
        """Create a plot comparing model performance.
        
        Args:
            comparison_results: Model comparison results
            
        Returns:
            Matplotlib figure with model comparison
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        try:
            if 'error' in comparison_results:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, f"Error: {comparison_results['error']}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            model_results = comparison_results.get('model_results', {})
            if not model_results:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No model comparison data available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            models = list(model_results.keys())
            
            # Plot 1: Accuracy Comparison
            accuracies = [model_results[model].get('accuracy_3f', 0) for model in models]
            bars1 = ax1.bar(models, accuracies, alpha=0.7)
            ax1.set_title('Model Accuracy Comparison (±3°F)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 2: MAE Comparison
            maes = [model_results[model].get('mae', 0) for model in models]
            bars2 = ax2.bar(models, maes, alpha=0.7, color='orange')
            ax2.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel('MAE (°F)', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mae in zip(bars2, maes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mae:.2f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 3: RMSE Comparison
            rmses = [model_results[model].get('rmse', 0) for model in models]
            bars3 = ax3.bar(models, rmses, alpha=0.7, color='green')
            ax3.set_title('Root Mean Square Error Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('RMSE (°F)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rmse in zip(bars3, rmses):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rmse:.2f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 4: Model Ranking
            # Create a composite score (higher is better)
            composite_scores = []
            for model in models:
                acc = model_results[model].get('accuracy_3f', 0)
                mae = model_results[model].get('mae', 10)  # Default high MAE
                rmse = model_results[model].get('rmse', 10)  # Default high RMSE
                
                # Composite score: accuracy is good, MAE and RMSE are bad
                score = acc - (mae / 10) - (rmse / 10)  # Normalize MAE and RMSE
                composite_scores.append(score)
            
            # Sort models by composite score
            model_scores = list(zip(models, composite_scores))
            model_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_models, sorted_scores = zip(*model_scores)
            
            bars4 = ax4.bar(sorted_models, sorted_scores, alpha=0.7, color='purple')
            ax4.set_title('Model Ranking (Composite Score)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Composite Score', fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars4, sorted_scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {e}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        return fig
    
    def create_optimization_results_plot(self, optimization_results: Dict[str, Any]) -> Figure:
        """Create a plot showing optimization results.
        
        Args:
            optimization_results: Strategy optimization results
            
        Returns:
            Matplotlib figure with optimization results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        try:
            if 'error' in optimization_results:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, f"Error: {optimization_results['error']}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            results = optimization_results.get('optimization_results', [])
            if not results:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No optimization results available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            # Extract data for plotting
            scores = [r['score'] for r in results]
            confidence_thresholds = [r['parameters'].get('confidence_threshold', 0.7) for r in results]
            position_sizes = [r['parameters'].get('position_size', 0.05) for r in results]
            retraining_freqs = [r['parameters'].get('retraining_frequency', 7) for r in results]
            
            # Plot 1: Score vs Confidence Threshold
            unique_conf_thresh = sorted(list(set(confidence_thresholds)))
            conf_scores = []
            for thresh in unique_conf_thresh:
                thresh_scores = [scores[i] for i, ct in enumerate(confidence_thresholds) if ct == thresh]
                conf_scores.append(np.mean(thresh_scores) if thresh_scores else 0)
            
            ax1.plot(unique_conf_thresh, conf_scores, 'o-', linewidth=2, markersize=6)
            ax1.set_title('Score vs Confidence Threshold', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Confidence Threshold', fontsize=12)
            ax1.set_ylabel('Average Score', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Score vs Position Size
            unique_pos_sizes = sorted(list(set(position_sizes)))
            pos_scores = []
            for pos_size in unique_pos_sizes:
                pos_size_scores = [scores[i] for i, ps in enumerate(position_sizes) if ps == pos_size]
                pos_scores.append(np.mean(pos_size_scores) if pos_size_scores else 0)
            
            ax2.plot(unique_pos_sizes, pos_scores, 's-', linewidth=2, markersize=6, color='orange')
            ax2.set_title('Score vs Position Size', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Position Size', fontsize=12)
            ax2.set_ylabel('Average Score', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Score vs Retraining Frequency
            unique_retrain_freqs = sorted(list(set(retraining_freqs)))
            retrain_scores = []
            for freq in unique_retrain_freqs:
                freq_scores = [scores[i] for i, rf in enumerate(retraining_freqs) if rf == freq]
                retrain_scores.append(np.mean(freq_scores) if freq_scores else 0)
            
            ax3.plot(unique_retrain_freqs, retrain_scores, '^-', linewidth=2, markersize=6, color='green')
            ax3.set_title('Score vs Retraining Frequency', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Retraining Frequency (days)', fontsize=12)
            ax3.set_ylabel('Average Score', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Best Parameters
            best_params = optimization_results.get('best_parameters', {})
            best_score = optimization_results.get('best_score', 0)
            
            if best_params:
                param_names = list(best_params.keys())
                param_values = list(best_params.values())
                
                # Convert values to strings for display
                param_labels = [f"{name}\n{value}" for name, value in zip(param_names, param_values)]
                
                ax4.text(0.5, 0.7, f'Best Score: {best_score:.4f}', 
                        ha='center', va='center', transform=ax4.transAxes, 
                        fontsize=16, fontweight='bold')
                
                ax4.text(0.5, 0.5, 'Best Parameters:', 
                        ha='center', va='center', transform=ax4.transAxes, 
                        fontsize=14, fontweight='bold')
                
                param_text = '\n'.join([f"{name}: {value}" for name, value in best_params.items()])
                ax4.text(0.5, 0.3, param_text, 
                        ha='center', va='center', transform=ax4.transAxes, 
                        fontsize=12)
                
                ax4.set_title('Optimization Results', fontsize=14, fontweight='bold')
                ax4.axis('off')
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating optimization results plot: {e}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        return fig
    
    def save_backtesting_plots(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save all backtesting dashboard plots to files.
        
        Args:
            output_dir: Directory to save plots (defaults to logs directory)
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            output_dir = config.logs_dir / 'dashboard_plots'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        try:
            # Run sample backtest
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            backtest_results = self.run_custom_backtest(start_date, end_date)
            
            if 'error' not in backtest_results:
                # Save backtest results plot
                fig1 = self.create_backtest_results_plot(backtest_results)
                path1 = output_dir / f'backtest_results_{date.today()}.png'
                fig1.savefig(path1, dpi=self.dpi, bbox_inches='tight')
                saved_files['backtest_results'] = path1
                plt.close(fig1)
            
            # Run model comparison
            comparison_results = self.compare_models(start_date, end_date)
            
            if 'error' not in comparison_results:
                # Save model comparison plot
                fig2 = self.create_model_comparison_plot(comparison_results)
                path2 = output_dir / f'model_comparison_{date.today()}.png'
                fig2.savefig(path2, dpi=self.dpi, bbox_inches='tight')
                saved_files['model_comparison'] = path2
                plt.close(fig2)
            
            logger.info(f"Backtesting dashboard plots saved to {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving backtesting plots: {e}")
            return saved_files
    
    def _calculate_trading_metrics(self, predictions_df: pd.DataFrame, 
                                  initial_bankroll: float) -> Dict[str, float]:
        """Calculate trading performance metrics.
        
        Args:
            predictions_df: DataFrame with predictions and actuals
            initial_bankroll: Initial bankroll amount
            
        Returns:
            Dictionary with trading metrics
        """
        try:
            # Simple trading simulation
            bankroll = initial_bankroll
            trades = []
            
            for _, row in predictions_df.iterrows():
                predicted = row['predicted_high']
                actual = row['actual_temperature']
                confidence = row.get('confidence', 0.7)
                
                # Simple strategy: bet on temperature being above/below 80°F
                threshold = 80.0
                
                if predicted > threshold and confidence > 0.7:
                    # Bet that temperature will be above threshold
                    position_size = bankroll * 0.05  # 5% of bankroll
                    won = actual > threshold
                    
                    if won:
                        bankroll += position_size  # Win $1 for every $1 bet
                    else:
                        bankroll -= position_size * 0.5  # Lose 50 cents for every $1 bet
                    
                    trades.append({
                        'position_size': position_size,
                        'won': won,
                        'return': position_size if won else -position_size * 0.5
                    })
            
            if not trades:
                return {
                    'total_return': 0,
                    'num_trades': 0,
                    'win_rate': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            
            # Calculate metrics
            total_return = (bankroll - initial_bankroll) / initial_bankroll
            num_trades = len(trades)
            win_rate = sum(1 for t in trades if t['won']) / num_trades
            
            # Calculate Sharpe ratio (simplified)
            returns = [t['return'] / initial_bankroll for t in trades]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown (simplified)
            running_bankroll = initial_bankroll
            peak_bankroll = initial_bankroll
            max_drawdown = 0
            
            for trade in trades:
                running_bankroll += trade['return']
                if running_bankroll > peak_bankroll:
                    peak_bankroll = running_bankroll
                
                drawdown = (peak_bankroll - running_bankroll) / peak_bankroll
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return {
                'total_return': total_return,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}


if __name__ == "__main__":
    # Demo usage
    dashboard = BacktestingDashboard()
    
    # Run sample backtest
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    print("Running sample backtest...")
    backtest_results = dashboard.run_custom_backtest(start_date, end_date)
    print(f"Backtest completed: {len(backtest_results.get('predictions', []))} predictions")
    
    # Compare models
    print("Comparing models...")
    comparison_results = dashboard.compare_models(start_date, end_date)
    print(f"Model comparison completed")
    
    # Save plots
    saved_files = dashboard.save_backtesting_plots()
    print(f"Plots saved: {saved_files}")
    
    print("Backtesting dashboard demo completed!")