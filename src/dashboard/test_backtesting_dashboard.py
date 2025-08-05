"""Tests for the backtesting dashboard."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Set matplotlib to use non-GUI backend for testing
import matplotlib
matplotlib.use('Agg')

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from src.dashboard.backtesting_dashboard import BacktestingDashboard
from src.utils.data_manager import DataManager


class TestBacktestingDashboard:
    """Test cases for BacktestingDashboard."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        mock_dm = Mock(spec=DataManager)
        return mock_dm
    
    @pytest.fixture
    def dashboard(self, mock_data_manager):
        """Create a dashboard instance with mocked dependencies."""
        with patch('src.dashboard.backtesting_dashboard.BacktestingFramework'), \
             patch('src.dashboard.backtesting_dashboard.ModelComparator'), \
             patch('src.dashboard.backtesting_dashboard.PerformanceMetricsCalculator'):
            return BacktestingDashboard(mock_data_manager)
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Create sample backtest results."""
        return {
            'predictions': [
                {
                    'date': '2025-01-01',
                    'predicted_high': 75.2,
                    'actual_temperature': 74.8,
                    'confidence': 0.85
                },
                {
                    'date': '2025-01-02',
                    'predicted_high': 78.1,
                    'actual_temperature': 79.2,
                    'confidence': 0.92
                }
            ],
            'performance_metrics': {
                'mae': 1.5,
                'rmse': 2.1,
                'accuracy_3f': 0.85,
                'accuracy_5f': 0.95
            },
            'trading_metrics': {
                'total_return': 0.15,
                'num_trades': 10,
                'win_rate': 0.7,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.05
            }
        }
    
    def test_initialization(self, mock_data_manager):
        """Test dashboard initialization."""
        with patch('src.dashboard.backtesting_dashboard.BacktestingFramework'), \
             patch('src.dashboard.backtesting_dashboard.ModelComparator'), \
             patch('src.dashboard.backtesting_dashboard.PerformanceMetricsCalculator'):
            dashboard = BacktestingDashboard(mock_data_manager)
            assert dashboard.data_manager == mock_data_manager
            assert dashboard.figure_size == (15, 10)
            assert dashboard.dpi == 100
    
    def test_run_custom_backtest_success(self, dashboard):
        """Test running custom backtest successfully."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 31)
        
        # Mock backtesting framework
        mock_backtest_result = {
            'predictions': [
                {'date': '2025-01-01', 'predicted_high': 75.0, 'actual_temperature': 74.5}
            ]
        }
        dashboard.backtesting_framework.run_backtest.return_value = mock_backtest_result
        
        # Mock performance metrics
        dashboard.performance_metrics.calculate_accuracy_metrics.return_value = {
            'mae': 1.0, 'rmse': 1.5, 'accuracy_3f': 0.9
        }
        
        result = dashboard.run_custom_backtest(start_date, end_date)
        
        assert 'error' not in result
        assert result['start_date'] == start_date
        assert result['end_date'] == end_date
        assert 'predictions' in result
        assert 'performance_metrics' in result
    
    def test_run_custom_backtest_error(self, dashboard):
        """Test running custom backtest with error."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 31)
        
        dashboard.backtesting_framework.run_backtest.side_effect = Exception("Backtest error")
        
        result = dashboard.run_custom_backtest(start_date, end_date)
        
        assert 'error' in result
        assert result['start_date'] == start_date
        assert result['end_date'] == end_date
    
    def test_compare_models_success(self, dashboard):
        """Test comparing models successfully."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 31)
        models = ['ensemble', 'xgboost']
        
        mock_comparison_result = {
            'model_results': {
                'ensemble': {'accuracy_3f': 0.85, 'mae': 1.5},
                'xgboost': {'accuracy_3f': 0.82, 'mae': 1.7}
            }
        }
        dashboard.model_comparison.compare_models.return_value = mock_comparison_result
        
        result = dashboard.compare_models(start_date, end_date, models)
        
        assert 'error' not in result
        assert 'model_results' in result
        dashboard.model_comparison.compare_models.assert_called_once_with(
            models=models, start_date=start_date, end_date=end_date
        )
    
    def test_compare_models_error(self, dashboard):
        """Test comparing models with error."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 31)
        
        dashboard.model_comparison.compare_models.side_effect = Exception("Comparison error")
        
        result = dashboard.compare_models(start_date, end_date)
        
        assert 'error' in result
    
    def test_optimize_strategy_success(self, dashboard):
        """Test strategy optimization successfully."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 31)
        
        # Mock run_custom_backtest to return different results for different parameters
        def mock_backtest(start, end, params):
            return {
                'trading_metrics': {
                    'sharpe_ratio': params.get('confidence_threshold', 0.7) * 2,
                    'total_return': params.get('position_size', 0.05) * 10
                },
                'performance_metrics': {'mae': 1.5}
            }
        
        with patch.object(dashboard, 'run_custom_backtest', side_effect=mock_backtest):
            result = dashboard.optimize_strategy(start_date, end_date)
            
            assert 'error' not in result
            assert 'best_parameters' in result
            assert 'best_score' in result
            assert 'optimization_results' in result
    
    def test_optimize_strategy_error(self, dashboard):
        """Test strategy optimization with error."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 31)
        
        with patch.object(dashboard, 'run_custom_backtest', side_effect=Exception("Optimization error")):
            result = dashboard.optimize_strategy(start_date, end_date)
            
            assert 'error' in result
    
    def test_run_what_if_analysis_success(self, dashboard):
        """Test what-if analysis successfully."""
        base_scenario = {
            'start_date': date(2025, 1, 1),
            'end_date': date(2025, 1, 31),
            'parameters': {'confidence_threshold': 0.7}
        }
        
        variations = [
            {'name': 'High Confidence', 'parameter_changes': {'confidence_threshold': 0.9}}
        ]
        
        mock_backtest_result = {'trading_metrics': {'total_return': 0.1}}
        
        with patch.object(dashboard, 'run_custom_backtest', return_value=mock_backtest_result):
            result = dashboard.run_what_if_analysis(base_scenario, variations)
            
            assert 'error' not in result
            assert 'results' in result
            assert len(result['results']) == 2  # Base + 1 variation
    
    def test_run_what_if_analysis_error(self, dashboard):
        """Test what-if analysis with error."""
        base_scenario = {
            'start_date': date(2025, 1, 1),
            'end_date': date(2025, 1, 31),
            'parameters': {}
        }
        
        with patch.object(dashboard, 'run_custom_backtest', side_effect=Exception("What-if error")):
            result = dashboard.run_what_if_analysis(base_scenario, [])
            
            assert 'error' in result
    
    def test_create_backtest_results_plot_success(self, dashboard, sample_backtest_results):
        """Test creating backtest results plot successfully."""
        fig = dashboard.create_backtest_results_plot(sample_backtest_results)
        
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
    
    def test_create_backtest_results_plot_error_data(self, dashboard):
        """Test creating backtest results plot with error data."""
        error_results = {'error': 'No data available'}
        
        fig = dashboard.create_backtest_results_plot(error_results)
        
        assert fig is not None
    
    def test_create_model_comparison_plot_success(self, dashboard):
        """Test creating model comparison plot successfully."""
        comparison_results = {
            'model_results': {
                'ensemble': {'accuracy_3f': 0.85, 'mae': 1.5, 'rmse': 2.0},
                'xgboost': {'accuracy_3f': 0.82, 'mae': 1.7, 'rmse': 2.2}
            }
        }
        
        fig = dashboard.create_model_comparison_plot(comparison_results)
        
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
    
    def test_create_model_comparison_plot_error_data(self, dashboard):
        """Test creating model comparison plot with error data."""
        error_results = {'error': 'No comparison data'}
        
        fig = dashboard.create_model_comparison_plot(error_results)
        
        assert fig is not None
    
    def test_create_optimization_results_plot_success(self, dashboard):
        """Test creating optimization results plot successfully."""
        optimization_results = {
            'optimization_results': [
                {
                    'parameters': {'confidence_threshold': 0.7, 'position_size': 0.05, 'retraining_frequency': 7},
                    'score': 1.2
                },
                {
                    'parameters': {'confidence_threshold': 0.8, 'position_size': 0.05, 'retraining_frequency': 7},
                    'score': 1.5
                }
            ],
            'best_parameters': {'confidence_threshold': 0.8, 'position_size': 0.05},
            'best_score': 1.5
        }
        
        fig = dashboard.create_optimization_results_plot(optimization_results)
        
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
    
    def test_create_optimization_results_plot_error_data(self, dashboard):
        """Test creating optimization results plot with error data."""
        error_results = {'error': 'No optimization data'}
        
        fig = dashboard.create_optimization_results_plot(error_results)
        
        assert fig is not None
    
    def test_save_backtesting_plots_success(self, dashboard, tmp_path):
        """Test saving backtesting plots successfully."""
        output_dir = tmp_path / "plots"
        
        # Mock plot creation methods and backtest methods
        mock_fig = MagicMock()
        mock_backtest_result = {'predictions': []}
        mock_comparison_result = {'model_results': {}}
        
        with patch.object(dashboard, 'run_custom_backtest', return_value=mock_backtest_result), \
             patch.object(dashboard, 'compare_models', return_value=mock_comparison_result), \
             patch.object(dashboard, 'create_backtest_results_plot', return_value=mock_fig), \
             patch.object(dashboard, 'create_model_comparison_plot', return_value=mock_fig), \
             patch('matplotlib.pyplot.close'):
            
            result = dashboard.save_backtesting_plots(output_dir)
            
            assert len(result) == 2
            assert 'backtest_results' in result
            assert 'model_comparison' in result
            
            # Check that savefig was called
            assert mock_fig.savefig.call_count == 2
    
    def test_save_backtesting_plots_error(self, dashboard, tmp_path):
        """Test saving backtesting plots with error."""
        output_dir = tmp_path / "plots"
        
        with patch.object(dashboard, 'run_custom_backtest', side_effect=Exception("Plot error")):
            result = dashboard.save_backtesting_plots(output_dir)
            
            # Should return empty dict on error
            assert result == {}
    
    def test_calculate_trading_metrics_success(self, dashboard):
        """Test calculating trading metrics successfully."""
        predictions_df = pd.DataFrame({
            'predicted_high': [75.0, 82.0, 78.0],
            'actual_temperature': [74.5, 83.2, 77.8],
            'confidence': [0.8, 0.9, 0.7]
        })
        
        result = dashboard._calculate_trading_metrics(predictions_df, 10000)
        
        assert 'total_return' in result
        assert 'num_trades' in result
        assert 'win_rate' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
    
    def test_calculate_trading_metrics_empty_data(self, dashboard):
        """Test calculating trading metrics with empty data."""
        predictions_df = pd.DataFrame()
        
        result = dashboard._calculate_trading_metrics(predictions_df, 10000)
        
        assert result['num_trades'] == 0
        assert result['total_return'] == 0
        assert result['win_rate'] == 0


if __name__ == "__main__":
    pytest.main([__file__])