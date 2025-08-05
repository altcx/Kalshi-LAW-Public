"""Tests for the prediction dashboard."""

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

from src.dashboard.prediction_dashboard import PredictionDashboard
from src.utils.data_manager import DataManager


class TestPredictionDashboard:
    """Test cases for PredictionDashboard."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        mock_dm = Mock(spec=DataManager)
        # Add the methods that the dashboard expects
        mock_dm.load_predictions = Mock(return_value=pd.DataFrame())
        mock_dm.load_actual_temperatures = Mock(return_value=pd.DataFrame())
        mock_dm.load_source_data = Mock(return_value=pd.DataFrame())
        return mock_dm
    
    @pytest.fixture
    def dashboard(self, mock_data_manager):
        """Create a dashboard instance with mocked dependencies."""
        with patch('src.dashboard.prediction_dashboard.PerformanceTracker'):
            return PredictionDashboard(mock_data_manager)
    
    @pytest.fixture
    def sample_predictions_df(self):
        """Create sample predictions dataframe."""
        dates = pd.date_range(start='2025-01-01', periods=7, freq='D')
        return pd.DataFrame({
            'date': dates,
            'predicted_high': [75.2, 78.1, 72.5, 80.3, 77.8, 74.6, 76.9],
            'confidence': [85.5, 92.1, 78.3, 88.7, 81.2, 86.4, 83.8],
            'model_contributions': [{}] * 7,
            'feature_importance': [{}] * 7,
            'created_at': [datetime.now()] * 7
        })
    
    @pytest.fixture
    def sample_actuals_df(self):
        """Create sample actual temperatures dataframe."""
        dates = pd.date_range(start='2025-01-01', periods=7, freq='D')
        return pd.DataFrame({
            'date': dates,
            'actual_high': [74.8, 79.2, 71.9, 81.1, 76.5, 75.2, 77.3]
        })
    
    def test_initialization(self, mock_data_manager):
        """Test dashboard initialization."""
        with patch('src.dashboard.prediction_dashboard.PerformanceTracker'):
            dashboard = PredictionDashboard(mock_data_manager)
            assert dashboard.data_manager == mock_data_manager
            assert dashboard.figure_size == (15, 10)
            assert dashboard.dpi == 100
    
    def test_get_latest_prediction_success(self, dashboard, sample_predictions_df):
        """Test getting latest prediction successfully."""
        dashboard.data_manager.load_predictions.return_value = sample_predictions_df
        
        result = dashboard.get_latest_prediction()
        
        assert result is not None
        assert result['date'] == sample_predictions_df.iloc[-1]['date']
        assert result['predicted_high'] == sample_predictions_df.iloc[-1]['predicted_high']
        assert result['confidence'] == sample_predictions_df.iloc[-1]['confidence']
    
    def test_get_latest_prediction_empty_data(self, dashboard):
        """Test getting latest prediction with empty data."""
        dashboard.data_manager.load_predictions.return_value = pd.DataFrame()
        
        result = dashboard.get_latest_prediction()
        
        assert result is None
    
    def test_get_latest_prediction_error(self, dashboard):
        """Test getting latest prediction with error."""
        dashboard.data_manager.load_predictions.side_effect = Exception("Database error")
        
        result = dashboard.get_latest_prediction()
        
        assert result is None
    
    def test_get_source_contributions_success(self, dashboard):
        """Test getting source contributions successfully."""
        target_date = date(2025, 1, 15)
        
        # Mock source data
        sample_data = pd.DataFrame({
            'date': [pd.Timestamp(target_date)],
            'predicted_high': [75.5]
        })
        
        dashboard.data_manager.load_source_data.return_value = sample_data
        
        result = dashboard.get_source_contributions(target_date)
        
        # Should call load_source_data for each source
        assert dashboard.data_manager.load_source_data.call_count == 5
        assert isinstance(result, dict)
    
    def test_get_source_contributions_empty_data(self, dashboard):
        """Test getting source contributions with empty data."""
        target_date = date(2025, 1, 15)
        dashboard.data_manager.load_source_data.return_value = pd.DataFrame()
        
        result = dashboard.get_source_contributions(target_date)
        
        assert result == {}
    
    def test_create_prediction_summary_plot_success(self, dashboard, sample_predictions_df, sample_actuals_df):
        """Test creating prediction summary plot successfully."""
        dashboard.data_manager.load_predictions.return_value = sample_predictions_df
        dashboard.data_manager.load_actual_temperatures.return_value = sample_actuals_df
        
        fig = dashboard.create_prediction_summary_plot(days_back=7)
        
        assert fig is not None
        assert len(fig.axes) == 2  # Should have 2 subplots
    
    def test_create_prediction_summary_plot_empty_data(self, dashboard):
        """Test creating prediction summary plot with empty data."""
        dashboard.data_manager.load_predictions.return_value = pd.DataFrame()
        dashboard.data_manager.load_actual_temperatures.return_value = pd.DataFrame()
        
        fig = dashboard.create_prediction_summary_plot(days_back=7)
        
        assert fig is not None
        # Should still create figure even with empty data
    
    def test_create_source_contributions_plot_success(self, dashboard):
        """Test creating source contributions plot successfully."""
        target_date = date(2025, 1, 15)
        
        # Mock get_source_contributions to return sample data
        with patch.object(dashboard, 'get_source_contributions') as mock_get_contrib:
            mock_get_contrib.return_value = {
                'nws': 75.2,
                'openweather': 76.1,
                'tomorrow': 74.8
            }
            
            fig = dashboard.create_source_contributions_plot(target_date)
            
            assert fig is not None
            assert len(fig.axes) == 2  # Should have 2 subplots
    
    def test_create_source_contributions_plot_empty_data(self, dashboard):
        """Test creating source contributions plot with empty data."""
        target_date = date(2025, 1, 15)
        
        with patch.object(dashboard, 'get_source_contributions') as mock_get_contrib:
            mock_get_contrib.return_value = {}
            
            fig = dashboard.create_source_contributions_plot(target_date)
            
            assert fig is not None
    
    def test_create_accuracy_trends_plot_success(self, dashboard):
        """Test creating accuracy trends plot successfully."""
        # Mock performance tracker
        mock_performance_data = {
            'daily_accuracy': {
                date(2025, 1, 1): 0.85,
                date(2025, 1, 2): 0.92,
                date(2025, 1, 3): 0.78
            },
            'source_performance': {
                'nws': {'accuracy_7d': 0.88},
                'openweather': {'accuracy_7d': 0.82}
            },
            'error_distribution': [-2.1, -1.5, 0.3, 1.2, -0.8, 2.3],
            'confidence_calibration': {
                70: 68,
                80: 82,
                90: 89
            }
        }
        
        dashboard.performance_tracker.get_performance_summary.return_value = mock_performance_data
        
        fig = dashboard.create_accuracy_trends_plot(days_back=30)
        
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
    
    def test_create_accuracy_trends_plot_empty_data(self, dashboard):
        """Test creating accuracy trends plot with empty data."""
        dashboard.performance_tracker.get_performance_summary.return_value = None
        
        fig = dashboard.create_accuracy_trends_plot(days_back=30)
        
        assert fig is not None
    
    def test_generate_prediction_report_success(self, dashboard):
        """Test generating prediction report successfully."""
        target_date = date(2025, 1, 15)
        
        # Mock dependencies
        mock_prediction = {
            'date': target_date,
            'predicted_high': 75.5,
            'confidence': 85.2
        }
        
        mock_contributions = {
            'nws': 75.2,
            'openweather': 76.1
        }
        
        mock_performance = {
            'accuracy_7d': 0.88,
            'rmse_7d': 2.1
        }
        
        with patch.object(dashboard, 'get_latest_prediction') as mock_get_pred, \
             patch.object(dashboard, 'get_source_contributions') as mock_get_contrib:
            
            mock_get_pred.return_value = mock_prediction
            mock_get_contrib.return_value = mock_contributions
            dashboard.performance_tracker.get_performance_summary.return_value = mock_performance
            
            report = dashboard.generate_prediction_report(target_date)
            
            assert report['date'] == target_date
            assert 'generated_at' in report
            assert report['prediction'] == mock_prediction
            assert report['source_contributions'] == mock_contributions
            assert report['performance_metrics'] == mock_performance
            assert len(report['recommendations']) > 0
    
    def test_generate_prediction_report_error(self, dashboard):
        """Test generating prediction report with error."""
        target_date = date(2025, 1, 15)
        
        with patch.object(dashboard, 'get_latest_prediction') as mock_get_pred:
            mock_get_pred.side_effect = Exception("Test error")
            
            report = dashboard.generate_prediction_report(target_date)
            
            assert report['date'] == target_date
            assert 'error' in report
    
    def test_save_dashboard_plots_success(self, dashboard, tmp_path):
        """Test saving dashboard plots successfully."""
        output_dir = tmp_path / "plots"
        
        # Mock plot creation methods to return mock figures
        mock_fig = MagicMock()
        
        with patch.object(dashboard, 'create_prediction_summary_plot') as mock_summary, \
             patch.object(dashboard, 'create_source_contributions_plot') as mock_contrib, \
             patch.object(dashboard, 'create_accuracy_trends_plot') as mock_trends, \
             patch('matplotlib.pyplot.close'):
            
            mock_summary.return_value = mock_fig
            mock_contrib.return_value = mock_fig
            mock_trends.return_value = mock_fig
            
            result = dashboard.save_dashboard_plots(output_dir)
            
            assert len(result) == 3
            assert 'prediction_summary' in result
            assert 'source_contributions' in result
            assert 'accuracy_trends' in result
            
            # Check that savefig was called
            assert mock_fig.savefig.call_count == 3
    
    def test_save_dashboard_plots_error(self, dashboard, tmp_path):
        """Test saving dashboard plots with error."""
        output_dir = tmp_path / "plots"
        
        with patch.object(dashboard, 'create_prediction_summary_plot') as mock_summary:
            mock_summary.side_effect = Exception("Plot creation error")
            
            result = dashboard.save_dashboard_plots(output_dir)
            
            # Should return empty dict on error
            assert result == {}


if __name__ == "__main__":
    pytest.main([__file__])