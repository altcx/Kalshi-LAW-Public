"""Tests for the trading dashboard."""

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

from src.dashboard.trading_dashboard import TradingDashboard
from src.utils.data_manager import DataManager


class TestTradingDashboard:
    """Test cases for TradingDashboard."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        mock_dm = Mock(spec=DataManager)
        mock_dm.load_predictions = Mock(return_value=pd.DataFrame())
        return mock_dm
    
    @pytest.fixture
    def mock_recommendation_engine(self):
        """Create a mock recommendation engine."""
        mock_engine = Mock()
        mock_engine.generate_recommendations = Mock(return_value=[])
        return mock_engine
    
    @pytest.fixture
    def mock_contract_analyzer(self):
        """Create a mock contract analyzer."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_temperature_contract = Mock(return_value={
            'expected_value': 0.1,
            'win_probability': 0.7,
            'contract_price': 0.5
        })
        return mock_analyzer
    
    @pytest.fixture
    def mock_position_sizing(self):
        """Create a mock position sizing."""
        mock_sizing = Mock()
        mock_sizing.calculate_kelly_fraction = Mock(return_value=0.05)
        mock_sizing.calculate_position_size = Mock(return_value=500)
        return mock_sizing
    
    @pytest.fixture
    def dashboard(self, mock_data_manager):
        """Create a dashboard instance with mocked dependencies."""
        with patch('src.dashboard.trading_dashboard.RecommendationEngine'), \
             patch('src.dashboard.trading_dashboard.KalshiContractAnalyzer'), \
             patch('src.dashboard.trading_dashboard.PositionSizer'):
            return TradingDashboard(mock_data_manager)
    
    @pytest.fixture
    def sample_predictions_df(self):
        """Create sample predictions dataframe."""
        return pd.DataFrame({
            'date': [pd.Timestamp('2025-01-15')],
            'predicted_high': [78.5],
            'confidence': [85.2],
            'model_contributions': [{}],
            'feature_importance': [{}],
            'created_at': [datetime.now()]
        })
    
    @pytest.fixture
    def sample_recommendations(self):
        """Create sample recommendations."""
        return [
            {
                'contract_type': 'LA_HIGH_TEMP_ABOVE_80F',
                'recommendation': 'BUY',
                'expected_value': 0.15,
                'confidence': 85.2,
                'contract_price': 0.45,
                'reasoning': 'High confidence prediction above threshold'
            },
            {
                'contract_type': 'LA_HIGH_TEMP_BELOW_75F',
                'recommendation': 'HOLD',
                'expected_value': -0.05,
                'confidence': 85.2,
                'contract_price': 0.25,
                'reasoning': 'Low probability of success'
            }
        ]
    
    def test_initialization(self, mock_data_manager):
        """Test dashboard initialization."""
        with patch('src.dashboard.trading_dashboard.RecommendationEngine'), \
             patch('src.dashboard.trading_dashboard.KalshiContractAnalyzer'), \
             patch('src.dashboard.trading_dashboard.PositionSizer'):
            dashboard = TradingDashboard(mock_data_manager)
            assert dashboard.data_manager == mock_data_manager
            assert dashboard.figure_size == (15, 10)
            assert dashboard.dpi == 100
    
    def test_get_latest_recommendations_success(self, dashboard, sample_predictions_df):
        """Test getting latest recommendations successfully."""
        dashboard.data_manager.load_predictions.return_value = sample_predictions_df
        dashboard.recommendation_engine.generate_recommendations.return_value = [
            {'contract_type': 'TEST', 'recommendation': 'BUY'}
        ]
        
        result = dashboard.get_latest_recommendations()
        
        assert len(result) == 1
        assert result[0]['contract_type'] == 'TEST'
        dashboard.recommendation_engine.generate_recommendations.assert_called_once()
    
    def test_get_latest_recommendations_empty_data(self, dashboard):
        """Test getting latest recommendations with empty data."""
        dashboard.data_manager.load_predictions.return_value = pd.DataFrame()
        
        result = dashboard.get_latest_recommendations()
        
        assert result == []
    
    def test_get_latest_recommendations_error(self, dashboard):
        """Test getting latest recommendations with error."""
        dashboard.data_manager.load_predictions.side_effect = Exception("Database error")
        
        result = dashboard.get_latest_recommendations()
        
        assert result == []
    
    def test_analyze_contract_opportunities_success(self, dashboard, sample_predictions_df):
        """Test analyzing contract opportunities successfully."""
        target_date = date(2025, 1, 15)
        dashboard.data_manager.load_predictions.return_value = sample_predictions_df
        
        # Mock contract analyzer
        dashboard.contract_analyzer.analyze_temperature_contract.return_value = {
            'expected_value': 0.1,
            'win_probability': 0.7
        }
        
        result = dashboard.analyze_contract_opportunities(target_date)
        
        assert 'error' not in result
        assert result['target_date'] == target_date
        assert result['predicted_temp'] == 78.5
        assert result['confidence'] == 85.2
        assert 'contracts' in result
    
    def test_analyze_contract_opportunities_no_prediction(self, dashboard):
        """Test analyzing contract opportunities with no prediction data."""
        target_date = date(2025, 1, 15)
        dashboard.data_manager.load_predictions.return_value = pd.DataFrame()
        
        result = dashboard.analyze_contract_opportunities(target_date)
        
        assert 'error' in result
        assert result['error'] == 'No predictions available'
    
    def test_analyze_contract_opportunities_error(self, dashboard):
        """Test analyzing contract opportunities with error."""
        target_date = date(2025, 1, 15)
        dashboard.data_manager.load_predictions.side_effect = Exception("Database error")
        
        result = dashboard.analyze_contract_opportunities(target_date)
        
        assert 'error' in result
    
    def test_calculate_position_sizing_success(self, dashboard, sample_recommendations):
        """Test calculating position sizing successfully."""
        bankroll = 10000
        
        # Mock position sizing methods
        dashboard.position_sizing.calculate_kelly_fraction.return_value = 0.05
        dashboard.position_sizing.calculate_position_size.return_value = 500
        
        result = dashboard.calculate_position_sizing(sample_recommendations, bankroll)
        
        assert len(result) == 2
        
        # Check BUY recommendation has position sizing
        buy_rec = next(r for r in result if r['recommendation'] == 'BUY')
        assert buy_rec['position_size_dollars'] == 500
        assert buy_rec['kelly_fraction'] == 0.05
        assert 'max_loss' in buy_rec
        assert 'max_gain' in buy_rec
        
        # Check HOLD recommendation has zero position
        hold_rec = next(r for r in result if r['recommendation'] == 'HOLD')
        assert hold_rec['position_size_dollars'] == 0
    
    def test_calculate_position_sizing_error(self, dashboard, sample_recommendations):
        """Test calculating position sizing with error."""
        bankroll = 10000
        dashboard.position_sizing.calculate_kelly_fraction.side_effect = Exception("Calculation error")
        
        result = dashboard.calculate_position_sizing(sample_recommendations, bankroll)
        
        # Should return original recommendations on error
        assert len(result) == 2
        assert result == sample_recommendations
    
    def test_create_recommendations_plot_success(self, dashboard, sample_recommendations):
        """Test creating recommendations plot successfully."""
        # Add position sizing data to recommendations
        sized_recs = sample_recommendations.copy()
        sized_recs[0].update({
            'position_size_dollars': 500,
            'max_loss': 225,
            'max_gain': 275
        })
        
        fig = dashboard.create_recommendations_plot(sized_recs)
        
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
    
    def test_create_recommendations_plot_empty_data(self, dashboard):
        """Test creating recommendations plot with empty data."""
        fig = dashboard.create_recommendations_plot([])
        
        assert fig is not None
        # Should still create figure even with empty data
    
    def test_create_contract_analysis_plot_success(self, dashboard):
        """Test creating contract analysis plot successfully."""
        contract_analysis = {
            'predicted_temp': 78.5,
            'confidence': 85.2,
            'contracts': {
                'above_80': {'win_probability': 0.3, 'expected_value': 0.1},
                'below_80': {'win_probability': 0.7, 'expected_value': -0.05},
                'above_85': {'win_probability': 0.1, 'expected_value': 0.2},
                'below_85': {'win_probability': 0.9, 'expected_value': -0.1}
            }
        }
        
        fig = dashboard.create_contract_analysis_plot(contract_analysis)
        
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
    
    def test_create_contract_analysis_plot_error_data(self, dashboard):
        """Test creating contract analysis plot with error data."""
        contract_analysis = {'error': 'No data available'}
        
        fig = dashboard.create_contract_analysis_plot(contract_analysis)
        
        assert fig is not None
    
    def test_generate_trading_report_success(self, dashboard, sample_recommendations):
        """Test generating trading report successfully."""
        target_date = date(2025, 1, 15)
        bankroll = 10000
        
        # Mock methods
        with patch.object(dashboard, 'get_latest_recommendations') as mock_get_recs, \
             patch.object(dashboard, 'calculate_position_sizing') as mock_calc_sizing, \
             patch.object(dashboard, 'analyze_contract_opportunities') as mock_analyze:
            
            mock_get_recs.return_value = sample_recommendations
            mock_calc_sizing.return_value = sample_recommendations
            mock_analyze.return_value = {'predicted_temp': 78.5}
            
            report = dashboard.generate_trading_report(target_date, bankroll)
            
            assert report['date'] == target_date
            assert report['bankroll'] == bankroll
            assert 'recommendations' in report
            assert 'contract_analysis' in report
            assert 'risk_metrics' in report
            assert 'summary' in report
    
    def test_generate_trading_report_error(self, dashboard):
        """Test generating trading report with error."""
        target_date = date(2025, 1, 15)
        
        with patch.object(dashboard, 'get_latest_recommendations') as mock_get_recs:
            mock_get_recs.side_effect = Exception("Test error")
            
            report = dashboard.generate_trading_report(target_date)
            
            assert report['date'] == target_date
            assert 'error' in report
    
    def test_save_trading_plots_success(self, dashboard, tmp_path):
        """Test saving trading plots successfully."""
        output_dir = tmp_path / "plots"
        
        # Mock plot creation methods and data methods
        mock_fig = MagicMock()
        
        with patch.object(dashboard, 'get_latest_recommendations') as mock_get_recs, \
             patch.object(dashboard, 'calculate_position_sizing') as mock_calc_sizing, \
             patch.object(dashboard, 'analyze_contract_opportunities') as mock_analyze, \
             patch.object(dashboard, 'create_recommendations_plot') as mock_create_recs, \
             patch.object(dashboard, 'create_contract_analysis_plot') as mock_create_analysis, \
             patch('matplotlib.pyplot.close'):
            
            mock_get_recs.return_value = [{'recommendation': 'BUY'}]
            mock_calc_sizing.return_value = [{'recommendation': 'BUY'}]
            mock_analyze.return_value = {'predicted_temp': 78.5}
            mock_create_recs.return_value = mock_fig
            mock_create_analysis.return_value = mock_fig
            
            result = dashboard.save_trading_plots(output_dir)
            
            assert len(result) == 2
            assert 'trading_recommendations' in result
            assert 'contract_analysis' in result
            
            # Check that savefig was called
            assert mock_fig.savefig.call_count == 2
    
    def test_save_trading_plots_error(self, dashboard, tmp_path):
        """Test saving trading plots with error."""
        output_dir = tmp_path / "plots"
        
        with patch.object(dashboard, 'get_latest_recommendations') as mock_get_recs:
            mock_get_recs.side_effect = Exception("Plot creation error")
            
            result = dashboard.save_trading_plots(output_dir)
            
            # Should return empty dict on error
            assert result == {}


if __name__ == "__main__":
    pytest.main([__file__])