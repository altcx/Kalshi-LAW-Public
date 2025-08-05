"""Test historical data loader functionality."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from src.backtesting.historical_data_loader import HistoricalDataLoader
from src.backtesting.performance_metrics import PerformanceMetricsCalculator, PredictionResult
from src.utils.data_manager import DataManager


def test_historical_data_loader_initialization():
    """Test that HistoricalDataLoader initializes correctly."""
    loader = HistoricalDataLoader()
    assert loader is not None
    assert loader.data_manager is not None
    assert loader.feature_pipeline is not None


def test_get_available_date_range():
    """Test getting available date range from data."""
    loader = HistoricalDataLoader()
    start_date, end_date = loader.get_available_date_range()
    
    # Should return dates or None
    if start_date is not None:
        assert isinstance(start_date, date)
    if end_date is not None:
        assert isinstance(end_date, date)
    
    # If both exist, start should be before end
    if start_date and end_date:
        assert start_date <= end_date


def test_create_walk_forward_splits():
    """Test creating walk-forward splits."""
    loader = HistoricalDataLoader()
    
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    
    splits = loader.create_walk_forward_splits(
        start_date, end_date, 
        train_window_days=90, 
        test_window_days=7, 
        step_days=7
    )
    
    assert len(splits) > 0
    
    # Check first split
    first_split = splits[0]
    assert 'split_id' in first_split
    assert 'train_start' in first_split
    assert 'train_end' in first_split
    assert 'test_start' in first_split
    assert 'test_end' in first_split


def test_performance_metrics_basic():
    """Test basic performance metrics calculation."""
    calculator = PerformanceMetricsCalculator()
    
    # Create sample predictions
    predictions = [
        PredictionResult(date(2024, 1, 1), 75.0, 73.0, 0.8),
        PredictionResult(date(2024, 1, 2), 78.0, 76.0, 0.9),
        PredictionResult(date(2024, 1, 3), 72.0, 74.0, 0.7),
    ]
    
    metrics = calculator.calculate_basic_metrics(predictions)
    
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'accuracy_within_3f' in metrics
    assert metrics['total_predictions'] == 3
    assert metrics['mae'] > 0


if __name__ == "__main__":
    test_historical_data_loader_initialization()
    test_get_available_date_range()
    test_create_walk_forward_splits()
    test_performance_metrics_basic()
    print("All tests passed!")