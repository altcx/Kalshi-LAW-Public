#!/usr/bin/env python3
"""Tests for the daily scheduler automation system."""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.automation.daily_scheduler import DailyScheduler
from src.automation.error_handler import ErrorHandler, retry_with_backoff, ErrorSeverity
from src.data_collection.base_client import WeatherData


class TestDailyScheduler(unittest.TestCase):
    """Test cases for the daily scheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = DailyScheduler()
        
        # Mock dependencies
        self.scheduler.data_manager = Mock()
        self.scheduler.weather_collector = Mock()
        self.scheduler.actual_temp_collector = Mock()
    
    def test_morning_data_collection_success(self):
        """Test successful morning data collection."""
        # Mock forecast data
        mock_weather_data = WeatherData(
            date=date.today(),
            forecast_date=date.today(),
            predicted_high=75.0,
            predicted_low=55.0,
            humidity=65.0,
            pressure=1013.2,
            wind_speed=8.5,
            data_quality_score=0.95
        )
        
        mock_forecasts = {
            'nws': [mock_weather_data],
            'openweather': [mock_weather_data],
            'tomorrow': []  # Empty to test handling
        }
        
        self.scheduler.weather_collector.get_all_forecasts.return_value = mock_forecasts
        self.scheduler.data_manager.append_daily_data.return_value = True
        
        # Execute morning collection
        result = self.scheduler.morning_data_collection()
        
        # Verify results
        self.assertTrue(result)
        self.scheduler.weather_collector.get_all_forecasts.assert_called_once_with(days=7)
        
        # Verify data manager was called for sources with data
        expected_calls = 2  # nws and openweather
        self.assertEqual(self.scheduler.data_manager.append_daily_data.call_count, expected_calls)
    
    def test_morning_data_collection_failure(self):
        """Test morning data collection with API failures."""
        # Mock API failure
        self.scheduler.weather_collector.get_all_forecasts.side_effect = Exception("API Error")
        
        # Execute morning collection
        result = self.scheduler.morning_data_collection()
        
        # Verify failure handling
        self.assertFalse(result)
    
    def test_evening_temperature_collection_success(self):
        """Test successful evening temperature collection."""
        yesterday = date.today() - timedelta(days=1)
        
        # Mock successful temperature collection
        self.scheduler.actual_temp_collector.collect_daily_high_temperature.return_value = 78.5
        
        # Execute evening collection
        result = self.scheduler.evening_actual_temperature_collection()
        
        # Verify results
        self.assertTrue(result)
        self.scheduler.actual_temp_collector.collect_daily_high_temperature.assert_called_once_with(yesterday)
    
    def test_evening_temperature_collection_failure(self):
        """Test evening temperature collection failure."""
        # Mock failed temperature collection
        self.scheduler.actual_temp_collector.collect_daily_high_temperature.return_value = None
        
        # Execute evening collection
        result = self.scheduler.evening_actual_temperature_collection()
        
        # Verify failure handling
        self.assertFalse(result)
    
    def test_health_check(self):
        """Test system health check."""
        # Mock health check dependencies
        self.scheduler.weather_collector.test_all_connections.return_value = {
            'nws': True,
            'openweather': True,
            'tomorrow': False
        }
        
        self.scheduler.data_manager.get_data_quality_summary.return_value = {
            'avg_quality_score': 0.85,
            'total_records': 100
        }
        
        self.scheduler.actual_temp_collector.get_temperature_collection_status.return_value = {
            'collection_rate': 0.9,
            'total_collected': 9,
            'total_expected': 10
        }
        
        # Execute health check
        health_status = self.scheduler.health_check()
        
        # Verify health status
        self.assertIsInstance(health_status, dict)
        self.assertIn('timestamp', health_status)
        self.assertIn('api_connections', health_status)
        self.assertIn('data_quality', health_status)
        self.assertIn('collection_status', health_status)
        self.assertIn('overall_health', health_status)
    
    def test_retry_failed_collection_morning(self):
        """Test retry logic for failed morning collection."""
        # Mock current time to be in morning retry window
        with patch('src.automation.daily_scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 15, 8, 0, 0)  # 8 AM
            
            # Mock failed previous run
            self.scheduler.last_morning_run = None
            
            # Mock successful retry
            self.scheduler.morning_data_collection = Mock(return_value=True)
            
            # Execute retry
            self.scheduler.retry_failed_collection(max_retries=1, backoff_minutes=1)
            
            # Verify retry was attempted
            self.scheduler.morning_data_collection.assert_called_once()
    
    def test_retry_failed_collection_evening(self):
        """Test retry logic for failed evening collection."""
        # Mock current time to be in evening retry window
        with patch('src.automation.daily_scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 15, 20, 0, 0)  # 8 PM
            
            # Mock failed previous run
            self.scheduler.last_evening_run = None
            
            # Mock successful retry
            self.scheduler.evening_actual_temperature_collection = Mock(return_value=True)
            
            # Execute retry
            self.scheduler.retry_failed_collection(max_retries=1, backoff_minutes=1)
            
            # Verify retry was attempted
            self.scheduler.evening_actual_temperature_collection.assert_called_once()
    
    @patch('src.automation.daily_scheduler.schedule')
    def test_setup_schedule(self, mock_schedule):
        """Test schedule setup."""
        # Mock schedule methods
        mock_schedule.every.return_value.day.at.return_value.do = Mock()
        mock_schedule.every.return_value.hours.do = Mock()
        
        # Execute schedule setup
        self.scheduler.setup_schedule()
        
        # Verify schedule was configured
        self.assertTrue(mock_schedule.every.called)
    
    def test_run_once_morning(self):
        """Test running morning task once."""
        self.scheduler.morning_data_collection = Mock(return_value=True)
        
        result = self.scheduler.run_once("morning")
        
        self.assertTrue(result)
        self.scheduler.morning_data_collection.assert_called_once()
    
    def test_run_once_evening(self):
        """Test running evening task once."""
        self.scheduler.evening_actual_temperature_collection = Mock(return_value=True)
        
        result = self.scheduler.run_once("evening")
        
        self.assertTrue(result)
        self.scheduler.evening_actual_temperature_collection.assert_called_once()
    
    def test_run_once_health(self):
        """Test running health check once."""
        self.scheduler.health_check = Mock(return_value={'overall_health': 'healthy'})
        
        result = self.scheduler.run_once("health")
        
        self.assertTrue(result)
        self.scheduler.health_check.assert_called_once()
    
    def test_run_once_invalid_task(self):
        """Test running invalid task."""
        result = self.scheduler.run_once("invalid_task")
        
        self.assertFalse(result)


class TestErrorHandler(unittest.TestCase):
    """Test cases for the error handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_log_error(self):
        """Test error logging."""
        test_error = ValueError("Test error")
        
        self.error_handler.log_error(
            test_error,
            "test_context",
            ErrorSeverity.MEDIUM,
            {"additional": "info"}
        )
        
        # Verify error was logged
        self.assertEqual(len(self.error_handler.error_history), 1)
        
        error_record = self.error_handler.error_history[0]
        self.assertEqual(error_record['error_type'], 'ValueError')
        self.assertEqual(error_record['error_message'], 'Test error')
        self.assertEqual(error_record['context'], 'test_context')
        self.assertEqual(error_record['severity'], 'medium')
        self.assertEqual(error_record['additional_info']['additional'], 'info')
    
    def test_get_error_summary(self):
        """Test error summary generation."""
        # Add some test errors
        for i in range(5):
            self.error_handler.log_error(
                ValueError(f"Error {i}"),
                f"context_{i % 2}",
                ErrorSeverity.LOW if i % 2 == 0 else ErrorSeverity.MEDIUM
            )
        
        summary = self.error_handler.get_error_summary(hours=24)
        
        # Verify summary
        self.assertEqual(summary['total_errors'], 5)
        self.assertEqual(summary['severity_breakdown']['low'], 3)
        self.assertEqual(summary['severity_breakdown']['medium'], 2)
        self.assertIn('context_0', summary['context_breakdown'])
        self.assertIn('context_1', summary['context_breakdown'])
    
    def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def successful_function():
            return "success"
        
        result = successful_function()
        self.assertEqual(result, "success")
    
    def test_retry_decorator_eventual_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = eventually_successful_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
    
    def test_retry_decorator_final_failure(self):
        """Test retry decorator with final failure."""
        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def always_failing_function():
            raise ValueError("Persistent error")
        
        with self.assertRaises(ValueError):
            always_failing_function()


class TestIntegration(unittest.TestCase):
    """Integration tests for the daily scheduler system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.scheduler = DailyScheduler()
    
    @patch('src.automation.daily_scheduler.WeatherDataCollector')
    @patch('src.automation.daily_scheduler.ActualTemperatureCollector')
    @patch('src.automation.daily_scheduler.DataManager')
    def test_full_morning_collection_workflow(self, mock_data_manager, 
                                            mock_temp_collector, mock_weather_collector):
        """Test complete morning collection workflow."""
        # Setup mocks
        mock_weather_instance = Mock()
        mock_weather_collector.return_value = mock_weather_instance
        
        mock_temp_instance = Mock()
        mock_temp_collector.return_value = mock_temp_instance
        
        mock_data_instance = Mock()
        mock_data_manager.return_value = mock_data_instance
        
        # Mock successful forecast collection
        mock_weather_data = WeatherData(
            date=date.today(),
            forecast_date=date.today(),
            predicted_high=75.0,
            predicted_low=55.0,
            data_quality_score=0.95
        )
        
        mock_weather_instance.get_all_forecasts.return_value = {
            'nws': [mock_weather_data],
            'openweather': [mock_weather_data]
        }
        
        mock_data_instance.append_daily_data.return_value = True
        
        # Create new scheduler instance (will use mocked classes)
        scheduler = DailyScheduler()
        
        # Execute morning collection
        result = scheduler.morning_data_collection()
        
        # Verify workflow
        self.assertTrue(result)
        mock_weather_instance.get_all_forecasts.assert_called_once_with(days=7)
        self.assertEqual(mock_data_instance.append_daily_data.call_count, 2)


def run_scheduler_tests():
    """Run all scheduler tests."""
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestDailyScheduler))
    test_suite.addTests(loader.loadTestsFromTestCase(TestErrorHandler))
    test_suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_scheduler_tests()
    sys.exit(0 if success else 1)