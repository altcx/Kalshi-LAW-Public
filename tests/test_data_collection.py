"""
Comprehensive unit tests for data collection components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.base_client import BaseWeatherClient, WeatherData
from src.data_collection.client_factory import WeatherClientFactory
from src.data_collection.actual_temperature_collector import ActualTemperatureCollector
from src.utils.data_manager import DataManager

# Try to import WeatherDataCollector if it exists
try:
    from src.data_collection.client_factory import WeatherDataCollector
except ImportError:
    WeatherDataCollector = None


class TestWeatherData:
    """Test WeatherData dataclass."""
    
    def test_weather_data_creation(self):
        """Test creating a weather data object."""
        data = WeatherData(
            date=date(2025, 1, 15),
            forecast_date=date(2025, 1, 14),
            predicted_high=75.0,
            predicted_low=58.0,
            humidity=65.0,
            pressure=1013.2,
            wind_speed=8.5,
            wind_direction=225.0,
            cloud_cover=30.0,
            precipitation_prob=10.0,
            data_quality_score=1.0
        )
        
        assert data.date == date(2025, 1, 15)
        assert data.predicted_high == 75.0
        assert data.data_quality_score == 1.0
    
    def test_weather_data_with_defaults(self):
        """Test creating weather data with default values."""
        data = WeatherData(
            date=date(2025, 1, 15),
            forecast_date=date(2025, 1, 14),
            predicted_high=75.0,
            predicted_low=58.0
        )
        
        assert data.date == date(2025, 1, 15)
        assert data.predicted_high == 75.0
        assert data.humidity is None
        assert data.data_quality_score == 1.0


class TestBaseWeatherClient:
    """Test BaseWeatherClient abstract class."""
    
    def test_base_client_cannot_be_instantiated(self):
        """Test that BaseWeatherClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseWeatherClient()
    
    def test_base_client_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        
        class TestClient(BaseWeatherClient):
            pass
        
        with pytest.raises(TypeError):
            TestClient()


class MockWeatherClient(BaseWeatherClient):
    """Mock weather client for testing."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.test_connection_result = True
        self.forecast_data = []
    
    @property
    def service_name(self) -> str:
        return "mock"
    
    @property
    def base_url(self) -> str:
        return "http://mock.api"
    
    def test_connection(self) -> bool:
        return self.test_connection_result
    
    def get_forecast(self, target_date=None, days: int = 7) -> list:
        return self.forecast_data[:days]
    
    def _parse_forecast_data(self, raw_data, forecast_date):
        return []
    
    def _fetch_forecast_data(self, target_date, days):
        return {}


class TestWeatherClientFactory:
    """Test WeatherClientFactory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = WeatherClientFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        assert self.factory is not None
        services = self.factory.get_available_services()
        assert len(services) > 0
        assert 'nws' in services
    
    def test_get_available_services(self):
        """Test getting available services."""
        services = self.factory.get_available_services()
        
        expected_services = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        for service in expected_services:
            assert service in services
    
    @patch.dict(os.environ, {'OPENWEATHER_API_KEY': 'test_key'})
    def test_create_client_with_api_key(self):
        """Test creating client with API key."""
        client = self.factory.create_client('openweather')
        assert client is not None
        assert client.api_key == 'test_key'
    
    def test_create_client_without_api_key(self):
        """Test creating client without required API key."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            client = self.factory.create_client('openweather')
            assert client is None
    
    def test_create_nws_client(self):
        """Test creating NWS client (no API key required)."""
        client = self.factory.create_client('nws')
        assert client is not None
        assert client.api_key is None
    
    def test_create_invalid_client(self):
        """Test creating client with invalid service name."""
        client = self.factory.create_client('invalid_service')
        assert client is None


class TestWeatherDataCollector:
    """Test WeatherDataCollector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        if WeatherDataCollector is None:
            pytest.skip("WeatherDataCollector not available")
        self.collector = WeatherDataCollector()
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        assert self.collector is not None
        assert hasattr(self.collector, 'factory')
    
    @patch('src.data_collection.client_factory.WeatherClientFactory.create_client')
    def test_test_all_connections(self, mock_create_client):
        """Test testing all client connections."""
        # Mock successful client
        mock_client = Mock()
        mock_client.test_connection.return_value = True
        mock_create_client.return_value = mock_client
        
        results = self.collector.test_all_connections()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # All should be True since we mocked successful connections
        for service, result in results.items():
            assert result is True
    
    @patch('src.data_collection.client_factory.WeatherClientFactory.create_client')
    def test_get_client_status(self, mock_create_client):
        """Test getting client status."""
        # Mock client creation
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        status = self.collector.get_client_status()
        
        assert isinstance(status, dict)
        assert len(status) > 0
        
        for service, info in status.items():
            assert 'enabled' in info
            assert 'has_api_key' in info
    
    @patch('src.data_collection.client_factory.WeatherClientFactory.create_client')
    def test_get_all_forecasts(self, mock_create_client):
        """Test getting forecasts from all clients."""
        # Mock client with forecast data
        mock_client = Mock()
        mock_forecast = WeatherData(
            date=date.today() + timedelta(days=1),
            forecast_date=date.today(),
            predicted_high=75.0,
            predicted_low=58.0
        )
        mock_client.get_forecast.return_value = [mock_forecast]
        mock_create_client.return_value = mock_client
        
        forecasts = self.collector.get_all_forecasts(days=3)
        
        assert isinstance(forecasts, dict)
        
        for service, forecast_data in forecasts.items():
            if forecast_data:  # If client was available
                assert len(forecast_data) > 0
                assert isinstance(forecast_data[0], WeatherData)
    
    @patch('src.data_collection.client_factory.WeatherClientFactory.create_client')
    def test_get_current_conditions(self, mock_create_client):
        """Test getting current conditions from all clients."""
        # Mock client with current weather
        mock_client = Mock()
        mock_current = WeatherData(
            date=date.today(),
            forecast_date=date.today(),
            predicted_high=72.0,
            predicted_low=55.0
        )
        mock_client.get_current_weather.return_value = mock_current
        mock_create_client.return_value = mock_client
        
        conditions = self.collector.get_current_conditions()
        
        assert isinstance(conditions, dict)
        
        for service, current_data in conditions.items():
            if current_data:  # If client was available
                assert isinstance(current_data, WeatherData)


class TestActualTemperatureCollector:
    """Test ActualTemperatureCollector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = ActualTemperatureCollector()
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        assert self.collector is not None
        assert hasattr(self.collector, 'station_id')
        assert self.collector.station_id == 'KLAX'  # LAX airport
    
    @patch('src.data_collection.actual_temperature_collector.requests.get')
    def test_get_daily_high_temperature_success(self, mock_get):
        """Test successful temperature collection."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'properties': {
                'periods': [
                    {
                        'name': 'Today',
                        'temperature': 75,
                        'temperatureUnit': 'F',
                        'isDaytime': True
                    },
                    {
                        'name': 'Tonight',
                        'temperature': 58,
                        'temperatureUnit': 'F',
                        'isDaytime': False
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        temperature = self.collector.get_daily_high_temperature(date.today())
        
        assert temperature == 75.0
        mock_get.assert_called_once()
    
    @patch('src.data_collection.actual_temperature_collector.requests.get')
    def test_get_daily_high_temperature_api_error(self, mock_get):
        """Test temperature collection with API error."""
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        temperature = self.collector.get_daily_high_temperature(date.today())
        
        assert temperature is None
    
    @patch('src.data_collection.actual_temperature_collector.requests.get')
    def test_get_daily_high_temperature_no_data(self, mock_get):
        """Test temperature collection with no temperature data."""
        # Mock response with no temperature data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'properties': {
                'periods': []
            }
        }
        mock_get.return_value = mock_response
        
        temperature = self.collector.get_daily_high_temperature(date.today())
        
        assert temperature is None
    
    @patch('src.data_collection.actual_temperature_collector.requests.get')
    def test_get_daily_high_temperature_connection_error(self, mock_get):
        """Test temperature collection with connection error."""
        # Mock connection error
        mock_get.side_effect = Exception("Connection error")
        
        temperature = self.collector.get_daily_high_temperature(date.today())
        
        assert temperature is None
    
    def test_validate_temperature(self):
        """Test temperature validation."""
        # Valid temperatures
        assert self.collector.validate_temperature(75.0) is True
        assert self.collector.validate_temperature(32.0) is True
        assert self.collector.validate_temperature(120.0) is True
        
        # Invalid temperatures
        assert self.collector.validate_temperature(-50.0) is False
        assert self.collector.validate_temperature(150.0) is False
        assert self.collector.validate_temperature(None) is False
    
    @patch('src.data_collection.actual_temperature_collector.ActualTemperatureCollector.get_daily_high_temperature')
    def test_collect_and_store_temperature(self, mock_get_temp):
        """Test collecting and storing temperature."""
        # Mock successful temperature collection
        mock_get_temp.return_value = 78.5
        
        # Create temporary data manager
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DataManager(data_dir=temp_dir)
            
            result = self.collector.collect_and_store_temperature(
                date.today(), data_manager
            )
            
            assert result is True
            mock_get_temp.assert_called_once()
    
    @patch('src.data_collection.actual_temperature_collector.ActualTemperatureCollector.get_daily_high_temperature')
    def test_collect_and_store_temperature_failure(self, mock_get_temp):
        """Test collecting and storing temperature with failure."""
        # Mock failed temperature collection
        mock_get_temp.return_value = None
        
        # Create temporary data manager
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DataManager(data_dir=temp_dir)
            
            result = self.collector.collect_and_store_temperature(
                date.today(), data_manager
            )
            
            assert result is False


class TestDataManagerIntegration:
    """Test data collection integration with DataManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_store_weather_data(self):
        """Test storing weather data."""
        # Create sample weather data
        weather_data = [
            WeatherData(
                date=date(2025, 1, 15),
                forecast_date=date(2025, 1, 14),
                predicted_high=75.0,
                predicted_low=58.0,
                humidity=65.0
            ),
            WeatherData(
                date=date(2025, 1, 16),
                forecast_date=date(2025, 1, 14),
                predicted_high=78.0,
                predicted_low=62.0,
                humidity=70.0
            )
        ]
        
        # Store weather data
        self.data_manager.store_weather_data('test_source', weather_data)
        
        # Verify file was created
        data_file = Path(self.temp_dir) / 'test_source_data.parquet'
        assert data_file.exists()
        
        # Load and verify data
        loaded_data = self.data_manager.load_source_data(
            'test_source',
            date(2025, 1, 15),
            date(2025, 1, 16)
        )
        
        assert len(loaded_data) == 2
        assert loaded_data.iloc[0]['predicted_high'] == 75.0
        assert loaded_data.iloc[1]['predicted_high'] == 78.0
    
    def test_store_actual_temperature(self):
        """Test storing actual temperature."""
        test_date = date(2025, 1, 15)
        test_temp = 76.5
        
        # Store actual temperature
        self.data_manager.store_actual_temperature(test_date, test_temp)
        
        # Verify file was created
        data_file = Path(self.temp_dir) / 'actual_temperatures.parquet'
        assert data_file.exists()
        
        # Load and verify data
        loaded_data = self.data_manager.load_actual_temperatures(
            test_date, test_date
        )
        
        assert len(loaded_data) == 1
        assert loaded_data.iloc[0]['actual_high'] == test_temp
        assert loaded_data.iloc[0]['date'] == test_date


class TestErrorHandling:
    """Test error handling in data collection components."""
    
    def test_weather_data_invalid_data(self):
        """Test WeatherData with invalid data."""
        # Should handle None values gracefully
        data = WeatherData(
            date=date(2025, 1, 15),
            forecast_date=date(2025, 1, 14),
            predicted_high=None,
            predicted_low=None
        )
        
        assert data.predicted_high is None
        assert data.predicted_low is None
    
    def test_client_factory_error_handling(self):
        """Test client factory error handling."""
        factory = WeatherClientFactory()
        
        # Invalid service should return None
        client = factory.create_client('nonexistent_service')
        assert client is None
        
        # Empty service name should return None
        client = factory.create_client('')
        assert client is None
        
        # None service name should return None
        client = factory.create_client(None)
        assert client is None
    
    @patch('src.data_collection.client_factory.WeatherClientFactory.create_client')
    def test_data_collector_error_handling(self, mock_create_client):
        """Test data collector error handling."""
        # Mock client that raises exception
        mock_client = Mock()
        mock_client.get_forecast.side_effect = Exception("API Error")
        mock_create_client.return_value = mock_client
        
        collector = WeatherDataCollector()
        
        # Should handle exceptions gracefully
        forecasts = collector.get_all_forecasts(days=3)
        
        # Should return empty results for failed clients
        assert isinstance(forecasts, dict)
        for service, forecast_data in forecasts.items():
            assert forecast_data == []  # Empty list for failed clients


def run_data_collection_tests():
    """Run all data collection tests."""
    print("Running data collection tests...")
    
    # Test WeatherData
    print("✓ Testing WeatherData...")
    data = WeatherData(
        date=date(2025, 1, 15),
        forecast_date=date(2025, 1, 14),
        predicted_high=75.0,
        predicted_low=58.0
    )
    assert data.predicted_high == 75.0
    
    # Test WeatherClientFactory
    print("✓ Testing WeatherClientFactory...")
    factory = WeatherClientFactory()
    services = factory.get_available_services()
    assert len(services) > 0
    
    # Test WeatherDataCollector
    print("✓ Testing WeatherDataCollector...")
    collector = WeatherDataCollector()
    assert collector is not None
    
    # Test ActualTemperatureCollector
    print("✓ Testing ActualTemperatureCollector...")
    temp_collector = ActualTemperatureCollector()
    assert temp_collector.validate_temperature(75.0) is True
    assert temp_collector.validate_temperature(-50.0) is False
    
    print("All data collection tests passed!")


if __name__ == '__main__':
    run_data_collection_tests()