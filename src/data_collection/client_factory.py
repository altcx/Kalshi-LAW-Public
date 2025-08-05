"""Factory for creating weather API clients."""

from typing import Dict, List, Optional, Type
from loguru import logger

from .base_client import BaseWeatherClient, WeatherData
from .nws_client import NWSClient
from .openweather_client import OpenWeatherClient
from .tomorrow_client import TomorrowClient
from .weatherbit_client import WeatherbitClient
from .visual_crossing_client import VisualCrossingClient
from ..utils.config import config


class WeatherClientFactory:
    """Factory for creating and managing weather API clients."""
    
    # Registry of available client classes
    CLIENT_REGISTRY: Dict[str, Type[BaseWeatherClient]] = {
        'nws': NWSClient,
        'openweather': OpenWeatherClient,
        'tomorrow': TomorrowClient,
        'weatherbit': WeatherbitClient,
        'visual_crossing': VisualCrossingClient
    }
    
    @classmethod
    def create_client(cls, service_name: str, api_key: Optional[str] = None) -> Optional[BaseWeatherClient]:
        """Create a weather API client for the specified service.
        
        Args:
            service_name: Name of the weather service
            api_key: Optional API key (will try to get from config if not provided)
            
        Returns:
            Weather client instance or None if creation failed
        """
        if service_name not in cls.CLIENT_REGISTRY:
            logger.error(f"Unknown weather service: {service_name}")
            return None
        
        client_class = cls.CLIENT_REGISTRY[service_name]
        
        try:
            if service_name == 'nws':
                # NWS doesn't require API key
                return client_class()
            else:
                # Other services require API key
                if api_key is None:
                    api_key = config.get_api_key(service_name)
                
                if not api_key:
                    logger.warning(f"No API key found for {service_name}")
                    return None
                
                return client_class(api_key=api_key)
                
        except Exception as e:
            logger.error(f"Failed to create {service_name} client: {e}")
            return None
    
    @classmethod
    def create_all_clients(self) -> Dict[str, BaseWeatherClient]:
        """Create all available weather API clients.
        
        Returns:
            Dictionary mapping service names to client instances
        """
        clients = {}
        
        for service_name in self.CLIENT_REGISTRY.keys():
            client = self.create_client(service_name)
            if client:
                clients[service_name] = client
                logger.info(f"Successfully created {service_name} client")
            else:
                logger.warning(f"Failed to create {service_name} client")
        
        return clients
    
    @classmethod
    def get_available_services(cls) -> List[str]:
        """Get list of available weather services.
        
        Returns:
            List of service names
        """
        return list(cls.CLIENT_REGISTRY.keys())
    
    @classmethod
    def test_all_clients(cls) -> Dict[str, bool]:
        """Test connection for all available clients.
        
        Returns:
            Dictionary mapping service names to connection test results
        """
        results = {}
        clients = cls.create_all_clients()
        
        for service_name, client in clients.items():
            try:
                logger.info(f"Testing {service_name} connection...")
                results[service_name] = client.test_connection()
                
                if results[service_name]:
                    logger.info(f"{service_name}: Connection test PASSED")
                else:
                    logger.warning(f"{service_name}: Connection test FAILED")
                    
            except Exception as e:
                logger.error(f"{service_name}: Connection test ERROR - {e}")
                results[service_name] = False
        
        return results


class WeatherDataCollector:
    """Coordinates data collection from multiple weather API clients."""
    
    def __init__(self, enabled_services: Optional[List[str]] = None):
        """Initialize weather data collector.
        
        Args:
            enabled_services: List of service names to enable (defaults to all available)
        """
        self.factory = WeatherClientFactory()
        
        if enabled_services is None:
            enabled_services = self.factory.get_available_services()
        
        self.clients = {}
        self.enabled_services = enabled_services
        
        # Create clients for enabled services
        for service_name in enabled_services:
            client = self.factory.create_client(service_name)
            if client:
                self.clients[service_name] = client
                logger.info(f"Enabled {service_name} client")
            else:
                logger.warning(f"Could not enable {service_name} client")
    
    def get_all_forecasts(self, days: int = 7) -> Dict[str, List]:
        """Get forecasts from all enabled clients.
        
        Args:
            days: Number of days to forecast
            
        Returns:
            Dictionary mapping service names to forecast data lists
        """
        forecasts = {}
        
        for service_name, client in self.clients.items():
            try:
                logger.info(f"Fetching {days}-day forecast from {service_name}...")
                forecast_data = client.get_forecast(days=days)
                forecasts[service_name] = forecast_data
                logger.info(f"{service_name}: Retrieved {len(forecast_data)} forecast records")
                
            except Exception as e:
                logger.error(f"Error getting forecast from {service_name}: {e}")
                forecasts[service_name] = []
        
        return forecasts
    
    def get_current_conditions(self) -> Dict[str, Optional[WeatherData]]:
        """Get current weather conditions from all enabled clients.
        
        Returns:
            Dictionary mapping service names to current weather data
        """
        current_conditions = {}
        
        for service_name, client in self.clients.items():
            try:
                if hasattr(client, 'get_current_weather'):
                    logger.info(f"Fetching current conditions from {service_name}...")
                    current_data = client.get_current_weather()
                    current_conditions[service_name] = current_data
                    
                    if current_data:
                        logger.info(f"{service_name}: Retrieved current conditions")
                    else:
                        logger.warning(f"{service_name}: No current conditions available")
                else:
                    logger.info(f"{service_name}: Current conditions not supported")
                    current_conditions[service_name] = None
                    
            except Exception as e:
                logger.error(f"Error getting current conditions from {service_name}: {e}")
                current_conditions[service_name] = None
        
        return current_conditions
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test connections for all enabled clients.
        
        Returns:
            Dictionary mapping service names to connection test results
        """
        results = {}
        
        for service_name, client in self.clients.items():
            try:
                logger.info(f"Testing {service_name} connection...")
                results[service_name] = client.test_connection()
                
                if results[service_name]:
                    logger.info(f"{service_name}: Connection test PASSED")
                else:
                    logger.warning(f"{service_name}: Connection test FAILED")
                    
            except Exception as e:
                logger.error(f"{service_name}: Connection test ERROR - {e}")
                results[service_name] = False
        
        return results
    
    def get_client_status(self) -> Dict[str, Dict]:
        """Get status information for all clients.
        
        Returns:
            Dictionary with client status information
        """
        status = {}
        
        for service_name, client in self.clients.items():
            status[service_name] = {
                'enabled': True,
                'service_name': client.service_name,
                'base_url': client.base_url,
                'has_api_key': client.api_key is not None,
                'rate_limiter': {
                    'calls_per_minute': client.rate_limiter.calls_per_minute,
                    'calls_per_day': client.rate_limiter.calls_per_day,
                    'daily_calls_made': client.rate_limiter.daily_calls
                }
            }
        
        # Add disabled services
        all_services = self.factory.get_available_services()
        for service_name in all_services:
            if service_name not in status:
                status[service_name] = {
                    'enabled': False,
                    'reason': 'Not enabled or failed to initialize'
                }
        
        return status