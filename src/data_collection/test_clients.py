"""Test script for weather API clients."""

import sys
from pathlib import Path
from datetime import date, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from data_collection import WeatherClientFactory, WeatherDataCollector


def test_individual_clients():
    """Test each weather API client individually."""
    logger.info("Testing individual weather API clients...")
    
    factory = WeatherClientFactory()
    services = factory.get_available_services()
    
    for service_name in services:
        logger.info(f"\n--- Testing {service_name} ---")
        
        try:
            client = factory.create_client(service_name)
            if not client:
                logger.warning(f"Could not create {service_name} client (likely missing API key)")
                continue
            
            # Test connection
            logger.info(f"Testing {service_name} connection...")
            connection_ok = client.test_connection()
            logger.info(f"{service_name} connection: {'OK' if connection_ok else 'FAILED'}")
            
            if not connection_ok:
                continue
            
            # Test forecast
            logger.info(f"Getting 3-day forecast from {service_name}...")
            forecast = client.get_forecast(days=3)
            logger.info(f"{service_name} forecast: {len(forecast)} records")
            
            if forecast:
                sample = forecast[0]
                logger.info(f"Sample forecast: {sample.date} - High: {sample.predicted_high}°F, Low: {sample.predicted_low}°F")
            
            # Test current weather (if supported)
            if hasattr(client, 'get_current_weather'):
                logger.info(f"Getting current weather from {service_name}...")
                current = client.get_current_weather()
                if current:
                    logger.info(f"Current temp: {current.predicted_high}°F")
                else:
                    logger.warning(f"No current weather data from {service_name}")
            
        except Exception as e:
            logger.error(f"Error testing {service_name}: {e}")


def test_data_collector():
    """Test the weather data collector."""
    logger.info("\n--- Testing Weather Data Collector ---")
    
    try:
        collector = WeatherDataCollector()
        
        # Test connections
        logger.info("Testing all connections...")
        connection_results = collector.test_all_connections()
        
        for service, result in connection_results.items():
            status = "OK" if result else "FAILED"
            logger.info(f"{service}: {status}")
        
        # Get client status
        logger.info("\nClient status:")
        status = collector.get_client_status()
        for service, info in status.items():
            if info.get('enabled'):
                logger.info(f"{service}: Enabled, API key: {'Yes' if info.get('has_api_key') else 'No'}")
            else:
                logger.info(f"{service}: Disabled - {info.get('reason', 'Unknown')}")
        
        # Get forecasts from all working clients
        logger.info("\nGetting forecasts from all clients...")
        forecasts = collector.get_all_forecasts(days=2)
        
        for service, forecast_data in forecasts.items():
            logger.info(f"{service}: {len(forecast_data)} forecast records")
            if forecast_data:
                sample = forecast_data[0]
                logger.info(f"  Sample: {sample.date} - High: {sample.predicted_high}°F")
        
        # Get current conditions
        logger.info("\nGetting current conditions...")
        current_conditions = collector.get_current_conditions()
        
        for service, current_data in current_conditions.items():
            if current_data:
                logger.info(f"{service}: Current temp {current_data.predicted_high}°F")
            else:
                logger.info(f"{service}: No current conditions available")
        
    except Exception as e:
        logger.error(f"Error testing data collector: {e}")


def main():
    """Main test function."""
    logger.info("Starting weather API client tests...")
    
    # Test individual clients
    test_individual_clients()
    
    # Test data collector
    test_data_collector()
    
    logger.info("\nWeather API client tests completed!")


if __name__ == "__main__":
    main()