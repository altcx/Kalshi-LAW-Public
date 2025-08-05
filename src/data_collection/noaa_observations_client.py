"""NOAA Observations API client for collecting actual temperature data."""

from typing import Dict, Any, Optional, List
from datetime import date, datetime, timedelta
import requests
from loguru import logger

from .base_client import RateLimiter
from ..utils.config import config


class NOAAObservationsClient:
    """Client for NOAA observations API to collect actual temperature data."""
    
    def __init__(self, station_id: Optional[str] = None):
        """Initialize NOAA observations client.
        
        Args:
            station_id: NOAA station ID (defaults to KLAX from config)
        """
        self.station_id = station_id or config.get('location.noaa_station_id', 'KLAX')
        self.base_url = "https://api.weather.gov"
        
        # NOAA has no official rate limits but recommends being reasonable
        self.rate_limiter = RateLimiter(calls_per_minute=30, calls_per_day=10000)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kalshi-Weather-Predictor/1.0'
        })
        
        logger.info(f"NOAA Observations: Initialized for station {self.station_id}")
    
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     max_retries: int = 3, backoff_factor: float = 1.0) -> Optional[Dict]:
        """Make HTTP request with retry logic.
        
        Args:
            url: Request URL
            params: Query parameters
            max_retries: Maximum number of retries
            backoff_factor: Backoff multiplier for retries
            
        Returns:
            JSON response data or None if failed
        """
        # Wait for rate limit if needed
        if not self.rate_limiter.can_make_call():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.info(f"NOAA Observations: Rate limit reached, waiting {wait_time:.1f}s")
                import time
                time.sleep(wait_time)
        
        for attempt in range(max_retries + 1):
            try:
                self.rate_limiter.record_call()
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"NOAA Observations: Rate limited, waiting {wait_time}s")
                    import time
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    logger.error(f"NOAA Observations: Station {self.station_id} not found")
                    return None
                else:
                    logger.warning(f"NOAA Observations: HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"NOAA Observations: Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    import time
                    time.sleep(wait_time)
                else:
                    logger.error(f"NOAA Observations: All retry attempts failed")
                    return None
        
        return None
    
    def get_daily_high_temperature(self, target_date: date) -> Optional[float]:
        """Get the daily high temperature for a specific date.
        
        Args:
            target_date: Date to get temperature for
            
        Returns:
            Daily high temperature in Fahrenheit, or None if not available
        """
        try:
            # Get observations for the entire day
            start_time = datetime.combine(target_date, datetime.min.time())
            end_time = start_time + timedelta(days=1)
            
            observations = self.get_observations(start_time, end_time)
            if not observations:
                logger.warning(f"NOAA Observations: No observations found for {target_date}")
                return None
            
            # Extract temperatures and find the maximum
            temperatures = []
            for obs in observations:
                temp = obs.get('temperature')
                if temp is not None:
                    # Convert from Celsius to Fahrenheit if needed
                    temp_f = self._convert_to_fahrenheit(temp, obs.get('temperature_unit', 'C'))
                    temperatures.append(temp_f)
            
            if not temperatures:
                logger.warning(f"NOAA Observations: No valid temperatures found for {target_date}")
                return None
            
            daily_high = max(temperatures)
            logger.info(f"NOAA Observations: Daily high for {target_date}: {daily_high:.1f}°F")
            return daily_high
            
        except Exception as e:
            logger.error(f"NOAA Observations: Error getting daily high for {target_date}: {e}")
            return None
    
    def get_observations(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get weather observations for a time period.
        
        Args:
            start_time: Start of time period
            end_time: End of time period
            
        Returns:
            List of observation dictionaries
        """
        try:
            # Format times for API (ISO 8601)
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            url = f"{self.base_url}/stations/{self.station_id}/observations"
            params = {
                'start': start_str,
                'end': end_str
            }
            
            response = self._make_request(url, params)
            if not response:
                return []
            
            # Extract observations from response
            features = response.get('features', [])
            observations = []
            
            for feature in features:
                properties = feature.get('properties', {})
                
                # Extract key observation data
                obs_data = {
                    'timestamp': properties.get('timestamp'),
                    'temperature': self._extract_temperature(properties),
                    'temperature_unit': self._get_temperature_unit(properties),
                    'humidity': properties.get('relativeHumidity', {}).get('value'),
                    'pressure': self._extract_pressure(properties),
                    'wind_speed': self._extract_wind_speed(properties),
                    'wind_direction': self._extract_wind_direction(properties),
                    'raw_data': properties
                }
                
                observations.append(obs_data)
            
            logger.info(f"NOAA Observations: Retrieved {len(observations)} observations from {start_time.date()} to {end_time.date()}")
            return observations
            
        except Exception as e:
            logger.error(f"NOAA Observations: Error getting observations: {e}")
            return []
    
    def _extract_temperature(self, properties: Dict) -> Optional[float]:
        """Extract temperature from observation properties."""
        temp_data = properties.get('temperature', {})
        if isinstance(temp_data, dict):
            return temp_data.get('value')
        return temp_data
    
    def _get_temperature_unit(self, properties: Dict) -> str:
        """Get temperature unit from observation properties."""
        temp_data = properties.get('temperature', {})
        if isinstance(temp_data, dict):
            unit_code = temp_data.get('unitCode', 'wmoUnit:degC')
            # NOAA typically uses Celsius
            return 'C' if 'degC' in unit_code else 'F'
        return 'C'  # Default to Celsius
    
    def _extract_pressure(self, properties: Dict) -> Optional[float]:
        """Extract pressure from observation properties."""
        pressure_data = properties.get('barometricPressure', {})
        if isinstance(pressure_data, dict):
            pressure_pa = pressure_data.get('value')
            if pressure_pa is not None:
                # Convert from Pa to hPa
                return pressure_pa / 100
        return None
    
    def _extract_wind_speed(self, properties: Dict) -> Optional[float]:
        """Extract wind speed from observation properties."""
        wind_data = properties.get('windSpeed', {})
        if isinstance(wind_data, dict):
            wind_ms = wind_data.get('value')
            if wind_ms is not None:
                # Convert from m/s to mph
                return wind_ms * 2.237
        return None
    
    def _extract_wind_direction(self, properties: Dict) -> Optional[float]:
        """Extract wind direction from observation properties."""
        wind_dir_data = properties.get('windDirection', {})
        if isinstance(wind_dir_data, dict):
            return wind_dir_data.get('value')
        return None
    
    def _convert_to_fahrenheit(self, temperature: float, unit: str) -> float:
        """Convert temperature to Fahrenheit.
        
        Args:
            temperature: Temperature value
            unit: Current unit ('C' or 'F')
            
        Returns:
            Temperature in Fahrenheit
        """
        if unit.upper() == 'C':
            return (temperature * 9/5) + 32
        return temperature
    
    def validate_temperature_reading(self, temperature: float, target_date: date) -> bool:
        """Validate that a temperature reading is reasonable for LA.
        
        Args:
            temperature: Temperature in Fahrenheit
            target_date: Date of the reading
            
        Returns:
            True if temperature is reasonable, False otherwise
        """
        # LA temperature ranges (very conservative bounds)
        min_temp = -10.0  # Extremely cold for LA
        max_temp = 130.0  # Extremely hot for LA
        
        if temperature < min_temp or temperature > max_temp:
            logger.warning(f"NOAA Observations: Temperature {temperature:.1f}°F is outside reasonable range for LA on {target_date}")
            return False
        
        # Additional seasonal checks could be added here
        # For now, just check basic bounds
        return True
    
    def get_station_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the weather station.
        
        Returns:
            Station information dictionary or None if failed
        """
        try:
            url = f"{self.base_url}/stations/{self.station_id}"
            response = self._make_request(url)
            
            if response and 'properties' in response:
                properties = response['properties']
                station_info = {
                    'station_id': self.station_id,
                    'name': properties.get('name'),
                    'elevation': properties.get('elevation', {}).get('value'),
                    'timezone': properties.get('timeZone'),
                    'coordinates': response.get('geometry', {}).get('coordinates', [])
                }
                logger.info(f"NOAA Observations: Station info - {station_info['name']} at elevation {station_info['elevation']}m")
                return station_info
            
        except Exception as e:
            logger.error(f"NOAA Observations: Error getting station info: {e}")
        
        return None
    
    def test_connection(self) -> bool:
        """Test connection to NOAA observations API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get station info
            station_info = self.get_station_info()
            if station_info:
                logger.info(f"NOAA Observations: Connection test successful for station {self.station_id}")
                return True
            else:
                logger.error(f"NOAA Observations: Connection test failed for station {self.station_id}")
                return False
                
        except Exception as e:
            logger.error(f"NOAA Observations: Connection test failed: {e}")
            return False