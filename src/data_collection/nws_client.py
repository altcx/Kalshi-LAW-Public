"""National Weather Service API client."""

from typing import Dict, Any, Optional, List
from datetime import date, datetime, timedelta
import re
from loguru import logger

from .base_client import BaseWeatherClient, WeatherData, RateLimiter


class NWSClient(BaseWeatherClient):
    """Client for National Weather Service API."""
    
    def __init__(self):
        """Initialize NWS client (no API key required)."""
        # NWS has no official rate limits but recommends being reasonable
        rate_limiter = RateLimiter(calls_per_minute=30, calls_per_day=10000)
        super().__init__(api_key=None, rate_limiter=rate_limiter)
        
        # LAX coordinates for Los Angeles
        self.lat = 33.9425
        self.lon = -118.4081
        self.grid_x = None
        self.grid_y = None
        self.office = None
        
        # Get grid information on initialization
        self._get_grid_info()
    
    @property
    def service_name(self) -> str:
        return "NWS"
    
    @property
    def base_url(self) -> str:
        return "https://api.weather.gov"
    
    def _get_grid_info(self) -> None:
        """Get grid information for LAX coordinates."""
        url = f"{self.base_url}/points/{self.lat},{self.lon}"
        
        try:
            response = self._make_request(url)
            if response and 'properties' in response:
                props = response['properties']
                self.grid_x = props.get('gridX')
                self.grid_y = props.get('gridY')
                self.office = props.get('gridId')
                logger.info(f"NWS: Grid info - Office: {self.office}, X: {self.grid_x}, Y: {self.grid_y}")
            else:
                logger.error("NWS: Failed to get grid information")
        except Exception as e:
            logger.error(f"NWS: Error getting grid info: {e}")
    
    def _fetch_forecast_data(self, target_date: date, days: int) -> Optional[Dict]:
        """Fetch forecast data from NWS API.
        
        Args:
            target_date: Target date for forecast
            days: Number of days to forecast
            
        Returns:
            Raw API response data
        """
        if not all([self.grid_x, self.grid_y, self.office]):
            logger.error("NWS: Grid information not available")
            return None
        
        # NWS provides 7-day forecast
        url = f"{self.base_url}/gridpoints/{self.office}/{self.grid_x},{self.grid_y}/forecast"
        
        return self._make_request(url)
    
    def _parse_forecast_data(self, raw_data: Dict, forecast_date: date) -> List[WeatherData]:
        """Parse NWS forecast data into standardized format.
        
        Args:
            raw_data: Raw NWS API response
            forecast_date: Date when forecast was made
            
        Returns:
            List of WeatherData objects
        """
        weather_data = []
        
        try:
            if 'properties' not in raw_data or 'periods' not in raw_data['properties']:
                logger.error("NWS: Invalid response format")
                return []
            
            periods = raw_data['properties']['periods']
            daily_data = {}
            
            # Group periods by date (NWS gives day/night periods)
            for period in periods:
                start_time = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
                period_date = start_time.date()
                
                if period_date not in daily_data:
                    daily_data[period_date] = {}
                
                is_daytime = period.get('isDaytime', True)
                period_key = 'day' if is_daytime else 'night'
                daily_data[period_date][period_key] = period
            
            # Convert to WeatherData objects
            for date_key, day_data in daily_data.items():
                day_period = day_data.get('day', {})
                night_period = day_data.get('night', {})
                
                # Extract temperature (NWS gives high for day, low for night)
                predicted_high = None
                predicted_low = None
                
                if day_period and day_period.get('isDaytime'):
                    predicted_high = float(day_period.get('temperature', 0))
                elif night_period and not night_period.get('isDaytime'):
                    predicted_low = float(night_period.get('temperature', 0))
                
                # If we only have one period, try to infer the other
                if predicted_high is None and night_period:
                    # Night temperature is typically the low
                    predicted_low = float(night_period.get('temperature', 0))
                elif predicted_low is None and day_period:
                    # Day temperature is typically the high
                    predicted_high = float(day_period.get('temperature', 0))
                
                # Extract other weather parameters
                humidity = self._extract_humidity(day_period or night_period)
                wind_speed = self._extract_wind_speed(day_period or night_period)
                wind_direction = self._extract_wind_direction(day_period or night_period)
                
                weather_data.append(WeatherData(
                    date=date_key,
                    forecast_date=forecast_date,
                    predicted_high=predicted_high,
                    predicted_low=predicted_low,
                    humidity=humidity,
                    pressure=None,  # NWS doesn't provide pressure in forecast
                    wind_speed=wind_speed,
                    wind_direction=wind_direction,
                    cloud_cover=None,  # Not directly available
                    precipitation_prob=None,  # Not in this endpoint
                    data_quality_score=0.95,  # NWS is generally high quality
                    raw_data={'day': day_period, 'night': night_period}
                ))
            
        except Exception as e:
            logger.error(f"NWS: Error parsing forecast data: {e}")
        
        return weather_data
    
    def _extract_humidity(self, period: Dict) -> Optional[float]:
        """Extract humidity from period data."""
        # NWS doesn't always provide humidity in forecast endpoint
        return None
    
    def _extract_wind_speed(self, period: Dict) -> Optional[float]:
        """Extract wind speed from period data."""
        wind_speed_str = period.get('windSpeed', '')
        if not wind_speed_str:
            return None
        
        # Parse wind speed (e.g., "10 mph", "5 to 10 mph")
        match = re.search(r'(\d+)(?:\s*to\s*(\d+))?\s*mph', wind_speed_str)
        if match:
            if match.group(2):  # Range given
                return (float(match.group(1)) + float(match.group(2))) / 2
            else:
                return float(match.group(1))
        
        return None
    
    def _extract_wind_direction(self, period: Dict) -> Optional[float]:
        """Extract wind direction from period data."""
        wind_dir_str = period.get('windDirection', '')
        if not wind_dir_str:
            return None
        
        # Convert compass direction to degrees
        direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        return direction_map.get(wind_dir_str.upper())
    
    def get_current_conditions(self) -> Optional[WeatherData]:
        """Get current weather conditions from NWS.
        
        Returns:
            Current weather data or None if failed
        """
        if not all([self.grid_x, self.grid_y, self.office]):
            logger.error("NWS: Grid information not available")
            return None
        
        # Get current conditions from gridpoints
        url = f"{self.base_url}/gridpoints/{self.office}/{self.grid_x},{self.grid_y}"
        
        try:
            response = self._make_request(url)
            if response and 'properties' in response:
                props = response['properties']
                
                # Extract current temperature
                temp_data = props.get('temperature', {})
                current_temp = None
                if 'values' in temp_data and temp_data['values']:
                    # Convert from Celsius to Fahrenheit
                    temp_celsius = temp_data['values'][0]['value']
                    current_temp = (temp_celsius * 9/5) + 32
                
                return WeatherData(
                    date=date.today(),
                    forecast_date=date.today(),
                    predicted_high=current_temp,
                    predicted_low=None,
                    humidity=self._extract_grid_humidity(props),
                    pressure=self._extract_grid_pressure(props),
                    wind_speed=self._extract_grid_wind_speed(props),
                    wind_direction=self._extract_grid_wind_direction(props),
                    cloud_cover=None,
                    precipitation_prob=None,
                    data_quality_score=0.95,
                    raw_data=props
                )
                
        except Exception as e:
            logger.error(f"NWS: Error getting current conditions: {e}")
        
        return None
    
    def _extract_grid_humidity(self, props: Dict) -> Optional[float]:
        """Extract humidity from gridpoint data."""
        humidity_data = props.get('relativeHumidity', {})
        if 'values' in humidity_data and humidity_data['values']:
            return humidity_data['values'][0]['value']
        return None
    
    def _extract_grid_pressure(self, props: Dict) -> Optional[float]:
        """Extract pressure from gridpoint data."""
        pressure_data = props.get('pressure', {})
        if 'values' in pressure_data and pressure_data['values']:
            # Convert from Pa to hPa
            return pressure_data['values'][0]['value'] / 100
        return None
    
    def _extract_grid_wind_speed(self, props: Dict) -> Optional[float]:
        """Extract wind speed from gridpoint data."""
        wind_data = props.get('windSpeed', {})
        if 'values' in wind_data and wind_data['values']:
            # Convert from m/s to mph
            return wind_data['values'][0]['value'] * 2.237
        return None
    
    def _extract_grid_wind_direction(self, props: Dict) -> Optional[float]:
        """Extract wind direction from gridpoint data."""
        wind_dir_data = props.get('windDirection', {})
        if 'values' in wind_dir_data and wind_dir_data['values']:
            return wind_dir_data['values'][0]['value']
        return None