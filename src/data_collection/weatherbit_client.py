"""Weatherbit API client."""

from typing import Dict, Any, Optional, List
from datetime import date, datetime, timedelta
from loguru import logger

from .base_client import BaseWeatherClient, WeatherData, RateLimiter
from ..utils.config import config


class WeatherbitClient(BaseWeatherClient):
    """Client for Weatherbit API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Weatherbit client.
        
        Args:
            api_key: Weatherbit API key (optional, will try to get from config)
        """
        if api_key is None:
            api_key = config.get_api_key('weatherbit')
        
        if not api_key:
            raise ValueError("Weatherbit API key is required")
        
        # Free tier: 500 calls/day, 1 call/second
        rate_limiter = RateLimiter(calls_per_minute=60, calls_per_day=500)
        super().__init__(api_key=api_key, rate_limiter=rate_limiter)
        
        # LAX coordinates for Los Angeles
        self.lat = 33.9425
        self.lon = -118.4081
    
    @property
    def service_name(self) -> str:
        return "Weatherbit"
    
    @property
    def base_url(self) -> str:
        return "https://api.weatherbit.io/v2.0"
    
    def _fetch_forecast_data(self, target_date: date, days: int) -> Optional[Dict]:
        """Fetch forecast data from Weatherbit API.
        
        Args:
            target_date: Target date for forecast
            days: Number of days to forecast (max 16 for free tier)
            
        Returns:
            Raw API response data
        """
        # Free tier supports up to 16-day forecast
        days = min(days, 16)
        
        url = f"{self.base_url}/forecast/daily"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'key': self.api_key,
            'days': days,
            'units': 'I'  # Imperial units (Fahrenheit)
        }
        
        return self._make_request(url, params)
    
    def _parse_forecast_data(self, raw_data: Dict, forecast_date: date) -> List[WeatherData]:
        """Parse Weatherbit forecast data into standardized format.
        
        Args:
            raw_data: Raw Weatherbit API response
            forecast_date: Date when forecast was made
            
        Returns:
            List of WeatherData objects
        """
        weather_data = []
        
        try:
            if 'data' not in raw_data:
                logger.error("Weatherbit: Invalid response format")
                return []
            
            # Process each day
            for day_data in raw_data['data']:
                # Parse date
                date_str = day_data.get('datetime')
                if not date_str:
                    continue
                
                day = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                # Extract weather data
                weather = day_data.get('weather', {})
                
                weather_data.append(WeatherData(
                    date=day,
                    forecast_date=forecast_date,
                    predicted_high=day_data.get('high_temp'),
                    predicted_low=day_data.get('low_temp'),
                    humidity=day_data.get('rh'),  # Relative humidity
                    pressure=day_data.get('pres'),  # Pressure in mb
                    wind_speed=day_data.get('wind_spd'),  # Wind speed in mph
                    wind_direction=day_data.get('wind_dir'),  # Wind direction in degrees
                    cloud_cover=day_data.get('clouds'),  # Cloud coverage percentage
                    precipitation_prob=day_data.get('pop'),  # Probability of precipitation
                    data_quality_score=0.82,  # Good quality commercial API
                    raw_data=day_data
                ))
                
        except Exception as e:
            logger.error(f"Weatherbit: Error parsing forecast data: {e}")
        
        return weather_data
    
    def get_current_weather(self) -> Optional[WeatherData]:
        """Get current weather conditions.
        
        Returns:
            Current weather data or None if failed
        """
        url = f"{self.base_url}/current"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'key': self.api_key,
            'units': 'I'  # Imperial units
        }
        
        try:
            response = self._make_request(url, params)
            if not response or 'data' not in response:
                return None
            
            data_list = response['data']
            if not data_list:
                return None
            
            current_data = data_list[0]  # First (and only) item
            
            return WeatherData(
                date=date.today(),
                forecast_date=date.today(),
                predicted_high=current_data.get('temp'),
                predicted_low=None,
                humidity=current_data.get('rh'),
                pressure=current_data.get('pres'),
                wind_speed=current_data.get('wind_spd'),
                wind_direction=current_data.get('wind_dir'),
                cloud_cover=current_data.get('clouds'),
                precipitation_prob=None,
                data_quality_score=0.85,
                raw_data=current_data
            )
            
        except Exception as e:
            logger.error(f"Weatherbit: Error getting current weather: {e}")
            return None
    
    def get_hourly_forecast(self, hours: int = 48) -> List[WeatherData]:
        """Get hourly forecast data.
        
        Args:
            hours: Number of hours to forecast (max 48 for free tier)
            
        Returns:
            List of WeatherData objects with hourly data
        """
        # Free tier supports up to 48-hour forecast
        hours = min(hours, 48)
        
        url = f"{self.base_url}/forecast/hourly"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'key': self.api_key,
            'hours': hours,
            'units': 'I'  # Imperial units
        }
        
        try:
            response = self._make_request(url, params)
            if not response or 'data' not in response:
                return []
            
            weather_data = []
            
            for hour_data in response['data']:
                # Parse datetime
                datetime_str = hour_data.get('datetime')
                if not datetime_str:
                    continue
                
                dt = datetime.strptime(datetime_str, '%Y-%m-%d:%H')
                
                weather_data.append(WeatherData(
                    date=dt.date(),
                    forecast_date=date.today(),
                    predicted_high=hour_data.get('temp'),
                    predicted_low=None,
                    humidity=hour_data.get('rh'),
                    pressure=hour_data.get('pres'),
                    wind_speed=hour_data.get('wind_spd'),
                    wind_direction=hour_data.get('wind_dir'),
                    cloud_cover=hour_data.get('clouds'),
                    precipitation_prob=hour_data.get('pop'),
                    data_quality_score=0.82,
                    raw_data=hour_data
                ))
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Weatherbit: Error getting hourly forecast: {e}")
            return []
    
    def get_historical_weather(self, start_date: date, end_date: date) -> List[WeatherData]:
        """Get historical weather data.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of WeatherData objects with historical data
        """
        url = f"{self.base_url}/history/daily"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'key': self.api_key,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'units': 'I'  # Imperial units
        }
        
        try:
            response = self._make_request(url, params)
            if not response or 'data' not in response:
                return []
            
            weather_data = []
            
            for day_data in response['data']:
                # Parse date
                date_str = day_data.get('datetime')
                if not date_str:
                    continue
                
                day = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                weather_data.append(WeatherData(
                    date=day,
                    forecast_date=day,  # Historical data
                    predicted_high=day_data.get('max_temp'),
                    predicted_low=day_data.get('min_temp'),
                    humidity=day_data.get('rh'),
                    pressure=day_data.get('pres'),
                    wind_speed=day_data.get('wind_spd'),
                    wind_direction=day_data.get('wind_dir'),
                    cloud_cover=day_data.get('clouds'),
                    precipitation_prob=None,  # Not available in historical data
                    data_quality_score=0.90,  # Historical data is typically accurate
                    raw_data=day_data
                ))
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Weatherbit: Error getting historical weather: {e}")
            return []