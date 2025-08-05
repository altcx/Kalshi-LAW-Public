"""OpenWeatherMap API client."""

from typing import Dict, Any, Optional, List
from datetime import date, datetime, timedelta
from loguru import logger

from .base_client import BaseWeatherClient, WeatherData, RateLimiter
from ..utils.config import config


class OpenWeatherClient(BaseWeatherClient):
    """Client for OpenWeatherMap API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenWeatherMap client.
        
        Args:
            api_key: OpenWeatherMap API key (optional, will try to get from config)
        """
        if api_key is None:
            api_key = config.get_api_key('openweathermap')
        
        if not api_key:
            raise ValueError("OpenWeatherMap API key is required")
        
        # Free tier: 1000 calls/day, 60 calls/minute
        rate_limiter = RateLimiter(calls_per_minute=60, calls_per_day=1000)
        super().__init__(api_key=api_key, rate_limiter=rate_limiter)
        
        # LAX coordinates for Los Angeles
        self.lat = 33.9425
        self.lon = -118.4081
    
    @property
    def service_name(self) -> str:
        return "OpenWeatherMap"
    
    @property
    def base_url(self) -> str:
        return "https://api.openweathermap.org/data/2.5"
    
    def _fetch_forecast_data(self, target_date: date, days: int) -> Optional[Dict]:
        """Fetch forecast data from OpenWeatherMap API.
        
        Args:
            target_date: Target date for forecast
            days: Number of days to forecast (max 5 for free tier)
            
        Returns:
            Raw API response data
        """
        # Free tier only supports 5-day forecast
        days = min(days, 5)
        
        url = f"{self.base_url}/forecast"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'imperial',  # Fahrenheit
            'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
        }
        
        return self._make_request(url, params)
    
    def _parse_forecast_data(self, raw_data: Dict, forecast_date: date) -> List[WeatherData]:
        """Parse OpenWeatherMap forecast data into standardized format.
        
        Args:
            raw_data: Raw OpenWeatherMap API response
            forecast_date: Date when forecast was made
            
        Returns:
            List of WeatherData objects
        """
        weather_data = []
        
        try:
            if 'list' not in raw_data:
                logger.error("OpenWeatherMap: Invalid response format")
                return []
            
            # Group 3-hour forecasts by day
            daily_forecasts = {}
            
            for item in raw_data['list']:
                dt = datetime.fromtimestamp(item['dt'])
                day = dt.date()
                
                if day not in daily_forecasts:
                    daily_forecasts[day] = []
                
                daily_forecasts[day].append(item)
            
            # Process each day
            for day, forecasts in daily_forecasts.items():
                # Calculate daily high/low from 3-hour forecasts
                temps = [f['main']['temp'] for f in forecasts]
                predicted_high = max(temps) if temps else None
                predicted_low = min(temps) if temps else None
                
                # Average other parameters
                humidity_values = [f['main']['humidity'] for f in forecasts]
                pressure_values = [f['main']['pressure'] for f in forecasts]
                wind_speed_values = [f['wind']['speed'] for f in forecasts]
                wind_dir_values = [f['wind'].get('deg', 0) for f in forecasts]
                cloud_values = [f['clouds']['all'] for f in forecasts]
                
                # Calculate precipitation probability (max of all forecasts for the day)
                precip_prob = 0
                for forecast in forecasts:
                    if 'rain' in forecast:
                        precip_prob = max(precip_prob, forecast.get('pop', 0) * 100)
                    if 'snow' in forecast:
                        precip_prob = max(precip_prob, forecast.get('pop', 0) * 100)
                
                weather_data.append(WeatherData(
                    date=day,
                    forecast_date=forecast_date,
                    predicted_high=predicted_high,
                    predicted_low=predicted_low,
                    humidity=sum(humidity_values) / len(humidity_values) if humidity_values else None,
                    pressure=sum(pressure_values) / len(pressure_values) if pressure_values else None,
                    wind_speed=sum(wind_speed_values) / len(wind_speed_values) if wind_speed_values else None,
                    wind_direction=self._average_wind_direction(wind_dir_values),
                    cloud_cover=sum(cloud_values) / len(cloud_values) if cloud_values else None,
                    precipitation_prob=precip_prob if precip_prob > 0 else None,
                    data_quality_score=0.85,  # Good quality commercial API
                    raw_data={'forecasts': forecasts}
                ))
                
        except Exception as e:
            logger.error(f"OpenWeatherMap: Error parsing forecast data: {e}")
        
        return weather_data
    
    def _average_wind_direction(self, directions: List[float]) -> Optional[float]:
        """Calculate average wind direction accounting for circular nature.
        
        Args:
            directions: List of wind directions in degrees
            
        Returns:
            Average wind direction in degrees
        """
        if not directions:
            return None
        
        import math
        
        # Convert to unit vectors and average
        x_sum = sum(math.cos(math.radians(d)) for d in directions)
        y_sum = sum(math.sin(math.radians(d)) for d in directions)
        
        avg_direction = math.degrees(math.atan2(y_sum, x_sum))
        
        # Normalize to 0-360 range
        if avg_direction < 0:
            avg_direction += 360
        
        return avg_direction
    
    def get_current_weather(self) -> Optional[WeatherData]:
        """Get current weather conditions.
        
        Returns:
            Current weather data or None if failed
        """
        url = f"{self.base_url}/weather"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'imperial'
        }
        
        try:
            response = self._make_request(url, params)
            if not response:
                return None
            
            main = response.get('main', {})
            wind = response.get('wind', {})
            clouds = response.get('clouds', {})
            
            return WeatherData(
                date=date.today(),
                forecast_date=date.today(),
                predicted_high=main.get('temp'),
                predicted_low=None,
                humidity=main.get('humidity'),
                pressure=main.get('pressure'),
                wind_speed=wind.get('speed'),
                wind_direction=wind.get('deg'),
                cloud_cover=clouds.get('all'),
                precipitation_prob=None,
                data_quality_score=0.90,
                raw_data=response
            )
            
        except Exception as e:
            logger.error(f"OpenWeatherMap: Error getting current weather: {e}")
            return None
    
    def get_daily_forecast(self, days: int = 5) -> List[WeatherData]:
        """Get daily forecast using One Call API (if available).
        
        Args:
            days: Number of days to forecast
            
        Returns:
            List of WeatherData objects
        """
        # Note: One Call API 3.0 requires subscription
        # This is a placeholder for potential future implementation
        logger.info("OpenWeatherMap: Daily forecast using standard 5-day/3-hour API")
        return self.get_forecast(days=days)