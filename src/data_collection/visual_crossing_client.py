"""Visual Crossing Weather API client."""

from typing import Dict, Any, Optional, List
from datetime import date, datetime, timedelta
from loguru import logger

from .base_client import BaseWeatherClient, WeatherData, RateLimiter
from ..utils.config import config


class VisualCrossingClient(BaseWeatherClient):
    """Client for Visual Crossing Weather API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Visual Crossing client.
        
        Args:
            api_key: Visual Crossing API key (optional, will try to get from config)
        """
        if api_key is None:
            api_key = config.get_api_key('visual_crossing')
        
        if not api_key:
            raise ValueError("Visual Crossing API key is required")
        
        # Free tier: 1000 records/day
        rate_limiter = RateLimiter(calls_per_minute=60, calls_per_day=100)  # Conservative
        super().__init__(api_key=api_key, rate_limiter=rate_limiter)
        
        # LAX coordinates for Los Angeles
        self.location = "33.9425,-118.4081"  # Visual Crossing uses lat,lon format
    
    @property
    def service_name(self) -> str:
        return "Visual Crossing"
    
    @property
    def base_url(self) -> str:
        return "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    def _fetch_forecast_data(self, target_date: date, days: int) -> Optional[Dict]:
        """Fetch forecast data from Visual Crossing API.
        
        Args:
            target_date: Target date for forecast
            days: Number of days to forecast
            
        Returns:
            Raw API response data
        """
        # Calculate end date
        end_date = target_date + timedelta(days=days-1)
        
        # Build URL with date range
        url = f"{self.base_url}/{self.location}/{target_date.isoformat()}/{end_date.isoformat()}"
        
        params = {
            'key': self.api_key,
            'unitGroup': 'us',  # US units (Fahrenheit, mph, etc.)
            'include': 'days',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,pressure,windspeed,winddir,cloudcover,precipprob'
        }
        
        return self._make_request(url, params)
    
    def _parse_forecast_data(self, raw_data: Dict, forecast_date: date) -> List[WeatherData]:
        """Parse Visual Crossing forecast data into standardized format.
        
        Args:
            raw_data: Raw Visual Crossing API response
            forecast_date: Date when forecast was made
            
        Returns:
            List of WeatherData objects
        """
        weather_data = []
        
        try:
            if 'days' not in raw_data:
                logger.error("Visual Crossing: Invalid response format")
                return []
            
            # Process each day
            for day_data in raw_data['days']:
                # Parse date
                date_str = day_data.get('datetime')
                if not date_str:
                    continue
                
                day = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                weather_data.append(WeatherData(
                    date=day,
                    forecast_date=forecast_date,
                    predicted_high=day_data.get('tempmax'),
                    predicted_low=day_data.get('tempmin'),
                    humidity=day_data.get('humidity'),
                    pressure=day_data.get('pressure'),  # Already in mb/hPa
                    wind_speed=day_data.get('windspeed'),
                    wind_direction=day_data.get('winddir'),
                    cloud_cover=day_data.get('cloudcover'),
                    precipitation_prob=day_data.get('precipprob'),
                    data_quality_score=0.80,  # Good quality commercial API
                    raw_data=day_data
                ))
                
        except Exception as e:
            logger.error(f"Visual Crossing: Error parsing forecast data: {e}")
        
        return weather_data
    
    def get_current_weather(self) -> Optional[WeatherData]:
        """Get current weather conditions.
        
        Returns:
            Current weather data or None if failed
        """
        # Get today's data which includes current conditions
        url = f"{self.base_url}/{self.location}/today"
        
        params = {
            'key': self.api_key,
            'unitGroup': 'us',
            'include': 'current',
            'elements': 'datetime,temp,humidity,pressure,windspeed,winddir,cloudcover'
        }
        
        try:
            response = self._make_request(url, params)
            if not response:
                return None
            
            current_conditions = response.get('currentConditions')
            if not current_conditions:
                # Fallback to today's data
                days = response.get('days', [])
                if days:
                    current_conditions = days[0]
                else:
                    return None
            
            return WeatherData(
                date=date.today(),
                forecast_date=date.today(),
                predicted_high=current_conditions.get('temp'),
                predicted_low=None,
                humidity=current_conditions.get('humidity'),
                pressure=current_conditions.get('pressure'),
                wind_speed=current_conditions.get('windspeed'),
                wind_direction=current_conditions.get('winddir'),
                cloud_cover=current_conditions.get('cloudcover'),
                precipitation_prob=None,
                data_quality_score=0.85,
                raw_data=current_conditions
            )
            
        except Exception as e:
            logger.error(f"Visual Crossing: Error getting current weather: {e}")
            return None
    
    def get_historical_weather(self, start_date: date, end_date: date) -> List[WeatherData]:
        """Get historical weather data.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of WeatherData objects with historical data
        """
        url = f"{self.base_url}/{self.location}/{start_date.isoformat()}/{end_date.isoformat()}"
        
        params = {
            'key': self.api_key,
            'unitGroup': 'us',
            'include': 'days',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,pressure,windspeed,winddir,cloudcover'
        }
        
        try:
            response = self._make_request(url, params)
            if not response or 'days' not in response:
                return []
            
            weather_data = []
            
            for day_data in response['days']:
                # Parse date
                date_str = day_data.get('datetime')
                if not date_str:
                    continue
                
                day = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                weather_data.append(WeatherData(
                    date=day,
                    forecast_date=day,  # Historical data
                    predicted_high=day_data.get('tempmax'),
                    predicted_low=day_data.get('tempmin'),
                    humidity=day_data.get('humidity'),
                    pressure=day_data.get('pressure'),
                    wind_speed=day_data.get('windspeed'),
                    wind_direction=day_data.get('winddir'),
                    cloud_cover=day_data.get('cloudcover'),
                    precipitation_prob=None,  # Not typically available in historical
                    data_quality_score=0.90,  # Historical data is typically accurate
                    raw_data=day_data
                ))
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Visual Crossing: Error getting historical weather: {e}")
            return []
    
    def get_hourly_forecast(self, hours: int = 24) -> List[WeatherData]:
        """Get hourly forecast data.
        
        Args:
            hours: Number of hours to forecast
            
        Returns:
            List of WeatherData objects with hourly data
        """
        # Calculate end time
        end_time = datetime.now() + timedelta(hours=hours)
        end_date = end_time.date()
        
        url = f"{self.base_url}/{self.location}/{date.today().isoformat()}/{end_date.isoformat()}"
        
        params = {
            'key': self.api_key,
            'unitGroup': 'us',
            'include': 'hours',
            'elements': 'datetime,temp,humidity,pressure,windspeed,winddir,cloudcover,precipprob'
        }
        
        try:
            response = self._make_request(url, params)
            if not response or 'days' not in response:
                return []
            
            weather_data = []
            
            for day in response['days']:
                hours_data = day.get('hours', [])
                for hour_data in hours_data:
                    # Parse datetime
                    datetime_str = hour_data.get('datetime')
                    if not datetime_str:
                        continue
                    
                    # Combine date and time
                    date_str = day.get('datetime')
                    full_datetime_str = f"{date_str}T{datetime_str}"
                    dt = datetime.fromisoformat(full_datetime_str)
                    
                    # Only include future hours
                    if dt <= datetime.now():
                        continue
                    
                    weather_data.append(WeatherData(
                        date=dt.date(),
                        forecast_date=date.today(),
                        predicted_high=hour_data.get('temp'),
                        predicted_low=None,
                        humidity=hour_data.get('humidity'),
                        pressure=hour_data.get('pressure'),
                        wind_speed=hour_data.get('windspeed'),
                        wind_direction=hour_data.get('winddir'),
                        cloud_cover=hour_data.get('cloudcover'),
                        precipitation_prob=hour_data.get('precipprob'),
                        data_quality_score=0.80,
                        raw_data=hour_data
                    ))
            
            return weather_data[:hours]  # Limit to requested number of hours
            
        except Exception as e:
            logger.error(f"Visual Crossing: Error getting hourly forecast: {e}")
            return []