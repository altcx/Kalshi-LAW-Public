"""Tomorrow.io API client."""

from typing import Dict, Any, Optional, List
from datetime import date, datetime, timedelta
from loguru import logger

from .base_client import BaseWeatherClient, WeatherData, RateLimiter
from ..utils.config import config


class TomorrowClient(BaseWeatherClient):
    """Client for Tomorrow.io API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tomorrow.io client.
        
        Args:
            api_key: Tomorrow.io API key (optional, will try to get from config)
        """
        if api_key is None:
            api_key = config.get_api_key('tomorrow_io')
        
        if not api_key:
            raise ValueError("Tomorrow.io API key is required")
        
        # Free tier: 1000 calls/day, 25 calls/hour
        rate_limiter = RateLimiter(calls_per_minute=25, calls_per_day=1000)
        super().__init__(api_key=api_key, rate_limiter=rate_limiter)
        
        # LAX coordinates for Los Angeles
        self.lat = 33.9425
        self.lon = -118.4081
    
    @property
    def service_name(self) -> str:
        return "Tomorrow.io"
    
    @property
    def base_url(self) -> str:
        return "https://api.tomorrow.io/v4"
    
    def _fetch_forecast_data(self, target_date: date, days: int) -> Optional[Dict]:
        """Fetch forecast data from Tomorrow.io API.
        
        Args:
            target_date: Target date for forecast
            days: Number of days to forecast
            
        Returns:
            Raw API response data
        """
        url = f"{self.base_url}/timelines"
        
        # Calculate end time
        end_time = target_date + timedelta(days=days)
        
        params = {
            'location': f"{self.lat},{self.lon}",
            'fields': [
                'temperatureMax',
                'temperatureMin',
                'temperature',
                'humidity',
                'pressureSeaLevel',
                'windSpeed',
                'windDirection',
                'cloudCover',
                'precipitationProbability'
            ],
            'timesteps': 'daily',
            'startTime': target_date.isoformat(),
            'endTime': end_time.isoformat(),
            'timezone': 'America/Los_Angeles',
            'apikey': self.api_key
        }
        
        # Convert fields list to comma-separated string
        params['fields'] = ','.join(params['fields'])
        
        return self._make_request(url, params)
    
    def _parse_forecast_data(self, raw_data: Dict, forecast_date: date) -> List[WeatherData]:
        """Parse Tomorrow.io forecast data into standardized format.
        
        Args:
            raw_data: Raw Tomorrow.io API response
            forecast_date: Date when forecast was made
            
        Returns:
            List of WeatherData objects
        """
        weather_data = []
        
        try:
            if 'data' not in raw_data or 'timelines' not in raw_data['data']:
                logger.error("Tomorrow.io: Invalid response format")
                return []
            
            timelines = raw_data['data']['timelines']
            if not timelines:
                logger.error("Tomorrow.io: No timeline data found")
                return []
            
            # Get daily timeline
            daily_timeline = None
            for timeline in timelines:
                if timeline.get('timestep') == 'daily':
                    daily_timeline = timeline
                    break
            
            if not daily_timeline or 'intervals' not in daily_timeline:
                logger.error("Tomorrow.io: No daily intervals found")
                return []
            
            # Process each day
            for interval in daily_timeline['intervals']:
                start_time = datetime.fromisoformat(interval['startTime'].replace('Z', '+00:00'))
                day = start_time.date()
                values = interval.get('values', {})
                
                # Convert temperature from Celsius to Fahrenheit
                temp_max_c = values.get('temperatureMax')
                temp_min_c = values.get('temperatureMin')
                temp_c = values.get('temperature')
                
                predicted_high = self._celsius_to_fahrenheit(temp_max_c) if temp_max_c is not None else None
                predicted_low = self._celsius_to_fahrenheit(temp_min_c) if temp_min_c is not None else None
                
                # If no max/min, use current temperature as high
                if predicted_high is None and temp_c is not None:
                    predicted_high = self._celsius_to_fahrenheit(temp_c)
                
                # Convert wind speed from m/s to mph
                wind_speed_ms = values.get('windSpeed')
                wind_speed_mph = wind_speed_ms * 2.237 if wind_speed_ms is not None else None
                
                weather_data.append(WeatherData(
                    date=day,
                    forecast_date=forecast_date,
                    predicted_high=predicted_high,
                    predicted_low=predicted_low,
                    humidity=values.get('humidity'),
                    pressure=values.get('pressureSeaLevel'),
                    wind_speed=wind_speed_mph,
                    wind_direction=values.get('windDirection'),
                    cloud_cover=values.get('cloudCover'),
                    precipitation_prob=values.get('precipitationProbability'),
                    data_quality_score=0.88,  # High quality commercial API
                    raw_data=values
                ))
                
        except Exception as e:
            logger.error(f"Tomorrow.io: Error parsing forecast data: {e}")
        
        return weather_data
    
    def _celsius_to_fahrenheit(self, celsius: float) -> float:
        """Convert Celsius to Fahrenheit.
        
        Args:
            celsius: Temperature in Celsius
            
        Returns:
            Temperature in Fahrenheit
        """
        return (celsius * 9/5) + 32
    
    def get_current_weather(self) -> Optional[WeatherData]:
        """Get current weather conditions.
        
        Returns:
            Current weather data or None if failed
        """
        url = f"{self.base_url}/timelines"
        
        params = {
            'location': f"{self.lat},{self.lon}",
            'fields': [
                'temperature',
                'humidity',
                'pressureSeaLevel',
                'windSpeed',
                'windDirection',
                'cloudCover'
            ],
            'timesteps': 'current',
            'timezone': 'America/Los_Angeles',
            'apikey': self.api_key
        }
        
        # Convert fields list to comma-separated string
        params['fields'] = ','.join(params['fields'])
        
        try:
            response = self._make_request(url, params)
            if not response or 'data' not in response:
                return None
            
            timelines = response['data'].get('timelines', [])
            if not timelines:
                return None
            
            current_timeline = timelines[0]
            intervals = current_timeline.get('intervals', [])
            if not intervals:
                return None
            
            values = intervals[0].get('values', {})
            
            # Convert temperature from Celsius to Fahrenheit
            temp_c = values.get('temperature')
            temp_f = self._celsius_to_fahrenheit(temp_c) if temp_c is not None else None
            
            # Convert wind speed from m/s to mph
            wind_speed_ms = values.get('windSpeed')
            wind_speed_mph = wind_speed_ms * 2.237 if wind_speed_ms is not None else None
            
            return WeatherData(
                date=date.today(),
                forecast_date=date.today(),
                predicted_high=temp_f,
                predicted_low=None,
                humidity=values.get('humidity'),
                pressure=values.get('pressureSeaLevel'),
                wind_speed=wind_speed_mph,
                wind_direction=values.get('windDirection'),
                cloud_cover=values.get('cloudCover'),
                precipitation_prob=None,
                data_quality_score=0.90,
                raw_data=values
            )
            
        except Exception as e:
            logger.error(f"Tomorrow.io: Error getting current weather: {e}")
            return None
    
    def get_hourly_forecast(self, hours: int = 24) -> List[WeatherData]:
        """Get hourly forecast data.
        
        Args:
            hours: Number of hours to forecast
            
        Returns:
            List of WeatherData objects with hourly data
        """
        url = f"{self.base_url}/timelines"
        
        end_time = datetime.now() + timedelta(hours=hours)
        
        params = {
            'location': f"{self.lat},{self.lon}",
            'fields': [
                'temperature',
                'humidity',
                'pressureSeaLevel',
                'windSpeed',
                'windDirection',
                'cloudCover',
                'precipitationProbability'
            ],
            'timesteps': 'hourly',
            'startTime': datetime.now().isoformat(),
            'endTime': end_time.isoformat(),
            'timezone': 'America/Los_Angeles',
            'apikey': self.api_key
        }
        
        # Convert fields list to comma-separated string
        params['fields'] = ','.join(params['fields'])
        
        try:
            response = self._make_request(url, params)
            if not response or 'data' not in response:
                return []
            
            weather_data = []
            timelines = response['data'].get('timelines', [])
            
            for timeline in timelines:
                if timeline.get('timestep') == 'hourly':
                    for interval in timeline.get('intervals', []):
                        start_time = datetime.fromisoformat(interval['startTime'].replace('Z', '+00:00'))
                        values = interval.get('values', {})
                        
                        # Convert temperature from Celsius to Fahrenheit
                        temp_c = values.get('temperature')
                        temp_f = self._celsius_to_fahrenheit(temp_c) if temp_c is not None else None
                        
                        # Convert wind speed from m/s to mph
                        wind_speed_ms = values.get('windSpeed')
                        wind_speed_mph = wind_speed_ms * 2.237 if wind_speed_ms is not None else None
                        
                        weather_data.append(WeatherData(
                            date=start_time.date(),
                            forecast_date=date.today(),
                            predicted_high=temp_f,
                            predicted_low=None,
                            humidity=values.get('humidity'),
                            pressure=values.get('pressureSeaLevel'),
                            wind_speed=wind_speed_mph,
                            wind_direction=values.get('windDirection'),
                            cloud_cover=values.get('cloudCover'),
                            precipitation_prob=values.get('precipitationProbability'),
                            data_quality_score=0.88,
                            raw_data=values
                        ))
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Tomorrow.io: Error getting hourly forecast: {e}")
            return []