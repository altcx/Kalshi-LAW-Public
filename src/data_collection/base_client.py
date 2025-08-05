"""Base weather API client with common functionality."""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, date
from dataclasses import dataclass
import requests
import aiohttp
from loguru import logger

from ..utils.config import config


@dataclass
class WeatherData:
    """Standardized weather data format for all API sources."""
    date: date
    forecast_date: date  # When forecast was made
    predicted_high: Optional[float] = None
    predicted_low: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None  # hPa
    wind_speed: Optional[float] = None  # mph
    wind_direction: Optional[float] = None  # degrees
    cloud_cover: Optional[float] = None  # percentage
    precipitation_prob: Optional[float] = None  # percentage
    data_quality_score: float = 1.0
    raw_data: Optional[Dict[str, Any]] = None


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_day: int = 1000):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls per minute
            calls_per_day: Maximum calls per day
        """
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls = []
        self.daily_calls = 0
        self.last_reset = datetime.now().date()
    
    def can_make_call(self) -> bool:
        """Check if we can make an API call."""
        now = datetime.now()
        current_date = now.date()
        
        # Reset daily counter if new day
        if current_date > self.last_reset:
            self.daily_calls = 0
            self.last_reset = current_date
        
        # Check daily limit
        if self.daily_calls >= self.calls_per_day:
            return False
        
        # Clean old minute calls
        minute_ago = now.timestamp() - 60
        self.minute_calls = [call_time for call_time in self.minute_calls if call_time > minute_ago]
        
        # Check minute limit
        return len(self.minute_calls) < self.calls_per_minute
    
    def record_call(self) -> None:
        """Record that an API call was made."""
        self.minute_calls.append(datetime.now().timestamp())
        self.daily_calls += 1
    
    def wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        if not self.minute_calls:
            return 0.0
        
        oldest_call = min(self.minute_calls)
        time_since_oldest = datetime.now().timestamp() - oldest_call
        
        if len(self.minute_calls) >= self.calls_per_minute:
            return max(0, 60 - time_since_oldest)
        
        return 0.0


class BaseWeatherClient(ABC):
    """Base class for weather API clients."""
    
    def __init__(self, api_key: Optional[str] = None, rate_limiter: Optional[RateLimiter] = None):
        """Initialize base client.
        
        Args:
            api_key: API key for the service
            rate_limiter: Rate limiter instance
        """
        self.api_key = api_key
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kalshi-Weather-Predictor/1.0'
        })
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        """Name of the weather service."""
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for the API."""
        pass
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit would be exceeded."""
        if not self.rate_limiter.can_make_call():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.info(f"{self.service_name}: Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
    
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
        self._wait_for_rate_limit()
        
        for attempt in range(max_retries + 1):
            try:
                self.rate_limiter.record_call()
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"{self.service_name}: Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [401, 403]:  # Auth error
                    logger.error(f"{self.service_name}: Authentication error: {response.status_code}")
                    return None
                else:
                    logger.warning(f"{self.service_name}: HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"{self.service_name}: Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    logger.error(f"{self.service_name}: All retry attempts failed")
                    return None
        
        return None
    
    @abstractmethod
    def _parse_forecast_data(self, raw_data: Dict, forecast_date: date) -> List[WeatherData]:
        """Parse raw API response into standardized format.
        
        Args:
            raw_data: Raw API response
            forecast_date: Date when forecast was made
            
        Returns:
            List of WeatherData objects
        """
        pass
    
    def get_forecast(self, target_date: Optional[date] = None, days: int = 7) -> List[WeatherData]:
        """Get weather forecast.
        
        Args:
            target_date: Target date for forecast (defaults to today)
            days: Number of days to forecast
            
        Returns:
            List of WeatherData objects
        """
        if target_date is None:
            target_date = date.today()
        
        try:
            raw_data = self._fetch_forecast_data(target_date, days)
            if raw_data is None:
                logger.error(f"{self.service_name}: Failed to fetch forecast data")
                return []
            
            forecast_date = date.today()
            parsed_data = self._parse_forecast_data(raw_data, forecast_date)
            
            logger.info(f"{self.service_name}: Successfully fetched {len(parsed_data)} forecast records")
            return parsed_data
            
        except Exception as e:
            logger.error(f"{self.service_name}: Error getting forecast: {e}")
            return []
    
    @abstractmethod
    def _fetch_forecast_data(self, target_date: date, days: int) -> Optional[Dict]:
        """Fetch raw forecast data from API.
        
        Args:
            target_date: Target date for forecast
            days: Number of days to forecast
            
        Returns:
            Raw API response data
        """
        pass
    
    def test_connection(self) -> bool:
        """Test API connection and authentication.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get a simple forecast to test connection
            result = self.get_forecast(days=1)
            return len(result) > 0
        except Exception as e:
            logger.error(f"{self.service_name}: Connection test failed: {e}")
            return False