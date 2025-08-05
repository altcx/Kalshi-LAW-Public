# Weather API Clients

This module provides a comprehensive set of weather API clients for collecting forecast data from multiple sources. All clients implement a standardized interface and include robust error handling, rate limiting, and retry logic.

## Supported APIs

### 1. National Weather Service (NWS)
- **API Key Required**: No (free government API)
- **Rate Limits**: No official limits (recommended: 30 calls/minute)
- **Coverage**: 7-day forecast
- **Data Quality**: Very high (0.95)
- **Features**: Official US government weather data, gridpoint forecasts

### 2. OpenWeatherMap
- **API Key Required**: Yes (free tier: 1000 calls/day)
- **Rate Limits**: 60 calls/minute, 1000 calls/day
- **Coverage**: 5-day forecast (3-hour intervals)
- **Data Quality**: High (0.85)
- **Features**: Current weather, forecasts, historical data

### 3. Tomorrow.io
- **API Key Required**: Yes (free tier: 1000 calls/day)
- **Rate Limits**: 25 calls/hour, 1000 calls/day
- **Coverage**: Up to 15-day forecast
- **Data Quality**: Very high (0.88)
- **Features**: High-quality commercial data, detailed parameters

### 4. Weatherbit
- **API Key Required**: Yes (free tier: 500 calls/day)
- **Rate Limits**: 60 calls/minute, 500 calls/day
- **Coverage**: 16-day forecast
- **Data Quality**: High (0.82)
- **Features**: Daily and hourly forecasts, historical data

### 5. Visual Crossing
- **API Key Required**: Yes (free tier: 1000 records/day)
- **Rate Limits**: Conservative (60 calls/minute, 100 calls/day)
- **Coverage**: Extended forecast periods
- **Data Quality**: Good (0.80)
- **Features**: Historical weather data, flexible queries

## Quick Start

### Basic Usage

```python
from src.data_collection import WeatherClientFactory, WeatherDataCollector

# Create individual client
factory = WeatherClientFactory()
nws_client = factory.create_client('nws')
forecast = nws_client.get_forecast(days=3)

# Use data collector for multiple sources
collector = WeatherDataCollector()
all_forecasts = collector.get_all_forecasts(days=7)
```

### Configuration

1. **API Keys**: Add your API keys to `.env` file:
```bash
OPENWEATHERMAP_API_KEY=your_key_here
TOMORROW_IO_API_KEY=your_key_here
WEATHERBIT_API_KEY=your_key_here
VISUAL_CROSSING_API_KEY=your_key_here
```

2. **Rate Limits**: Configure in `config/config.yaml`:
```yaml
data_collection:
  rate_limits:
    openweathermap: 1000
    tomorrow_io: 1000
    weatherbit: 500
    visual_crossing: 1000
```

## Data Format

All clients return data in a standardized `WeatherData` format:

```python
@dataclass
class WeatherData:
    date: date                          # Forecast date
    forecast_date: date                 # When forecast was made
    predicted_high: Optional[float]     # High temperature (°F)
    predicted_low: Optional[float]      # Low temperature (°F)
    humidity: Optional[float]           # Relative humidity (%)
    pressure: Optional[float]           # Atmospheric pressure (hPa)
    wind_speed: Optional[float]         # Wind speed (mph)
    wind_direction: Optional[float]     # Wind direction (degrees)
    cloud_cover: Optional[float]        # Cloud coverage (%)
    precipitation_prob: Optional[float] # Precipitation probability (%)
    data_quality_score: float           # Quality score (0-1)
    raw_data: Optional[Dict]            # Original API response
```

## Error Handling

### Rate Limiting
- Automatic rate limiting with configurable limits
- Exponential backoff for rate limit violations
- Graceful degradation when limits are reached

### API Failures
- Automatic retry with exponential backoff
- Graceful handling of authentication errors
- Fallback to cached data when available

### Data Quality
- Outlier detection and filtering
- Data validation and consistency checks
- Quality scoring for each data point

## Testing

### Test Individual Clients
```python
python src/data_collection/test_clients.py
```

### Test Specific Service
```python
from src.data_collection import WeatherClientFactory

factory = WeatherClientFactory()
client = factory.create_client('nws')
result = client.test_connection()
print(f"Connection test: {'PASSED' if result else 'FAILED'}")
```

## Advanced Usage

### Custom Rate Limiting
```python
from src.data_collection import RateLimiter, OpenWeatherClient

# Custom rate limiter
rate_limiter = RateLimiter(calls_per_minute=30, calls_per_day=500)
client = OpenWeatherClient(rate_limiter=rate_limiter)
```

### Historical Data
```python
from datetime import date, timedelta

# Get historical data (supported by some APIs)
start_date = date.today() - timedelta(days=30)
end_date = date.today()

weatherbit_client = factory.create_client('weatherbit')
historical_data = weatherbit_client.get_historical_weather(start_date, end_date)
```

### Current Conditions
```python
# Get current weather conditions
collector = WeatherDataCollector()
current_conditions = collector.get_current_conditions()

for service, data in current_conditions.items():
    if data:
        print(f"{service}: {data.predicted_high}°F")
```

## API-Specific Features

### NWS Client
- No API key required
- Official US government data
- Gridpoint-based forecasts for high accuracy
- Current conditions from observation stations

### OpenWeatherMap Client
- 5-day/3-hour forecast aggregated to daily
- Current weather conditions
- Support for One Call API (subscription required)

### Tomorrow.io Client
- High-quality commercial data
- Hourly and daily forecasts
- Advanced atmospheric parameters
- Timeline-based API structure

### Weatherbit Client
- Extended 16-day forecasts
- Historical weather data
- Hourly forecasts (48 hours)
- Comprehensive weather parameters

### Visual Crossing Client
- Flexible date range queries
- Historical weather data
- Hourly forecasts
- Weather timeline visualization

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys in `.env` file
   - Check API key validity and quotas
   - Ensure correct service name mapping

2. **Rate Limit Exceeded**
   - Check daily/hourly quotas
   - Adjust rate limiting configuration
   - Consider upgrading API plans

3. **Connection Timeouts**
   - Check internet connectivity
   - Verify API endpoint availability
   - Increase timeout values if needed

4. **Data Quality Issues**
   - Review data validation thresholds
   - Check for API service outages
   - Verify location coordinates

### Debugging

Enable detailed logging:
```python
from loguru import logger
logger.add("weather_api_debug.log", level="DEBUG")
```

Check client status:
```python
collector = WeatherDataCollector()
status = collector.get_client_status()
print(status)
```

## Performance Considerations

- **Caching**: Implement caching for frequently requested data
- **Batch Requests**: Use data collector for efficient multi-source collection
- **Rate Management**: Monitor API usage to stay within limits
- **Error Recovery**: Implement fallback strategies for critical operations

## Contributing

When adding new weather API clients:

1. Inherit from `BaseWeatherClient`
2. Implement required abstract methods
3. Add proper error handling and rate limiting
4. Include comprehensive tests
5. Update documentation and factory registry