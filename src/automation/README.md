# Daily Automation System

This module provides comprehensive automation for the Kalshi Weather Predictor, including scheduled data collection, error handling, and system monitoring.

## Overview

The automation system handles:
- **Morning Data Collection**: Automated forecast collection from all weather APIs (6:00 AM PT)
- **Evening Temperature Collection**: Actual temperature collection for validation (8:00 PM PT)
- **Health Monitoring**: Regular system health checks and API status monitoring
- **Error Handling**: Robust error handling with retry logic and graceful degradation
- **Service Management**: Background service management with process monitoring

## Components

### 1. Daily Scheduler (`daily_scheduler.py`)

The main automation engine that orchestrates daily data collection tasks.

**Key Features:**
- Scheduled morning forecast collection from all weather APIs
- Evening actual temperature collection for model validation
- Comprehensive health checks every 4 hours
- Automatic retry logic for failed operations
- Graceful error handling and logging

**Usage:**
```bash
# Run scheduler continuously
python src/automation/daily_scheduler.py

# Run specific task once
python src/automation/daily_scheduler.py --run-once morning
python src/automation/daily_scheduler.py --run-once evening
python src/automation/daily_scheduler.py --run-once health
```

### 2. Service Manager (`service_manager.py`)

Manages the daily scheduler as a background service with process monitoring.

**Key Features:**
- Start/stop/restart scheduler as background service
- Process monitoring and automatic restart on crashes
- Service status reporting with resource usage
- Individual task execution for testing

**Usage:**
```bash
# Start service
python src/automation/service_manager.py start

# Stop service
python src/automation/service_manager.py stop

# Restart service
python src/automation/service_manager.py restart

# Check status
python src/automation/service_manager.py status

# Monitor service (auto-restart on failure)
python src/automation/service_manager.py monitor

# Run specific task once
python src/automation/service_manager.py start --task morning
```

### 3. Error Handler (`error_handler.py`)

Comprehensive error handling system with retry logic and graceful degradation.

**Key Features:**
- Structured error logging with severity levels
- Configurable retry strategies (exponential backoff, linear, fixed interval)
- Circuit breaker pattern for failing services
- Graceful degradation with fallback strategies
- Error trend analysis and alerting

**Usage:**
```python
from src.automation.error_handler import error_handler, retry_with_backoff, ErrorSeverity

# Log errors with context
error_handler.log_error(exception, "context", ErrorSeverity.HIGH)

# Use retry decorator
@retry_with_backoff(max_retries=3, base_delay=1.0)
def api_call():
    # Your API call here
    pass

# Get error summary
summary = error_handler.get_error_summary(hours=24)
```

## Daily Schedule

The automation system runs on the following schedule:

| Time | Task | Description |
|------|------|-------------|
| 06:00 PT | Morning Collection | Fetch forecasts from all weather APIs |
| 20:00 PT | Evening Collection | Collect actual temperatures for validation |
| Every 4 hours | Health Check | Monitor API status and data quality |
| Every 2 hours | Retry Failed | Retry any failed collection attempts |

## Configuration

### Environment Variables

Required API keys in `.env` file:
```bash
OPENWEATHERMAP_API_KEY=your_key_here
TOMORROW_IO_API_KEY=your_key_here
WEATHERBIT_API_KEY=your_key_here
VISUAL_CROSSING_API_KEY=your_key_here
```

### Rate Limits

Configure API rate limits in `config/config.yaml`:
```yaml
data_collection:
  rate_limits:
    openweathermap: 1000  # calls per day
    tomorrow_io: 1000
    weatherbit: 500
    visual_crossing: 1000
```

## Error Handling

### Severity Levels

- **LOW**: Minor issues that don't affect core functionality
- **MEDIUM**: Issues that may impact data quality or availability
- **HIGH**: Significant problems affecting system operation
- **CRITICAL**: System-threatening issues requiring immediate attention

### Retry Strategies

1. **Exponential Backoff**: Delay doubles with each retry (default)
2. **Linear Backoff**: Delay increases linearly
3. **Fixed Interval**: Constant delay between retries

### Circuit Breaker

Automatically disables failing services to prevent cascade failures:
- Opens after 5 consecutive failures
- Attempts recovery after 60 seconds
- Logs all state transitions

## Monitoring and Alerts

### Health Checks

The system performs comprehensive health checks including:
- API connection testing
- Data quality analysis
- Collection success rates
- System resource usage

### Logging

Structured logging with multiple levels:
- **Console**: INFO level for real-time monitoring
- **File**: DEBUG level for detailed troubleshooting
- **Rotation**: 10MB files with 30-day retention

Log files:
- `logs/daily_scheduler.log` - Main scheduler operations
- `logs/service_manager.log` - Service management events
- `logs/actual_temperature_collection.log` - Temperature collection

### Error Tracking

- Error history with full context and stack traces
- Trend analysis to identify recurring issues
- Automatic alerting when error thresholds are exceeded

## Data Quality Monitoring

### Quality Metrics

- **Data Quality Score**: 0-1 score based on validation checks
- **Outlier Detection**: Statistical outlier identification
- **Completeness**: Missing data detection and reporting
- **Consistency**: Cross-source data validation

### Quality Thresholds

Temperature data validation:
- Range: -20°F to 130°F for Los Angeles
- Consistency: High temperature >= Low temperature
- Outliers: Values beyond 1.5 IQR from median

## Backup and Recovery

### Graceful Degradation

When APIs fail, the system:
1. Continues with available data sources
2. Adjusts confidence scores appropriately
3. Uses cached data when possible
4. Logs degraded operation status

### Data Recovery

- Automatic backfill for missing temperature data
- Historical data validation and correction
- Duplicate detection and removal

## Testing

### Unit Tests

```bash
python src/automation/test_daily_scheduler.py
```

### Integration Demo

```bash
python src/automation/demo_daily_scheduler.py
```

### Manual Testing

Test individual components:
```bash
# Test morning collection
python src/automation/service_manager.py start --task morning

# Test evening collection
python src/automation/service_manager.py start --task evening

# Test health check
python src/automation/service_manager.py start --task health
```

## Production Deployment

### System Service Setup

For production deployment, create a systemd service:

```ini
[Unit]
Description=Kalshi Weather Predictor Daily Scheduler
After=network.target

[Service]
Type=simple
User=weather
WorkingDirectory=/path/to/kalshi-weather-predictor
ExecStart=/usr/bin/python3 src/automation/daily_scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Monitoring Integration

The system provides metrics for integration with monitoring tools:
- Process health and resource usage
- API response times and success rates
- Data quality metrics
- Error rates and trends

### Backup Strategy

- Daily data backups to separate storage
- Configuration file versioning
- Log file archival and rotation
- Model checkpoint preservation

## Troubleshooting

### Common Issues

1. **API Rate Limits Exceeded**
   - Check rate limit configuration
   - Verify API key quotas
   - Adjust collection frequency if needed

2. **Service Won't Start**
   - Check log files for errors
   - Verify Python environment and dependencies
   - Ensure proper file permissions

3. **Data Quality Issues**
   - Review API response validation
   - Check for network connectivity issues
   - Verify location coordinates

4. **High Error Rates**
   - Check error summary for patterns
   - Review API service status
   - Verify system resources

### Debug Mode

Enable verbose logging:
```bash
python src/automation/daily_scheduler.py --verbose
```

### Log Analysis

Key log patterns to monitor:
- `ERROR` - Failed operations requiring attention
- `CRITICAL` - System-threatening issues
- `Rate limit` - API quota management
- `Circuit breaker` - Service degradation

## Performance Considerations

### Resource Usage

- Memory: ~50-100MB typical usage
- CPU: Low usage except during collection periods
- Disk: Log rotation prevents unbounded growth
- Network: Respects API rate limits

### Optimization

- Parallel API calls where possible
- Efficient data storage with Parquet format
- Intelligent retry backoff to avoid overwhelming APIs
- Circuit breakers to prevent cascade failures

## Security

### API Key Management

- Store keys in environment variables
- Never commit keys to version control
- Rotate keys regularly
- Monitor for unauthorized usage

### Data Protection

- Local data storage (no cloud dependencies)
- Encrypted log files for sensitive information
- Access control for configuration files
- Audit trail for all operations

## Future Enhancements

Planned improvements:
- Web dashboard for real-time monitoring
- Email/SMS alerting for critical issues
- Machine learning for predictive failure detection
- Integration with external monitoring systems
- Automated model retraining triggers