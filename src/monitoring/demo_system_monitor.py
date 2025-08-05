"""Demo script for the comprehensive logging and monitoring system."""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import time
import json

from .system_monitor import SystemMonitor
from .performance_tracker import PerformanceTracker
from .model_retrainer import ModelRetrainer
from ..utils.data_manager import DataManager


def create_sample_system_activity():
    """Create sample system activity for demonstration."""
    print("Creating sample system activity...")
    
    data_manager = DataManager()
    
    # Create some sample data
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create weather data
    for source in ['nws', 'openweather', 'tomorrow']:
        data = []
        for i, date_val in enumerate(dates):
            base_temp = 75 + 5 * np.sin(i * 0.3)
            noise = np.random.normal(0, 2)
            
            data.append({
                'date': date_val,
                'forecast_date': date_val - timedelta(days=1),
                'predicted_high': base_temp + noise,
                'predicted_low': base_temp - 15 + noise,
                'humidity': 60 + np.random.normal(0, 10),
                'pressure': 1013 + np.random.normal(0, 5),
                'wind_speed': 8 + np.random.normal(0, 3),
                'wind_direction': np.random.uniform(0, 360),
                'cloud_cover': np.random.uniform(0, 100),
                'precipitation_prob': np.random.uniform(0, 30),
                'data_quality_score': 0.85 + np.random.normal(0, 0.1)
            })
        
        source_df = pd.DataFrame(data)
        data_manager.save_source_data(source, source_df)
    
    # Create actual temperatures
    actual_data = []
    for i, date_val in enumerate(dates):
        base_temp = 75 + 5 * np.sin(i * 0.3)
        actual_noise = np.random.normal(0, 1)
        
        actual_data.append({
            'date': date_val,
            'actual_high': base_temp + actual_noise,
            'actual_low': base_temp - 15 + actual_noise,
            'source': 'NOAA'
        })
    
    actual_df = pd.DataFrame(actual_data)
    data_manager.save_source_data('actual_temperatures', actual_df)
    
    return data_manager, len(dates)


def demo_system_monitoring():
    """Demonstrate the comprehensive logging and monitoring system."""
    print("=== Comprehensive Logging and Monitoring System Demo ===\n")
    
    # Create sample data
    data_manager, data_points = create_sample_system_activity()
    
    # Initialize monitoring components
    performance_tracker = PerformanceTracker(data_manager)
    model_retrainer = ModelRetrainer(data_manager, performance_tracker)
    system_monitor = SystemMonitor(data_manager, performance_tracker, model_retrainer)
    
    print(f"Initialized monitoring system with {data_points} days of sample data\n")
    
    # Demonstrate prediction logging
    print("=== Prediction Logging ===")
    target_date = date.today() + timedelta(days=1)
    prediction = 78.5
    confidence = 0.85
    
    model_info = {
        'model_type': 'ensemble',
        'models_used': ['xgboost', 'lightgbm', 'linear_regression'],
        'weights': [0.5, 0.3, 0.2]
    }
    
    features_used = [
        'nws_predicted_high', 'openweather_predicted_high', 'tomorrow_predicted_high',
        'humidity_avg', 'pressure_avg', 'day_of_year', 'month'
    ]
    
    system_monitor.log_prediction(
        prediction, confidence, target_date, model_info, features_used
    )
    print(f"Logged prediction: {prediction}°F (confidence: {confidence:.1%}) for {target_date}")
    
    # Demonstrate actual temperature logging
    print("\n=== Actual Temperature Logging ===")
    yesterday = date.today() - timedelta(days=1)
    actual_temp = 77.2
    
    system_monitor.log_actual_temperature(yesterday, actual_temp, "NOAA")
    print(f"Logged actual temperature: {actual_temp}°F for {yesterday}")
    
    # Demonstrate API call logging
    print("\n=== API Call Logging ===")
    
    # Successful API call
    system_monitor.log_api_call(
        api_name="openweather",
        endpoint="/weather/forecast",
        success=True,
        response_time=1.2,
        data_quality=0.92
    )
    print("Logged successful API call to OpenWeatherMap")
    
    # Failed API call
    system_monitor.log_api_call(
        api_name="weatherbit",
        endpoint="/forecast/daily",
        success=False,
        response_time=5.0,
        error_message="Connection timeout",
        data_quality=None
    )
    print("Logged failed API call to Weatherbit")
    
    # Slow API call
    system_monitor.log_api_call(
        api_name="tomorrow",
        endpoint="/weather/forecast",
        success=True,
        response_time=35.0,  # Exceeds threshold
        data_quality=0.88
    )
    print("Logged slow API call to Tomorrow.io (will trigger alert)")
    
    # Demonstrate error logging
    print("\n=== Error Logging ===")
    
    # Warning level error
    try:
        # Simulate a warning condition
        raise ValueError("Data quality below threshold for visual_crossing")
    except Exception as e:
        system_monitor.log_error(
            e, 
            context={'source': 'visual_crossing', 'quality_score': 0.65},
            severity='warning'
        )
        print("Logged warning-level error")
    
    # Critical error
    try:
        # Simulate a critical error
        raise ConnectionError("Unable to connect to primary database")
    except Exception as e:
        system_monitor.log_error(
            e,
            context={'component': 'database', 'retry_count': 3},
            severity='critical'
        )
        print("Logged critical error (will trigger alert)")
    
    # Demonstrate performance metrics logging
    print("\n=== Performance Metrics Logging ===")
    
    performance_metrics = {
        'prediction_accuracy_3f': 0.82,
        'average_mae': 1.8,
        'api_success_rate': 0.95,
        'data_quality_avg': 0.87,
        'response_time_avg': 2.1,
        'memory_usage_mb': 245,
        'disk_usage_mb': 150,
        'active_models': 4,
        'predictions_made_today': 12
    }
    
    system_monitor.log_performance_metrics(performance_metrics)
    print("Logged performance metrics")
    
    # Wait a moment for logs to be written
    time.sleep(1)
    
    # Demonstrate system health monitoring
    print("\n=== System Health Monitoring ===")
    health_report = system_monitor.monitor_system_health()
    
    print(f"Overall system status: {health_report['overall_status'].upper()}")
    print(f"Health check timestamp: {health_report['timestamp']}")
    
    print("\nComponent Health:")
    for component, health in health_report['components'].items():
        status = health.get('status', 'unknown')
        print(f"  - {component.capitalize()}: {status.upper()}")
        
        if component == 'apis' and 'apis' in health:
            for api, api_health in health['apis'].items():
                print(f"    - {api}: {api_health['status']} ({api_health['failure_count']} failures)")
        
        elif component == 'data' and 'sources' in health:
            for source, source_health in health['sources'].items():
                records = source_health.get('records', 0)
                quality = source_health.get('quality_score', 0)
                print(f"    - {source}: {source_health['status']} ({records} records, {quality:.2f} quality)")
        
        elif component == 'models' and 'models' in health:
            for model, model_health in health['models'].items():
                accuracy = model_health.get('accuracy_3f', 0)
                mae = model_health.get('mae', 0)
                print(f"    - {model}: {model_health['status']} ({accuracy:.1%} accuracy, {mae:.1f}°F MAE)")
    
    # Show alerts
    if health_report.get('alerts'):
        print(f"\nRecent Alerts ({len(health_report['alerts'])}):")
        for alert in health_report['alerts'][-5:]:  # Show last 5 alerts
            timestamp = alert['timestamp'][:19]  # Remove microseconds
            print(f"  - [{timestamp}] {alert['severity'].upper()}: {alert['message']}")
    
    # Show recommendations
    if health_report.get('recommendations'):
        print(f"\nRecommendations:")
        for i, rec in enumerate(health_report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Demonstrate system dashboard
    print("\n=== System Dashboard ===")
    dashboard = system_monitor.get_system_dashboard()
    
    print(f"Dashboard generated at: {dashboard['timestamp']}")
    print(f"Recent metrics snapshots: {len(dashboard.get('recent_metrics', []))}")
    print(f"Recent errors: {len(dashboard.get('recent_errors', []))}")
    print(f"Recent alerts: {len(dashboard.get('recent_alerts', []))}")
    
    # Show API failure summary
    api_failures = dashboard.get('api_failure_counts', {})
    if api_failures:
        print(f"\nAPI Failure Summary:")
        for api, count in api_failures.items():
            print(f"  - {api}: {count} failures")
    
    # Show data summary
    data_summary = dashboard.get('data_summary', {})
    if data_summary:
        print(f"\nData Summary:")
        for source, info in data_summary.items():
            if 'records' in info:
                size_mb = info.get('file_size_mb', 0)
                print(f"  - {source}: {info['records']} records ({size_mb:.1f} MB)")
    
    # Demonstrate log file information
    print("\n=== Log Files Created ===")
    logs_dir = system_monitor.logs_dir
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        for log_file in sorted(log_files):
            size_kb = log_file.stat().st_size / 1024
            print(f"  - {log_file.name}: {size_kb:.1f} KB")
    
    # Show recent log entries (last few lines from main log)
    main_log = logs_dir / "weather_predictor.log"
    if main_log.exists():
        print(f"\n=== Recent Log Entries (from {main_log.name}) ===")
        try:
            with open(main_log, 'r') as f:
                lines = f.readlines()
                for line in lines[-5:]:  # Show last 5 lines
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"  Error reading log file: {e}")
    
    # Demonstrate cleanup functionality
    print("\n=== Log Cleanup ===")
    print("Log cleanup functionality available (demo skipped to preserve logs)")
    print("Use system_monitor.cleanup_old_logs(days_to_keep=30) to clean old logs")
    
    print("\n=== Demo Complete ===")
    print("Comprehensive logging and monitoring system successfully demonstrated!")
    print(f"Check the {logs_dir} directory for detailed log files.")
    print("The system provides:")
    print("  - Structured logging with multiple log files")
    print("  - Real-time system health monitoring")
    print("  - Automated alerting for issues")
    print("  - Performance metrics tracking")
    print("  - Error tracking and analysis")
    print("  - API monitoring and failure detection")
    print("  - Comprehensive system dashboard")


if __name__ == "__main__":
    demo_system_monitoring()