#!/usr/bin/env python3
"""Daily automation scheduler for weather data collection and prediction pipeline."""

import sys
import time
import schedule
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
from loguru import logger
import threading
import signal

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.client_factory import WeatherDataCollector
from src.data_collection.actual_temperature_collector import ActualTemperatureCollector
from src.utils.data_manager import DataManager
from src.utils.config import config
from src.automation.prediction_pipeline import PredictionPipeline
from src.automation.alert_system import alert_system


class DailyScheduler:
    """Manages daily automation tasks for weather data collection."""
    
    def __init__(self):
        """Initialize the daily scheduler."""
        self.data_manager = DataManager()
        self.weather_collector = WeatherDataCollector()
        self.actual_temp_collector = ActualTemperatureCollector()
        self.prediction_pipeline = PredictionPipeline()
        self.running = False
        self.setup_logging()
        
        # Track last successful runs
        self.last_morning_run = None
        self.last_evening_run = None
        self.last_prediction_run = None
        
        logger.info("Daily scheduler initialized")
    
    def setup_logging(self):
        """Setup logging for the scheduler."""
        logger.remove()
        
        # Console logging
        logger.add(
            sys.stdout, 
            level="INFO", 
            format="{time:HH:mm:ss} | {level} | {message}"
        )
        
        # File logging
        log_file = Path("logs/daily_scheduler.log")
        log_file.parent.mkdir(exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="10 MB",
            retention="30 days"
        )
    
    def morning_data_collection(self):
        """Execute morning forecast data collection from all APIs."""
        logger.info("=== Starting Morning Data Collection ===")
        
        try:
            # Get forecasts from all available sources
            forecasts = self.weather_collector.get_all_forecasts(days=7)
            
            successful_sources = 0
            total_records = 0
            
            for source, forecast_data in forecasts.items():
                if forecast_data:
                    # Convert to DataFrame format for storage
                    import pandas as pd
                    
                    records = []
                    for weather_data in forecast_data:
                        record = {
                            'date': weather_data.date,
                            'forecast_date': weather_data.forecast_date,
                            'predicted_high': weather_data.predicted_high,
                            'predicted_low': weather_data.predicted_low,
                            'humidity': weather_data.humidity,
                            'pressure': weather_data.pressure,
                            'wind_speed': weather_data.wind_speed,
                            'wind_direction': weather_data.wind_direction,
                            'cloud_cover': weather_data.cloud_cover,
                            'precipitation_prob': weather_data.precipitation_prob,
                            'data_quality_score': weather_data.data_quality_score
                        }
                        records.append(record)
                    
                    if records:
                        df = pd.DataFrame(records)
                        success = self.data_manager.append_daily_data(source, df, validate=True)
                        
                        if success:
                            successful_sources += 1
                            total_records += len(records)
                            logger.info(f"✓ {source}: Collected {len(records)} forecast records")
                        else:
                            logger.error(f"✗ {source}: Failed to store forecast data")
                    else:
                        logger.warning(f"⚠ {source}: No forecast records to store")
                else:
                    logger.warning(f"⚠ {source}: No forecast data received")
            
            # Update last successful run
            self.last_morning_run = datetime.now()
            
            logger.info(f"Morning collection completed: {successful_sources} sources, {total_records} total records")
            return successful_sources > 0
            
        except Exception as e:
            logger.error(f"Error during morning data collection: {e}")
            return False
    
    def evening_actual_temperature_collection(self):
        """Execute evening actual temperature collection."""
        logger.info("=== Starting Evening Actual Temperature Collection ===")
        
        try:
            # Collect actual temperature for yesterday
            yesterday = date.today() - timedelta(days=1)
            
            temperature = self.actual_temp_collector.collect_daily_high_temperature(yesterday)
            
            if temperature is not None:
                logger.info(f"✓ Successfully collected actual temperature: {temperature:.1f}°F for {yesterday}")
                
                # Update last successful run
                self.last_evening_run = datetime.now()
                return True
            else:
                logger.error(f"✗ Failed to collect actual temperature for {yesterday}")
                return False
                
        except Exception as e:
            logger.error(f"Error during evening temperature collection: {e}")
            return False
    
    def daily_prediction_and_trading_pipeline(self):
        """Execute daily prediction and trading recommendation generation."""
        logger.info("=== Starting Daily Prediction and Trading Pipeline ===")
        
        try:
            # Run prediction pipeline for today
            today = date.today()
            results = self.prediction_pipeline.run_daily_prediction_pipeline(today)
            
            if results['success']:
                prediction = results['prediction']
                recommendations = results['recommendations']
                alerts = results['alerts']
                
                logger.info(f"✓ Prediction generated: {prediction['predicted_high']:.1f}°F "
                           f"(confidence: {prediction['confidence']:.3f})")
                
                # Check for high-confidence opportunities
                if prediction['confidence'] >= 0.85 and recommendations:
                    high_value_recs = [r for r in recommendations if r.get('expected_value', 0) > 0.1]
                    if high_value_recs:
                        alert_system.check_high_confidence_opportunity(prediction, high_value_recs)
                
                # Generate alerts for significant changes
                for alert_data in alerts:
                    logger.info(f"Alert: {alert_data.get('message', 'Unknown alert')}")
                
                logger.info(f"Generated {len(recommendations)} trading recommendations")
                logger.info(f"Generated {len(alerts)} alerts")
                
                # Update last successful run
                self.last_prediction_run = datetime.now()
                return True
            else:
                logger.error("✗ Prediction pipeline failed")
                for error in results.get('errors', []):
                    logger.error(f"Pipeline error: {error}")
                return False
                
        except Exception as e:
            logger.error(f"Error during prediction and trading pipeline: {e}")
            return False
    
    def generate_trading_alerts(self):
        """Generate and process trading alerts."""
        logger.info("=== Processing Trading Alerts ===")
        
        try:
            # Get active high-priority alerts
            critical_alerts = alert_system.get_active_alerts(severity_filter=None)
            
            high_priority_alerts = [
                alert for alert in critical_alerts
                if alert.type in ['high_confidence_opportunity', 'significant_prediction_change']
                and not alert.acknowledged
            ]
            
            if high_priority_alerts:
                logger.info(f"Found {len(high_priority_alerts)} high-priority trading alerts")
                
                for alert in high_priority_alerts:
                    logger.info(f"🚨 {alert.title}")
                    logger.info(f"   {alert.message}")
                    
                    # Auto-acknowledge info alerts after processing
                    if alert.severity.value in ['info', 'success']:
                        alert_system.acknowledge_alert(alert.id)
            else:
                logger.info("No high-priority trading alerts at this time")
            
            return len(high_priority_alerts) > 0
            
        except Exception as e:
            logger.error(f"Error processing trading alerts: {e}")
            return False
    
    def health_check(self):
        """Perform system health check."""
        logger.info("=== Performing System Health Check ===")
        
        try:
            health_status = {
                'timestamp': datetime.now(),
                'api_connections': {},
                'data_quality': {},
                'collection_status': {},
                'overall_health': 'healthy'
            }
            
            # Test API connections
            connection_results = self.weather_collector.test_all_connections()
            health_status['api_connections'] = connection_results
            
            healthy_apis = sum(1 for result in connection_results.values() if result)
            total_apis = len(connection_results)
            
            logger.info(f"API Health: {healthy_apis}/{total_apis} APIs responding")
            
            # Check data quality for recent collections
            for source in ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']:
                quality_summary = self.data_manager.get_data_quality_summary(source, days=7)
                health_status['data_quality'][source] = quality_summary
                
                if 'error' not in quality_summary:
                    avg_quality = quality_summary.get('avg_quality_score', 0)
                    logger.info(f"{source} quality: {avg_quality:.3f}")
            
            # Check collection status
            temp_status = self.actual_temp_collector.get_temperature_collection_status(days=7)
            health_status['collection_status'] = temp_status
            
            if 'error' not in temp_status:
                collection_rate = temp_status.get('collection_rate', 0)
                logger.info(f"Temperature collection rate: {collection_rate:.1%}")
                
                # Determine overall health
                if healthy_apis < total_apis * 0.5:  # Less than 50% APIs working
                    health_status['overall_health'] = 'degraded'
                elif collection_rate < 0.7:  # Less than 70% temperature collection
                    health_status['overall_health'] = 'degraded'
            
            logger.info(f"Overall system health: {health_status['overall_health']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {'error': str(e), 'overall_health': 'error'}
    
    def retry_failed_collection(self, max_retries: int = 3, backoff_minutes: int = 30):
        """Retry failed data collection with exponential backoff."""
        logger.info("=== Retry Failed Collection ===")
        
        current_hour = datetime.now().hour
        
        # Determine what to retry based on time of day
        if 6 <= current_hour <= 12:  # Morning retry window
            if self.last_morning_run is None or (datetime.now() - self.last_morning_run).total_seconds() > 3600:
                logger.info("Retrying morning data collection...")
                for attempt in range(max_retries):
                    if self.morning_data_collection():
                        logger.info(f"Morning collection retry successful on attempt {attempt + 1}")
                        break
                    else:
                        if attempt < max_retries - 1:
                            wait_time = backoff_minutes * (2 ** attempt)
                            logger.info(f"Retry attempt {attempt + 1} failed, waiting {wait_time} minutes")
                            time.sleep(wait_time * 60)
        
        elif 7 <= current_hour <= 14:  # Prediction retry window
            if self.last_prediction_run is None or (datetime.now() - self.last_prediction_run).total_seconds() > 3600:
                logger.info("Retrying prediction and trading pipeline...")
                for attempt in range(max_retries):
                    if self.daily_prediction_and_trading_pipeline():
                        logger.info(f"Prediction pipeline retry successful on attempt {attempt + 1}")
                        break
                    else:
                        if attempt < max_retries - 1:
                            wait_time = backoff_minutes * (2 ** attempt)
                            logger.info(f"Retry attempt {attempt + 1} failed, waiting {wait_time} minutes")
                            time.sleep(wait_time * 60)
        
        elif 18 <= current_hour <= 23:  # Evening retry window
            if self.last_evening_run is None or (datetime.now() - self.last_evening_run).total_seconds() > 3600:
                logger.info("Retrying evening temperature collection...")
                for attempt in range(max_retries):
                    if self.evening_actual_temperature_collection():
                        logger.info(f"Evening collection retry successful on attempt {attempt + 1}")
                        break
                    else:
                        if attempt < max_retries - 1:
                            wait_time = backoff_minutes * (2 ** attempt)
                            logger.info(f"Retry attempt {attempt + 1} failed, waiting {wait_time} minutes")
                            time.sleep(wait_time * 60)
    
    def setup_schedule(self):
        """Setup the daily schedule for automated tasks."""
        logger.info("Setting up daily schedule...")
        
        # Morning forecast collection (6:00 AM PT)
        schedule.every().day.at("06:00").do(self.morning_data_collection)
        
        # Daily prediction and trading pipeline (7:00 AM PT - after morning collection)
        schedule.every().day.at("07:00").do(self.daily_prediction_and_trading_pipeline)
        
        # Trading alerts processing (every 3 hours during trading hours)
        schedule.every(3).hours.do(self.generate_trading_alerts)
        
        # Evening actual temperature collection (8:00 PM PT)
        schedule.every().day.at("20:00").do(self.evening_actual_temperature_collection)
        
        # Health checks (every 4 hours)
        schedule.every(4).hours.do(self.health_check)
        
        # Retry failed collections (every 2 hours during business hours)
        schedule.every(2).hours.do(self.retry_failed_collection)
        
        logger.info("Daily schedule configured:")
        logger.info("  - 06:00: Morning forecast collection")
        logger.info("  - 07:00: Daily prediction and trading pipeline")
        logger.info("  - 20:00: Evening actual temperature collection")
        logger.info("  - Every 3 hours: Trading alerts processing")
        logger.info("  - Every 4 hours: Health check")
        logger.info("  - Every 2 hours: Retry failed collections")
    
    def run_scheduler(self):
        """Run the scheduler in a loop."""
        logger.info("Starting daily scheduler...")
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Setup the schedule
        self.setup_schedule()
        
        # Run initial health check
        self.health_check()
        
        # Main scheduler loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
        
        logger.info("Daily scheduler stopped")
    
    def run_once(self, task: str):
        """Run a specific task once for testing."""
        logger.info(f"Running task once: {task}")
        
        if task == "morning":
            return self.morning_data_collection()
        elif task == "evening":
            return self.evening_actual_temperature_collection()
        elif task == "prediction":
            return self.daily_prediction_and_trading_pipeline()
        elif task == "alerts":
            return self.generate_trading_alerts()
        elif task == "health":
            result = self.health_check()
            return 'error' not in result
        elif task == "retry":
            self.retry_failed_collection()
            return True
        else:
            logger.error(f"Unknown task: {task}")
            return False


def main():
    """Main entry point for the daily scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily automation scheduler for weather data collection")
    parser.add_argument("--run-once", choices=["morning", "evening", "prediction", "alerts", "health", "retry"], 
                       help="Run a specific task once instead of starting the scheduler")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create scheduler
    scheduler = DailyScheduler()
    
    if args.run_once:
        # Run specific task once
        success = scheduler.run_once(args.run_once)
        return 0 if success else 1
    else:
        # Run continuous scheduler
        try:
            scheduler.run_scheduler()
            return 0
        except Exception as e:
            logger.error(f"Scheduler failed: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())