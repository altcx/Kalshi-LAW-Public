#!/usr/bin/env python3
"""Daily script for automated actual temperature collection."""

import sys
from pathlib import Path
from datetime import date, timedelta
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.actual_temperature_collector import ActualTemperatureCollector


def setup_logging():
    """Setup logging for daily collection."""
    logger.remove()
    
    # Console logging (minimal)
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    # File logging (detailed)
    log_file = Path("logs/actual_temperature_collection.log")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(
        log_file, 
        level="DEBUG", 
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        retention="30 days"
    )


def collect_yesterday_temperature() -> bool:
    """Collect actual temperature for yesterday.
    
    Returns:
        True if collection was successful, False otherwise
    """
    yesterday = date.today() - timedelta(days=1)
    
    try:
        logger.info(f"Starting daily actual temperature collection for {yesterday}")
        
        collector = ActualTemperatureCollector()
        temperature = collector.collect_daily_high_temperature(yesterday)
        
        if temperature is not None:
            logger.info(f"✓ Successfully collected actual temperature: {temperature:.1f}°F for {yesterday}")
            return True
        else:
            logger.error(f"✗ Failed to collect actual temperature for {yesterday}")
            return False
            
    except Exception as e:
        logger.error(f"Error during daily temperature collection: {e}")
        return False


def check_collection_health() -> bool:
    """Check the health of temperature collection over recent days.
    
    Returns:
        True if collection health is good, False if there are issues
    """
    try:
        collector = ActualTemperatureCollector()
        
        # Check collection status for last 7 days
        status = collector.get_temperature_collection_status(days=7)
        
        if 'error' in status:
            logger.error(f"Error checking collection status: {status['error']}")
            return False
        
        collection_rate = status['collection_rate']
        missing_count = len(status['missing_dates'])
        
        logger.info(f"Collection health check: {status['total_collected']}/{status['total_expected']} collected ({collection_rate:.1%})")
        
        # Alert if collection rate is low
        if collection_rate < 0.8:  # Less than 80%
            logger.warning(f"⚠️  Low collection rate: {collection_rate:.1%} (missing {missing_count} days)")
            if status['missing_dates']:
                logger.warning(f"Missing dates: {', '.join(status['missing_dates'][:5])}")
        
        # Check data quality
        quality = collector.validate_temperature_data_quality(days=7)
        
        if 'error' not in quality:
            outlier_percentage = quality.get('outlier_percentage', 0)
            if outlier_percentage > 10:  # More than 10% outliers
                logger.warning(f"⚠️  High outlier rate: {outlier_percentage:.1f}%")
        
        return collection_rate >= 0.6  # At least 60% collection rate is acceptable
        
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return False


def main():
    """Main entry point for daily collection."""
    setup_logging()
    
    logger.info("=== Daily Actual Temperature Collection ===")
    
    # Collect yesterday's temperature
    collection_success = collect_yesterday_temperature()
    
    # Check overall collection health
    health_good = check_collection_health()
    
    # Summary
    if collection_success and health_good:
        logger.info("✅ Daily collection completed successfully")
        return 0
    elif collection_success:
        logger.warning("⚠️  Daily collection completed but health check shows issues")
        return 0
    else:
        logger.error("❌ Daily collection failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())