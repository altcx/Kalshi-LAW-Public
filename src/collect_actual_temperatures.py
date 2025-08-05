#!/usr/bin/env python3
"""Script to collect actual temperature data for validation."""

import sys
from pathlib import Path
from datetime import date, timedelta
import argparse
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.actual_temperature_collector import ActualTemperatureCollector


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    
    level = "DEBUG" if verbose else "INFO"
    format_str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    
    # Console logging
    logger.add(sys.stdout, level=level, format=format_str)
    
    # File logging
    log_file = Path("logs/actual_temperature_collection.log")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, level="DEBUG", format=format_str, rotation="10 MB")


def collect_yesterday_temperature():
    """Collect actual temperature for yesterday."""
    yesterday = date.today() - timedelta(days=1)
    
    logger.info(f"Collecting actual temperature for {yesterday}")
    
    collector = ActualTemperatureCollector()
    temperature = collector.collect_daily_high_temperature(yesterday)
    
    if temperature is not None:
        logger.info(f"✓ Successfully collected temperature: {temperature:.1f}°F")
        return True
    else:
        logger.error("✗ Failed to collect temperature")
        return False


def collect_recent_temperatures(days: int):
    """Collect actual temperatures for recent days."""
    logger.info(f"Collecting actual temperatures for the last {days} days")
    
    collector = ActualTemperatureCollector()
    results = collector.collect_recent_temperatures(days=days)
    
    successful = 0
    for date_key, temp in results.items():
        if temp is not None:
            logger.info(f"✓ {date_key}: {temp:.1f}°F")
            successful += 1
        else:
            logger.warning(f"✗ {date_key}: No data available")
    
    logger.info(f"Collected {successful}/{len(results)} temperatures")
    return successful == len(results)


def backfill_temperatures(start_date: str, end_date: str):
    """Backfill temperatures for a date range."""
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return False
    
    if start > end:
        logger.error("Start date must be before end date")
        return False
    
    logger.info(f"Backfilling temperatures from {start} to {end}")
    
    collector = ActualTemperatureCollector()
    results = collector.backfill_missing_temperatures(start, end)
    
    successful = 0
    for date_key, temp in results.items():
        if temp is not None:
            logger.info(f"✓ {date_key}: {temp:.1f}°F")
            successful += 1
        else:
            logger.warning(f"✗ {date_key}: No data available")
    
    logger.info(f"Backfilled {successful}/{len(results)} temperatures")
    return successful > 0


def show_collection_status(days: int):
    """Show temperature collection status."""
    logger.info(f"Checking collection status for the last {days} days")
    
    collector = ActualTemperatureCollector()
    status = collector.get_temperature_collection_status(days=days)
    
    if 'error' in status:
        logger.error(f"Error getting status: {status['error']}")
        return False
    
    logger.info(f"Collection Status:")
    logger.info(f"  Period: {status['period_days']} days")
    logger.info(f"  Collected: {status['total_collected']}/{status['total_expected']} ({status['collection_rate']:.1%})")
    
    if status['latest_collection']:
        logger.info(f"  Latest: {status['latest_collection']}")
    
    if status['missing_dates']:
        missing_count = len(status['missing_dates'])
        if missing_count <= 5:
            logger.info(f"  Missing: {', '.join(status['missing_dates'])}")
        else:
            logger.info(f"  Missing: {', '.join(status['missing_dates'][:5])} and {missing_count - 5} more")
    
    return True


def validate_data_quality(days: int):
    """Validate temperature data quality."""
    logger.info(f"Validating data quality for the last {days} days")
    
    collector = ActualTemperatureCollector()
    quality = collector.validate_temperature_data_quality(days=days)
    
    if 'error' in quality:
        logger.error(f"Error validating quality: {quality['error']}")
        return False
    
    logger.info(f"Data Quality Report:")
    logger.info(f"  Total readings: {quality['total_readings']}")
    logger.info(f"  Average temperature: {quality['avg_temperature']:.1f}°F")
    logger.info(f"  Temperature range: {quality['min_temperature']:.1f}°F to {quality['max_temperature']:.1f}°F")
    logger.info(f"  Standard deviation: {quality['std_temperature']:.1f}°F")
    logger.info(f"  Outliers: {quality['outlier_count']} ({quality['outlier_percentage']:.1f}%)")
    
    if quality['data_sources']:
        logger.info(f"  Sources: {', '.join(f'{k}({v})' for k, v in quality['data_sources'].items())}")
    
    return True


def test_system():
    """Test the actual temperature collection system."""
    logger.info("Testing actual temperature collection system")
    
    collector = ActualTemperatureCollector()
    success = collector.test_collection_system()
    
    if success:
        logger.info("✓ System test passed")
    else:
        logger.error("✗ System test failed")
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect actual temperature data for validation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Yesterday command
    subparsers.add_parser("yesterday", help="Collect temperature for yesterday")
    
    # Recent command
    recent_parser = subparsers.add_parser("recent", help="Collect temperatures for recent days")
    recent_parser.add_argument("--days", type=int, default=7, help="Number of recent days (default: 7)")
    
    # Backfill command
    backfill_parser = subparsers.add_parser("backfill", help="Backfill temperatures for date range")
    backfill_parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    backfill_parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show collection status")
    status_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    
    # Quality command
    quality_parser = subparsers.add_parser("quality", help="Validate data quality")
    quality_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    
    # Test command
    subparsers.add_parser("test", help="Test the collection system")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    success = False
    
    if args.command == "yesterday":
        success = collect_yesterday_temperature()
    elif args.command == "recent":
        success = collect_recent_temperatures(args.days)
    elif args.command == "backfill":
        success = backfill_temperatures(args.start_date, args.end_date)
    elif args.command == "status":
        success = show_collection_status(args.days)
    elif args.command == "quality":
        success = validate_data_quality(args.days)
    elif args.command == "test":
        success = test_system()
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())