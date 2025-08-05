"""Service for collecting actual temperature data from NOAA observations."""

from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import pandas as pd
from loguru import logger

from .noaa_observations_client import NOAAObservationsClient
from ..utils.data_manager import DataManager
from ..utils.config import config


class ActualTemperatureCollector:
    """Service for collecting and storing actual temperature observations."""
    
    def __init__(self, station_id: Optional[str] = None):
        """Initialize the actual temperature collector.
        
        Args:
            station_id: NOAA station ID (defaults to config value)
        """
        self.noaa_client = NOAAObservationsClient(station_id)
        self.data_manager = DataManager()
        
        logger.info("ActualTemperatureCollector initialized")
    
    def collect_daily_high_temperature(self, target_date: date) -> Optional[float]:
        """Collect and store the daily high temperature for a specific date.
        
        Args:
            target_date: Date to collect temperature for
            
        Returns:
            Daily high temperature in Fahrenheit, or None if collection failed
        """
        try:
            # Check if we already have this temperature
            existing_temp = self.data_manager.get_actual_temperature(target_date)
            if existing_temp is not None:
                logger.info(f"Actual temperature for {target_date} already exists: {existing_temp:.1f}°F")
                return existing_temp
            
            # Collect temperature from NOAA
            logger.info(f"Collecting actual temperature for {target_date}")
            daily_high = self.noaa_client.get_daily_high_temperature(target_date)
            
            if daily_high is None:
                logger.warning(f"Failed to collect actual temperature for {target_date}")
                return None
            
            # Validate the temperature reading
            if not self.data_manager.validate_actual_temperature(daily_high, target_date):
                logger.error(f"Invalid temperature reading for {target_date}: {daily_high:.1f}°F")
                return None
            
            # Store the temperature
            self.data_manager.store_actual_temperature(target_date, daily_high, source="NOAA")
            
            logger.info(f"Successfully collected and stored actual temperature for {target_date}: {daily_high:.1f}°F")
            return daily_high
            
        except Exception as e:
            logger.error(f"Error collecting actual temperature for {target_date}: {e}")
            return None
    
    def collect_recent_temperatures(self, days: int = 7) -> Dict[date, Optional[float]]:
        """Collect actual temperatures for recent days.
        
        Args:
            days: Number of recent days to collect (default: 7)
            
        Returns:
            Dictionary mapping dates to temperatures (None if collection failed)
        """
        results = {}
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days-1)
        
        logger.info(f"Collecting actual temperatures from {start_date} to {end_date}")
        
        current_date = start_date
        while current_date <= end_date:
            temp = self.collect_daily_high_temperature(current_date)
            results[current_date] = temp
            current_date += timedelta(days=1)
        
        successful_collections = sum(1 for temp in results.values() if temp is not None)
        logger.info(f"Collected {successful_collections}/{len(results)} actual temperatures")
        
        return results
    
    def backfill_missing_temperatures(self, start_date: date, end_date: date) -> Dict[date, Optional[float]]:
        """Backfill missing actual temperatures for a date range.
        
        Args:
            start_date: Start date for backfill
            end_date: End date for backfill
            
        Returns:
            Dictionary mapping dates to temperatures (None if collection failed)
        """
        results = {}
        
        logger.info(f"Backfilling actual temperatures from {start_date} to {end_date}")
        
        current_date = start_date
        while current_date <= end_date:
            # Only collect if we don't already have the temperature
            existing_temp = self.data_manager.get_actual_temperature(current_date)
            if existing_temp is None:
                temp = self.collect_daily_high_temperature(current_date)
                results[current_date] = temp
            else:
                results[current_date] = existing_temp
                logger.debug(f"Temperature for {current_date} already exists: {existing_temp:.1f}°F")
            
            current_date += timedelta(days=1)
        
        new_collections = sum(1 for date_key, temp in results.items() 
                             if temp is not None and self.data_manager.get_actual_temperature(date_key) != temp)
        logger.info(f"Backfilled {new_collections} new actual temperatures")
        
        return results
    
    def get_temperature_collection_status(self, days: int = 30) -> Dict[str, Any]:
        """Get status of temperature collection over recent days.
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            Dictionary with collection status information
        """
        try:
            end_date = date.today() - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=days-1)
            
            # Load actual temperatures data
            actual_temps_df = self.data_manager.load_source_data('actual_temperatures', start_date, end_date)
            
            if actual_temps_df.empty:
                return {
                    'period_days': days,
                    'total_expected': days,
                    'total_collected': 0,
                    'collection_rate': 0.0,
                    'missing_dates': [],
                    'latest_collection': None
                }
            
            # Convert date column for analysis
            actual_temps_df['date'] = pd.to_datetime(actual_temps_df['date']).dt.date
            
            # Find missing dates
            expected_dates = set()
            current_date = start_date
            while current_date <= end_date:
                expected_dates.add(current_date)
                current_date += timedelta(days=1)
            
            collected_dates = set(actual_temps_df['date'].tolist())
            missing_dates = sorted(expected_dates - collected_dates)
            
            # Get latest collection date
            latest_collection = actual_temps_df['date'].max() if not actual_temps_df.empty else None
            
            return {
                'period_days': days,
                'total_expected': len(expected_dates),
                'total_collected': len(collected_dates),
                'collection_rate': len(collected_dates) / len(expected_dates),
                'missing_dates': [d.strftime('%Y-%m-%d') for d in missing_dates],
                'latest_collection': latest_collection.strftime('%Y-%m-%d') if latest_collection else None
            }
            
        except Exception as e:
            logger.error(f"Error getting temperature collection status: {e}")
            return {'error': str(e)}
    
    def validate_temperature_data_quality(self, days: int = 30) -> Dict[str, Any]:
        """Validate the quality of collected temperature data.
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=days-1)
            
            # Load actual temperatures
            actual_temps_df = self.data_manager.load_source_data('actual_temperatures', start_date, end_date)
            
            if actual_temps_df.empty:
                return {'error': 'No temperature data available for analysis'}
            
            # Convert date column
            actual_temps_df['date'] = pd.to_datetime(actual_temps_df['date']).dt.date
            
            # Calculate quality metrics
            temperatures = actual_temps_df['actual_high'].dropna()
            
            if temperatures.empty:
                return {'error': 'No valid temperature readings found'}
            
            # Detect outliers using IQR method
            Q1 = temperatures.quantile(0.25)
            Q3 = temperatures.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = temperatures[(temperatures < lower_bound) | (temperatures > upper_bound)]
            
            return {
                'period_days': days,
                'total_readings': len(temperatures),
                'avg_temperature': temperatures.mean(),
                'min_temperature': temperatures.min(),
                'max_temperature': temperatures.max(),
                'std_temperature': temperatures.std(),
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(temperatures) * 100,
                'temperature_range': temperatures.max() - temperatures.min(),
                'data_sources': actual_temps_df['source'].value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error validating temperature data quality: {e}")
            return {'error': str(e)}
    
    def test_collection_system(self) -> bool:
        """Test the actual temperature collection system.
        
        Returns:
            True if system is working correctly, False otherwise
        """
        try:
            logger.info("Testing actual temperature collection system")
            
            # Test NOAA client connection
            if not self.noaa_client.test_connection():
                logger.error("NOAA client connection test failed")
                return False
            
            # Test station info retrieval
            station_info = self.noaa_client.get_station_info()
            if not station_info:
                logger.error("Failed to retrieve station information")
                return False
            
            # Test temperature collection for yesterday
            yesterday = date.today() - timedelta(days=1)
            test_temp = self.noaa_client.get_daily_high_temperature(yesterday)
            
            if test_temp is None:
                logger.warning(f"Could not retrieve temperature for {yesterday} (may be normal if data not yet available)")
            else:
                logger.info(f"Test temperature collection successful: {test_temp:.1f}°F for {yesterday}")
            
            # Test data manager operations
            test_date = date.today() - timedelta(days=2)
            self.data_manager.store_actual_temperature(test_date, 75.0, source="TEST")
            retrieved_temp = self.data_manager.get_actual_temperature(test_date)
            
            if retrieved_temp != 75.0:
                logger.error("Data manager test failed")
                return False
            
            logger.info("Actual temperature collection system test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error testing collection system: {e}")
            return False