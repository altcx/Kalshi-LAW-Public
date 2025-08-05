"""Historical data loading and simulation for backtesting."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

from src.utils.data_manager import DataManager
from src.feature_engineering.feature_pipeline import FeaturePipeline


class HistoricalDataLoader:
    """Loads and manages historical weather data for backtesting."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize the historical data loader.
        
        Args:
            data_manager: DataManager instance (creates new if None)
        """
        self.data_manager = data_manager or DataManager()
        self.feature_pipeline = FeaturePipeline()
        
        # Cache for loaded data to improve performance
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_valid_until: Dict[str, datetime] = {}
        self.cache_duration = timedelta(hours=1)  # Cache data for 1 hour
        
        logger.info("HistoricalDataLoader initialized")
    
    def get_available_date_range(self) -> Tuple[Optional[date], Optional[date]]:
        """Get the available date range across all data sources.
        
        Returns:
            Tuple of (earliest_date, latest_date) or (None, None) if no data
        """
        earliest_date = None
        latest_date = None
        
        # Check all weather data sources
        weather_sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        
        for source in weather_sources:
            try:
                data = self._load_source_with_cache(source)
                if not data.empty and 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date']).dt.date
                    source_min = data['date'].min()
                    source_max = data['date'].max()
                    
                    if earliest_date is None or source_min < earliest_date:
                        earliest_date = source_min
                    if latest_date is None or source_max > latest_date:
                        latest_date = source_max
                        
            except Exception as e:
                logger.warning(f"Error checking date range for {source}: {e}")
                continue
        
        logger.info(f"Available data range: {earliest_date} to {latest_date}")
        return earliest_date, latest_date
    
    def _load_source_with_cache(self, source: str) -> pd.DataFrame:
        """Load data from source with caching.
        
        Args:
            source: Data source name
            
        Returns:
            DataFrame with source data
        """
        cache_key = f"source_{source}"
        
        # Check if cache is valid
        if (cache_key in self._data_cache and 
            cache_key in self._cache_valid_until and
            datetime.now() < self._cache_valid_until[cache_key]):
            return self._data_cache[cache_key]
        
        # Load fresh data
        data = self.data_manager.load_source_data(source)
        
        # Update cache
        self._data_cache[cache_key] = data
        self._cache_valid_until[cache_key] = datetime.now() + self.cache_duration
        
        return data
    
    def load_historical_data(self, start_date: date, end_date: date, 
                           sources: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load historical data for a specific date range.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            sources: List of sources to load (None for all sources)
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        if sources is None:
            sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        
        historical_data = {}
        
        for source in sources:
            try:
                data = self.data_manager.load_source_data(source, start_date, end_date)
                if not data.empty:
                    # Ensure date column is properly formatted
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                    historical_data[source] = data
                    logger.info(f"Loaded {len(data)} records from {source} for {start_date} to {end_date}")
                else:
                    logger.warning(f"No data available for {source} in date range {start_date} to {end_date}")
                    
            except Exception as e:
                logger.error(f"Error loading historical data from {source}: {e}")
                continue
        
        return historical_data
    
    def load_actual_temperatures(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Load actual temperature observations for a date range.
        
        Args:
            start_date: Start date for temperature data
            end_date: End date for temperature data
            
        Returns:
            DataFrame with actual temperature data
        """
        try:
            actual_temps = self.data_manager.load_source_data('actual_temperatures', start_date, end_date)
            
            if not actual_temps.empty and 'date' in actual_temps.columns:
                actual_temps['date'] = pd.to_datetime(actual_temps['date'])
                logger.info(f"Loaded {len(actual_temps)} actual temperature records for {start_date} to {end_date}")
            else:
                logger.warning(f"No actual temperature data available for {start_date} to {end_date}")
                
            return actual_temps
            
        except Exception as e:
            logger.error(f"Error loading actual temperatures: {e}")
            return pd.DataFrame()
    
    def get_data_completeness_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate a report on data completeness for a date range.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with completeness statistics
        """
        date_range = pd.date_range(start_date, end_date, freq='D')
        total_days = len(date_range)
        
        sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing', 'actual_temperatures']
        completeness_report = {
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'total_days': total_days
            },
            'sources': {}
        }
        
        for source in sources:
            try:
                data = self.data_manager.load_source_data(source, start_date, end_date)
                
                if data.empty:
                    completeness_report['sources'][source] = {
                        'available_days': 0,
                        'completeness_pct': 0.0,
                        'missing_days': total_days,
                        'data_quality_avg': 0.0
                    }
                    continue
                
                # Count available days
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date']).dt.date
                    available_dates = set(data['date'].unique())
                    expected_dates = set(date_range.date)
                    
                    available_days = len(available_dates)
                    missing_days = total_days - available_days
                    completeness_pct = (available_days / total_days) * 100
                    
                    # Calculate average data quality if available
                    avg_quality = 0.0
                    if 'data_quality_score' in data.columns:
                        avg_quality = data['data_quality_score'].mean()
                    
                    completeness_report['sources'][source] = {
                        'available_days': available_days,
                        'completeness_pct': completeness_pct,
                        'missing_days': missing_days,
                        'data_quality_avg': avg_quality,
                        'total_records': len(data)
                    }
                else:
                    completeness_report['sources'][source] = {
                        'error': 'No date column found',
                        'total_records': len(data)
                    }
                    
            except Exception as e:
                completeness_report['sources'][source] = {
                    'error': str(e)
                }
        
        # Calculate overall completeness
        source_completeness = [
            info.get('completeness_pct', 0) 
            for info in completeness_report['sources'].values() 
            if isinstance(info, dict) and 'completeness_pct' in info
        ]
        
        if source_completeness:
            completeness_report['overall'] = {
                'avg_completeness_pct': np.mean(source_completeness),
                'min_completeness_pct': np.min(source_completeness),
                'max_completeness_pct': np.max(source_completeness),
                'sources_with_data': len(source_completeness)
            }
        
        return completeness_report
    
    def create_walk_forward_splits(self, start_date: date, end_date: date, 
                                 train_window_days: int = 365,
                                 test_window_days: int = 30,
                                 step_days: int = 7) -> List[Dict[str, Any]]:
        """Create walk-forward analysis splits for backtesting.
        
        Args:
            start_date: Overall start date
            end_date: Overall end date
            train_window_days: Number of days to use for training
            test_window_days: Number of days to test on
            step_days: Number of days to step forward between splits
            
        Returns:
            List of dictionaries with train/test date ranges
        """
        splits = []
        current_date = start_date + timedelta(days=train_window_days)
        
        while current_date + timedelta(days=test_window_days) <= end_date:
            train_start = current_date - timedelta(days=train_window_days)
            train_end = current_date - timedelta(days=1)
            test_start = current_date
            test_end = current_date + timedelta(days=test_window_days - 1)
            
            split_info = {
                'split_id': len(splits) + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_days': train_window_days,
                'test_days': test_window_days
            }
            
            splits.append(split_info)
            current_date += timedelta(days=step_days)
        
        logger.info(f"Created {len(splits)} walk-forward splits from {start_date} to {end_date}")
        return splits
    
    def simulate_real_time_prediction(self, target_date: date, 
                                    forecast_horizon_days: int = 1) -> Dict[str, Any]:
        """Simulate making a prediction as if it were real-time on a historical date.
        
        Args:
            target_date: Date to make prediction for
            forecast_horizon_days: How many days ahead to predict (1 = next day)
            
        Returns:
            Dictionary with available data and metadata for that historical point
        """
        # Calculate the forecast date (when the prediction would be made)
        forecast_date = target_date - timedelta(days=forecast_horizon_days)
        
        # Load data that would have been available at forecast time
        # We need data from before the forecast date
        data_cutoff_date = forecast_date - timedelta(days=1)
        
        # Load historical weather data up to the cutoff
        available_data = {}
        sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        
        for source in sources:
            try:
                # Load data from a reasonable lookback period
                lookback_start = data_cutoff_date - timedelta(days=30)  # 30 days of history
                source_data = self.data_manager.load_source_data(source, lookback_start, data_cutoff_date)
                
                if not source_data.empty:
                    # Filter to only data that would have been available
                    if 'forecast_date' in source_data.columns:
                        source_data['forecast_date'] = pd.to_datetime(source_data['forecast_date'])
                        # Only include forecasts made before our cutoff
                        source_data = source_data[source_data['forecast_date'] <= pd.to_datetime(data_cutoff_date)]
                    
                    available_data[source] = source_data
                    
            except Exception as e:
                logger.warning(f"Error loading historical data for {source} at {forecast_date}: {e}")
                continue
        
        # Load actual temperature for validation (if available)
        actual_temp = None
        try:
            actual_temps = self.data_manager.load_source_data('actual_temperatures')
            if not actual_temps.empty and 'date' in actual_temps.columns:
                actual_temps['date'] = pd.to_datetime(actual_temps['date']).dt.date
                temp_mask = actual_temps['date'] == target_date
                if temp_mask.any():
                    actual_temp = actual_temps[temp_mask].iloc[0]['actual_high']
        except Exception as e:
            logger.warning(f"Error loading actual temperature for {target_date}: {e}")
        
        simulation_info = {
            'target_date': target_date,
            'forecast_date': forecast_date,
            'forecast_horizon_days': forecast_horizon_days,
            'data_cutoff_date': data_cutoff_date,
            'available_sources': list(available_data.keys()),
            'total_records': sum(len(df) for df in available_data.values()),
            'actual_temperature': actual_temp,
            'data': available_data
        }
        
        logger.info(f"Simulated real-time prediction for {target_date}: "
                   f"{len(available_data)} sources, {simulation_info['total_records']} records")
        
        return simulation_info
    
    def prepare_features_for_date(self, target_date: date, 
                                available_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare features for a specific prediction date using available historical data.
        
        Args:
            target_date: Date to prepare features for
            available_data: Dictionary of available data from different sources
            
        Returns:
            DataFrame with engineered features for the target date
        """
        try:
            # Use the feature pipeline to create features
            features = self.feature_pipeline.create_features(available_data, target_date)
            
            if features.empty:
                logger.warning(f"No features could be created for {target_date}")
                return pd.DataFrame()
            
            logger.info(f"Prepared {len(features.columns)} features for {target_date}")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features for {target_date}: {e}")
            return pd.DataFrame()
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame], 
                            min_sources: int = 2,
                            min_quality_score: float = 0.5) -> Tuple[bool, List[str]]:
        """Validate that historical data meets quality requirements for backtesting.
        
        Args:
            data: Dictionary of data from different sources
            min_sources: Minimum number of sources required
            min_quality_score: Minimum average quality score required
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check number of sources
        valid_sources = [source for source, df in data.items() if not df.empty]
        if len(valid_sources) < min_sources:
            issues.append(f"Only {len(valid_sources)} sources available, need at least {min_sources}")
        
        # Check data quality scores
        for source, df in data.items():
            if df.empty:
                issues.append(f"No data available for source: {source}")
                continue
            
            if 'data_quality_score' in df.columns:
                avg_quality = df['data_quality_score'].mean()
                if avg_quality < min_quality_score:
                    issues.append(f"Low quality data for {source}: {avg_quality:.3f} < {min_quality_score}")
            else:
                issues.append(f"No quality scores available for source: {source}")
        
        # Check for required columns
        required_columns = ['date', 'predicted_high']
        for source, df in data.items():
            if df.empty:
                continue
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                issues.append(f"Missing required columns in {source}: {missing_cols}")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Data quality validation failed: {issues}")
        
        return is_valid, issues
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        self._cache_valid_until.clear()
        logger.info("Data cache cleared")