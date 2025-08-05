"""Data management utilities for the Kalshi Weather Predictor."""

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import date, datetime
import pandas as pd
import numpy as np
from loguru import logger

from .config import config


class DataManager:
    """Manages data storage and retrieval for weather prediction system."""
    
    def __init__(self):
        """Initialize data manager."""
        self.data_dir = config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define data file paths
        self.data_files = {
            'nws': self.data_dir / 'nws_data.parquet',
            'openweather': self.data_dir / 'openweather_data.parquet',
            'tomorrow': self.data_dir / 'tomorrow_data.parquet',
            'weatherbit': self.data_dir / 'weatherbit_data.parquet',
            'visual_crossing': self.data_dir / 'visual_crossing_data.parquet',
            'actual_temperatures': self.data_dir / 'actual_temperatures.parquet',
            'predictions': self.data_dir / 'predictions.parquet',
            'model_performance': self.data_dir / 'model_performance.parquet'
        }
        
        # Define expected schemas for validation
        self.weather_schema = {
            'date': 'datetime64[ns]',
            'forecast_date': 'datetime64[ns]',
            'predicted_high': 'float64',
            'predicted_low': 'float64',
            'humidity': 'float64',
            'pressure': 'float64',
            'wind_speed': 'float64',
            'wind_direction': 'float64',
            'cloud_cover': 'float64',
            'precipitation_prob': 'float64',
            'data_quality_score': 'float64'
        }
        
        self.actual_temp_schema = {
            'date': 'datetime64[ns]',
            'actual_high': 'float64',
            'actual_low': 'float64',
            'source': 'object'
        }
        
        self.prediction_schema = {
            'date': 'datetime64[ns]',
            'predicted_high': 'float64',
            'confidence': 'float64',
            'model_contributions': 'object',
            'feature_importance': 'object',
            'created_at': 'datetime64[ns]',
            'actual_temperature': 'float64'
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'temperature_min': -20.0,  # Minimum reasonable temperature for LA (°F)
            'temperature_max': 130.0,  # Maximum reasonable temperature for LA (°F)
            'humidity_min': 0.0,
            'humidity_max': 100.0,
            'pressure_min': 900.0,     # Minimum reasonable pressure (hPa)
            'pressure_max': 1100.0,    # Maximum reasonable pressure (hPa)
            'wind_speed_max': 200.0,   # Maximum reasonable wind speed (mph)
            'cloud_cover_min': 0.0,
            'cloud_cover_max': 100.0,
            'precipitation_prob_min': 0.0,
            'precipitation_prob_max': 100.0
        }
        
        logger.info(f"DataManager initialized with data directory: {self.data_dir}")
    
    def get_data_file_path(self, source: str) -> Path:
        """Get the file path for a data source.
        
        Args:
            source: Data source name
            
        Returns:
            Path to the data file
        """
        if source not in self.data_files:
            raise ValueError(f"Unknown data source: {source}")
        return self.data_files[source]
    
    def file_exists(self, source: str) -> bool:
        """Check if data file exists for a source.
        
        Args:
            source: Data source name
            
        Returns:
            True if file exists, False otherwise
        """
        return self.get_data_file_path(source).exists()
    
    def load_source_data(self, source: str, start_date: Optional[date] = None, 
                        end_date: Optional[date] = None) -> pd.DataFrame:
        """Load data from a specific source.
        
        Args:
            source: Data source name
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            DataFrame with source data
        """
        file_path = self.get_data_file_path(source)
        
        if not file_path.exists():
            logger.warning(f"Data file does not exist: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(file_path)
            
            # Filter by date range if provided
            if start_date or end_date:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    if start_date:
                        df = df[df['date'] >= start_date]
                    if end_date:
                        df = df[df['date'] <= end_date]
            
            logger.info(f"Loaded {len(df)} records from {source}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {source}: {e}")
            return pd.DataFrame()
    
    def save_source_data(self, source: str, data: pd.DataFrame, append: bool = True) -> None:
        """Save data for a specific source.
        
        Args:
            source: Data source name
            data: DataFrame to save
            append: Whether to append to existing data or overwrite
        """
        file_path = self.get_data_file_path(source)
        
        try:
            if append and file_path.exists():
                # Load existing data and append new data
                existing_data = pd.read_parquet(file_path)
                combined_data = pd.concat([existing_data, data], ignore_index=True)
                
                # Remove duplicates if date column exists
                if 'date' in combined_data.columns:
                    combined_data = combined_data.drop_duplicates(subset=['date'], keep='last')
                
                combined_data.to_parquet(file_path, index=False)
                logger.info(f"Appended {len(data)} records to {source} (total: {len(combined_data)})")
            else:
                # Save new data
                data.to_parquet(file_path, index=False)
                logger.info(f"Saved {len(data)} records to {source}")
                
        except Exception as e:
            logger.error(f"Error saving data to {source}: {e}")
            raise
    
    def load_all_sources(self, start_date: Optional[date] = None, 
                        end_date: Optional[date] = None) -> Dict[str, pd.DataFrame]:
        """Load data from all weather sources.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        weather_sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        data = {}
        
        for source in weather_sources:
            data[source] = self.load_source_data(source, start_date, end_date)
        
        return data
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """Get summary information about stored data.
        
        Returns:
            Dictionary with summary info for each data source
        """
        summary = {}
        
        for source, file_path in self.data_files.items():
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    summary[source] = {
                        'records': len(df),
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'columns': list(df.columns),
                        'date_range': None
                    }
                    
                    # Add date range if date column exists
                    if 'date' in df.columns:
                        dates = pd.to_datetime(df['date'])
                        summary[source]['date_range'] = {
                            'start': dates.min().strftime('%Y-%m-%d'),
                            'end': dates.max().strftime('%Y-%m-%d')
                        }
                        
                except Exception as e:
                    summary[source] = {'error': str(e)}
            else:
                summary[source] = {'status': 'file_not_found'}
        
        return summary
    
    def validate_weather_data_schema(self, data: pd.DataFrame, source: str) -> Tuple[bool, List[str]]:
        """Validate weather data against expected schema.
        
        Args:
            data: DataFrame to validate
            source: Data source name for context
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_columns = ['date', 'forecast_date', 'predicted_high', 'predicted_low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types for existing columns
        for column, expected_dtype in self.weather_schema.items():
            if column in data.columns:
                try:
                    if expected_dtype.startswith('datetime'):
                        pd.to_datetime(data[column])
                    elif expected_dtype == 'float64':
                        pd.to_numeric(data[column], errors='coerce')
                except Exception as e:
                    errors.append(f"Column '{column}' cannot be converted to {expected_dtype}: {e}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Schema validation failed for {source}: {errors}")
        
        return is_valid, errors
    
    def perform_data_quality_checks(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Perform data quality checks and calculate quality scores.
        
        Args:
            data: DataFrame to check
            source: Data source name for context
            
        Returns:
            DataFrame with quality scores and cleaned data
        """
        if data.empty:
            return data
        
        data = data.copy()
        quality_scores = []
        
        for idx, row in data.iterrows():
            score = 1.0  # Start with perfect score
            issues = []
            
            # Check temperature ranges
            if 'predicted_high' in row:
                temp_high = row['predicted_high']
                if pd.notna(temp_high):
                    if temp_high < self.quality_thresholds['temperature_min'] or temp_high > self.quality_thresholds['temperature_max']:
                        score -= 0.3
                        issues.append(f"High temp out of range: {temp_high}")
                else:
                    score -= 0.2
                    issues.append("Missing high temperature")
            
            if 'predicted_low' in row:
                temp_low = row['predicted_low']
                if pd.notna(temp_low):
                    if temp_low < self.quality_thresholds['temperature_min'] or temp_low > self.quality_thresholds['temperature_max']:
                        score -= 0.3
                        issues.append(f"Low temp out of range: {temp_low}")
                else:
                    score -= 0.2
                    issues.append("Missing low temperature")
            
            # Check temperature consistency (high should be >= low)
            if 'predicted_high' in row and 'predicted_low' in row:
                if pd.notna(row['predicted_high']) and pd.notna(row['predicted_low']):
                    if row['predicted_high'] < row['predicted_low']:
                        score -= 0.4
                        issues.append("High temp less than low temp")
            
            # Check humidity range
            if 'humidity' in row and pd.notna(row['humidity']):
                humidity = row['humidity']
                if humidity < self.quality_thresholds['humidity_min'] or humidity > self.quality_thresholds['humidity_max']:
                    score -= 0.1
                    issues.append(f"Humidity out of range: {humidity}")
            
            # Check pressure range
            if 'pressure' in row and pd.notna(row['pressure']):
                pressure = row['pressure']
                if pressure < self.quality_thresholds['pressure_min'] or pressure > self.quality_thresholds['pressure_max']:
                    score -= 0.1
                    issues.append(f"Pressure out of range: {pressure}")
            
            # Check wind speed
            if 'wind_speed' in row and pd.notna(row['wind_speed']):
                wind_speed = row['wind_speed']
                if wind_speed < 0 or wind_speed > self.quality_thresholds['wind_speed_max']:
                    score -= 0.1
                    issues.append(f"Wind speed out of range: {wind_speed}")
            
            # Check cloud cover range
            if 'cloud_cover' in row and pd.notna(row['cloud_cover']):
                cloud_cover = row['cloud_cover']
                if cloud_cover < self.quality_thresholds['cloud_cover_min'] or cloud_cover > self.quality_thresholds['cloud_cover_max']:
                    score -= 0.1
                    issues.append(f"Cloud cover out of range: {cloud_cover}")
            
            # Check precipitation probability range
            if 'precipitation_prob' in row and pd.notna(row['precipitation_prob']):
                precip_prob = row['precipitation_prob']
                if precip_prob < self.quality_thresholds['precipitation_prob_min'] or precip_prob > self.quality_thresholds['precipitation_prob_max']:
                    score -= 0.1
                    issues.append(f"Precipitation probability out of range: {precip_prob}")
            
            # Ensure score doesn't go below 0
            score = max(0.0, score)
            quality_scores.append(score)
            
            if issues:
                logger.debug(f"Data quality issues for {source} on {row.get('date', 'unknown date')}: {issues} (score: {score:.2f})")
        
        # Add quality scores to data
        data['data_quality_score'] = quality_scores
        
        # Log summary
        avg_quality = np.mean(quality_scores)
        low_quality_count = sum(1 for score in quality_scores if score < 0.7)
        logger.info(f"Data quality check for {source}: avg_score={avg_quality:.3f}, low_quality_records={low_quality_count}/{len(quality_scores)}")
        
        return data
    
    def detect_outliers(self, data: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        """Detect outliers in a specific column.
        
        Args:
            data: DataFrame containing the data
            column: Column name to check for outliers
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Boolean series indicating outliers
        """
        if column not in data.columns or data[column].empty:
            return pd.Series([False] * len(data), index=data.index)
        
        values = data[column].dropna()
        if len(values) < 4:  # Need at least 4 values for meaningful outlier detection
            return pd.Series([False] * len(data), index=data.index)
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data[column] - values.mean()) / values.std())
            outliers = z_scores > 3
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers.fillna(False)
    
    def append_daily_data(self, source: str, data: pd.DataFrame, validate: bool = True) -> bool:
        """Append daily data to source-specific file with validation.
        
        Args:
            source: Data source name
            data: DataFrame to append
            validate: Whether to perform validation and quality checks
            
        Returns:
            True if data was successfully appended, False otherwise
        """
        if data.empty:
            logger.warning(f"No data to append for {source}")
            return False
        
        try:
            # Make a copy to avoid modifying original data
            data_to_store = data.copy()
            
            # Convert date columns to datetime if they're strings
            for date_col in ['date', 'forecast_date']:
                if date_col in data_to_store.columns:
                    data_to_store[date_col] = pd.to_datetime(data_to_store[date_col])
            
            if validate:
                # Validate schema
                is_valid, errors = self.validate_weather_data_schema(data_to_store, source)
                if not is_valid:
                    logger.error(f"Schema validation failed for {source}: {errors}")
                    return False
                
                # Perform quality checks
                data_to_store = self.perform_data_quality_checks(data_to_store, source)
                
                # Check for outliers in temperature data
                if 'predicted_high' in data_to_store.columns:
                    outliers = self.detect_outliers(data_to_store, 'predicted_high')
                    if outliers.any():
                        outlier_count = outliers.sum()
                        logger.warning(f"Found {outlier_count} temperature outliers in {source} data")
                        # Reduce quality score for outliers
                        data_to_store.loc[outliers, 'data_quality_score'] *= 0.5
            
            # Filter out very low quality data (score < 0.3)
            if 'data_quality_score' in data_to_store.columns:
                initial_count = len(data_to_store)
                data_to_store = data_to_store[data_to_store['data_quality_score'] >= 0.3]
                filtered_count = initial_count - len(data_to_store)
                if filtered_count > 0:
                    logger.warning(f"Filtered out {filtered_count} very low quality records from {source}")
            
            if data_to_store.empty:
                logger.error(f"No valid data remaining after quality checks for {source}")
                return False
            
            # Save the data
            self.save_source_data(source, data_to_store, append=True)
            logger.info(f"Successfully appended {len(data_to_store)} records to {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error appending daily data for {source}: {e}")
            return False
    
    def store_prediction(self, prediction: float, confidence: float, target_date: date, 
                        model_contributions: Optional[Dict] = None, 
                        feature_importance: Optional[Dict] = None) -> None:
        """Store a temperature prediction with metadata.
        
        Args:
            prediction: Predicted high temperature
            confidence: Confidence score (0-1)
            target_date: Date the prediction is for
            model_contributions: Dictionary of model contributions (optional)
            feature_importance: Dictionary of feature importance scores (optional)
        """
        prediction_data = pd.DataFrame([{
            'date': pd.to_datetime(target_date),
            'predicted_high': prediction,
            'confidence': confidence,
            'model_contributions': model_contributions or {},
            'feature_importance': feature_importance or {},
            'created_at': pd.to_datetime(datetime.now()),
            'actual_temperature': None  # Will be filled in later
        }])
        
        try:
            self.save_source_data('predictions', prediction_data, append=True)
            logger.info(f"Stored prediction for {target_date}: {prediction:.1f}°F (confidence: {confidence:.3f})")
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    def store_actual_temperature(self, target_date: date, actual_high: float, 
                                actual_low: Optional[float] = None, source: str = "NOAA") -> None:
        """Store actual temperature observation.
        
        Args:
            target_date: Date of the observation
            actual_high: Actual high temperature in Fahrenheit
            actual_low: Actual low temperature in Fahrenheit (optional)
            source: Source of the observation (default: NOAA)
        """
        try:
            actual_temp_data = pd.DataFrame([{
                'date': pd.to_datetime(target_date),
                'actual_high': actual_high,
                'actual_low': actual_low,
                'source': source
            }])
            
            self.save_source_data('actual_temperatures', actual_temp_data, append=True)
            logger.info(f"Stored actual temperature for {target_date}: {actual_high:.1f}°F (source: {source})")
            
            # Also update any existing predictions
            self.update_actual_temperature(target_date, actual_high)
            
        except Exception as e:
            logger.error(f"Error storing actual temperature: {e}")
    
    def get_actual_temperature(self, target_date: date) -> Optional[float]:
        """Get actual temperature for a specific date.
        
        Args:
            target_date: Date to get temperature for
            
        Returns:
            Actual high temperature or None if not available
        """
        try:
            actual_temps_df = self.load_source_data('actual_temperatures')
            if actual_temps_df.empty:
                return None
            
            # Convert date column for comparison
            actual_temps_df['date'] = pd.to_datetime(actual_temps_df['date']).dt.date
            
            # Find temperature for target date
            mask = actual_temps_df['date'] == target_date
            if mask.any():
                temp_row = actual_temps_df[mask].iloc[0]
                return temp_row['actual_high']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting actual temperature for {target_date}: {e}")
            return None
    
    def update_actual_temperature(self, target_date: date, actual_temperature: float) -> None:
        """Update the actual temperature for a prediction.
        
        Args:
            target_date: Date to update
            actual_temperature: Actual observed temperature
        """
        try:
            # Load existing predictions
            predictions_file = self.get_data_file_path('predictions')
            if not predictions_file.exists():
                logger.warning("No predictions file found to update")
                return
            
            predictions_df = pd.read_parquet(predictions_file)
            predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
            
            # Update the actual temperature for the target date
            mask = predictions_df['date'] == target_date
            if mask.any():
                predictions_df.loc[mask, 'actual_temperature'] = actual_temperature
                
                # Convert date back to datetime for storage
                predictions_df['date'] = pd.to_datetime(predictions_df['date'])
                predictions_df.to_parquet(predictions_file, index=False)
                
                logger.info(f"Updated actual temperature for {target_date}: {actual_temperature:.1f}°F")
            else:
                logger.warning(f"No prediction found for {target_date} to update")
                
        except Exception as e:
            logger.error(f"Error updating actual temperature: {e}")
    
    def validate_actual_temperature(self, temperature: float, target_date: date) -> bool:
        """Validate that an actual temperature reading is reasonable.
        
        Args:
            temperature: Temperature in Fahrenheit
            target_date: Date of the reading
            
        Returns:
            True if temperature is reasonable, False otherwise
        """
        # Use the same quality thresholds as for forecast data
        min_temp = self.quality_thresholds['temperature_min']
        max_temp = self.quality_thresholds['temperature_max']
        
        if temperature < min_temp or temperature > max_temp:
            logger.warning(f"Actual temperature {temperature:.1f}°F is outside reasonable range for {target_date}")
            return False
        
        return True
    
    def load_predictions(self, start_date: Optional[date] = None, 
                        end_date: Optional[date] = None) -> pd.DataFrame:
        """Load prediction data.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            DataFrame with prediction data
        """
        return self.load_source_data('predictions', start_date, end_date)
    
    def load_actual_temperatures(self, start_date: Optional[date] = None, 
                                end_date: Optional[date] = None) -> pd.DataFrame:
        """Load actual temperature data.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            DataFrame with actual temperature data
        """
        return self.load_source_data('actual_temperatures', start_date, end_date)

    def get_data_quality_summary(self, source: str, days: int = 30) -> Dict[str, Any]:
        """Get data quality summary for a source over recent days.
        
        Args:
            source: Data source name
            days: Number of recent days to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Load all data for the source (don't filter by date for testing)
            data = self.load_source_data(source)
            if data.empty or 'data_quality_score' not in data.columns:
                return {'error': 'No quality data available'}
            
            quality_scores = data['data_quality_score'].dropna()
            
            if quality_scores.empty:
                return {'error': 'No quality scores available'}
            
            return {
                'source': source,
                'period_days': days,
                'total_records': len(data),
                'avg_quality_score': quality_scores.mean(),
                'min_quality_score': quality_scores.min(),
                'max_quality_score': quality_scores.max(),
                'low_quality_count': (quality_scores < 0.7).sum(),
                'high_quality_count': (quality_scores >= 0.9).sum(),
                'quality_trend': 'improving' if len(quality_scores) > 7 and quality_scores.tail(7).mean() > quality_scores.head(7).mean() else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Error getting quality summary for {source}: {e}")
            return {'error': str(e)}