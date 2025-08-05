"""Basic feature extraction from individual weather data sources."""

from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class BasicFeatureExtractor:
    """Extracts basic features from individual weather data sources."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.weather_sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        
        # Define feature columns for each source
        self.temperature_features = ['predicted_high', 'predicted_low']
        self.atmospheric_features = ['humidity', 'pressure', 'wind_speed', 'wind_direction', 'cloud_cover', 'precipitation_prob']
        self.quality_features = ['data_quality_score']
        
        logger.info("BasicFeatureExtractor initialized")
    
    def extract_temperature_features(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Extract temperature-related features from a data source.
        
        Args:
            data: DataFrame with weather data from a single source
            source: Name of the weather data source
            
        Returns:
            DataFrame with temperature features
        """
        if data.empty:
            return pd.DataFrame()
        
        features = data[['date']].copy()
        
        # Basic temperature features
        if 'predicted_high' in data.columns:
            features[f'{source}_temp_high'] = data['predicted_high']
        if 'predicted_low' in data.columns:
            features[f'{source}_temp_low'] = data['predicted_low']
        
        # Derived temperature features
        if 'predicted_high' in data.columns and 'predicted_low' in data.columns:
            # Temperature range (diurnal temperature variation)
            features[f'{source}_temp_range'] = data['predicted_high'] - data['predicted_low']
            
            # Average temperature
            features[f'{source}_temp_avg'] = (data['predicted_high'] + data['predicted_low']) / 2
        
        # Temperature trends (if we have multiple days of data)
        if len(data) > 1 and 'predicted_high' in data.columns:
            # Day-over-day temperature change
            features[f'{source}_temp_high_change'] = data['predicted_high'].diff()
            
            # 3-day rolling average (if enough data)
            if len(data) >= 3:
                features[f'{source}_temp_high_3day_avg'] = data['predicted_high'].rolling(window=3, min_periods=1).mean()
        
        logger.debug(f"Extracted {len(features.columns)-1} temperature features from {source}")
        return features
    
    def extract_atmospheric_features(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Extract atmospheric features from a data source.
        
        Args:
            data: DataFrame with weather data from a single source
            source: Name of the weather data source
            
        Returns:
            DataFrame with atmospheric features
        """
        if data.empty:
            return pd.DataFrame()
        
        features = data[['date']].copy()
        
        # Basic atmospheric features
        for feature in self.atmospheric_features:
            if feature in data.columns:
                features[f'{source}_{feature}'] = data[feature]
        
        # Derived atmospheric features
        if 'wind_speed' in data.columns and 'wind_direction' in data.columns:
            # Wind components (useful for ML models)
            wind_speed = data['wind_speed'].fillna(0)
            wind_direction_rad = np.radians(data['wind_direction'].fillna(0))
            
            features[f'{source}_wind_u'] = wind_speed * np.cos(wind_direction_rad)  # East-west component
            features[f'{source}_wind_v'] = wind_speed * np.sin(wind_direction_rad)  # North-south component
        
        # Pressure tendency (if we have multiple days)
        if len(data) > 1 and 'pressure' in data.columns:
            features[f'{source}_pressure_change'] = data['pressure'].diff()
        
        # Humidity categories (useful for some models)
        if 'humidity' in data.columns:
            humidity = data['humidity'].fillna(50)  # Default to moderate humidity
            features[f'{source}_humidity_low'] = (humidity < 30).astype(int)
            features[f'{source}_humidity_high'] = (humidity > 80).astype(int)
        
        logger.debug(f"Extracted {len(features.columns)-1} atmospheric features from {source}")
        return features
    
    def extract_date_features(self, dates: pd.Series) -> pd.DataFrame:
        """Extract date-based features.
        
        Args:
            dates: Series of dates
            
        Returns:
            DataFrame with date-based features
        """
        features = pd.DataFrame()
        features['date'] = dates
        
        # Convert to datetime if needed
        dt_dates = pd.to_datetime(dates)
        
        # Basic date features
        features['day_of_year'] = dt_dates.dt.dayofyear
        features['month'] = dt_dates.dt.month
        features['day_of_month'] = dt_dates.dt.day
        features['day_of_week'] = dt_dates.dt.dayofweek  # Monday=0, Sunday=6
        features['week_of_year'] = dt_dates.dt.isocalendar().week
        
        # Season features (meteorological seasons)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:  # [9, 10, 11]
                return 'fall'
        
        features['season'] = features['month'].apply(get_season)
        
        # Season as numeric (for some models)
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        features['season_numeric'] = features['season'].map(season_map)
        
        # Cyclical encoding for periodic features (better for ML models)
        # Day of year (365-day cycle)
        features['day_of_year_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365.25)
        features['day_of_year_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365.25)
        
        # Month (12-month cycle)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Day of week (7-day cycle)
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Weekend indicator
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # LA-specific seasonal patterns
        # Summer heat season (June-September)
        features['is_heat_season'] = features['month'].isin([6, 7, 8, 9]).astype(int)
        
        # Fire season (October-April, when Santa Ana winds are common)
        features['is_fire_season'] = features['month'].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
        
        # Marine layer season (May-August, when marine layer is most common)
        features['is_marine_layer_season'] = features['month'].isin([5, 6, 7, 8]).astype(int)
        
        logger.debug(f"Extracted {len(features.columns)-1} date-based features")
        return features
    
    def extract_quality_features(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Extract data quality features from a source.
        
        Args:
            data: DataFrame with weather data from a single source
            source: Name of the weather data source
            
        Returns:
            DataFrame with quality features
        """
        if data.empty:
            return pd.DataFrame()
        
        features = data[['date']].copy()
        
        # Basic quality score
        if 'data_quality_score' in data.columns:
            features[f'{source}_quality_score'] = data['data_quality_score']
        
        # Missing data indicators (useful for ensemble weighting)
        for feature in self.temperature_features + self.atmospheric_features:
            if feature in data.columns:
                features[f'{source}_{feature}_missing'] = data[feature].isna().astype(int)
        
        # Data completeness score (percentage of non-missing values)
        feature_cols = [col for col in self.temperature_features + self.atmospheric_features if col in data.columns]
        if feature_cols:
            completeness = data[feature_cols].notna().mean(axis=1)
            features[f'{source}_completeness'] = completeness
        
        logger.debug(f"Extracted {len(features.columns)-1} quality features from {source}")
        return features
    
    def detect_and_clean_outliers(self, data: pd.DataFrame, source: str, 
                                 method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Detect and clean outliers in weather data.
        
        Args:
            data: DataFrame with weather data
            source: Name of the weather data source
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers cleaned
        """
        if data.empty:
            return data
        
        cleaned_data = data.copy()
        outlier_counts = {}
        
        # Define columns to check for outliers
        numeric_columns = ['predicted_high', 'predicted_low', 'humidity', 'pressure', 
                          'wind_speed', 'cloud_cover', 'precipitation_prob']
        
        for column in numeric_columns:
            if column not in data.columns:
                continue
                
            values = data[column].dropna()
            if len(values) < 4:  # Need at least 4 values for meaningful outlier detection
                continue
            
            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((data[column] - values.mean()) / values.std())
                outliers = z_scores > threshold
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_counts[column] = outlier_count
                
                # For temperature data, cap outliers rather than removing them
                if column in ['predicted_high', 'predicted_low']:
                    if method == 'iqr':
                        cleaned_data.loc[outliers & (data[column] < lower_bound), column] = lower_bound
                        cleaned_data.loc[outliers & (data[column] > upper_bound), column] = upper_bound
                    else:  # zscore method
                        # Cap at 3 standard deviations
                        mean_val = values.mean()
                        std_val = values.std()
                        cleaned_data.loc[outliers & (data[column] < mean_val - 3*std_val), column] = mean_val - 3*std_val
                        cleaned_data.loc[outliers & (data[column] > mean_val + 3*std_val), column] = mean_val + 3*std_val
                else:
                    # For other features, set outliers to NaN
                    cleaned_data.loc[outliers, column] = np.nan
                
                # Reduce quality score for outliers
                if 'data_quality_score' in cleaned_data.columns:
                    cleaned_data.loc[outliers, 'data_quality_score'] *= 0.7
        
        if outlier_counts:
            logger.info(f"Cleaned outliers in {source}: {outlier_counts}")
        
        return cleaned_data
    
    def extract_source_features(self, data: pd.DataFrame, source: str, 
                               clean_outliers: bool = True) -> pd.DataFrame:
        """Extract all features from a single weather data source.
        
        Args:
            data: DataFrame with weather data from a single source
            source: Name of the weather data source
            clean_outliers: Whether to clean outliers
            
        Returns:
            DataFrame with all extracted features
        """
        if data.empty:
            logger.warning(f"No data available for source: {source}")
            return pd.DataFrame()
        
        # Clean outliers if requested
        if clean_outliers:
            data = self.detect_and_clean_outliers(data, source)
        
        # Extract different types of features
        temp_features = self.extract_temperature_features(data, source)
        atmo_features = self.extract_atmospheric_features(data, source)
        quality_features = self.extract_quality_features(data, source)
        
        # Merge all features
        all_features = temp_features
        for features_df in [atmo_features, quality_features]:
            if not features_df.empty:
                all_features = all_features.merge(features_df, on='date', how='outer')
        
        logger.info(f"Extracted {len(all_features.columns)-1} features from {source} ({len(all_features)} records)")
        return all_features
    
    def extract_all_source_features(self, source_data: Dict[str, pd.DataFrame], 
                                   clean_outliers: bool = True) -> pd.DataFrame:
        """Extract features from all weather data sources.
        
        Args:
            source_data: Dictionary mapping source names to DataFrames
            clean_outliers: Whether to clean outliers
            
        Returns:
            DataFrame with features from all sources
        """
        all_source_features = []
        
        for source, data in source_data.items():
            if data.empty:
                logger.warning(f"No data available for source: {source}")
                continue
            
            source_features = self.extract_source_features(data, source, clean_outliers)
            if not source_features.empty:
                all_source_features.append(source_features)
        
        if not all_source_features:
            logger.error("No features extracted from any source")
            return pd.DataFrame()
        
        # Merge all source features on date
        combined_features = all_source_features[0]
        for features_df in all_source_features[1:]:
            combined_features = combined_features.merge(features_df, on='date', how='outer')
        
        # Add date-based features
        if not combined_features.empty:
            date_features = self.extract_date_features(combined_features['date'])
            combined_features = combined_features.merge(date_features, on='date', how='left')
        
        # Sort by date
        combined_features = combined_features.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Combined features from all sources: {len(combined_features.columns)-1} features, {len(combined_features)} records")
        return combined_features
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Get summary statistics for extracted features.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Dictionary with feature summary statistics
        """
        if features_df.empty:
            return {'error': 'No features to summarize'}
        
        # Exclude date column from summary
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        summary = {
            'total_features': len(features_df.columns) - 1,  # Exclude date column
            'total_records': len(features_df),
            'numeric_features': len(numeric_features.columns),
            'missing_data_percentage': (features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))) * 100,
            'date_range': {
                'start': features_df['date'].min().strftime('%Y-%m-%d') if 'date' in features_df.columns else None,
                'end': features_df['date'].max().strftime('%Y-%m-%d') if 'date' in features_df.columns else None
            }
        }
        
        # Feature categories
        feature_categories = {
            'temperature': len([col for col in features_df.columns if 'temp' in col.lower()]),
            'atmospheric': len([col for col in features_df.columns if any(x in col.lower() for x in ['humidity', 'pressure', 'wind', 'cloud', 'precipitation'])]),
            'date_based': len([col for col in features_df.columns if any(x in col.lower() for x in ['day', 'month', 'season', 'weekend'])]),
            'quality': len([col for col in features_df.columns if 'quality' in col.lower() or 'missing' in col.lower() or 'completeness' in col.lower()])
        }
        summary['feature_categories'] = feature_categories
        
        # Top features by variance (excluding date features)
        if len(numeric_features.columns) > 0:
            feature_variance = numeric_features.var().sort_values(ascending=False)
            summary['top_variance_features'] = feature_variance.head(10).to_dict()
        
        return summary
    
    def validate_features(self, features_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate extracted features for quality and completeness.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if features_df.empty:
            issues.append("Features DataFrame is empty")
            return False, issues
        
        # Check for date column
        if 'date' not in features_df.columns:
            issues.append("Missing required 'date' column")
        
        # Check for minimum number of features
        if len(features_df.columns) < 5:  # date + at least 4 features
            issues.append(f"Too few features extracted: {len(features_df.columns)-1}")
        
        # Check for excessive missing data
        missing_percentage = (features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))) * 100
        if missing_percentage > 50:
            issues.append(f"Excessive missing data: {missing_percentage:.1f}%")
        
        # Check for duplicate dates
        if 'date' in features_df.columns:
            duplicate_dates = features_df['date'].duplicated().sum()
            if duplicate_dates > 0:
                issues.append(f"Found {duplicate_dates} duplicate dates")
        
        # Check for constant features (no variance)
        numeric_features = features_df.select_dtypes(include=[np.number])
        constant_features = []
        for col in numeric_features.columns:
            if numeric_features[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"Found constant features (no variance): {constant_features}")
        
        # Check for reasonable feature ranges
        if 'day_of_year' in features_df.columns:
            doy_range = features_df['day_of_year'].agg(['min', 'max'])
            if doy_range['min'] < 1 or doy_range['max'] > 366:
                issues.append(f"Day of year out of range: {doy_range['min']}-{doy_range['max']}")
        
        if 'month' in features_df.columns:
            month_range = features_df['month'].agg(['min', 'max'])
            if month_range['min'] < 1 or month_range['max'] > 12:
                issues.append(f"Month out of range: {month_range['min']}-{month_range['max']}")
        
        is_valid = len(issues) == 0
        return is_valid, issues