"""Feature engineering pipeline for weather prediction."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, Optional, Tuple, List
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from src.utils.data_manager import DataManager
from src.feature_engineering.basic_features import BasicFeatureExtractor
from src.feature_engineering.ensemble_features import EnsembleFeatureExtractor
from src.feature_engineering.la_weather_patterns import LAWeatherPatternExtractor


class FeaturePipeline:
    """Complete feature engineering pipeline for weather prediction."""
    
    def __init__(self):
        """Initialize the feature pipeline."""
        self.data_manager = DataManager()
        self.feature_extractor = BasicFeatureExtractor()
        self.ensemble_extractor = EnsembleFeatureExtractor()
        self.la_pattern_extractor = LAWeatherPatternExtractor()
        logger.info("FeaturePipeline initialized")
    
    def create_features_for_date_range(self, start_date: date, end_date: date, 
                                     clean_outliers: bool = True) -> pd.DataFrame:
        """Create features for a specific date range.
        
        Args:
            start_date: Start date for feature extraction
            end_date: End date for feature extraction
            clean_outliers: Whether to clean outliers
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Creating features for date range: {start_date} to {end_date}")
        
        # Load weather data for the date range
        source_data = self.data_manager.load_all_sources(start_date, end_date)
        
        # Check if we have any data
        available_sources = [source for source, data in source_data.items() if not data.empty]
        if not available_sources:
            logger.warning(f"No weather data available for date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        logger.info(f"Found data from {len(available_sources)} sources: {available_sources}")
        
        # Extract features from all sources
        features = self.feature_extractor.extract_all_source_features(source_data, clean_outliers)
        
        if not features.empty:
            logger.info(f"Created {len(features.columns)-1} features for {len(features)} records")
        else:
            logger.warning("No features extracted")
        
        return features
    
    def create_features_for_prediction(self, target_date: date, 
                                     lookback_days: int = 30) -> pd.DataFrame:
        """Create features for making a prediction on a specific date.
        
        Args:
            target_date: Date to make prediction for
            lookback_days: Number of days to look back for feature creation
            
        Returns:
            DataFrame with features for the target date
        """
        logger.info(f"Creating features for prediction on {target_date}")
        
        # Calculate date range (include some historical data for trends)
        start_date = target_date - timedelta(days=lookback_days)
        end_date = target_date
        
        # Create features for the date range
        all_features = self.create_features_for_date_range(start_date, end_date)
        
        if all_features.empty:
            logger.error(f"No features available for prediction on {target_date}")
            return pd.DataFrame()
        
        # Filter to just the target date
        target_features = all_features[all_features['date'] == pd.to_datetime(target_date)]
        
        if target_features.empty:
            logger.error(f"No features available for target date {target_date}")
            return pd.DataFrame()
        
        logger.info(f"Created {len(target_features.columns)-1} features for prediction on {target_date}")
        return target_features
    
    def create_training_dataset(self, start_date: date, end_date: date, 
                               include_targets: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Create a complete training dataset with features and targets.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            include_targets: Whether to include target values (actual temperatures)
            
        Returns:
            Tuple of (features_df, targets_series) or (features_df, None)
        """
        logger.info(f"Creating training dataset from {start_date} to {end_date}")
        
        # Create features
        features = self.create_features_for_date_range(start_date, end_date)
        
        if features.empty:
            logger.error("No features available for training dataset")
            return pd.DataFrame(), None
        
        targets = None
        if include_targets:
            # Load actual temperatures
            actual_temps = self.data_manager.load_source_data('actual_temperatures', start_date, end_date)
            
            if not actual_temps.empty:
                # Convert date column for merging
                actual_temps['date'] = pd.to_datetime(actual_temps['date'])
                features['date'] = pd.to_datetime(features['date'])
                
                # Merge features with actual temperatures
                merged = features.merge(actual_temps[['date', 'actual_high']], on='date', how='left')
                
                # Extract targets
                targets = merged['actual_high']
                
                # Remove rows without targets for training
                valid_mask = targets.notna()
                features = features[valid_mask].reset_index(drop=True)
                targets = targets[valid_mask].reset_index(drop=True)
                
                logger.info(f"Training dataset: {len(features)} records with targets")
            else:
                logger.warning("No actual temperature data available for targets")
        
        return features, targets
    
    def create_complete_features(self, start_date: date, end_date: date, 
                                include_ensemble: bool = True, 
                                include_la_patterns: bool = True,
                                clean_outliers: bool = True) -> pd.DataFrame:
        """Create complete feature set including basic, ensemble, and LA-specific features.
        
        Args:
            start_date: Start date for feature extraction
            end_date: End date for feature extraction
            include_ensemble: Whether to include ensemble and meta-features
            include_la_patterns: Whether to include LA-specific weather pattern features
            clean_outliers: Whether to clean outliers
            
        Returns:
            DataFrame with complete feature set
        """
        logger.info(f"Creating complete features for date range: {start_date} to {end_date}")
        
        # Load weather data for the date range
        source_data = self.data_manager.load_all_sources(start_date, end_date)
        
        # Check if we have any data
        available_sources = [source for source, data in source_data.items() if not data.empty]
        if not available_sources:
            logger.warning(f"No weather data available for date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        logger.info(f"Found data from {len(available_sources)} sources: {available_sources}")
        
        # Extract basic features from all sources
        basic_features = self.feature_extractor.extract_all_source_features(source_data, clean_outliers)
        
        if basic_features.empty:
            logger.warning("No basic features extracted")
            return pd.DataFrame()
        
        complete_features = basic_features
        
        # Add ensemble features if requested
        if include_ensemble and len(available_sources) > 1:
            logger.info("Adding ensemble and meta-features")
            ensemble_features = self.ensemble_extractor.create_all_ensemble_features(source_data)
            
            if not ensemble_features.empty:
                # Ensure date columns have consistent types
                complete_features['date'] = pd.to_datetime(complete_features['date'])
                ensemble_features['date'] = pd.to_datetime(ensemble_features['date'])
                
                # Merge basic and ensemble features
                complete_features = complete_features.merge(ensemble_features, on='date', how='outer')
                logger.info(f"Added ensemble features: {len(ensemble_features.columns)-1} features")
            else:
                logger.warning("No ensemble features created")
        else:
            if not include_ensemble:
                logger.info("Ensemble features disabled")
            else:
                logger.info("Only one data source available, skipping ensemble features")
        
        # Add LA-specific weather pattern features if requested
        if include_la_patterns:
            logger.info("Adding LA-specific weather pattern features")
            
            # Create a combined dataset for LA pattern analysis
            combined_source_data = pd.DataFrame()
            for source, data in source_data.items():
                if not data.empty:
                    if combined_source_data.empty:
                        combined_source_data = data.copy()
                    else:
                        # Take the mean of overlapping columns for pattern analysis
                        merged = combined_source_data.merge(data, on='date', how='outer', suffixes=('', '_temp'))
                        
                        # Average numeric columns
                        numeric_cols = ['predicted_high', 'predicted_low', 'humidity', 'pressure', 
                                       'wind_speed', 'wind_direction', 'cloud_cover', 'precipitation_prob']
                        
                        for col in numeric_cols:
                            if col in merged.columns and f'{col}_temp' in merged.columns:
                                merged[col] = merged[[col, f'{col}_temp']].mean(axis=1)
                                merged = merged.drop(f'{col}_temp', axis=1)
                            elif f'{col}_temp' in merged.columns:
                                merged[col] = merged[f'{col}_temp']
                                merged = merged.drop(f'{col}_temp', axis=1)
                        
                        combined_source_data = merged
            
            if not combined_source_data.empty:
                la_pattern_features = self.la_pattern_extractor.create_all_la_pattern_features(combined_source_data)
                
                if not la_pattern_features.empty:
                    # Ensure date columns have consistent types
                    complete_features['date'] = pd.to_datetime(complete_features['date'])
                    la_pattern_features['date'] = pd.to_datetime(la_pattern_features['date'])
                    
                    # Merge with existing features
                    complete_features = complete_features.merge(la_pattern_features, on='date', how='outer')
                    logger.info(f"Added LA pattern features: {len(la_pattern_features.columns)-1} features")
                else:
                    logger.warning("No LA pattern features created")
            else:
                logger.warning("No combined data available for LA pattern analysis")
        else:
            logger.info("LA-specific pattern features disabled")
        
        # Sort by date
        complete_features = complete_features.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Created {len(complete_features.columns)-1} total features for {len(complete_features)} records")
        return complete_features
    
    def get_feature_importance_analysis(self, features: pd.DataFrame) -> Dict:
        """Analyze feature importance and correlations.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Dictionary with feature analysis
        """
        if features.empty:
            return {'error': 'No features to analyze'}
        
        # Get numeric features only
        numeric_features = features.select_dtypes(include=[np.number])
        if 'date' in numeric_features.columns:
            numeric_features = numeric_features.drop('date', axis=1)
        
        analysis = {
            'total_features': len(features.columns) - 1,  # Exclude date
            'numeric_features': len(numeric_features.columns),
            'feature_variance': numeric_features.var().sort_values(ascending=False).to_dict(),
            'feature_correlations': {},
            'missing_data_by_feature': features.isnull().sum().to_dict()
        }
        
        # Calculate correlations between features
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr()
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            analysis['high_correlation_pairs'] = high_corr_pairs
        
        # Categorize features
        feature_categories = {
            'temperature': [col for col in features.columns if 'temp' in col.lower()],
            'atmospheric': [col for col in features.columns if any(x in col.lower() for x in ['humidity', 'pressure', 'wind', 'cloud', 'precipitation'])],
            'date_based': [col for col in features.columns if any(x in col.lower() for x in ['day', 'month', 'season', 'weekend'])],
            'quality': [col for col in features.columns if 'quality' in col.lower() or 'missing' in col.lower() or 'completeness' in col.lower()],
            'derived': [col for col in features.columns if any(x in col.lower() for x in ['change', 'avg', 'range', 'sin', 'cos'])],
            'ensemble': [col for col in features.columns if any(x in col.lower() for x in ['consensus', 'agreement', 'rolling', 'trend'])],
            'la_patterns': [col for col in features.columns if any(x in col.lower() for x in ['marine_layer', 'santa_ana', 'heat_island', 'fire_season'])]
        }
        analysis['feature_categories'] = {cat: len(features) for cat, features in feature_categories.items()}
        
        return analysis
    
    def validate_feature_quality(self, features: pd.DataFrame) -> Dict:
        """Validate the quality of extracted features.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if features.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Features DataFrame is empty")
            return validation_results
        
        # Basic validation using the feature extractor
        is_valid, issues = self.feature_extractor.validate_features(features)
        if not is_valid:
            validation_results['is_valid'] = False
            validation_results['errors'].extend(issues)
        
        # Additional quality checks
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Check for excessive missing data
        missing_percentage = (features.isnull().sum().sum() / (len(features) * len(features.columns))) * 100
        validation_results['statistics']['missing_data_percentage'] = missing_percentage
        
        if missing_percentage > 30:
            validation_results['warnings'].append(f"High missing data percentage: {missing_percentage:.1f}%")
        
        # Check for features with very low variance
        if len(numeric_features.columns) > 0:
            low_variance_features = []
            for col in numeric_features.columns:
                if numeric_features[col].var() < 0.01:
                    low_variance_features.append(col)
            
            if low_variance_features:
                validation_results['warnings'].append(f"Low variance features: {len(low_variance_features)}")
                validation_results['statistics']['low_variance_features'] = low_variance_features
        
        # Check for reasonable date range
        if 'date' in features.columns:
            date_range = features['date'].agg(['min', 'max'])
            validation_results['statistics']['date_range'] = {
                'start': date_range['min'].strftime('%Y-%m-%d'),
                'end': date_range['max'].strftime('%Y-%m-%d'),
                'days': (date_range['max'] - date_range['min']).days
            }
        
        # Check feature distribution
        validation_results['statistics']['feature_count_by_source'] = {}
        for source in ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']:
            source_features = [col for col in features.columns if col.startswith(f'{source}_')]
            if source_features:
                validation_results['statistics']['feature_count_by_source'][source] = len(source_features)
        
        return validation_results
    
    def save_features_to_file(self, features: pd.DataFrame, filepath: str) -> bool:
        """Save features to a file for later use.
        
        Args:
            features: DataFrame with features
            filepath: Path to save the features
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filepath.endswith('.parquet'):
                features.to_parquet(filepath, index=False)
            elif filepath.endswith('.csv'):
                features.to_csv(filepath, index=False)
            else:
                logger.error(f"Unsupported file format: {filepath}")
                return False
            
            logger.info(f"Saved {len(features)} feature records to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving features to {filepath}: {e}")
            return False
    
    def load_features_from_file(self, filepath: str) -> pd.DataFrame:
        """Load features from a file.
        
        Args:
            filepath: Path to load the features from
            
        Returns:
            DataFrame with features
        """
        try:
            if filepath.endswith('.parquet'):
                features = pd.read_parquet(filepath)
            elif filepath.endswith('.csv'):
                features = pd.read_csv(filepath)
                # Convert date column back to datetime
                if 'date' in features.columns:
                    features['date'] = pd.to_datetime(features['date'])
            else:
                logger.error(f"Unsupported file format: {filepath}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(features)} feature records from {filepath}")
            return features
            
        except Exception as e:
            logger.error(f"Error loading features from {filepath}: {e}")
            return pd.DataFrame()
    
    def get_pipeline_status(self) -> Dict:
        """Get the current status of the feature pipeline.
        
        Returns:
            Dictionary with pipeline status information
        """
        # Get data summary
        data_summary = self.data_manager.get_data_summary()
        
        # Count available sources
        available_sources = []
        total_records = 0
        for source, info in data_summary.items():
            if isinstance(info, dict) and 'records' in info:
                available_sources.append(source)
                total_records += info['records']
        
        status = {
            'available_data_sources': len(available_sources),
            'total_weather_records': total_records,
            'data_sources': available_sources,
            'feature_extractor_ready': True,
            'pipeline_ready': len(available_sources) > 0
        }
        
        # Get date range of available data
        if available_sources:
            all_dates = []
            for source in available_sources:
                if source in ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']:
                    data = self.data_manager.load_source_data(source)
                    if not data.empty and 'date' in data.columns:
                        all_dates.extend(data['date'].tolist())
            
            if all_dates:
                all_dates = pd.to_datetime(all_dates)
                status['data_date_range'] = {
                    'start': all_dates.min().strftime('%Y-%m-%d'),
                    'end': all_dates.max().strftime('%Y-%m-%d'),
                    'days': (all_dates.max() - all_dates.min()).days
                }
        
        return status


def main():
    """Demonstrate the complete feature pipeline."""
    print("=== Feature Pipeline Demo ===\n")
    
    pipeline = FeaturePipeline()
    
    # Get pipeline status
    print("1. Pipeline Status:")
    status = pipeline.get_pipeline_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    print()
    
    if not status['pipeline_ready']:
        print("Pipeline not ready - no weather data available")
        return
    
    # Create features for recent data
    print("2. Creating Complete Features (Basic + Ensemble + LA Patterns) for Recent Data:")
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    features = pipeline.create_complete_features(start_date, end_date, include_ensemble=True, include_la_patterns=True)
    
    if not features.empty:
        print(f"   Created features: {features.shape}")
        print(f"   Date range: {features['date'].min()} to {features['date'].max()}")
        
        # Validate features
        print("\n3. Feature Quality Validation:")
        validation = pipeline.validate_feature_quality(features)
        print(f"   Valid: {validation['is_valid']}")
        if validation['errors']:
            print(f"   Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"   Warnings: {validation['warnings']}")
        
        # Feature analysis
        print("\n4. Feature Analysis:")
        analysis = pipeline.get_feature_importance_analysis(features)
        print(f"   Total features: {analysis['total_features']}")
        print(f"   Numeric features: {analysis['numeric_features']}")
        print(f"   Feature categories: {analysis['feature_categories']}")
        
        if 'high_correlation_pairs' in analysis and analysis['high_correlation_pairs']:
            print(f"   High correlation pairs: {len(analysis['high_correlation_pairs'])}")
        
        # Show top variance features
        if 'feature_variance' in analysis:
            print("\n   Top 5 features by variance:")
            for i, (feature, variance) in enumerate(list(analysis['feature_variance'].items())[:5]):
                print(f"   {i+1}. {feature}: {variance:.2f}")
    
    print("\n=== Feature Pipeline Demo Complete ===")


if __name__ == '__main__':
    main()