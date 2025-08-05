"""Ensemble and meta-feature extraction for weather prediction."""

from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class EnsembleFeatureExtractor:
    """Extracts ensemble and meta-features from multiple weather data sources."""
    
    def __init__(self):
        """Initialize the ensemble feature extractor."""
        self.weather_sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
        
        # Define core weather parameters for ensemble features
        self.ensemble_parameters = [
            'predicted_high', 'predicted_low', 'humidity', 'pressure', 
            'wind_speed', 'cloud_cover', 'precipitation_prob'
        ]
        
        logger.info("EnsembleFeatureExtractor initialized")
    
    def create_consensus_features(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create consensus features (mean, median, std dev) across all sources.
        
        Args:
            source_data: Dictionary mapping source names to DataFrames
            
        Returns:
            DataFrame with consensus features
        """
        logger.info("Creating consensus features across all sources")
        
        # Get all unique dates across sources
        all_dates = set()
        for source, data in source_data.items():
            if not data.empty and 'date' in data.columns:
                all_dates.update(pd.to_datetime(data['date']).dt.date)
        
        if not all_dates:
            logger.warning("No dates found in source data")
            return pd.DataFrame()
        
        all_dates = sorted(list(all_dates))
        consensus_features = pd.DataFrame({'date': pd.to_datetime(all_dates)})
        
        # Create consensus features for each parameter
        for parameter in self.ensemble_parameters:
            logger.debug(f"Creating consensus features for {parameter}")
            
            # Collect values from all sources for this parameter
            parameter_data = []
            for target_date in all_dates:
                date_values = []
                source_count = 0
                
                for source, data in source_data.items():
                    if data.empty or parameter not in data.columns:
                        continue
                    
                    # Get value for this date from this source
                    date_mask = pd.to_datetime(data['date']).dt.date == target_date
                    if date_mask.any():
                        value = data.loc[date_mask, parameter].iloc[0]
                        if pd.notna(value):
                            date_values.append(value)
                            source_count += 1
                
                # Calculate consensus statistics
                if date_values:
                    consensus_stats = {
                        f'{parameter}_consensus_mean': np.mean(date_values),
                        f'{parameter}_consensus_median': np.median(date_values),
                        f'{parameter}_consensus_std': np.std(date_values) if len(date_values) > 1 else 0.0,
                        f'{parameter}_consensus_min': np.min(date_values),
                        f'{parameter}_consensus_max': np.max(date_values),
                        f'{parameter}_consensus_range': np.max(date_values) - np.min(date_values),
                        f'{parameter}_source_count': source_count
                    }
                else:
                    # No data available for this date/parameter
                    consensus_stats = {
                        f'{parameter}_consensus_mean': np.nan,
                        f'{parameter}_consensus_median': np.nan,
                        f'{parameter}_consensus_std': np.nan,
                        f'{parameter}_consensus_min': np.nan,
                        f'{parameter}_consensus_max': np.nan,
                        f'{parameter}_consensus_range': np.nan,
                        f'{parameter}_source_count': 0
                    }
                
                parameter_data.append(consensus_stats)
            
            # Add parameter consensus features to main DataFrame
            parameter_df = pd.DataFrame(parameter_data)
            for col in parameter_df.columns:
                consensus_features[col] = parameter_df[col]
        
        logger.info(f"Created {len(consensus_features.columns)-1} consensus features for {len(consensus_features)} dates")
        return consensus_features
    
    def create_agreement_metrics(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create API agreement/disagreement metrics.
        
        Args:
            source_data: Dictionary mapping source names to DataFrames
            
        Returns:
            DataFrame with agreement metrics
        """
        logger.info("Creating API agreement/disagreement metrics")
        
        # Get all unique dates
        all_dates = set()
        for source, data in source_data.items():
            if not data.empty and 'date' in data.columns:
                all_dates.update(pd.to_datetime(data['date']).dt.date)
        
        if not all_dates:
            return pd.DataFrame()
        
        all_dates = sorted(list(all_dates))
        agreement_features = pd.DataFrame({'date': pd.to_datetime(all_dates)})
        
        # Focus on key parameters for agreement analysis
        key_parameters = ['predicted_high', 'predicted_low', 'humidity', 'pressure']
        
        for parameter in key_parameters:
            logger.debug(f"Creating agreement metrics for {parameter}")
            
            agreement_data = []
            for target_date in all_dates:
                # Collect values from all sources
                source_values = {}
                for source, data in source_data.items():
                    if data.empty or parameter not in data.columns:
                        continue
                    
                    date_mask = pd.to_datetime(data['date']).dt.date == target_date
                    if date_mask.any():
                        value = data.loc[date_mask, parameter].iloc[0]
                        if pd.notna(value):
                            source_values[source] = value
                
                # Calculate agreement metrics
                if len(source_values) >= 2:
                    values = list(source_values.values())
                    
                    # Coefficient of variation (std/mean) - lower means more agreement
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / mean_val if mean_val != 0 else 0
                    
                    # Pairwise agreement (average absolute difference between all pairs)
                    pairwise_diffs = []
                    for i in range(len(values)):
                        for j in range(i+1, len(values)):
                            pairwise_diffs.append(abs(values[i] - values[j]))
                    avg_pairwise_diff = np.mean(pairwise_diffs) if pairwise_diffs else 0
                    
                    # Agreement score (inverse of coefficient of variation, scaled 0-1)
                    agreement_score = 1 / (1 + cv) if cv > 0 else 1.0
                    
                    # Outlier detection - sources that deviate significantly from median
                    median_val = np.median(values)
                    mad = np.median([abs(v - median_val) for v in values])  # Median Absolute Deviation
                    outlier_threshold = median_val + 2 * mad if mad > 0 else median_val
                    outlier_count = sum(1 for v in values if abs(v - median_val) > outlier_threshold)
                    
                    metrics = {
                        f'{parameter}_agreement_cv': cv,
                        f'{parameter}_agreement_score': agreement_score,
                        f'{parameter}_pairwise_diff': avg_pairwise_diff,
                        f'{parameter}_outlier_count': outlier_count,
                        f'{parameter}_agreement_sources': len(source_values)
                    }
                else:
                    # Insufficient data for agreement analysis
                    metrics = {
                        f'{parameter}_agreement_cv': np.nan,
                        f'{parameter}_agreement_score': np.nan,
                        f'{parameter}_pairwise_diff': np.nan,
                        f'{parameter}_outlier_count': 0,
                        f'{parameter}_agreement_sources': len(source_values)
                    }
                
                agreement_data.append(metrics)
            
            # Add agreement metrics to main DataFrame
            agreement_df = pd.DataFrame(agreement_data)
            for col in agreement_df.columns:
                agreement_features[col] = agreement_df[col]
        
        # Create overall agreement metrics across all parameters
        logger.debug("Creating overall agreement metrics")
        
        # Average agreement score across all parameters
        agreement_cols = [col for col in agreement_features.columns if '_agreement_score' in col]
        if agreement_cols:
            agreement_features['overall_agreement_score'] = agreement_features[agreement_cols].mean(axis=1)
        
        # Total number of sources with data
        source_count_cols = [col for col in agreement_features.columns if '_agreement_sources' in col]
        if source_count_cols:
            agreement_features['total_active_sources'] = agreement_features[source_count_cols].max(axis=1)
        
        # Disagreement flag - high when sources disagree significantly
        cv_cols = [col for col in agreement_features.columns if '_agreement_cv' in col]
        if cv_cols:
            # High disagreement when average CV > 0.1 (10%)
            agreement_features['high_disagreement_flag'] = (
                agreement_features[cv_cols].mean(axis=1) > 0.1
            ).astype(int)
        
        logger.info(f"Created {len(agreement_features.columns)-1} agreement metrics for {len(agreement_features)} dates")
        return agreement_features
    
    def create_rolling_features(self, source_data: Dict[str, pd.DataFrame], 
                               windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """Create rolling average and trend features.
        
        Args:
            source_data: Dictionary mapping source names to DataFrames
            windows: List of rolling window sizes in days
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features with windows: {windows}")
        
        # First create consensus features to use as base for rolling calculations
        consensus_features = self.create_consensus_features(source_data)
        
        if consensus_features.empty:
            logger.warning("No consensus features available for rolling calculations")
            return pd.DataFrame()
        
        # Sort by date for proper rolling calculations
        consensus_features = consensus_features.sort_values('date').reset_index(drop=True)
        rolling_features = consensus_features[['date']].copy()
        
        # Parameters to create rolling features for
        rolling_parameters = ['predicted_high', 'predicted_low', 'humidity', 'pressure', 'wind_speed']
        
        for parameter in rolling_parameters:
            consensus_col = f'{parameter}_consensus_mean'
            if consensus_col not in consensus_features.columns:
                continue
            
            logger.debug(f"Creating rolling features for {parameter}")
            
            # Get the consensus values
            values = consensus_features[consensus_col]
            
            for window in windows:
                if len(values) < window:
                    logger.debug(f"Insufficient data for {window}-day window (need {window}, have {len(values)})")
                    continue
                
                # Rolling mean
                rolling_mean = values.rolling(window=window, min_periods=1).mean()
                rolling_features[f'{parameter}_rolling_{window}d_mean'] = rolling_mean
                
                # Rolling standard deviation
                rolling_std = values.rolling(window=window, min_periods=2).std()
                rolling_features[f'{parameter}_rolling_{window}d_std'] = rolling_std
                
                # Rolling trend (slope of linear regression over window)
                rolling_trend = values.rolling(window=window, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
                    raw=False
                )
                rolling_features[f'{parameter}_rolling_{window}d_trend'] = rolling_trend
                
                # Rolling momentum (current value vs rolling mean)
                rolling_momentum = values - rolling_mean
                rolling_features[f'{parameter}_rolling_{window}d_momentum'] = rolling_momentum
                
                # Rolling volatility (coefficient of variation)
                rolling_cv = rolling_std / rolling_mean
                rolling_features[f'{parameter}_rolling_{window}d_volatility'] = rolling_cv
        
        # Create cross-parameter rolling features
        logger.debug("Creating cross-parameter rolling features")
        
        # Temperature range stability (rolling std of daily temperature range)
        if ('predicted_high_consensus_mean' in consensus_features.columns and 
            'predicted_low_consensus_mean' in consensus_features.columns):
            
            temp_range = (consensus_features['predicted_high_consensus_mean'] - 
                         consensus_features['predicted_low_consensus_mean'])
            
            for window in windows:
                if len(temp_range) >= window:
                    range_stability = temp_range.rolling(window=window, min_periods=1).std()
                    rolling_features[f'temp_range_stability_{window}d'] = range_stability
        
        # Weather pattern consistency (how consistent are the weather conditions)
        weather_params = ['humidity_consensus_mean', 'pressure_consensus_mean', 'wind_speed_consensus_mean']
        available_params = [p for p in weather_params if p in consensus_features.columns]
        
        if len(available_params) >= 2:
            for window in windows:
                if len(consensus_features) >= window:
                    # Calculate the average coefficient of variation across weather parameters
                    cv_values = []
                    for param in available_params:
                        values = consensus_features[param]
                        rolling_mean = values.rolling(window=window, min_periods=1).mean()
                        rolling_std = values.rolling(window=window, min_periods=2).std()
                        cv = rolling_std / rolling_mean
                        cv_values.append(cv)
                    
                    if cv_values:
                        avg_cv = pd.concat(cv_values, axis=1).mean(axis=1)
                        rolling_features[f'weather_consistency_{window}d'] = 1 / (1 + avg_cv)
        
        logger.info(f"Created {len(rolling_features.columns)-1} rolling features for {len(rolling_features)} dates")
        return rolling_features
    
    def create_trend_features(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create advanced trend features across sources.
        
        Args:
            source_data: Dictionary mapping source names to DataFrames
            
        Returns:
            DataFrame with trend features
        """
        logger.info("Creating advanced trend features")
        
        # Get consensus features as base
        consensus_features = self.create_consensus_features(source_data)
        
        if consensus_features.empty or len(consensus_features) < 3:
            logger.warning("Insufficient data for trend analysis")
            return pd.DataFrame()
        
        # Sort by date
        consensus_features = consensus_features.sort_values('date').reset_index(drop=True)
        trend_features = consensus_features[['date']].copy()
        
        # Parameters for trend analysis
        trend_parameters = ['predicted_high', 'predicted_low', 'pressure', 'humidity']
        
        for parameter in trend_parameters:
            consensus_col = f'{parameter}_consensus_mean'
            if consensus_col not in consensus_features.columns:
                continue
            
            logger.debug(f"Creating trend features for {parameter}")
            
            values = consensus_features[consensus_col].dropna()
            if len(values) < 3:
                continue
            
            # Short-term trend (3-day slope)
            if len(values) >= 3:
                trend_3d = values.rolling(window=3, min_periods=3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0],
                    raw=False
                )
                trend_features[f'{parameter}_trend_3d'] = trend_3d
            
            # Medium-term trend (7-day slope)
            if len(values) >= 7:
                trend_7d = values.rolling(window=7, min_periods=7).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0],
                    raw=False
                )
                trend_features[f'{parameter}_trend_7d'] = trend_7d
            
            # Trend acceleration (change in trend)
            if f'{parameter}_trend_3d' in trend_features.columns:
                trend_accel = trend_features[f'{parameter}_trend_3d'].diff()
                trend_features[f'{parameter}_trend_acceleration'] = trend_accel
            
            # Trend reversal detection
            if f'{parameter}_trend_3d' in trend_features.columns:
                trend_3d_values = trend_features[f'{parameter}_trend_3d']
                # Reversal when trend changes sign
                trend_reversal = ((trend_3d_values > 0) & (trend_3d_values.shift(1) < 0)) | \
                                ((trend_3d_values < 0) & (trend_3d_values.shift(1) > 0))
                trend_features[f'{parameter}_trend_reversal'] = trend_reversal.astype(int)
        
        # Cross-parameter trend features
        logger.debug("Creating cross-parameter trend features")
        
        # Temperature-pressure relationship trend
        if ('predicted_high_trend_3d' in trend_features.columns and 
            'pressure_trend_3d' in trend_features.columns):
            
            # Correlation between temperature and pressure trends
            temp_trend = trend_features['predicted_high_trend_3d']
            pressure_trend = trend_features['pressure_trend_3d']
            
            # Rolling correlation (when both trends are available)
            correlation_window = 7
            if len(trend_features) >= correlation_window:
                rolling_corr = temp_trend.rolling(window=correlation_window).corr(pressure_trend)
                trend_features['temp_pressure_trend_correlation'] = rolling_corr
        
        logger.info(f"Created {len(trend_features.columns)-1} trend features for {len(trend_features)} dates")
        return trend_features
    
    def create_source_reliability_features(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features that measure individual source reliability and performance.
        
        Args:
            source_data: Dictionary mapping source names to DataFrames
            
        Returns:
            DataFrame with source reliability features
        """
        logger.info("Creating source reliability features")
        
        # Get all unique dates
        all_dates = set()
        for source, data in source_data.items():
            if not data.empty and 'date' in data.columns:
                all_dates.update(pd.to_datetime(data['date']).dt.date)
        
        if not all_dates:
            return pd.DataFrame()
        
        all_dates = sorted(list(all_dates))
        reliability_features = pd.DataFrame({'date': pd.to_datetime(all_dates)})
        
        # For each source, create reliability metrics
        for source in self.weather_sources:
            if source not in source_data or source_data[source].empty:
                continue
            
            logger.debug(f"Creating reliability features for {source}")
            
            source_reliability = []
            for target_date in all_dates:
                data = source_data[source]
                date_mask = pd.to_datetime(data['date']).dt.date == target_date
                
                if date_mask.any():
                    row = data.loc[date_mask].iloc[0]
                    
                    # Data completeness for this source on this date
                    total_fields = len(self.ensemble_parameters)
                    available_fields = sum(1 for param in self.ensemble_parameters 
                                         if param in row and pd.notna(row[param]))
                    completeness = available_fields / total_fields
                    
                    # Quality score (if available)
                    quality_score = row.get('data_quality_score', 0.5)
                    
                    # Freshness (how recent is the forecast)
                    if 'forecast_date' in row and pd.notna(row['forecast_date']):
                        forecast_age = (target_date - pd.to_datetime(row['forecast_date']).date()).days
                        freshness = max(0, 1 - forecast_age / 7)  # Decay over 7 days
                    else:
                        freshness = 0.5  # Default if no forecast date
                    
                    reliability_metrics = {
                        f'{source}_completeness': completeness,
                        f'{source}_quality_score': quality_score,
                        f'{source}_freshness': freshness,
                        f'{source}_available': 1
                    }
                else:
                    # Source not available for this date
                    reliability_metrics = {
                        f'{source}_completeness': 0,
                        f'{source}_quality_score': 0,
                        f'{source}_freshness': 0,
                        f'{source}_available': 0
                    }
                
                source_reliability.append(reliability_metrics)
            
            # Add source reliability features
            reliability_df = pd.DataFrame(source_reliability)
            for col in reliability_df.columns:
                reliability_features[col] = reliability_df[col]
        
        # Create aggregate reliability metrics
        logger.debug("Creating aggregate reliability metrics")
        
        # Total sources available
        available_cols = [col for col in reliability_features.columns if col.endswith('_available')]
        if available_cols:
            reliability_features['total_sources_available'] = reliability_features[available_cols].sum(axis=1)
        
        # Average quality across available sources
        quality_cols = [col for col in reliability_features.columns if col.endswith('_quality_score')]
        available_quality_cols = []
        for i, date_idx in enumerate(reliability_features.index):
            date_quality_scores = []
            for col in quality_cols:
                source = col.replace('_quality_score', '')
                if reliability_features.loc[date_idx, f'{source}_available'] == 1:
                    date_quality_scores.append(reliability_features.loc[date_idx, col])
            
            avg_quality = np.mean(date_quality_scores) if date_quality_scores else 0
            available_quality_cols.append(avg_quality)
        
        reliability_features['average_source_quality'] = available_quality_cols
        
        # Source diversity score (how many different sources we have)
        if available_cols:
            max_sources = len(available_cols)
            reliability_features['source_diversity_score'] = (
                reliability_features['total_sources_available'] / max_sources
            )
        
        logger.info(f"Created {len(reliability_features.columns)-1} reliability features for {len(reliability_features)} dates")
        return reliability_features
    
    def create_all_ensemble_features(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create all ensemble and meta-features.
        
        Args:
            source_data: Dictionary mapping source names to DataFrames
            
        Returns:
            DataFrame with all ensemble features
        """
        logger.info("Creating all ensemble and meta-features")
        
        if not source_data or all(data.empty for data in source_data.values()):
            logger.error("No source data available for ensemble feature creation")
            return pd.DataFrame()
        
        # Create different types of ensemble features
        consensus_features = self.create_consensus_features(source_data)
        agreement_features = self.create_agreement_metrics(source_data)
        rolling_features = self.create_rolling_features(source_data)
        trend_features = self.create_trend_features(source_data)
        reliability_features = self.create_source_reliability_features(source_data)
        
        # Merge all features
        all_features = consensus_features
        
        for features_df in [agreement_features, rolling_features, trend_features, reliability_features]:
            if not features_df.empty:
                all_features = all_features.merge(features_df, on='date', how='outer')
        
        if not all_features.empty:
            # Sort by date
            all_features = all_features.sort_values('date').reset_index(drop=True)
            logger.info(f"Created {len(all_features.columns)-1} total ensemble features for {len(all_features)} dates")
        else:
            logger.warning("No ensemble features created")
        
        return all_features
    
    def get_ensemble_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Get summary of ensemble features.
        
        Args:
            features_df: DataFrame with ensemble features
            
        Returns:
            Dictionary with feature summary
        """
        if features_df.empty:
            return {'error': 'No features to summarize'}
        
        # Categorize features
        feature_categories = {
            'consensus': len([col for col in features_df.columns if 'consensus' in col]),
            'agreement': len([col for col in features_df.columns if 'agreement' in col or 'disagreement' in col]),
            'rolling': len([col for col in features_df.columns if 'rolling' in col]),
            'trend': len([col for col in features_df.columns if 'trend' in col]),
            'reliability': len([col for col in features_df.columns if any(x in col for x in ['completeness', 'quality', 'freshness', 'available'])])
        }
        
        summary = {
            'total_features': len(features_df.columns) - 1,  # Exclude date
            'total_records': len(features_df),
            'feature_categories': feature_categories,
            'date_range': {
                'start': features_df['date'].min().strftime('%Y-%m-%d') if 'date' in features_df.columns else None,
                'end': features_df['date'].max().strftime('%Y-%m-%d') if 'date' in features_df.columns else None
            },
            'missing_data_percentage': (features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))) * 100
        }
        
        return summary