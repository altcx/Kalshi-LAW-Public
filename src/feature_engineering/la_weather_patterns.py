"""LA-specific weather pattern feature extraction for enhanced temperature prediction."""

from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class LAWeatherPatternExtractor:
    """Extracts Los Angeles-specific weather pattern features for temperature prediction."""
    
    def __init__(self):
        """Initialize the LA weather pattern extractor."""
        logger.info("LAWeatherPatternExtractor initialized")
    
    def detect_marine_layer_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect marine layer conditions based on weather parameters.
        
        Marine layer characteristics in LA:
        - High humidity (>80%) near coast
        - Low cloud cover or overcast conditions
        - Cooler temperatures in morning
        - Low wind speeds
        - Pressure gradient from coast to inland
        - Most common May-August, especially June-July
        
        Args:
            data: DataFrame with weather data including humidity, cloud_cover, wind_speed
            
        Returns:
            DataFrame with marine layer detection features
        """
        logger.debug("Detecting marine layer conditions")
        
        if data.empty:
            return pd.DataFrame()
        
        features = data[['date']].copy()
        
        # Marine layer probability based on multiple indicators
        marine_layer_score = pd.Series(0.0, index=data.index)
        
        # High humidity indicator (marine layer brings moisture)
        if 'humidity' in data.columns:
            humidity = data['humidity'].fillna(50)
            # Marine layer typically brings humidity >75%
            humidity_score = np.where(humidity > 75, 
                                    (humidity - 75) / 25,  # Scale 75-100% to 0-1
                                    0)
            marine_layer_score += np.clip(humidity_score, 0, 1) * 0.3
        
        # Cloud cover indicator (marine layer creates low clouds/overcast)
        if 'cloud_cover' in data.columns:
            cloud_cover = data['cloud_cover'].fillna(50)
            # Marine layer typically creates 70-100% cloud cover
            cloud_score = np.where(cloud_cover > 70,
                                 (cloud_cover - 70) / 30,  # Scale 70-100% to 0-1
                                 0)
            marine_layer_score += np.clip(cloud_score, 0, 1) * 0.25
        
        # Low wind speed indicator (marine layer associated with calm conditions)
        if 'wind_speed' in data.columns:
            wind_speed = data['wind_speed'].fillna(10)
            # Marine layer typically has wind speeds <8 mph
            wind_score = np.where(wind_speed < 8,
                                (8 - wind_speed) / 8,  # Scale 0-8 mph to 1-0
                                0)
            marine_layer_score += np.clip(wind_score, 0, 1) * 0.2
        
        # Seasonal adjustment (marine layer most common May-August)
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
            months = dates.dt.month
            # Peak marine layer months: May(0.7), June(1.0), July(1.0), August(0.8)
            seasonal_multiplier = np.where(months == 5, 0.7,
                                 np.where(months == 6, 1.0,
                                 np.where(months == 7, 1.0,
                                 np.where(months == 8, 0.8,
                                 np.where(months.isin([4, 9]), 0.4, 0.1)))))
            marine_layer_score *= seasonal_multiplier
        
        # Temperature depression indicator (marine layer cools temperatures)
        if 'predicted_high' in data.columns and len(data) > 1:
            temp_high = data['predicted_high'].fillna(data['predicted_high'].mean())
            # Calculate rolling 7-day average temperature
            temp_avg = temp_high.rolling(window=7, min_periods=1).mean()
            temp_depression = temp_avg - temp_high
            # Marine layer typically depresses temps by 5-15°F
            temp_score = np.where(temp_depression > 3,
                                np.clip((temp_depression - 3) / 12, 0, 1),  # Scale 3-15°F to 0-1
                                0)
            marine_layer_score += temp_score * 0.25
        
        # Final marine layer features
        features['marine_layer_probability'] = np.clip(marine_layer_score, 0, 1)
        features['marine_layer_likely'] = (marine_layer_score > 0.6).astype(int)
        features['marine_layer_possible'] = (marine_layer_score > 0.3).astype(int)
        
        # Marine layer strength categories
        features['marine_layer_strength'] = pd.cut(marine_layer_score, 
                                                  bins=[0, 0.3, 0.6, 1.0],
                                                  labels=['weak', 'moderate', 'strong'],
                                                  include_lowest=True)
        
        # Encode strength as numeric for ML models
        strength_map = {'weak': 0, 'moderate': 1, 'strong': 2}
        features['marine_layer_strength_numeric'] = features['marine_layer_strength'].map(strength_map).fillna(0)
        
        logger.debug(f"Created {len(features.columns)-1} marine layer features")
        return features
    
    def detect_santa_ana_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect Santa Ana wind conditions based on weather parameters.
        
        Santa Ana wind characteristics:
        - High wind speeds (>25 mph) from northeast/east
        - Low humidity (<20%)
        - High pressure inland, low pressure coast (pressure gradient)
        - Clear skies (low cloud cover)
        - Warm, dry conditions
        - Most common October-April, peak December-February
        
        Args:
            data: DataFrame with weather data including wind_speed, wind_direction, humidity
            
        Returns:
            DataFrame with Santa Ana wind detection features
        """
        logger.debug("Detecting Santa Ana wind conditions")
        
        if data.empty:
            return pd.DataFrame()
        
        features = data[['date']].copy()
        
        # Santa Ana probability based on multiple indicators
        santa_ana_score = pd.Series(0.0, index=data.index)
        
        # Wind speed indicator (Santa Ana winds are strong)
        if 'wind_speed' in data.columns:
            wind_speed = data['wind_speed'].fillna(5)
            # Santa Ana winds typically >20 mph, strongest >35 mph
            wind_score = np.where(wind_speed > 20,
                                np.clip((wind_speed - 20) / 25, 0, 1),  # Scale 20-45 mph to 0-1
                                0)
            santa_ana_score += wind_score * 0.35
        
        # Wind direction indicator (Santa Ana winds from northeast/east)
        if 'wind_direction' in data.columns:
            wind_direction = data['wind_direction'].fillna(180)
            # Santa Ana winds typically from 45-135 degrees (NE to SE)
            # Create a score that peaks at 90 degrees (east)
            direction_diff = np.abs(wind_direction - 90)
            direction_score = np.where(direction_diff <= 45,
                                     (45 - direction_diff) / 45,  # Scale 0-45° diff to 1-0
                                     0)
            santa_ana_score += direction_score * 0.25
        
        # Low humidity indicator (Santa Ana winds are very dry)
        if 'humidity' in data.columns:
            humidity = data['humidity'].fillna(50)
            # Santa Ana conditions typically have humidity <25%
            humidity_score = np.where(humidity < 30,
                                    (30 - humidity) / 30,  # Scale 0-30% to 1-0
                                    0)
            santa_ana_score += humidity_score * 0.25
        
        # Clear sky indicator (Santa Ana winds bring clear conditions)
        if 'cloud_cover' in data.columns:
            cloud_cover = data['cloud_cover'].fillna(50)
            # Santa Ana conditions typically have <20% cloud cover
            clear_score = np.where(cloud_cover < 30,
                                 (30 - cloud_cover) / 30,  # Scale 0-30% to 1-0
                                 0)
            santa_ana_score += clear_score * 0.15
        
        # Seasonal adjustment (Santa Ana winds most common October-April)
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
            months = dates.dt.month
            # Peak Santa Ana months: Dec(1.0), Jan(1.0), Feb(0.9), Nov(0.8), Oct(0.7), Mar(0.6)
            seasonal_multiplier = np.where(months.isin([12, 1]), 1.0,
                                 np.where(months == 2, 0.9,
                                 np.where(months == 11, 0.8,
                                 np.where(months == 10, 0.7,
                                 np.where(months == 3, 0.6,
                                 np.where(months == 4, 0.3, 0.1))))))
            santa_ana_score *= seasonal_multiplier
        
        # Final Santa Ana features
        features['santa_ana_probability'] = np.clip(santa_ana_score, 0, 1)
        features['santa_ana_likely'] = (santa_ana_score > 0.6).astype(int)
        features['santa_ana_possible'] = (santa_ana_score > 0.3).astype(int)
        
        # Santa Ana strength categories
        features['santa_ana_strength'] = pd.cut(santa_ana_score,
                                               bins=[0, 0.3, 0.6, 1.0],
                                               labels=['weak', 'moderate', 'strong'],
                                               include_lowest=True)
        
        # Encode strength as numeric for ML models
        strength_map = {'weak': 0, 'moderate': 1, 'strong': 2}
        features['santa_ana_strength_numeric'] = features['santa_ana_strength'].map(strength_map).fillna(0)
        
        # Santa Ana fire danger indicator (combines wind speed, low humidity, dry conditions)
        if 'wind_speed' in data.columns and 'humidity' in data.columns:
            wind_speed = data['wind_speed'].fillna(5)
            humidity = data['humidity'].fillna(50)
            
            # Fire danger increases with wind speed and decreases with humidity
            fire_danger = (wind_speed / 50) * (1 - humidity / 100)  # Normalized 0-1
            features['santa_ana_fire_danger'] = np.clip(fire_danger, 0, 1)
            features['santa_ana_high_fire_risk'] = (fire_danger > 0.4).astype(int)
        
        logger.debug(f"Created {len(features.columns)-1} Santa Ana wind features")
        return features
    
    def detect_heat_island_effects(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect urban heat island effects and seasonal temperature adjustments.
        
        Heat island characteristics in LA:
        - Temperature differences between urban core and surrounding areas
        - More pronounced during clear, calm nights
        - Seasonal variations in intensity
        - Correlation with development density and vegetation
        
        Args:
            data: DataFrame with weather data including temperature, wind_speed, cloud_cover
            
        Returns:
            DataFrame with heat island effect features
        """
        logger.debug("Detecting heat island effects")
        
        if data.empty:
            return pd.DataFrame()
        
        features = data[['date']].copy()
        
        # Heat island intensity based on conditions that enhance the effect
        heat_island_score = pd.Series(0.0, index=data.index)
        
        # Clear sky enhancement (heat island stronger with clear skies)
        if 'cloud_cover' in data.columns:
            cloud_cover = data['cloud_cover'].fillna(50)
            # Heat island effect strongest with <30% cloud cover
            clear_enhancement = np.where(cloud_cover < 30,
                                       (30 - cloud_cover) / 30,  # Scale 0-30% to 1-0
                                       0.3)  # Minimum effect even with clouds
            heat_island_score += clear_enhancement * 0.3
        
        # Calm wind enhancement (heat island stronger with low wind speeds)
        if 'wind_speed' in data.columns:
            wind_speed = data['wind_speed'].fillna(10)
            # Heat island effect strongest with wind speeds <5 mph
            calm_enhancement = np.where(wind_speed < 5,
                                      (5 - wind_speed) / 5,  # Scale 0-5 mph to 1-0
                                      0.2)  # Reduced effect with higher winds
            heat_island_score += calm_enhancement * 0.25
        
        # Seasonal intensity (heat island stronger in summer)
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
            months = dates.dt.month
            # Heat island intensity by month (summer peak)
            seasonal_intensity = np.where(months.isin([6, 7, 8]), 1.0,  # Summer peak
                               np.where(months.isin([5, 9]), 0.8,        # Shoulder months
                               np.where(months.isin([4, 10]), 0.6,       # Spring/fall
                               np.where(months.isin([3, 11]), 0.4, 0.3)))) # Winter minimum
            heat_island_score += seasonal_intensity * 0.25
        
        # Temperature-based enhancement (heat island more pronounced at higher temps)
        if 'predicted_high' in data.columns:
            temp_high = data['predicted_high'].fillna(75)
            # Heat island effect increases with base temperature
            temp_enhancement = np.where(temp_high > 80,
                                      np.clip((temp_high - 80) / 20, 0, 1),  # Scale 80-100°F to 0-1
                                      0.2)  # Minimum effect at lower temps
            heat_island_score += temp_enhancement * 0.2
        
        # Final heat island features
        features['heat_island_intensity'] = np.clip(heat_island_score, 0, 1)
        
        # Heat island temperature adjustment (estimated additional warming)
        # Typical LA heat island effect: 2-8°F additional warming
        features['heat_island_temp_adjustment'] = heat_island_score * 6  # 0-6°F adjustment
        
        # Heat island categories
        features['heat_island_effect'] = pd.cut(heat_island_score,
                                              bins=[0, 0.3, 0.6, 1.0],
                                              labels=['low', 'moderate', 'high'],
                                              include_lowest=True)
        
        # Encode effect as numeric for ML models
        effect_map = {'low': 0, 'moderate': 1, 'high': 2}
        features['heat_island_effect_numeric'] = features['heat_island_effect'].map(effect_map).fillna(0)
        
        # Nighttime heat island indicator (stronger at night)
        # This is a proxy since we don't have time-of-day data
        features['nighttime_heat_island_risk'] = (heat_island_score > 0.5).astype(int)
        
        # Extreme heat amplification (heat island effect during heat waves)
        if 'predicted_high' in data.columns:
            temp_high = data['predicted_high'].fillna(75)
            extreme_heat = (temp_high > 95).astype(int)
            features['heat_island_extreme_amplification'] = extreme_heat * heat_island_score
        
        logger.debug(f"Created {len(features.columns)-1} heat island features")
        return features
    
    def create_seasonal_adjustments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create LA-specific seasonal adjustment features.
        
        Args:
            data: DataFrame with weather data including date and temperature
            
        Returns:
            DataFrame with seasonal adjustment features
        """
        logger.debug("Creating LA-specific seasonal adjustments")
        
        if data.empty:
            return pd.DataFrame()
        
        features = data[['date']].copy()
        
        if 'date' not in data.columns:
            logger.warning("No date column found for seasonal adjustments")
            return features
        
        dates = pd.to_datetime(data['date'])
        months = dates.dt.month
        day_of_year = dates.dt.dayofyear
        
        # LA-specific seasonal temperature patterns
        # Based on historical LA temperature climatology
        
        # Monthly temperature normals for LA (approximate)
        monthly_normals = {
            1: 68,   # January
            2: 70,   # February  
            3: 72,   # March
            4: 75,   # April
            5: 78,   # May
            6: 82,   # June
            7: 85,   # July
            8: 86,   # August
            9: 84,   # September
            10: 79,  # October
            11: 73,  # November
            12: 68   # December
        }
        
        features['la_monthly_normal'] = months.map(monthly_normals)
        
        # Seasonal temperature deviation (how much current temp deviates from normal)
        if 'predicted_high' in data.columns:
            predicted_temp = data['predicted_high'].fillna(features['la_monthly_normal'])
            features['temp_deviation_from_normal'] = predicted_temp - features['la_monthly_normal']
            
            # Categorize deviations
            features['temp_much_above_normal'] = (features['temp_deviation_from_normal'] > 10).astype(int)
            features['temp_above_normal'] = (features['temp_deviation_from_normal'] > 5).astype(int)
            features['temp_below_normal'] = (features['temp_deviation_from_normal'] < -5).astype(int)
            features['temp_much_below_normal'] = (features['temp_deviation_from_normal'] < -10).astype(int)
        
        # Seasonal weather pattern indicators
        features['is_peak_summer'] = months.isin([7, 8]).astype(int)  # Hottest months
        features['is_shoulder_season'] = months.isin([5, 6, 9, 10]).astype(int)  # Transition months
        features['is_cool_season'] = months.isin([11, 12, 1, 2]).astype(int)  # Coolest months
        
        # Fire season refinement (more specific than basic features)
        # Peak fire danger: October-December, especially with Santa Ana conditions
        features['is_peak_fire_season'] = months.isin([10, 11, 12]).astype(int)
        features['is_extended_fire_season'] = months.isin([9, 10, 11, 12, 1]).astype(int)
        
        # Marine layer season refinement
        # Peak marine layer: June-August, especially June-July
        features['is_peak_marine_layer_season'] = months.isin([6, 7]).astype(int)
        features['is_extended_marine_layer_season'] = months.isin([5, 6, 7, 8, 9]).astype(int)
        
        # Monsoon influence (July-September, brings humidity and thunderstorms)
        features['is_monsoon_season'] = months.isin([7, 8, 9]).astype(int)
        
        # El Niño/La Niña seasonal adjustments (simplified)
        # This would ideally use actual ENSO index data
        # For now, create a proxy based on seasonal patterns
        features['winter_wet_season'] = months.isin([12, 1, 2, 3]).astype(int)
        features['spring_dry_season'] = months.isin([4, 5, 6]).astype(int)
        
        # Day length effects (longer days = more heating potential)
        # Approximate day length for LA latitude (34°N)
        day_length_hours = 12 + 2.5 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        features['day_length_hours'] = day_length_hours
        features['is_long_day'] = (day_length_hours > 13).astype(int)  # Summer long days
        features['is_short_day'] = (day_length_hours < 11).astype(int)  # Winter short days
        
        # Solar angle effects (higher sun angle = more intense heating)
        # Approximate solar declination
        solar_declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        features['solar_declination'] = solar_declination
        features['high_sun_angle'] = (solar_declination > 15).astype(int)  # Summer high sun
        features['low_sun_angle'] = (solar_declination < -15).astype(int)  # Winter low sun
        
        logger.debug(f"Created {len(features.columns)-1} seasonal adjustment features")
        return features
    
    def create_all_la_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all LA-specific weather pattern features.
        
        Args:
            data: DataFrame with weather data
            
        Returns:
            DataFrame with all LA-specific pattern features
        """
        logger.info("Creating all LA-specific weather pattern features")
        
        if data.empty:
            logger.error("No data available for LA pattern feature creation")
            return pd.DataFrame()
        
        # Create different types of LA-specific features
        marine_layer_features = self.detect_marine_layer_conditions(data)
        santa_ana_features = self.detect_santa_ana_conditions(data)
        heat_island_features = self.detect_heat_island_effects(data)
        seasonal_features = self.create_seasonal_adjustments(data)
        
        # Merge all features
        all_features = data[['date']].copy()
        
        for features_df in [marine_layer_features, santa_ana_features, heat_island_features, seasonal_features]:
            if not features_df.empty:
                all_features = all_features.merge(features_df, on='date', how='left')
        
        if not all_features.empty:
            logger.info(f"Created {len(all_features.columns)-1} total LA-specific pattern features for {len(all_features)} dates")
        else:
            logger.warning("No LA-specific pattern features created")
        
        return all_features
    
    def get_pattern_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Get summary of LA-specific pattern features.
        
        Args:
            features_df: DataFrame with LA pattern features
            
        Returns:
            Dictionary with feature summary
        """
        if features_df.empty:
            return {'error': 'No features to summarize'}
        
        # Categorize features
        feature_categories = {
            'marine_layer': len([col for col in features_df.columns if 'marine_layer' in col]),
            'santa_ana': len([col for col in features_df.columns if 'santa_ana' in col]),
            'heat_island': len([col for col in features_df.columns if 'heat_island' in col]),
            'seasonal': len([col for col in features_df.columns if any(x in col for x in ['season', 'normal', 'deviation', 'solar', 'day_length'])])
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
        
        # Pattern detection statistics
        if 'marine_layer_likely' in features_df.columns:
            summary['marine_layer_days'] = features_df['marine_layer_likely'].sum()
        
        if 'santa_ana_likely' in features_df.columns:
            summary['santa_ana_days'] = features_df['santa_ana_likely'].sum()
        
        if 'heat_island_effect_numeric' in features_df.columns:
            heat_island_counts = features_df['heat_island_effect_numeric'].value_counts().to_dict()
            summary['heat_island_distribution'] = heat_island_counts
        
        return summary
    
    def validate_pattern_features(self, features_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate LA-specific pattern features for quality and reasonableness.
        
        Args:
            features_df: DataFrame with LA pattern features
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if features_df.empty:
            issues.append("Features DataFrame is empty")
            return False, issues
        
        # Check for required date column
        if 'date' not in features_df.columns:
            issues.append("Missing required 'date' column")
        
        # Validate probability features are in [0, 1] range
        probability_features = [col for col in features_df.columns if 'probability' in col]
        for col in probability_features:
            if col in features_df.columns:
                min_val = features_df[col].min()
                max_val = features_df[col].max()
                if min_val < 0 or max_val > 1:
                    issues.append(f"Probability feature {col} out of range [0,1]: {min_val:.3f} to {max_val:.3f}")
        
        # Validate binary features are 0 or 1
        binary_features = [col for col in features_df.columns if col.endswith('_likely') or col.endswith('_possible') or col.startswith('is_')]
        for col in binary_features:
            if col in features_df.columns:
                unique_vals = features_df[col].dropna().unique()
                if not all(val in [0, 1] for val in unique_vals):
                    issues.append(f"Binary feature {col} contains non-binary values: {unique_vals}")
        
        # Validate temperature adjustments are reasonable
        if 'heat_island_temp_adjustment' in features_df.columns:
            temp_adj = features_df['heat_island_temp_adjustment']
            if temp_adj.max() > 15 or temp_adj.min() < 0:
                issues.append(f"Heat island temperature adjustment out of reasonable range: {temp_adj.min():.1f} to {temp_adj.max():.1f}")
        
        # Check for seasonal consistency
        if 'date' in features_df.columns and 'is_peak_summer' in features_df.columns:
            dates = pd.to_datetime(features_df['date'])
            summer_months = dates.dt.month.isin([7, 8])
            peak_summer_flag = features_df['is_peak_summer'] == 1
            
            # Should have high correlation between summer months and peak summer flag
            if len(summer_months) > 0:
                consistency = (summer_months == peak_summer_flag).mean()
                if consistency < 0.95:
                    issues.append(f"Seasonal consistency issue: peak summer flag only {consistency:.1%} consistent with actual months")
        
        # Validate feature completeness
        expected_categories = ['marine_layer', 'santa_ana', 'heat_island']
        for category in expected_categories:
            category_features = [col for col in features_df.columns if category in col]
            if not category_features:
                issues.append(f"Missing {category} features")
        
        # Check for seasonal features (more flexible matching)
        seasonal_indicators = ['season', 'normal', 'solar', 'day_length', 'fire_season', 'marine_layer_season']
        seasonal_features = [col for col in features_df.columns if any(indicator in col for indicator in seasonal_indicators)]
        if not seasonal_features:
            issues.append("Missing seasonal features")
        
        is_valid = len(issues) == 0
        return is_valid, issues


def main():
    """Demonstrate LA-specific weather pattern feature extraction."""
    print("=== LA Weather Pattern Features Demo ===\n")
    
    # Create sample data for demonstration
    from datetime import timedelta
    
    dates = [date.today() - timedelta(days=i) for i in range(30, 0, -1)]
    
    # Create sample weather data with various conditions
    sample_data = []
    for i, d in enumerate(dates):
        # Simulate different weather patterns throughout the year
        month = d.month
        
        # Base conditions
        base_temp = 75 + 10 * np.sin(2 * np.pi * (d.timetuple().tm_yday - 81) / 365)  # Seasonal variation
        
        # Add some variation
        if month in [6, 7]:  # Summer - potential marine layer
            humidity = 80 + np.random.normal(0, 10)
            cloud_cover = 70 + np.random.normal(0, 20)
            wind_speed = 5 + np.random.normal(0, 3)
            wind_direction = 270 + np.random.normal(0, 30)  # Westerly (marine)
        elif month in [12, 1]:  # Winter - potential Santa Ana
            humidity = 15 + np.random.normal(0, 10)
            cloud_cover = 10 + np.random.normal(0, 15)
            wind_speed = 30 + np.random.normal(0, 10)
            wind_direction = 90 + np.random.normal(0, 30)  # Easterly (Santa Ana)
        else:  # Normal conditions
            humidity = 50 + np.random.normal(0, 15)
            cloud_cover = 40 + np.random.normal(0, 25)
            wind_speed = 10 + np.random.normal(0, 5)
            wind_direction = 180 + np.random.normal(0, 60)
        
        # Ensure reasonable ranges
        humidity = np.clip(humidity, 5, 95)
        cloud_cover = np.clip(cloud_cover, 0, 100)
        wind_speed = np.clip(wind_speed, 0, 50)
        wind_direction = wind_direction % 360
        
        sample_data.append({
            'date': d,
            'predicted_high': base_temp + np.random.normal(0, 3),
            'predicted_low': base_temp - 15 + np.random.normal(0, 2),
            'humidity': humidity,
            'cloud_cover': cloud_cover,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'pressure': 1013 + np.random.normal(0, 10)
        })
    
    sample_df = pd.DataFrame(sample_data)
    
    print("1. Sample Data Created:")
    print(f"   Records: {len(sample_df)}")
    print(f"   Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
    print(f"   Columns: {list(sample_df.columns)}")
    print()
    
    # Initialize extractor
    extractor = LAWeatherPatternExtractor()
    
    # Extract all LA-specific features
    print("2. Extracting LA-Specific Weather Pattern Features:")
    la_features = extractor.create_all_la_pattern_features(sample_df)
    
    if not la_features.empty:
        print(f"   Created features: {la_features.shape}")
        print(f"   Feature categories:")
        
        # Show feature categories
        categories = {
            'Marine Layer': [col for col in la_features.columns if 'marine_layer' in col],
            'Santa Ana': [col for col in la_features.columns if 'santa_ana' in col],
            'Heat Island': [col for col in la_features.columns if 'heat_island' in col],
            'Seasonal': [col for col in la_features.columns if any(x in col for x in ['season', 'normal', 'solar', 'day_length'])]
        }
        
        for category, features in categories.items():
            print(f"     {category}: {len(features)} features")
        print()
        
        # Validate features
        print("3. Feature Validation:")
        is_valid, issues = extractor.validate_pattern_features(la_features)
        print(f"   Valid: {is_valid}")
        if issues:
            print(f"   Issues: {issues}")
        print()
        
        # Feature summary
        print("4. Feature Summary:")
        summary = extractor.get_pattern_feature_summary(la_features)
        for key, value in summary.items():
            if key != 'feature_categories':
                print(f"   {key}: {value}")
        print()
        
        # Show some example feature values
        print("5. Example Feature Values (first 3 records):")
        example_features = [
            'marine_layer_probability', 'santa_ana_probability', 'heat_island_intensity',
            'temp_deviation_from_normal', 'is_peak_summer', 'is_peak_fire_season'
        ]
        
        for feature in example_features:
            if feature in la_features.columns:
                values = la_features[feature].head(3).tolist()
                print(f"   {feature}: {values}")
        print()
        
        # Pattern detection results
        print("6. Pattern Detection Results:")
        if 'marine_layer_likely' in la_features.columns:
            marine_days = la_features['marine_layer_likely'].sum()
            print(f"   Marine layer likely days: {marine_days}/{len(la_features)}")
        
        if 'santa_ana_likely' in la_features.columns:
            santa_ana_days = la_features['santa_ana_likely'].sum()
            print(f"   Santa Ana likely days: {santa_ana_days}/{len(la_features)}")
        
        if 'heat_island_effect_numeric' in la_features.columns:
            heat_island_dist = la_features['heat_island_effect_numeric'].value_counts().sort_index()
            print(f"   Heat island effect distribution: {dict(heat_island_dist)}")
        
    else:
        print("   No features created")
    
    print("\n=== LA Weather Pattern Features Demo Complete ===")


if __name__ == '__main__':
    main()