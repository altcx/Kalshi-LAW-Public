"""Demo script for LA-specific weather pattern features."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.feature_engineering.la_weather_patterns import LAWeatherPatternExtractor


def create_realistic_sample_data():
    """Create realistic sample weather data for demonstration."""
    print("Creating realistic sample weather data...")
    
    # Create a year's worth of data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    # Simulate seasonal patterns
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Base temperature with seasonal variation (LA climate)
    base_temp = 75 + 10 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    
    # Humidity patterns (higher in summer due to marine layer potential)
    base_humidity = 50 + 20 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
    
    # Wind patterns (higher in winter for Santa Ana potential)
    base_wind = 8 + 5 * np.sin(2 * np.pi * (day_of_year - 350) / 365)
    
    # Cloud cover (varies seasonally)
    base_clouds = 40 + 15 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
    
    # Create sample data with realistic variations
    sample_data = pd.DataFrame({
        'date': dates,
        'predicted_high': base_temp + np.random.normal(0, 3, n_days),
        'predicted_low': base_temp - 15 + np.random.normal(0, 2, n_days),
        'humidity': np.clip(base_humidity + np.random.normal(0, 10, n_days), 10, 95),
        'cloud_cover': np.clip(base_clouds + np.random.normal(0, 20, n_days), 0, 100),
        'wind_speed': np.clip(base_wind + np.random.normal(0, 5, n_days), 0, 50),
        'wind_direction': np.random.uniform(0, 360, n_days),
        'pressure': 1013 + np.random.normal(0, 8, n_days)
    })
    
    # Add some extreme weather events
    # Summer marine layer events (high humidity, low wind, overcast)
    marine_layer_days = np.random.choice(
        np.where((day_of_year >= 150) & (day_of_year <= 240))[0], 
        size=20, replace=False
    )
    for day_idx in marine_layer_days:
        sample_data.loc[day_idx, 'humidity'] = np.random.uniform(80, 95)
        sample_data.loc[day_idx, 'cloud_cover'] = np.random.uniform(80, 100)
        sample_data.loc[day_idx, 'wind_speed'] = np.random.uniform(2, 6)
        sample_data.loc[day_idx, 'wind_direction'] = np.random.uniform(240, 300)  # Westerly
    
    # Winter Santa Ana events (low humidity, high wind from east, clear skies)
    santa_ana_days = np.random.choice(
        np.where((day_of_year <= 90) | (day_of_year >= 300))[0], 
        size=15, replace=False
    )
    for day_idx in santa_ana_days:
        sample_data.loc[day_idx, 'humidity'] = np.random.uniform(5, 25)
        sample_data.loc[day_idx, 'cloud_cover'] = np.random.uniform(0, 20)
        sample_data.loc[day_idx, 'wind_speed'] = np.random.uniform(25, 45)
        sample_data.loc[day_idx, 'wind_direction'] = np.random.uniform(60, 120)  # Easterly
    
    print(f"Created {len(sample_data)} days of sample weather data")
    return sample_data


def demonstrate_marine_layer_detection(extractor, data):
    """Demonstrate marine layer detection capabilities."""
    print("\n" + "="*60)
    print("MARINE LAYER DETECTION DEMONSTRATION")
    print("="*60)
    
    marine_features = extractor.detect_marine_layer_conditions(data)
    
    # Find days with high marine layer probability
    high_marine_days = marine_features[marine_features['marine_layer_probability'] > 0.6]
    
    print(f"\nFound {len(high_marine_days)} days with high marine layer probability (>60%)")
    
    if not high_marine_days.empty:
        print("\nTop 5 marine layer days:")
        top_marine = high_marine_days.nlargest(5, 'marine_layer_probability')
        
        for idx, row in top_marine.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            prob = row['marine_layer_probability']
            strength = row['marine_layer_strength']
            print(f"  {date_str}: {prob:.1%} probability, {strength} strength")
        
        # Show seasonal distribution
        high_marine_months = pd.to_datetime(high_marine_days['date']).dt.month
        monthly_counts = high_marine_months.value_counts().sort_index()
        
        print(f"\nMarine layer days by month:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in monthly_counts.items():
            print(f"  {month_names[month-1]}: {count} days")


def demonstrate_santa_ana_detection(extractor, data):
    """Demonstrate Santa Ana wind detection capabilities."""
    print("\n" + "="*60)
    print("SANTA ANA WIND DETECTION DEMONSTRATION")
    print("="*60)
    
    santa_ana_features = extractor.detect_santa_ana_conditions(data)
    
    # Find days with high Santa Ana probability
    high_santa_ana_days = santa_ana_features[santa_ana_features['santa_ana_probability'] > 0.6]
    
    print(f"\nFound {len(high_santa_ana_days)} days with high Santa Ana probability (>60%)")
    
    if not high_santa_ana_days.empty:
        print("\nTop 5 Santa Ana days:")
        top_santa_ana = high_santa_ana_days.nlargest(5, 'santa_ana_probability')
        
        for idx, row in top_santa_ana.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            prob = row['santa_ana_probability']
            strength = row['santa_ana_strength']
            fire_danger = row.get('santa_ana_fire_danger', 0)
            print(f"  {date_str}: {prob:.1%} probability, {strength} strength, {fire_danger:.1%} fire danger")
        
        # Show seasonal distribution
        high_santa_ana_months = pd.to_datetime(high_santa_ana_days['date']).dt.month
        monthly_counts = high_santa_ana_months.value_counts().sort_index()
        
        print(f"\nSanta Ana days by month:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in monthly_counts.items():
            print(f"  {month_names[month-1]}: {count} days")


def demonstrate_heat_island_effects(extractor, data):
    """Demonstrate heat island effect detection."""
    print("\n" + "="*60)
    print("HEAT ISLAND EFFECT DEMONSTRATION")
    print("="*60)
    
    heat_island_features = extractor.detect_heat_island_effects(data)
    
    # Analyze heat island intensity by season
    dates = pd.to_datetime(heat_island_features['date'])
    seasons = dates.dt.month.map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                 3: 'Spring', 4: 'Spring', 5: 'Spring',
                                 6: 'Summer', 7: 'Summer', 8: 'Summer',
                                 9: 'Fall', 10: 'Fall', 11: 'Fall'})
    
    heat_island_features['season'] = seasons
    
    print("\nAverage heat island intensity by season:")
    seasonal_intensity = heat_island_features.groupby('season')['heat_island_intensity'].mean()
    for season, intensity in seasonal_intensity.items():
        print(f"  {season}: {intensity:.2f} (0-1 scale)")
    
    print("\nAverage temperature adjustment by season:")
    seasonal_adjustment = heat_island_features.groupby('season')['heat_island_temp_adjustment'].mean()
    for season, adjustment in seasonal_adjustment.items():
        print(f"  {season}: +{adjustment:.1f}°F")
    
    # Find days with highest heat island effect
    high_heat_island = heat_island_features.nlargest(5, 'heat_island_intensity')
    
    print(f"\nTop 5 heat island effect days:")
    for idx, row in high_heat_island.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        intensity = row['heat_island_intensity']
        temp_adj = row['heat_island_temp_adjustment']
        effect = row['heat_island_effect']
        print(f"  {date_str}: {intensity:.2f} intensity, +{temp_adj:.1f}°F adjustment, {effect} effect")


def demonstrate_seasonal_adjustments(extractor, data):
    """Demonstrate seasonal adjustment features."""
    print("\n" + "="*60)
    print("SEASONAL ADJUSTMENT DEMONSTRATION")
    print("="*60)
    
    seasonal_features = extractor.create_seasonal_adjustments(data)
    
    # Show monthly temperature normals
    print("\nLA Monthly Temperature Normals:")
    monthly_normals = seasonal_features.groupby(pd.to_datetime(seasonal_features['date']).dt.month)['la_monthly_normal'].first()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month, normal in monthly_normals.items():
        print(f"  {month_names[month-1]}: {normal}°F")
    
    # Show seasonal pattern indicators
    print(f"\nSeasonal Pattern Summary:")
    peak_summer_days = seasonal_features['is_peak_summer'].sum()
    peak_fire_days = seasonal_features['is_peak_fire_season'].sum()
    peak_marine_days = seasonal_features['is_peak_marine_layer_season'].sum()
    monsoon_days = seasonal_features['is_monsoon_season'].sum()
    
    print(f"  Peak summer days (Jul-Aug): {peak_summer_days}")
    print(f"  Peak fire season days (Oct-Dec): {peak_fire_days}")
    print(f"  Peak marine layer season days (Jun-Jul): {peak_marine_days}")
    print(f"  Monsoon season days (Jul-Sep): {monsoon_days}")
    
    # Show day length variation
    min_day_length = seasonal_features['day_length_hours'].min()
    max_day_length = seasonal_features['day_length_hours'].max()
    print(f"\nDay length variation:")
    print(f"  Shortest day: {min_day_length:.1f} hours")
    print(f"  Longest day: {max_day_length:.1f} hours")
    print(f"  Variation: {max_day_length - min_day_length:.1f} hours")


def demonstrate_pattern_interactions(extractor, data):
    """Demonstrate interactions between different weather patterns."""
    print("\n" + "="*60)
    print("WEATHER PATTERN INTERACTIONS")
    print("="*60)
    
    all_features = extractor.create_all_la_pattern_features(data)
    
    # Analyze pattern co-occurrence
    marine_layer_days = all_features['marine_layer_likely'] == 1
    santa_ana_days = all_features['santa_ana_likely'] == 1
    high_heat_island_days = all_features['heat_island_effect_numeric'] >= 2
    
    print(f"\nPattern occurrence summary:")
    print(f"  Marine layer likely days: {marine_layer_days.sum()}")
    print(f"  Santa Ana likely days: {santa_ana_days.sum()}")
    print(f"  High heat island effect days: {high_heat_island_days.sum()}")
    
    # Check for mutual exclusivity (marine layer and Santa Ana shouldn't co-occur)
    conflicting_days = marine_layer_days & santa_ana_days
    print(f"  Days with both marine layer and Santa Ana (should be rare): {conflicting_days.sum()}")
    
    # Analyze seasonal timing
    dates = pd.to_datetime(all_features['date'])
    months = dates.dt.month
    
    print(f"\nSeasonal timing analysis:")
    
    # Marine layer timing
    if marine_layer_days.any():
        marine_months = months[marine_layer_days].value_counts().sort_index()
        peak_marine_month = marine_months.idxmax()
        print(f"  Marine layer peak month: {peak_marine_month} ({marine_months[peak_marine_month]} days)")
    
    # Santa Ana timing
    if santa_ana_days.any():
        santa_ana_months = months[santa_ana_days].value_counts().sort_index()
        peak_santa_ana_month = santa_ana_months.idxmax()
        print(f"  Santa Ana peak month: {peak_santa_ana_month} ({santa_ana_months[peak_santa_ana_month]} days)")
    
    # Temperature impact analysis
    if 'temp_deviation_from_normal' in all_features.columns:
        marine_temp_impact = all_features.loc[marine_layer_days, 'temp_deviation_from_normal'].mean()
        santa_ana_temp_impact = all_features.loc[santa_ana_days, 'temp_deviation_from_normal'].mean()
        
        print(f"\nTemperature impact analysis:")
        print(f"  Marine layer average temp deviation: {marine_temp_impact:.1f}°F from normal")
        print(f"  Santa Ana average temp deviation: {santa_ana_temp_impact:.1f}°F from normal")


def main():
    """Main demonstration function."""
    print("="*80)
    print("LA-SPECIFIC WEATHER PATTERN FEATURES DEMONSTRATION")
    print("="*80)
    
    # Create sample data
    sample_data = create_realistic_sample_data()
    
    # Initialize extractor
    extractor = LAWeatherPatternExtractor()
    
    # Run demonstrations
    demonstrate_marine_layer_detection(extractor, sample_data)
    demonstrate_santa_ana_detection(extractor, sample_data)
    demonstrate_heat_island_effects(extractor, sample_data)
    demonstrate_seasonal_adjustments(extractor, sample_data)
    demonstrate_pattern_interactions(extractor, sample_data)
    
    # Final summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    
    all_features = extractor.create_all_la_pattern_features(sample_data)
    summary = extractor.get_pattern_feature_summary(all_features)
    
    print(f"\nTotal LA-specific features created: {summary['total_features']}")
    print(f"Feature categories:")
    for category, count in summary['feature_categories'].items():
        print(f"  {category.replace('_', ' ').title()}: {count} features")
    
    print(f"\nData quality:")
    print(f"  Records processed: {summary['total_records']}")
    print(f"  Missing data: {summary['missing_data_percentage']:.1f}%")
    print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    
    # Validation
    is_valid, issues = extractor.validate_pattern_features(all_features)
    print(f"\nFeature validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    if issues:
        print(f"Issues: {issues}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe LA-specific weather pattern features are now ready for integration")
    print("with the machine learning pipeline for enhanced temperature prediction!")


if __name__ == '__main__':
    main()