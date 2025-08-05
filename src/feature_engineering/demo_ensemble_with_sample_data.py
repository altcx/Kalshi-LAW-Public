"""Demo ensemble features with sample data to show functionality."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, timedelta
import pandas as pd
import numpy as np
from src.feature_engineering.ensemble_features import EnsembleFeatureExtractor


def create_sample_weather_data():
    """Create realistic sample weather data for demonstration."""
    print("Creating sample weather data for demonstration...")
    
    # Create 14 days of sample data
    dates = [date.today() - timedelta(days=i) for i in range(14, 0, -1)]
    
    # Base temperature pattern (seasonal + daily variation)
    base_temps = [72 + 5*np.sin(i/14 * 2*np.pi) + np.random.normal(0, 1) for i in range(14)]
    
    sample_data = {}
    
    # NWS data (most reliable)
    sample_data['nws'] = pd.DataFrame({
        'date': dates,
        'forecast_date': [d - timedelta(days=1) for d in dates],
        'predicted_high': [temp + np.random.normal(0, 0.5) for temp in base_temps],
        'predicted_low': [temp - 15 + np.random.normal(0, 0.5) for temp in base_temps],
        'humidity': [65 + np.random.normal(0, 5) for _ in range(14)],
        'pressure': [1013 + np.random.normal(0, 2) for _ in range(14)],
        'wind_speed': [8 + np.random.normal(0, 2) for _ in range(14)],
        'wind_direction': [225 + np.random.normal(0, 30) for _ in range(14)],
        'cloud_cover': [40 + np.random.normal(0, 15) for _ in range(14)],
        'precipitation_prob': [25 + np.random.normal(0, 10) for _ in range(14)],
        'data_quality_score': [0.95 + np.random.normal(0, 0.02) for _ in range(14)]
    })
    
    # OpenWeatherMap data (slightly different bias)
    sample_data['openweather'] = pd.DataFrame({
        'date': dates,
        'forecast_date': [d - timedelta(days=1) for d in dates],
        'predicted_high': [temp + 1 + np.random.normal(0, 0.8) for temp in base_temps],  # Slightly warmer bias
        'predicted_low': [temp - 14 + np.random.normal(0, 0.8) for temp in base_temps],
        'humidity': [62 + np.random.normal(0, 6) for _ in range(14)],
        'pressure': [1014 + np.random.normal(0, 2.5) for _ in range(14)],
        'wind_speed': [9 + np.random.normal(0, 2.5) for _ in range(14)],
        'wind_direction': [220 + np.random.normal(0, 35) for _ in range(14)],
        'cloud_cover': [38 + np.random.normal(0, 18) for _ in range(14)],
        'precipitation_prob': [22 + np.random.normal(0, 12) for _ in range(14)],
        'data_quality_score': [0.88 + np.random.normal(0, 0.03) for _ in range(14)]
    })
    
    # Tomorrow.io data (different characteristics)
    sample_data['tomorrow'] = pd.DataFrame({
        'date': dates,
        'forecast_date': [d - timedelta(days=1) for d in dates],
        'predicted_high': [temp - 0.5 + np.random.normal(0, 1.2) for temp in base_temps],  # Slightly cooler bias
        'predicted_low': [temp - 16 + np.random.normal(0, 1.2) for temp in base_temps],
        'humidity': [68 + np.random.normal(0, 7) for _ in range(14)],
        'pressure': [1012 + np.random.normal(0, 3) for _ in range(14)],
        'wind_speed': [7.5 + np.random.normal(0, 3) for _ in range(14)],
        'wind_direction': [230 + np.random.normal(0, 40) for _ in range(14)],
        'cloud_cover': [42 + np.random.normal(0, 20) for _ in range(14)],
        'precipitation_prob': [28 + np.random.normal(0, 15) for _ in range(14)],
        'data_quality_score': [0.85 + np.random.normal(0, 0.04) for _ in range(14)]
    })
    
    # Weatherbit data
    sample_data['weatherbit'] = pd.DataFrame({
        'date': dates,
        'forecast_date': [d - timedelta(days=1) for d in dates],
        'predicted_high': [temp + 0.3 + np.random.normal(0, 0.9) for temp in base_temps],
        'predicted_low': [temp - 15.5 + np.random.normal(0, 0.9) for temp in base_temps],
        'humidity': [64 + np.random.normal(0, 5.5) for _ in range(14)],
        'pressure': [1013.5 + np.random.normal(0, 2.2) for _ in range(14)],
        'wind_speed': [8.2 + np.random.normal(0, 2.2) for _ in range(14)],
        'wind_direction': [227 + np.random.normal(0, 32) for _ in range(14)],
        'cloud_cover': [39 + np.random.normal(0, 16) for _ in range(14)],
        'precipitation_prob': [24 + np.random.normal(0, 11) for _ in range(14)],
        'data_quality_score': [0.90 + np.random.normal(0, 0.025) for _ in range(14)]
    })
    
    # Add some missing data to make it realistic
    sample_data['tomorrow'].loc[2, 'humidity'] = np.nan
    sample_data['weatherbit'].loc[5, 'wind_speed'] = np.nan
    sample_data['openweather'].loc[8, 'pressure'] = np.nan
    
    # Add one day with high disagreement (weather event)
    event_day = 7
    sample_data['nws'].loc[event_day, 'predicted_high'] += 8  # Much warmer prediction
    sample_data['tomorrow'].loc[event_day, 'predicted_high'] -= 3  # Much cooler prediction
    
    return sample_data


def main():
    """Demonstrate ensemble feature extraction."""
    print("=== Ensemble Feature Extraction Demo ===\n")
    
    # Create sample data
    sample_data = create_sample_weather_data()
    
    print(f"Created sample data from {len(sample_data)} weather sources:")
    for source, data in sample_data.items():
        date_range = f"{data['date'].min()} to {data['date'].max()}"
        print(f"   {source}: {len(data)} records ({date_range})")
    
    # Initialize ensemble extractor
    ensemble_extractor = EnsembleFeatureExtractor()
    
    print("\n" + "="*60)
    print("1. CONSENSUS FEATURES")
    print("="*60)
    
    consensus_features = ensemble_extractor.create_consensus_features(sample_data)
    print(f"Created {len(consensus_features.columns)-1} consensus features")
    
    # Show temperature consensus example
    print("\nTemperature Consensus Example (last 5 days):")
    temp_cols = ['date', 'predicted_high_consensus_mean', 'predicted_high_consensus_median', 
                 'predicted_high_consensus_std', 'predicted_high_source_count']
    if all(col in consensus_features.columns for col in temp_cols):
        sample_consensus = consensus_features[temp_cols].tail(5)
        print(sample_consensus.to_string(index=False, float_format='%.2f'))
    
    print("\n" + "="*60)
    print("2. AGREEMENT METRICS")
    print("="*60)
    
    agreement_features = ensemble_extractor.create_agreement_metrics(sample_data)
    print(f"Created {len(agreement_features.columns)-1} agreement metrics")
    
    # Show agreement analysis
    if 'overall_agreement_score' in agreement_features.columns:
        agreement_scores = agreement_features['overall_agreement_score']
        print(f"\nAPI Agreement Analysis:")
        print(f"   Average agreement score: {agreement_scores.mean():.3f}")
        print(f"   Agreement range: {agreement_scores.min():.3f} to {agreement_scores.max():.3f}")
        
        # Find days with high/low agreement
        high_agreement_day = agreement_features.loc[agreement_scores.idxmax()]
        low_agreement_day = agreement_features.loc[agreement_scores.idxmin()]
        
        print(f"\n   Highest agreement: {high_agreement_day['date'].strftime('%Y-%m-%d')} (score: {agreement_scores.max():.3f})")
        print(f"   Lowest agreement: {low_agreement_day['date'].strftime('%Y-%m-%d')} (score: {agreement_scores.min():.3f})")
        print("   (Lower scores indicate weather sources disagree more - often during weather events)")
    
    print("\n" + "="*60)
    print("3. ROLLING FEATURES")
    print("="*60)
    
    rolling_features = ensemble_extractor.create_rolling_features(sample_data)
    print(f"Created {len(rolling_features.columns)-1} rolling features")
    
    # Show rolling temperature trends
    rolling_temp_cols = ['date', 'predicted_high_rolling_3d_mean', 'predicted_high_rolling_3d_trend', 
                        'predicted_high_rolling_7d_mean', 'predicted_high_rolling_7d_trend']
    available_rolling_cols = ['date'] + [col for col in rolling_temp_cols[1:] if col in rolling_features.columns]
    
    if len(available_rolling_cols) > 1:
        print("\nTemperature Rolling Trends (last 5 days):")
        sample_rolling = rolling_features[available_rolling_cols].tail(5)
        print(sample_rolling.to_string(index=False, float_format='%.3f'))
        
        # Interpret trends
        if 'predicted_high_rolling_3d_trend' in rolling_features.columns:
            latest_trend = rolling_features['predicted_high_rolling_3d_trend'].iloc[-1]
            if latest_trend > 0.5:
                trend_desc = "warming trend"
            elif latest_trend < -0.5:
                trend_desc = "cooling trend"
            else:
                trend_desc = "stable temperatures"
            print(f"\n   Latest 3-day trend: {latest_trend:.3f}°F/day ({trend_desc})")
    
    print("\n" + "="*60)
    print("4. SOURCE RELIABILITY")
    print("="*60)
    
    reliability_features = ensemble_extractor.create_source_reliability_features(sample_data)
    print(f"Created {len(reliability_features.columns)-1} source reliability features")
    
    # Show source reliability comparison
    reliability_cols = [col for col in reliability_features.columns if '_quality_score' in col]
    if reliability_cols:
        print("\nSource Quality Comparison (average scores):")
        for col in reliability_cols:
            source = col.replace('_quality_score', '')
            avg_quality = reliability_features[col].mean()
            print(f"   {source}: {avg_quality:.3f}")
    
    # Show data availability
    availability_cols = [col for col in reliability_features.columns if '_available' in col]
    if availability_cols:
        print("\nData Availability (% of days with data):")
        for col in availability_cols:
            source = col.replace('_available', '')
            availability = reliability_features[col].mean() * 100
            print(f"   {source}: {availability:.1f}%")
    
    print("\n" + "="*60)
    print("5. COMPLETE ENSEMBLE FEATURE SET")
    print("="*60)
    
    all_ensemble_features = ensemble_extractor.create_all_ensemble_features(sample_data)
    summary = ensemble_extractor.get_ensemble_feature_summary(all_ensemble_features)
    
    print(f"Total ensemble features: {summary['total_features']}")
    print(f"Records: {summary['total_records']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Missing data: {summary['missing_data_percentage']:.1f}%")
    
    print("\nFeature breakdown by category:")
    for category, count in summary['feature_categories'].items():
        print(f"   {category}: {count} features")
    
    print("\n" + "="*60)
    print("6. FEATURE INSIGHTS")
    print("="*60)
    
    # Show most variable features (potentially most informative)
    numeric_features = all_ensemble_features.select_dtypes(include=['float64', 'int64'])
    if len(numeric_features.columns) > 0:
        feature_variance = numeric_features.var().sort_values(ascending=False)
        
        print("Top 10 highest variance features (potentially most informative):")
        for i, (feature, variance) in enumerate(feature_variance.head(10).items()):
            print(f"   {i+1:2d}. {feature}: {variance:.3f}")
    
    # Show correlation insights
    if len(numeric_features.columns) > 1:
        corr_matrix = numeric_features.corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:  # Very high correlation
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"\nHighly correlated feature pairs (|r| > 0.9):")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # Show top 5
                print(f"   {feat1} <-> {feat2}: {corr:.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("Ensemble features provide valuable meta-information:")
    print("✓ Consensus: Combines predictions from multiple sources")
    print("✓ Agreement: Measures how much sources agree (low = weather event)")
    print("✓ Rolling: Captures trends and momentum over time")
    print("✓ Reliability: Tracks data quality and source performance")
    print("\nThese features can significantly improve ML model performance by:")
    print("- Reducing noise through consensus")
    print("- Detecting uncertain weather conditions")
    print("- Capturing temporal patterns")
    print("- Weighting sources by reliability")
    
    print(f"\n=== Demo Complete: {summary['total_features']} ensemble features created! ===")


if __name__ == '__main__':
    main()