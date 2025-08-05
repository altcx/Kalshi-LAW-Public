"""Integration test for LA weather patterns with the feature pipeline."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.feature_engineering.la_weather_patterns import LAWeatherPatternExtractor
from src.utils.data_manager import DataManager


def create_test_weather_data():
    """Create test weather data for integration testing."""
    print("Creating test weather data for integration testing...")
    
    # Create 30 days of test data
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create realistic weather data
    n_days = len(dates)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Base temperature with seasonal variation
    base_temp = 75 + 10 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    
    # Create sample data for multiple sources
    sources = ['nws', 'openweather', 'tomorrow']
    
    data_manager = DataManager()
    
    for source in sources:
        # Add some variation per source
        source_variation = np.random.normal(0, 1, n_days)
        
        source_data = pd.DataFrame({
            'date': dates,
            'forecast_date': dates - timedelta(days=1),  # Forecast made day before
            'predicted_high': base_temp + source_variation + np.random.normal(0, 2, n_days),
            'predicted_low': base_temp - 15 + source_variation + np.random.normal(0, 1.5, n_days),
            'humidity': np.clip(50 + 20 * np.sin(2 * np.pi * (day_of_year - 150) / 365) + np.random.normal(0, 10, n_days), 10, 95),
            'pressure': 1013 + np.random.normal(0, 8, n_days),
            'wind_speed': np.clip(8 + 5 * np.sin(2 * np.pi * (day_of_year - 350) / 365) + np.random.normal(0, 5, n_days), 0, 50),
            'wind_direction': np.random.uniform(0, 360, n_days),
            'cloud_cover': np.clip(40 + np.random.normal(0, 20, n_days), 0, 100),
            'precipitation_prob': np.clip(20 + np.random.normal(0, 15, n_days), 0, 100),
            'data_quality_score': np.random.uniform(0.8, 1.0, n_days)
        })
        
        # Store the data
        data_manager.save_source_data(source, source_data, append=False)
        print(f"  Created {len(source_data)} records for {source}")
    
    # Create some actual temperature data for validation
    actual_data = pd.DataFrame({
        'date': dates[:-5],  # Don't include most recent 5 days (not yet observed)
        'actual_high': base_temp[:-5] + np.random.normal(0, 1.5, n_days-5),
        'data_quality_score': np.random.uniform(0.9, 1.0, n_days-5)
    })
    
    data_manager.save_source_data('actual_temperatures', actual_data, append=False)
    print(f"  Created {len(actual_data)} actual temperature records")
    
    return start_date, end_date


def test_la_patterns_integration():
    """Test integration of LA weather patterns with the feature pipeline."""
    print("\n" + "="*70)
    print("LA WEATHER PATTERNS INTEGRATION TEST")
    print("="*70)
    
    # Create test data
    start_date, end_date = create_test_weather_data()
    
    # Initialize pipeline
    pipeline = FeaturePipeline()
    
    print(f"\n1. Testing feature pipeline with LA patterns...")
    print(f"   Date range: {start_date} to {end_date}")
    
    # Test complete feature creation with LA patterns
    features = pipeline.create_complete_features(
        start_date, end_date, 
        include_ensemble=True, 
        include_la_patterns=True
    )
    
    if features.empty:
        print("   ❌ No features created")
        return False
    
    print(f"   ✅ Created features: {features.shape}")
    
    # Analyze feature categories
    print(f"\n2. Feature Analysis:")
    analysis = pipeline.get_feature_importance_analysis(features)
    
    print(f"   Total features: {analysis['total_features']}")
    print(f"   Feature categories:")
    for category, count in analysis['feature_categories'].items():
        print(f"     {category.replace('_', ' ').title()}: {count}")
    
    # Validate LA-specific features are present
    print(f"\n3. LA-Specific Feature Validation:")
    
    la_pattern_features = [col for col in features.columns if any(x in col.lower() for x in 
                          ['marine_layer', 'santa_ana', 'heat_island', 'fire_season', 'la_monthly'])]
    
    if la_pattern_features:
        print(f"   ✅ Found {len(la_pattern_features)} LA-specific features")
        
        # Show some example LA features
        example_features = la_pattern_features[:5]
        print(f"   Example LA features: {example_features}")
        
        # Check feature ranges
        print(f"\n4. LA Feature Quality Check:")
        
        # Check probability features
        prob_features = [col for col in la_pattern_features if 'probability' in col]
        for feature in prob_features:
            if feature in features.columns:
                min_val = features[feature].min()
                max_val = features[feature].max()
                print(f"   {feature}: [{min_val:.3f}, {max_val:.3f}] ✅" if 0 <= min_val <= max_val <= 1 else f"   {feature}: [{min_val:.3f}, {max_val:.3f}] ❌")
        
        # Check binary features
        binary_features = [col for col in la_pattern_features if col.endswith('_likely') or col.startswith('is_')]
        valid_binary = 0
        for feature in binary_features[:3]:  # Check first 3
            if feature in features.columns:
                unique_vals = set(features[feature].dropna().unique())
                if unique_vals.issubset({0, 1}):
                    valid_binary += 1
                    print(f"   {feature}: Binary values ✅")
                else:
                    print(f"   {feature}: Invalid values {unique_vals} ❌")
        
        print(f"   Binary feature validation: {valid_binary}/{min(3, len(binary_features))} passed")
        
    else:
        print(f"   ❌ No LA-specific features found")
        return False
    
    # Test feature pipeline validation
    print(f"\n5. Pipeline Validation:")
    validation = pipeline.validate_feature_quality(features)
    
    print(f"   Valid: {'✅' if validation['is_valid'] else '❌'}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings'][:2]}")  # Show first 2 warnings
    
    # Test with training dataset creation
    print(f"\n6. Training Dataset Creation:")
    train_features, train_targets = pipeline.create_training_dataset(start_date, end_date)
    
    if not train_features.empty:
        print(f"   ✅ Training features: {train_features.shape}")
        if train_targets is not None:
            print(f"   ✅ Training targets: {len(train_targets)} values")
            print(f"   Target range: {train_targets.min():.1f}°F to {train_targets.max():.1f}°F")
        else:
            print(f"   ⚠️  No training targets available")
    else:
        print(f"   ❌ No training features created")
    
    # Test standalone LA pattern extractor
    print(f"\n7. Standalone LA Pattern Extractor Test:")
    la_extractor = LAWeatherPatternExtractor()
    
    # Create combined data for LA pattern analysis
    source_data = pipeline.data_manager.load_all_sources(start_date, end_date)
    
    if source_data:
        # Use first available source for testing
        first_source = next(iter(source_data.values()))
        if not first_source.empty:
            la_features = la_extractor.create_all_la_pattern_features(first_source)
            
            if not la_features.empty:
                print(f"   ✅ Standalone LA features: {la_features.shape}")
                
                # Validate standalone features
                is_valid, issues = la_extractor.validate_pattern_features(la_features)
                print(f"   Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
                if issues:
                    print(f"   Issues: {issues[:2]}")  # Show first 2 issues
                
                # Get summary
                summary = la_extractor.get_pattern_feature_summary(la_features)
                print(f"   Feature categories: {summary['feature_categories']}")
                
            else:
                print(f"   ❌ No standalone LA features created")
        else:
            print(f"   ❌ No source data available for LA pattern testing")
    else:
        print(f"   ❌ No source data loaded")
    
    print(f"\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    success_criteria = [
        not features.empty,
        len(la_pattern_features) > 0,
        validation['is_valid'] or len(validation['errors']) == 0,
        not train_features.empty
    ]
    
    passed_criteria = sum(success_criteria)
    total_criteria = len(success_criteria)
    
    print(f"Success criteria: {passed_criteria}/{total_criteria}")
    print(f"Overall result: {'✅ PASSED' if passed_criteria >= 3 else '❌ FAILED'}")
    
    if passed_criteria >= 3:
        print(f"\n🎉 LA weather patterns successfully integrated with feature pipeline!")
        print(f"   - {analysis['total_features']} total features created")
        print(f"   - {len(la_pattern_features)} LA-specific features")
        print(f"   - Ready for machine learning model training")
    else:
        print(f"\n❌ Integration test failed. Check the issues above.")
    
    return passed_criteria >= 3


def cleanup_test_data():
    """Clean up test data files."""
    print(f"\nCleaning up test data...")
    
    import os
    test_files = [
        'data/nws_data.parquet',
        'data/openweather_data.parquet', 
        'data/tomorrow_data.parquet',
        'data/actual_temperatures.parquet'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   Removed {file_path}")
            except Exception as e:
                print(f"   Failed to remove {file_path}: {e}")


def main():
    """Run the integration test."""
    try:
        success = test_la_patterns_integration()
        return success
    finally:
        cleanup_test_data()


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)