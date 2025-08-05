"""Demonstration of feature extraction with real data."""

import sys
import os
# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.data_manager import DataManager
from src.feature_engineering.basic_features import BasicFeatureExtractor
import pandas as pd
from datetime import date, timedelta


def main():
    """Demonstrate feature extraction with real data."""
    print("=== Kalshi Weather Predictor - Feature Extraction Demo ===\n")
    
    # Initialize components
    data_manager = DataManager()
    feature_extractor = BasicFeatureExtractor()
    
    # Get data summary
    print("1. Current Data Summary:")
    summary = data_manager.get_data_summary()
    for source, info in summary.items():
        if 'records' in info:
            print(f"   {source}: {info['records']} records, {info['date_range']}")
        else:
            print(f"   {source}: {info}")
    print()
    
    # Load all available weather data
    print("2. Loading Weather Data:")
    all_source_data = data_manager.load_all_sources()
    
    available_sources = []
    for source, data in all_source_data.items():
        if not data.empty:
            print(f"   {source}: {len(data)} records")
            available_sources.append(source)
        else:
            print(f"   {source}: No data available")
    print()
    
    if not available_sources:
        print("No weather data available for feature extraction.")
        return
    
    # Extract features from individual sources
    print("3. Extracting Features from Individual Sources:")
    individual_features = {}
    
    for source in available_sources:
        data = all_source_data[source]
        print(f"\n   Processing {source}:")
        print(f"   - Raw data shape: {data.shape}")
        print(f"   - Date range: {data['date'].min()} to {data['date'].max()}")
        
        # Extract features for this source
        source_features = feature_extractor.extract_source_features(data, source, clean_outliers=True)
        individual_features[source] = source_features
        
        if not source_features.empty:
            print(f"   - Extracted features: {len(source_features.columns)-1}")
            print(f"   - Feature records: {len(source_features)}")
            
            # Show some example features
            feature_cols = [col for col in source_features.columns if col != 'date']
            print(f"   - Sample features: {feature_cols[:5]}...")
        else:
            print(f"   - No features extracted")
    
    # Combine features from all sources
    print("\n4. Combining Features from All Sources:")
    if individual_features:
        combined_features = feature_extractor.extract_all_source_features(all_source_data, clean_outliers=True)
        
        if not combined_features.empty:
            print(f"   - Combined features shape: {combined_features.shape}")
            print(f"   - Total features: {len(combined_features.columns)-1}")
            print(f"   - Date range: {combined_features['date'].min()} to {combined_features['date'].max()}")
            
            # Show feature categories
            feature_cols = [col for col in combined_features.columns if col != 'date']
            temp_features = [col for col in feature_cols if 'temp' in col.lower()]
            atmo_features = [col for col in feature_cols if any(x in col.lower() for x in ['humidity', 'pressure', 'wind', 'cloud', 'precipitation'])]
            date_features = [col for col in feature_cols if any(x in col.lower() for x in ['day', 'month', 'season', 'weekend'])]
            quality_features = [col for col in feature_cols if 'quality' in col.lower() or 'missing' in col.lower() or 'completeness' in col.lower()]
            
            print(f"   - Temperature features: {len(temp_features)}")
            print(f"   - Atmospheric features: {len(atmo_features)}")
            print(f"   - Date-based features: {len(date_features)}")
            print(f"   - Quality features: {len(quality_features)}")
            
            # Show sample of combined features
            print("\n   Sample of combined features:")
            print(combined_features.head(2).to_string())
        else:
            print("   - No combined features extracted")
    
    # Generate feature summary
    print("\n5. Feature Summary and Validation:")
    if 'combined_features' in locals() and not combined_features.empty:
        summary = feature_extractor.get_feature_summary(combined_features)
        print(f"   - Total features: {summary['total_features']}")
        print(f"   - Total records: {summary['total_records']}")
        print(f"   - Missing data: {summary['missing_data_percentage']:.1f}%")
        print(f"   - Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        print("\n   Feature categories:")
        for category, count in summary['feature_categories'].items():
            print(f"   - {category}: {count}")
        
        print("\n   Top features by variance:")
        if 'top_variance_features' in summary:
            for feature, variance in list(summary['top_variance_features'].items())[:5]:
                print(f"   - {feature}: {variance:.2f}")
        
        # Validate features
        is_valid, issues = feature_extractor.validate_features(combined_features)
        print(f"\n   Features valid: {is_valid}")
        if issues:
            print("   Issues found:")
            for issue in issues:
                print(f"   - {issue}")
    
    # Demonstrate outlier detection
    print("\n6. Outlier Detection and Cleaning:")
    for source in available_sources:
        data = all_source_data[source]
        if not data.empty and 'predicted_high' in data.columns:
            print(f"\n   {source} outlier analysis:")
            
            # Check for outliers before cleaning
            outliers_before = data_manager.detect_outliers(data, 'predicted_high', method='iqr')
            print(f"   - Outliers detected (IQR): {outliers_before.sum()}")
            
            if outliers_before.sum() > 0:
                outlier_values = data.loc[outliers_before, 'predicted_high'].tolist()
                print(f"   - Outlier values: {outlier_values}")
            
            # Clean outliers
            cleaned_data = feature_extractor.detect_and_clean_outliers(data, source, method='iqr')
            
            # Check quality scores
            if 'data_quality_score' in cleaned_data.columns:
                avg_quality = cleaned_data['data_quality_score'].mean()
                print(f"   - Average quality score after cleaning: {avg_quality:.3f}")
    
    # Show date-based features in detail
    print("\n7. Date-Based Features Example:")
    if 'combined_features' in locals() and not combined_features.empty:
        date_cols = [col for col in combined_features.columns if any(x in col.lower() for x in ['day', 'month', 'season', 'weekend', 'heat', 'fire', 'marine'])]
        if date_cols:
            print("   Date-based features for first record:")
            first_record = combined_features.iloc[0]
            for col in date_cols:
                if col in first_record:
                    print(f"   - {col}: {first_record[col]}")
    
    print("\n=== Feature Extraction Demo Complete ===")


if __name__ == '__main__':
    main()