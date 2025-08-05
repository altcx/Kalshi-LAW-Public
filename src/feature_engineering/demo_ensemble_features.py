"""Demo script for ensemble feature extraction with real data."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, timedelta
import pandas as pd
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.feature_engineering.ensemble_features import EnsembleFeatureExtractor
from src.utils.data_manager import DataManager


def main():
    """Demonstrate ensemble feature extraction with real data."""
    print("=== Ensemble Feature Extraction Demo ===\n")
    
    # Initialize components
    pipeline = FeaturePipeline()
    data_manager = DataManager()
    ensemble_extractor = EnsembleFeatureExtractor()
    
    # Check data availability
    print("1. Checking Data Availability:")
    data_summary = data_manager.get_data_summary()
    
    available_sources = []
    for source, info in data_summary.items():
        if isinstance(info, dict) and 'records' in info and info['records'] > 0:
            available_sources.append(source)
            print(f"   {source}: {info['records']} records")
    
    if len(available_sources) < 2:
        print(f"\nInsufficient data sources for ensemble features (need 2+, have {len(available_sources)})")
        print("Ensemble features work best with multiple weather data sources.")
        return
    
    print(f"\nFound {len(available_sources)} data sources - good for ensemble features!")
    
    # Load recent data
    print("\n2. Loading Recent Weather Data:")
    end_date = date.today()
    start_date = end_date - timedelta(days=14)  # Last 2 weeks
    
    source_data = data_manager.load_all_sources(start_date, end_date)
    
    # Filter to sources with actual data
    source_data = {source: data for source, data in source_data.items() 
                   if not data.empty and len(data) > 0}
    
    if not source_data:
        print("No recent data available for demonstration")
        return
    
    print(f"Loaded data from {len(source_data)} sources:")
    for source, data in source_data.items():
        if not data.empty:
            date_range = f"{data['date'].min()} to {data['date'].max()}"
            print(f"   {source}: {len(data)} records ({date_range})")
    
    # Create ensemble features
    print("\n3. Creating Ensemble Features:")
    
    # Consensus features
    print("\n   a) Consensus Features (mean, median, std across sources):")
    consensus_features = ensemble_extractor.create_consensus_features(source_data)
    if not consensus_features.empty:
        consensus_cols = [col for col in consensus_features.columns if 'consensus' in col]
        print(f"      Created {len(consensus_cols)} consensus features")
        
        # Show sample consensus features for temperature
        temp_consensus_cols = [col for col in consensus_cols if 'predicted_high' in col][:5]
        if temp_consensus_cols and len(consensus_features) > 0:
            print("      Sample temperature consensus features:")
            sample_data = consensus_features[['date'] + temp_consensus_cols].head(3)
            for _, row in sample_data.iterrows():
                print(f"        {row['date'].strftime('%Y-%m-%d')}: mean={row.get('predicted_high_consensus_mean', 'N/A'):.1f}°F, "
                      f"std={row.get('predicted_high_consensus_std', 'N/A'):.2f}°F")
    
    # Agreement metrics
    print("\n   b) API Agreement/Disagreement Metrics:")
    agreement_features = ensemble_extractor.create_agreement_metrics(source_data)
    if not agreement_features.empty:
        agreement_cols = [col for col in agreement_features.columns if 'agreement' in col]
        print(f"      Created {len(agreement_cols)} agreement metrics")
        
        # Show sample agreement scores
        if 'overall_agreement_score' in agreement_features.columns and len(agreement_features) > 0:
            avg_agreement = agreement_features['overall_agreement_score'].mean()
            print(f"      Average API agreement score: {avg_agreement:.3f} (higher = more agreement)")
            
            # Show days with high/low agreement
            high_agreement = agreement_features['overall_agreement_score'].max()
            low_agreement = agreement_features['overall_agreement_score'].min()
            print(f"      Agreement range: {low_agreement:.3f} to {high_agreement:.3f}")
    
    # Rolling features
    print("\n   c) Rolling Average and Trend Features:")
    rolling_features = ensemble_extractor.create_rolling_features(source_data)
    if not rolling_features.empty:
        rolling_cols = [col for col in rolling_features.columns if 'rolling' in col]
        print(f"      Created {len(rolling_cols)} rolling features")
        
        # Show sample rolling features
        temp_rolling_cols = [col for col in rolling_cols if 'predicted_high' in col and '3d' in col][:3]
        if temp_rolling_cols and len(rolling_features) > 0:
            print("      Sample 3-day rolling temperature features:")
            sample_data = rolling_features[['date'] + temp_rolling_cols].tail(3)
            for _, row in sample_data.iterrows():
                print(f"        {row['date'].strftime('%Y-%m-%d')}: "
                      f"3d_mean={row.get('predicted_high_rolling_3d_mean', 'N/A'):.1f}°F, "
                      f"trend={row.get('predicted_high_rolling_3d_trend', 'N/A'):.3f}")
    
    # All ensemble features combined
    print("\n4. Complete Ensemble Feature Set:")
    all_ensemble_features = ensemble_extractor.create_all_ensemble_features(source_data)
    
    if not all_ensemble_features.empty:
        summary = ensemble_extractor.get_ensemble_feature_summary(all_ensemble_features)
        
        print(f"   Total ensemble features: {summary['total_features']}")
        print(f"   Records: {summary['total_records']}")
        print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"   Missing data: {summary['missing_data_percentage']:.1f}%")
        
        print("\n   Feature breakdown by category:")
        for category, count in summary['feature_categories'].items():
            print(f"     {category}: {count} features")
    
    # Compare with basic features
    print("\n5. Comparison with Basic Features:")
    basic_features = pipeline.create_features_for_date_range(start_date, end_date)
    complete_features = pipeline.create_complete_features(start_date, end_date, include_ensemble=True)
    
    if not basic_features.empty and not complete_features.empty:
        basic_count = len(basic_features.columns) - 1  # Exclude date
        complete_count = len(complete_features.columns) - 1  # Exclude date
        ensemble_added = complete_count - basic_count
        
        print(f"   Basic features only: {basic_count}")
        print(f"   Complete features (basic + ensemble): {complete_count}")
        print(f"   Ensemble features added: {ensemble_added}")
        print(f"   Feature increase: {(ensemble_added/basic_count)*100:.1f}%")
    
    # Show feature importance insights
    print("\n6. Feature Quality Insights:")
    if not all_ensemble_features.empty:
        # Check for high-variance features (potentially most informative)
        numeric_features = all_ensemble_features.select_dtypes(include=['float64', 'int64'])
        if len(numeric_features.columns) > 0:
            feature_variance = numeric_features.var().sort_values(ascending=False)
            
            print("   Top 5 highest variance ensemble features:")
            for i, (feature, variance) in enumerate(feature_variance.head(5).items()):
                print(f"     {i+1}. {feature}: {variance:.2f}")
        
        # Check agreement consistency
        if 'overall_agreement_score' in all_ensemble_features.columns:
            agreement_scores = all_ensemble_features['overall_agreement_score'].dropna()
            if len(agreement_scores) > 0:
                stable_days = (agreement_scores > 0.8).sum()
                total_days = len(agreement_scores)
                print(f"\n   API Agreement Analysis:")
                print(f"     Days with high agreement (>0.8): {stable_days}/{total_days} ({(stable_days/total_days)*100:.1f}%)")
                print(f"     This indicates how often weather sources agree on forecasts")
    
    print("\n=== Ensemble Feature Demo Complete ===")
    print("\nEnsemble features provide valuable meta-information about:")
    print("- Consensus across multiple weather sources")
    print("- Agreement/disagreement between APIs") 
    print("- Rolling trends and momentum")
    print("- Source reliability and data quality")
    print("- These features can significantly improve ML model performance!")


if __name__ == '__main__':
    main()