"""Demo script for the performance tracking system."""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

from .performance_tracker import PerformanceTracker
from ..utils.data_manager import DataManager


def create_sample_data():
    """Create sample weather and actual temperature data for demonstration."""
    print("Creating sample data...")
    
    # Create date range for the last 30 days
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create sample weather data for multiple sources
    sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
    weather_data = {}
    
    for source in sources:
        data = []
        for i, date_val in enumerate(dates):
            # Create realistic temperature predictions with source-specific bias
            base_temp = 75 + 10 * np.sin(i * 0.2)  # Seasonal variation
            
            # Add source-specific bias and accuracy
            source_bias = {
                'nws': 0.5,           # Slight warm bias
                'openweather': -0.3,  # Slight cool bias
                'tomorrow': 0.8,      # Warm bias
                'weatherbit': -0.5,   # Cool bias
                'visual_crossing': 0.2 # Minimal bias
            }
            
            source_accuracy = {
                'nws': 1.5,           # High accuracy (low noise)
                'openweather': 2.0,   # Good accuracy
                'tomorrow': 2.5,      # Moderate accuracy
                'weatherbit': 3.0,    # Lower accuracy
                'visual_crossing': 1.8 # Good accuracy
            }
            
            bias = source_bias.get(source, 0)
            noise_std = source_accuracy.get(source, 2.0)
            noise = np.random.normal(bias, noise_std)
            
            data.append({
                'date': date_val,
                'forecast_date': date_val - timedelta(days=1),
                'predicted_high': base_temp + noise,
                'predicted_low': base_temp - 15 + noise,
                'humidity': 60 + np.random.normal(0, 10),
                'pressure': 1013 + np.random.normal(0, 5),
                'wind_speed': 8 + np.random.normal(0, 3),
                'wind_direction': np.random.uniform(0, 360),
                'cloud_cover': np.random.uniform(0, 100),
                'precipitation_prob': np.random.uniform(0, 30),
                'data_quality_score': 0.85 + np.random.normal(0, 0.1)
            })
        
        weather_data[source] = pd.DataFrame(data)
    
    # Create actual temperature data (ground truth)
    actual_data = []
    for i, date_val in enumerate(dates):
        base_temp = 75 + 10 * np.sin(i * 0.2)
        actual_noise = np.random.normal(0, 1.0)  # Less noise than predictions
        
        actual_data.append({
            'date': date_val,
            'actual_high': base_temp + actual_noise,
            'actual_low': base_temp - 15 + actual_noise,
            'source': 'NOAA'
        })
    
    actual_temps = pd.DataFrame(actual_data)
    
    return weather_data, actual_temps


def demo_performance_tracking():
    """Demonstrate the performance tracking system."""
    print("=== Performance Tracking System Demo ===\n")
    
    # Initialize components
    data_manager = DataManager()
    performance_tracker = PerformanceTracker(data_manager)
    
    # Create and store sample data
    weather_data, actual_temps = create_sample_data()
    
    print("Storing sample data...")
    for source, data in weather_data.items():
        data_manager.save_source_data(source, data)
        print(f"  - Stored {len(data)} records for {source}")
    
    data_manager.save_source_data('actual_temperatures', actual_temps)
    print(f"  - Stored {len(actual_temps)} actual temperature records\n")
    
    # Track performance for individual sources
    print("=== Individual Source Performance ===")
    for source in weather_data.keys():
        print(f"\n{source.upper()} Performance:")
        metrics = performance_tracker.track_source_performance(source, window_days=30)
        
        if 'error' not in metrics:
            print(f"  - Count: {metrics['count']} predictions")
            print(f"  - MAE: {metrics['mae']:.2f}°F")
            print(f"  - RMSE: {metrics['rmse']:.2f}°F")
            print(f"  - Accuracy (±1°F): {metrics['accuracy_1f']:.1%}")
            print(f"  - Accuracy (±3°F): {metrics['accuracy_3f']:.1%}")
            print(f"  - Performance Category: {metrics['performance_category']}")
            print(f"  - Average Data Quality: {metrics.get('data_quality_avg', 0):.3f}")
        else:
            print(f"  - Error: {metrics['error']}")
    
    # Calculate and display source weights
    print("\n=== Source Weight Calculation ===")
    performance_data = performance_tracker.track_all_sources_performance(window_days=30)
    weights = performance_tracker.calculate_source_weights(performance_data)
    
    print("Dynamic source weights based on recent performance:")
    for source, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {source}: {weight:.3f} ({weight*100:.1f}%)")
    
    # Store performance metrics
    print("\n=== Storing Performance Metrics ===")
    for source, metrics in performance_data.items():
        if 'error' not in metrics:
            performance_tracker.store_performance_metrics(metrics)
            print(f"  - Stored metrics for {source}")
    
    # Generate comprehensive performance report
    print("\n=== Comprehensive Performance Report ===")
    report = performance_tracker.generate_performance_report(window_days=30)
    
    if 'summary' in report and 'error' not in report['summary']:
        summary = report['summary']
        print(f"Best performing source: {summary['best_source']}")
        print(f"Worst performing source: {summary['worst_source']}")
        print(f"Average ±3°F accuracy: {summary['avg_accuracy_3f']:.1%}")
        print(f"Average MAE: {summary['avg_mae']:.2f}°F")
        print(f"Sources with performance issues: {summary['sources_with_issues']}")
    
    # Demonstrate performance degradation detection
    print("\n=== Performance Degradation Detection ===")
    for source in weather_data.keys():
        degradation = performance_tracker.detect_performance_degradation(source)
        status = "DEGRADED" if degradation else "STABLE"
        print(f"  - {source}: {status}")
    
    # Run full daily performance tracking
    print("\n=== Daily Performance Tracking ===")
    results = performance_tracker.run_daily_performance_tracking()
    
    print("Daily tracking completed with results for:")
    for window_name in results.keys():
        if window_name not in ['updated_weights', 'performance_report']:
            print(f"  - {window_name} ({performance_tracker.tracking_windows[window_name]} days)")
    
    if results['updated_weights']:
        print(f"\nUpdated weights: {len(results['updated_weights'])} sources")
    
    # Display data summary
    print("\n=== Data Summary ===")
    summary = data_manager.get_data_summary()
    for source, info in summary.items():
        if 'error' not in info and 'status' not in info:
            print(f"{source}: {info['records']} records, {info['file_size_mb']:.1f} MB")
    
    print("\n=== Demo Complete ===")
    print("Performance tracking system successfully demonstrated!")
    print("Check the data/ directory for stored performance metrics.")


if __name__ == "__main__":
    demo_performance_tracking()