#!/usr/bin/env python3
"""Demonstration of DataManager functionality."""

import sys
from pathlib import Path
from datetime import date, datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_manager import DataManager
from loguru import logger

def demo_data_manager():
    """Demonstrate DataManager capabilities."""
    logger.info("=== DataManager Demonstration ===")
    
    # Initialize DataManager
    dm = DataManager()
    
    # Create sample weather data
    sample_data = pd.DataFrame([
        {
            'date': '2025-01-28',
            'forecast_date': '2025-01-27',
            'predicted_high': 72.5,
            'predicted_low': 55.2,
            'humidity': 68.0,
            'pressure': 1014.5,
            'wind_speed': 6.8,
            'wind_direction': 270.0,
            'cloud_cover': 25.0,
            'precipitation_prob': 5.0
        }
    ])
    
    logger.info("1. Appending weather data with validation...")
    success = dm.append_daily_data('nws', sample_data, validate=True)
    logger.info(f"Data append success: {success}")
    
    logger.info("2. Storing a prediction...")
    dm.store_prediction(
        prediction=73.1,
        confidence=0.92,
        target_date=date(2025, 1, 28),
        model_contributions={'xgboost': 0.7, 'lightgbm': 0.3},
        feature_importance={'temp_consensus': 0.4, 'pressure': 0.3, 'humidity': 0.3}
    )
    
    logger.info("3. Getting data quality summary...")
    quality_summary = dm.get_data_quality_summary('nws')
    logger.info(f"Quality summary: {quality_summary}")
    
    logger.info("4. Getting overall data summary...")
    data_summary = dm.get_data_summary()
    for source, info in data_summary.items():
        if 'records' in info:
            logger.info(f"{source}: {info['records']} records, {info['file_size_mb']:.3f} MB")
    
    logger.info("=== Demo Complete ===")

if __name__ == "__main__":
    demo_data_manager()