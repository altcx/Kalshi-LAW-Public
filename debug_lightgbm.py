#!/usr/bin/env python3

import sys
import os

# Add project root to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

print("Testing LightGBM availability...")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✓ LightGBM import successful")
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    print(f"✗ LightGBM import failed: {e}")

print(f"LIGHTGBM_AVAILABLE = {LIGHTGBM_AVAILABLE}")

# Now try to import the class
if LIGHTGBM_AVAILABLE:
    print("Attempting to define LightGBMTemperatureModel...")
    
    try:
        # Import dependencies
        from typing import Dict, Optional, Tuple, List, Any
        from datetime import date, datetime, timedelta
        import pandas as pd
        import numpy as np
        from loguru import logger
        import joblib
        from pathlib import Path
        
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.preprocessing import StandardScaler
        import optuna
        
        from src.utils.data_manager import DataManager
        from src.feature_engineering.feature_pipeline import FeaturePipeline
        
        print("✓ All dependencies imported successfully")
        
        # Try to define a minimal class
        class LightGBMTemperatureModel:
            def __init__(self):
                self.is_trained = False
                print("✓ LightGBMTemperatureModel class defined successfully")
        
        # Test instantiation
        model = LightGBMTemperatureModel()
        print("✓ LightGBMTemperatureModel instantiated successfully")
        
    except Exception as e:
        print(f"✗ Error defining class: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Skipping class definition due to missing LightGBM")