#!/usr/bin/env python3

import sys
import os

# Add project root to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

try:
    print("Testing imports...")
    
    # Test LightGBM model
    try:
        from src.models.lightgbm_model import LightGBMTemperatureModel
        print("✓ LightGBM model import successful")
    except Exception as e:
        print(f"✗ LightGBM model import failed: {e}")
    
    # Test Linear Regression model
    try:
        from src.models.linear_regression_model import LinearRegressionTemperatureModel
        print("✓ Linear Regression model import successful")
    except Exception as e:
        print(f"✗ Linear Regression model import failed: {e}")
    
    # Test Prophet model
    try:
        from src.models.prophet_model import ProphetTemperatureModel
        print("✓ Prophet model import successful")
    except Exception as e:
        print(f"✗ Prophet model import failed: {e}")
    
    # Test model adapters
    try:
        from src.models.model_adapters import (
            LightGBMModelAdapter, 
            LinearRegressionModelAdapter, 
            ProphetModelAdapter
        )
        print("✓ Model adapters import successful")
    except Exception as e:
        print(f"✗ Model adapters import failed: {e}")
    
    print("Import testing complete!")
    
except Exception as e:
    print(f"General error: {e}")
    import traceback
    traceback.print_exc()