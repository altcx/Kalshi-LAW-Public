"""Minimal LightGBM model for testing."""

print("Starting minimal LightGBM model import...")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✓ LightGBM imported")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("✗ LightGBM not available")

print("Defining class...")

class LightGBMTemperatureModel:
    """Minimal LightGBM model for testing."""
    
    def __init__(self):
        self.is_trained = False
        print("✓ LightGBMTemperatureModel initialized")

print("✓ Class defined successfully")