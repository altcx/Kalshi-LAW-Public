"""Main entry point for the Kalshi Weather Predictor."""

import sys
from pathlib import Path
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import config
from utils.log_config import setup_logging
from utils.data_manager import DataManager


def main():
    """Main application entry point."""
    logger.info("Starting Kalshi Weather Predictor")
    
    # Initialize components
    data_manager = DataManager()
    
    # Display configuration summary
    logger.info(f"Configuration loaded from: {config.config_path}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Location: {config.location.get('city', 'Unknown')}, {config.location.get('state', 'Unknown')}")
    
    # Display data summary
    data_summary = data_manager.get_data_summary()
    logger.info("Data summary:")
    for source, info in data_summary.items():
        if 'records' in info:
            logger.info(f"  {source}: {info['records']} records, {info['file_size_mb']:.2f} MB")
        else:
            logger.info(f"  {source}: {info.get('status', 'unknown status')}")
    
    logger.info("Kalshi Weather Predictor initialized successfully")


if __name__ == "__main__":
    main()