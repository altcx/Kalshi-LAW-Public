"""Logging configuration for the Kalshi Weather Predictor."""

import sys
from pathlib import Path
from loguru import logger
from .config import config


def setup_logging():
    """Configure logging for the application."""
    
    # Remove default handler
    logger.remove()
    
    # Get logging configuration
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.log_file', 'logs/weather_predictor.log')
    max_file_size = config.get('logging.max_file_size', '10MB')
    backup_count = config.get('logging.backup_count', 5)
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Console handler with colored output
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler with rotation
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=max_file_size,
        retention=backup_count,
        compression="zip"
    )
    
    logger.info("Logging configured successfully")
    return logger


# Initialize logging
setup_logging()