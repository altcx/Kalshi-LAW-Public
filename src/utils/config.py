"""Configuration management for the Kalshi Weather Predictor."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for the weather predictor system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config file. Defaults to config/config.yaml
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Set default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data_collection.rate_limits.openweathermap')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment variables.
        
        Args:
            service: Service name (e.g., 'openweathermap', 'tomorrow_io')
            
        Returns:
            API key or None if not found
        """
        env_var_map = {
            'openweathermap': 'OPENWEATHERMAP_API_KEY',
            'tomorrow_io': 'TOMORROW_IO_API_KEY',
            'weatherbit': 'WEATHERBIT_API_KEY',
            'visual_crossing': 'VISUAL_CROSSING_API_KEY',
            'kalshi': 'KALSHI_API_KEY'
        }
        
        env_var = env_var_map.get(service.lower())
        if env_var:
            return os.getenv(env_var)
        return None
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        data_dir = self.get('data_collection.data_dir', 'data')
        return Path(data_dir)
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        logs_dir = self.get('logging.logs_dir', 'logs')
        return Path(logs_dir)
    
    @property
    def location(self) -> Dict[str, Any]:
        """Get location configuration."""
        return self.get('location', {})
    
    @property
    def rate_limits(self) -> Dict[str, Any]:
        """Get API rate limits."""
        return self.get('data_collection.rate_limits', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('models', {})
    
    @property
    def trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.get('trading', {})
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value and save to file.
        
        Args:
            key: Configuration key using dot notation
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save to file
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)


# Global configuration instance
config = Config()