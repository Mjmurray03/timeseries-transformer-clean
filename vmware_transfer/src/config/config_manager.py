"""
Configuration management system for the time-series transformer project.
Handles loading, validation, and access to configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class ConfigPaths:
    """Configuration file paths."""
    root: Path
    data: Path
    model: Path
    training: Path
    deployment: Path

class ConfigManager:
    """
    Centralized configuration management system.
    
    Handles loading and merging configuration from multiple sources:
    1. YAML configuration files
    2. Environment variables
    3. Command line arguments (future)
    """
    
    def __init__(self, config_root: Union[str, Path] = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_root: Root directory for configuration files
        """
        self.config_root = Path(config_root)
        self.paths = ConfigPaths(
            root=self.config_root,
            data=self.config_root / "data_config.yaml",
            model=self.config_root / "model",
            training=self.config_root / "training", 
            deployment=self.config_root / "deployment"
        )
        
        # Load environment variables
        load_dotenv()
        
        # Cache for loaded configurations
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        
        # Check cache first
        cache_key = str(config_path.absolute())
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Apply environment variable overrides
            config = self._apply_env_overrides(config)
            
            # Cache the configuration
            self._config_cache[cache_key] = config
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def get_data_config(self) -> Dict[str, Any]:
        """Load data collection configuration."""
        return self.load_config(self.paths.data)
    
    def get_model_config(self, model_name: str = "transformer_base") -> Dict[str, Any]:
        """
        Load model configuration.
        
        Args:
            model_name: Name of the model configuration file (without .yaml)
        """
        model_config_path = self.paths.model / f"{model_name}.yaml"
        return self.load_config(model_config_path)
    
    def get_training_config(self, training_name: str = "full_training") -> Dict[str, Any]:
        """
        Load training configuration.
        
        Args:
            training_name: Name of the training configuration file (without .yaml)
        """
        training_config_path = self.paths.training / f"{training_name}.yaml"
        return self.load_config(training_config_path)
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables should follow the pattern:
        CONFIG_SECTION_SUBSECTION_KEY=value
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        env_overrides = {}
        
        # Common environment variable mappings
        env_mappings = {
            'ALPHA_VANTAGE_API_KEY': ['data_sources', 'alpha_vantage', 'api_key'],
            'NEWS_API_KEY': ['data_sources', 'news_api', 'api_key'],
            'WANDB_API_KEY': ['monitoring', 'wandb', 'api_key'],
            'DATA_ROOT_PATH': ['storage', 'data_root'],
            'CACHE_ROOT_PATH': ['storage', 'cache_root'],
            'MODEL_ROOT_PATH': ['storage', 'model_root'],
            'DATABASE_URL': ['storage', 'metadata', 'database_url'],
            'REDIS_URL': ['caching', 'redis_url'],
            'LOG_LEVEL': ['logging', 'level'],
            'DEBUG': ['development', 'debug_mode']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_value.lower() in ('true', 'false'):
                    env_value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif self._is_float(env_value):
                    env_value = float(env_value)
                
                # Set nested configuration value
                self._set_nested_config(config, config_path, env_value)
                logger.debug(f"Applied environment override: {env_var} -> {'.'.join(config_path)}")
        
        return config
    
    def _set_nested_config(self, config: Dict[str, Any], path: list, value: Any):
        """
        Set a nested configuration value.
        
        Args:
            config: Configuration dictionary to modify
            path: List of keys representing the nested path
            value: Value to set
        """
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema.
        
        Args:
            config: Configuration to validate
            schema: Validation schema
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation implementation
        # In production, consider using jsonschema or cerberus
        try:
            for key, expected_type in schema.items():
                if key in config:
                    if not isinstance(config[key], expected_type):
                        logger.error(f"Configuration key '{key}' has wrong type. Expected {expected_type}, got {type(config[key])}")
                        return False
                else:
                    logger.warning(f"Configuration key '{key}' is missing")
            return True
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def clear_cache(self):
        """Clear configuration cache."""
        self._config_cache.clear()
        logger.info("Configuration cache cleared")
    
    def reload_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Reload configuration from file, bypassing cache.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Reloaded configuration dictionary
        """
        cache_key = str(Path(config_path).absolute())
        if cache_key in self._config_cache:
            del self._config_cache[cache_key]
        return self.load_config(config_path)

# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(config_type: str = "data", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to get configuration.
    
    Args:
        config_type: Type of configuration ('data', 'model', 'training')
        **kwargs: Additional arguments passed to specific config loaders
        
    Returns:
        Configuration dictionary
    """
    manager = get_config_manager()
    
    if config_type == "data":
        return manager.get_data_config()
    elif config_type == "model":
        return manager.get_model_config(kwargs.get("model_name", "transformer_base"))
    elif config_type == "training":
        return manager.get_training_config(kwargs.get("training_name", "full_training"))
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")