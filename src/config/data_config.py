"""
Data configuration classes and validation.
Provides typed access to data collection configuration.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""

    enabled: bool = True
    rate_limit: int = 5
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class DateRangeConfig:
    """Configuration for date ranges."""

    start_date: str
    end_date: Optional[str] = None

    def get_start_date(self) -> date:
        """Convert start_date string to date object."""
        return datetime.strptime(self.start_date, "%Y-%m-%d").date()

    def get_end_date(self) -> Optional[date]:
        """Convert end_date string to date object."""
        if self.end_date:
            return datetime.strptime(self.end_date, "%Y-%m-%d").date()
        return None


@dataclass
class DataQualityConfig:
    """Configuration for data quality parameters."""

    min_trading_days: int = 252
    max_missing_ratio: float = 0.05
    outlier_threshold: float = 10.0
    min_volume: int = 100000
    min_price: float = 1.0
    max_price: float = 10000.0


@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicators."""

    enabled: bool = True
    period: Optional[int] = None
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    signal_period: Optional[int] = None
    std_dev: Optional[float] = None
    periods: Optional[List[int]] = None


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    technical_indicators: Dict[str, TechnicalIndicatorConfig] = field(default_factory=dict)
    price_features: Dict[str, Any] = field(default_factory=dict)
    temporal_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceConfig:
    """Configuration for sequence generation."""

    window_size: int = 60
    forecast_horizon: int = 5
    stride: int = 1
    min_sequences: int = 100


@dataclass
class StorageConfig:
    """Configuration for data storage."""

    raw_data: Dict[str, str] = field(
        default_factory=lambda: {
            "format": "parquet",
            "compression": "snappy",
            "partition_by": "ticker",
        }
    )
    processed_data: Dict[str, Any] = field(
        default_factory=lambda: {"format": "pytorch", "compression": True}
    )
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "database_type": "sqlite",
            "backup_enabled": True,
            "backup_frequency": "daily",
        }
    )


@dataclass
class CachingConfig:
    """Configuration for caching."""

    enabled: bool = True
    cache_type: str = "memory"
    max_size_mb: int = 1024
    ttl_seconds: int = 3600


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "structured"
    log_to_file: bool = True
    log_file: str = "logs/data_collection.log"
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing."""

    enabled: bool = True
    max_workers: int = 4
    chunk_size: int = 10


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and alerts."""

    enabled: bool = True
    metrics_collection: bool = True
    alert_on_failures: bool = True
    max_consecutive_failures: int = 3


@dataclass
class DevelopmentConfig:
    """Configuration for development and testing."""

    use_sample_data: bool = False
    mock_api_calls: bool = False
    reduced_date_range: bool = False
    debug_mode: bool = False


class DataConfig:
    """
    Main data configuration class.
    Provides typed access to all data collection configuration.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize data configuration from dictionary.

        Args:
            config_dict: Configuration dictionary loaded from YAML
        """
        self.raw_config = config_dict

        # Parse data sources
        self.data_sources = {}
        for name, source_config in config_dict.get("data_sources", {}).items():
            self.data_sources[name] = DataSourceConfig(**source_config)

        # Parse tickers
        self.tickers = config_dict.get("tickers", {})

        # Parse date ranges
        self.date_ranges = {}
        for name, range_config in config_dict.get("date_ranges", {}).items():
            self.date_ranges[name] = DateRangeConfig(**range_config)

        # Parse other configurations
        self.data_quality = DataQualityConfig(**config_dict.get("data_quality", {}))
        self.sequences = SequenceConfig(**config_dict.get("sequences", {}))
        self.storage = StorageConfig(**config_dict.get("storage", {}))
        self.caching = CachingConfig(**config_dict.get("caching", {}))
        self.logging = LoggingConfig(**config_dict.get("logging", {}))
        self.parallel_processing = ParallelProcessingConfig(
            **config_dict.get("parallel_processing", {})
        )
        self.monitoring = MonitoringConfig(**config_dict.get("monitoring", {}))
        self.development = DevelopmentConfig(**config_dict.get("development", {}))

        # Parse features configuration
        self._parse_features_config(config_dict.get("features", {}))

    def _parse_features_config(self, features_config: Dict[str, Any]):
        """Parse features configuration."""
        self.features = FeatureConfig()

        # Parse technical indicators
        tech_indicators = features_config.get("technical_indicators", {})
        for name, indicator_config in tech_indicators.items():
            if isinstance(indicator_config, dict):
                self.features.technical_indicators[name] = TechnicalIndicatorConfig(
                    **indicator_config
                )

        # Store other feature configs as-is for now
        self.features.price_features = features_config.get("price_features", {})
        self.features.temporal_features = features_config.get("temporal_features", {})

    def get_ticker_list(self, ticker_set: str = "large_cap") -> List[str]:
        """
        Get list of tickers for a specific set.

        Args:
            ticker_set: Name of the ticker set

        Returns:
            List of ticker symbols
        """
        return self.tickers.get(ticker_set, [])

    def get_enabled_data_sources(self) -> List[str]:
        """Get list of enabled data sources."""
        return [name for name, config in self.data_sources.items() if config.enabled]

    def get_data_source_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """Get configuration for a specific data source."""
        return self.data_sources.get(source_name)

    def get_date_range(self, range_name: str = "full_history") -> Optional[DateRangeConfig]:
        """Get date range configuration."""
        return self.date_ranges.get(range_name)

    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.development.debug_mode or self.development.use_sample_data

    def validate(self) -> bool:
        """
        Validate configuration for consistency and completeness.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check that at least one data source is enabled
            if not self.get_enabled_data_sources():
                logger.error("No data sources are enabled")
                return False

            # Check that at least one ticker set is defined
            if not self.tickers:
                logger.error("No ticker sets defined")
                return False

            # Validate date ranges
            for name, date_range in self.date_ranges.items():
                try:
                    start_date = date_range.get_start_date()
                    end_date = date_range.get_end_date()
                    if end_date and start_date >= end_date:
                        logger.error(f"Invalid date range '{name}': start_date >= end_date")
                        return False
                except ValueError as e:
                    logger.error(f"Invalid date format in range '{name}': {e}")
                    return False

            # Validate sequence parameters
            if self.sequences.window_size <= 0:
                logger.error("Sequence window_size must be positive")
                return False

            if self.sequences.forecast_horizon <= 0:
                logger.error("Sequence forecast_horizon must be positive")
                return False

            # Validate data quality parameters
            if not (0 <= self.data_quality.max_missing_ratio <= 1):
                logger.error("max_missing_ratio must be between 0 and 1")
                return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration back to dictionary format."""
        return self.raw_config
