#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Historical Data Download Script for Time-Series Transformer

Downloads and processes historical stock market data following the 
data collection architecture specified in .kiro/specs/data-collection/design.md

Usage:
    doppler run -- python scripts/data/download_historical.py --tickers AAPL,MSFT --years 5
    ./scripts/run_with_doppler.sh scripts/data/download_historical.py --years 5
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import pandas as pd
from tqdm.asyncio import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config import Config
from src.config.data_config import DataConfig
from src.data.collectors.yahoo_finance import YahooFinanceCollector
from src.data.processors.feature_engineering import FeatureEngineer
from src.data.storage import DataStorage
from src.data.validators import DataValidator, create_validation_report

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for historical data download.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Download historical stock data using DataCollectionOrchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 5 years of data for specific tickers
    doppler run -- python scripts/data/download_historical.py --tickers AAPL,MSFT,NVDA,GOOGL,TSLA --years 5
    
    # Use helper script with default tickers
    ./scripts/run_with_doppler.sh scripts/data/download_historical.py --years 5
    
    # Download with feature engineering enabled
    doppler run -- python scripts/data/download_historical.py --tickers AAPL --years 2 --features
        """
    )
    
    parser.add_argument(
        '--tickers', 
        type=str, 
        help='Comma-separated list of tickers (e.g., AAPL,MSFT,GOOGL)',
        default=','.join(getattr(Config, 'DEFAULT_TICKERS', ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']))
    )
    
    parser.add_argument(
        '--years', 
        type=int, 
        default=getattr(Config, 'YEARS_OF_DATA', 5),
        help='Number of years of historical data (default: 5)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=getattr(Config, 'RAW_DATA_DIR', Path('data/raw')),
        help='Output directory for raw data'
    )
    
    parser.add_argument(
        '--features',
        action='store_true',
        help='Enable feature engineering during download'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Enable data validation (default: True)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maximum concurrent downloads (default: 5)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--format',
        choices=['parquet', 'csv', 'hdf5'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    
    return parser.parse_args()

class DataCollectionOrchestrator:
    """
    Coordinates data collection from multiple sources following design specifications.
    
    Implements the DataCollectionOrchestrator pattern from design.md with proper
    error handling, validation, and feature engineering pipeline.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize data collection orchestrator.
        
        Args:
            config: Data configuration object
        """
        self.config = config or self._create_default_config()
        
        # Initialize components following design specification
        self.collectors = {
            'yahoo': YahooFinanceCollector(self.config)
        }
        
        self.validator = DataValidator(self.config)
        self.engineer = FeatureEngineer()
        self.storage = DataStorage(self.config)
        
        # Statistics tracking
        self.stats = {
            'tickers_requested': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'validation_failures': 0,
            'feature_engineering_time': 0.0,
            'total_data_points': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataCollectionOrchestrator initialized")
    
    def _create_default_config(self) -> DataConfig:
        """Create default configuration for data collection."""
        default_config = {
            'data_sources': {
                'yahoo_finance': {
                    'enabled': True,
                    'rate_limit': 5,
                    'timeout': 30,
                    'retry_attempts': 3,
                    'retry_delay': 1
                }
            },
            'data_quality': {
                'min_trading_days': 252,
                'max_missing_ratio': 0.05,
                'outlier_threshold': 10.0,
                'min_volume': 100000,
                'min_price': 0.01,
                'max_price': 100000.0
            },
            'features': {
                'technical_indicators': {
                    'rsi': {'enabled': True, 'period': 14},
                    'macd': {
                        'enabled': True,
                        'fast_period': 12,
                        'slow_period': 26,
                        'signal_period': 9
                    },
                    'bollinger_bands': {
                        'enabled': True,
                        'period': 20,
                        'std_dev': 2.0
                    }
                }
            }
        }
        return DataConfig(default_config)
    
    async def collect_all(self, tickers: List[str], start_date: date, 
                         end_date: date, enable_features: bool = False,
                         enable_validation: bool = True,
                         output_format: str = 'parquet') -> Dict[str, pd.DataFrame]:
        """
        Collect data for all specified tickers following the orchestration pattern.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            enable_features: Whether to perform feature engineering
            enable_validation: Whether to validate data
            output_format: Output format (parquet, csv, hdf5)
            
        Returns:
            Dictionary mapping ticker to processed DataFrame
        """
        self.logger.info(f"Starting data collection for {len(tickers)} tickers")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Features enabled: {enable_features}, Validation: {enable_validation}")
        
        self.stats['tickers_requested'] = len(tickers)
        
        # Step 1: Collect raw data using Yahoo Finance collector
        self.logger.info("Step 1: Collecting raw data...")
        collected_data = await self.collectors['yahoo'].collect_multiple(
            tickers, start_date, end_date
        )
        
        self.logger.info(f"Raw data collected for {len(collected_data)}/{len(tickers)} tickers")
        
        # Step 2: Validate collected data
        validated_data = {}
        validation_results = {}
        
        if enable_validation:
            self.logger.info("Step 2: Validating collected data...")
            validation_results = self.validator.validate_batch(collected_data)
            
            for ticker, result in validation_results.items():
                if result.is_valid:
                    validated_data[ticker] = collected_data[ticker]
                    self.stats['successful_downloads'] += 1
                else:
                    self.logger.warning(f"Validation failed for {ticker}: "
                                      f"{len(result.get_issues_by_severity('ERROR'))} errors")
                    self.stats['validation_failures'] += 1
        else:
            validated_data = collected_data
            self.stats['successful_downloads'] = len(collected_data)
        
        self.stats['failed_downloads'] = len(tickers) - self.stats['successful_downloads']
        
        # Step 3: Feature engineering (if enabled)
        processed_data = {}
        
        if enable_features:
            self.logger.info("Step 3: Engineering features...")
            
            for ticker, data in validated_data.items():
                try:
                    start_time = datetime.now()
                    
                    # Engineer features
                    engineered_data = self.engineer.engineer_features(data)
                    
                    # Validate engineered features
                    if self.engineer.validate_features(engineered_data):
                        processed_data[ticker] = engineered_data
                        self.stats['total_data_points'] += len(engineered_data)
                        
                        feature_time = (datetime.now() - start_time).total_seconds()
                        self.stats['feature_engineering_time'] += feature_time
                        
                        self.logger.debug(f"Features engineered for {ticker} in {feature_time:.2f}s")
                    else:
                        self.logger.error(f"Feature validation failed for {ticker}")
                        processed_data[ticker] = data  # Use original data
                        
                except Exception as e:
                    self.logger.error(f"Feature engineering failed for {ticker}: {e}")
                    processed_data[ticker] = data  # Use original data
        else:
            processed_data = validated_data
            for data in processed_data.values():
                self.stats['total_data_points'] += len(data)
        
        # Step 4: Save processed data
        if processed_data:
            self.logger.info(f"Step 4: Saving {len(processed_data)} datasets...")
            await self._save_datasets(processed_data, output_format)
        
        # Log final statistics
        self._log_collection_summary(validation_results if enable_validation else {})
        
        return processed_data
    
    async def _save_datasets(self, datasets: Dict[str, pd.DataFrame], 
                           output_format: str) -> None:
        """
        Save datasets using the configured storage backend.
        
        Args:
            datasets: Dictionary mapping ticker to DataFrame
            output_format: Output format (parquet, csv, hdf5)
        """
        try:
            # Use storage component to save data
            for ticker, data in datasets.items():
                await self.storage.save_raw_data(ticker, data, format=output_format)
                self.logger.debug(f"Saved {len(data)} records for {ticker} in {output_format} format")
                
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise
    
    def _log_collection_summary(self, validation_results: Dict) -> None:
        """
        Log comprehensive collection summary.
        
        Args:
            validation_results: Validation results dictionary
        """
        self.logger.info("=== Data Collection Summary ===")
        self.logger.info(f"Tickers requested: {self.stats['tickers_requested']}")
        self.logger.info(f"Successful downloads: {self.stats['successful_downloads']}")
        self.logger.info(f"Failed downloads: {self.stats['failed_downloads']}")
        self.logger.info(f"Validation failures: {self.stats['validation_failures']}")
        self.logger.info(f"Total data points: {self.stats['total_data_points']}")
        
        if self.stats['feature_engineering_time'] > 0:
            self.logger.info(f"Feature engineering time: {self.stats['feature_engineering_time']:.2f}s")
        
        # Log validation report if available
        if validation_results:
            validation_report = create_validation_report(validation_results)
            self.logger.info("\n" + validation_report)
    
    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        stats = self.stats.copy()
        
        # Add collector statistics
        for name, collector in self.collectors.items():
            stats[f'{name}_collector'] = collector.get_statistics()
        
        # Add validator statistics
        stats['validator'] = self.validator.get_statistics()
        
        return stats

def setup_logging(level: str) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level string
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/data_download.log', mode='a')
        ]
    )
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)


def calculate_date_range(years: int) -> tuple[date, date]:
    """
    Calculate start and end dates for data collection.
    
    Args:
        years: Number of years of historical data
        
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365 + 30)  # Add buffer for weekends/holidays
    return start_date, end_date


def print_header(tickers: List[str], years: int, output_dir: Path, 
                features: bool, validation: bool) -> None:
    """
    Print formatted header with configuration details.
    
    Args:
        tickers: List of ticker symbols
        years: Years of data to download
        output_dir: Output directory path
        features: Whether feature engineering is enabled
        validation: Whether validation is enabled
    """
    print("\n" + "=" * 80)
    print("    TIME-SERIES TRANSFORMER - HISTORICAL DATA DOWNLOAD")
    print("=" * 80)
    print(f"Tickers:           {len(tickers)} symbols ({', '.join(tickers[:5])}{', ...' if len(tickers) > 5 else ''})")
    print(f"Historical Period: {years} years")
    print(f"Output Directory:  {output_dir}")
    print(f"Feature Engineering: {'Enabled' if features else 'Disabled'}")
    print(f"Data Validation:   {'Enabled' if validation else 'Disabled'}")
    print("=" * 80 + "\n")


async def main() -> None:
    """
    Main entry point for historical data download script.
    
    Implements the complete data collection pipeline following the
    DataCollectionOrchestrator pattern from design specifications.
    """
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        # Parse tickers
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        
        # Calculate date range
        start_date, end_date = calculate_date_range(args.years)
        
        # Print configuration header
        print_header(tickers, args.years, args.output_dir, args.features, args.validate)
        
        # Create output directories
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collection orchestrator
        logger.info("Initializing DataCollectionOrchestrator...")
        orchestrator = DataCollectionOrchestrator()
        
        # Execute data collection pipeline
        logger.info("Starting data collection pipeline...")
        start_time = datetime.now()
        
        collected_data = await orchestrator.collect_all(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            enable_features=args.features,
            enable_validation=args.validate,
            output_format=args.format
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        print("\n" + "=" * 80)
        print("                        DOWNLOAD COMPLETE")
        print("=" * 80)
        
        stats = orchestrator.get_statistics()
        print(f"[SUCCESS] Successful downloads: {stats['successful_downloads']}/{stats['tickers_requested']}")
        
        if stats['failed_downloads'] > 0:
            print(f"[FAILED] Failed downloads: {stats['failed_downloads']}")
        
        if stats['validation_failures'] > 0:
            print(f"[WARNING] Validation failures: {stats['validation_failures']}")
        
        print(f"[DATA] Total data points: {stats['total_data_points']:,}")
        print(f"[TIME] Total time: {total_time:.2f} seconds")
        print(f"[OUTPUT] Data saved to: {args.output_dir.absolute()}")
        
        # Log detailed statistics
        logger.info(f"Collection completed in {total_time:.2f} seconds")
        logger.info(f"Final statistics: {stats}")
        
        if stats['successful_downloads'] == 0:
            logger.error("No data was successfully downloaded")
            sys.exit(1)
        
        print("\n[SUCCESS] Historical data download completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        print("\n[INTERRUPTED] Download interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Download failed with error: {e}", exc_info=True)
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())