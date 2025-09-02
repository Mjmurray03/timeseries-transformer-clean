"""
Yahoo Finance data collector implementation.
Handles downloading stock data from Yahoo Finance with proper error handling and retries.
"""

import asyncio
import logging
import time
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

from src.config.data_config import DataConfig, DataSourceConfig
from src.data.rate_limiter import RateLimiter
from src.data.validators import DataValidator

logger = logging.getLogger(__name__)


class YahooFinanceError(Exception):
    """Custom exception for Yahoo Finance data collection errors."""
    pass


# RateLimiter and DataValidator are now imported from separate modules


class YahooFinanceCollector:
    """
    Yahoo Finance data collector with rate limiting, retries, and validation.
    Follows the DataCollectionOrchestrator pattern from design.md.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize Yahoo Finance collector.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.source_config = config.get_data_source_config('yahoo_finance')
        
        if not self.source_config:
            raise YahooFinanceError("Yahoo Finance configuration not found")
        
        if not self.source_config.enabled:
            raise YahooFinanceError("Yahoo Finance collector is disabled")
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rate=self.source_config.rate_limit,
            period=1  # per second
        )
        
        # Initialize validator
        self.validator = DataValidator(config)
        
        # Track statistics
        self.stats = {
            'requests_made': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'validation_failures': 0,
            'total_retries': 0
        }
        
        logger.info(f"YahooFinanceCollector initialized with rate limit: {self.source_config.rate_limit}/sec")
    
    async def collect_ticker(self, ticker: str, start_date: Optional[date] = None, 
                           end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        Collect data for a single ticker with error handling and validation.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with stock data or None if collection failed
        """
        logger.info(f"Starting data collection for {ticker}")
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Download data with retries
            data = await self._download_with_retry(ticker, start_date, end_date)
            
            if data is None:
                logger.error(f"Failed to download data for {ticker}")
                self.stats['failed_downloads'] += 1
                return None
            
            # Validate data
            validation_result = self.validator.validate(data, ticker)
            
            if not validation_result.is_valid:
                logger.error(f"Data validation failed for {ticker}: {[str(issue) for issue in validation_result.issues]}")
                self.stats['validation_failures'] += 1
                return None
            
            # Clean and prepare data
            cleaned_data = self._clean_data(data, ticker)
            
            self.stats['successful_downloads'] += 1
            logger.info(f"Successfully collected {len(cleaned_data)} days of data for {ticker}")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Unexpected error collecting data for {ticker}: {e}")
            self.stats['failed_downloads'] += 1
            return None
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, YahooFinanceError))
    )
    async def _download_with_retry(self, ticker: str, start_date: Optional[date], 
                                 end_date: Optional[date]) -> Optional[pd.DataFrame]:
        """
        Download data with exponential backoff retry logic.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Raw DataFrame from yfinance or None if failed
        """
        try:
            self.stats['requests_made'] += 1
            
            # Convert dates to strings for yfinance
            start_str = start_date.strftime('%Y-%m-%d') if start_date else None
            end_str = end_date.strftime('%Y-%m-%d') if end_date else None
            
            logger.debug(f"Downloading {ticker} from {start_str} to {end_str}")
            
            # Use asyncio to run the synchronous yfinance call
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    ticker,
                    start=start_str,
                    end=end_str,
                    progress=False,
                    timeout=self.source_config.timeout,
                    auto_adjust=True
                )
            )
            
            if data.empty:
                raise YahooFinanceError(f"No data returned for {ticker}")
            
            # Flatten MultiIndex columns if present (happens with single ticker downloads)
            if isinstance(data.columns, pd.MultiIndex):
                # For single ticker, just use the first level (price names)
                data.columns = data.columns.get_level_values(0)
            
            return data
            
        except Exception as e:
            self.stats['total_retries'] += 1
            logger.warning(f"Download attempt failed for {ticker}: {e}")
            raise YahooFinanceError(f"Download failed for {ticker}: {e}")
    
    def _clean_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Clean and prepare downloaded data.
        
        Args:
            data: Raw data from yfinance
            ticker: Ticker symbol
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned = data.copy()
        
        # Ensure index is datetime
        if not isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = pd.to_datetime(cleaned.index)
        
        # Sort by date
        cleaned = cleaned.sort_index()
        
        # Remove any duplicate dates
        cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
        
        # Handle missing values
        cleaned = self._handle_missing_values(cleaned)
        
        # Add ticker column
        cleaned['Ticker'] = ticker
        
        # Round prices to reasonable precision
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].round(4)
        
        # Ensure volume is integer
        if 'Volume' in cleaned.columns:
            cleaned['Volume'] = cleaned['Volume'].astype('int64')
        
        logger.debug(f"Cleaned data for {ticker}: {len(cleaned)} rows, {len(cleaned.columns)} columns")
        
        return cleaned
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Forward fill missing values (use previous day's value)
        data = data.ffill()
        
        # If there are still missing values at the beginning, backward fill
        data = data.bfill()
        
        # Drop any rows that still have missing values
        initial_rows = len(data)
        data = data.dropna()
        
        if len(data) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(data)} rows with missing values")
        
        return data
    
    async def collect_multiple(self, tickers: List[str], start_date: Optional[date] = None,
                             end_date: Optional[date] = None, 
                             max_concurrent: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple tickers concurrently.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info(f"Starting concurrent collection for {len(tickers)} tickers")
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def collect_with_semaphore(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
            async with semaphore:
                data = await self.collect_ticker(ticker, start_date, end_date)
                return ticker, data
        
        # Create tasks for all tickers
        tasks = [collect_with_semaphore(ticker) for ticker in tickers]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        collected_data = {}
        successful = 0
        failed = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                failed += 1
                continue
            
            ticker, data = result
            if data is not None:
                collected_data[ticker] = data
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Collection completed: {successful} successful, {failed} failed")
        
        return collected_data
    
    def get_statistics(self) -> Dict[str, int]:
        """Get collection statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset collection statistics."""
        self.stats = {
            'requests_made': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'validation_failures': 0,
            'total_retries': 0
        }
        logger.info("Statistics reset")


# Convenience functions for common use cases
async def download_ticker(ticker: str, config: DataConfig, 
                         date_range: str = "full_history") -> Optional[pd.DataFrame]:
    """
    Convenience function to download a single ticker.
    
    Args:
        ticker: Ticker symbol
        config: Data configuration
        date_range: Name of date range configuration
        
    Returns:
        DataFrame with stock data or None if failed
    """
    collector = YahooFinanceCollector(config)
    
    # Get date range
    range_config = config.get_date_range(date_range)
    if not range_config:
        logger.error(f"Date range '{date_range}' not found in configuration")
        return None
    
    start_date = range_config.get_start_date()
    end_date = range_config.get_end_date()
    
    return await collector.collect_ticker(ticker, start_date, end_date)


async def download_ticker_set(ticker_set: str, config: DataConfig,
                            date_range: str = "full_history") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to download a set of tickers.
    
    Args:
        ticker_set: Name of ticker set in configuration
        config: Data configuration
        date_range: Name of date range configuration
        
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    collector = YahooFinanceCollector(config)
    
    # Get tickers
    tickers = config.get_ticker_list(ticker_set)
    if not tickers:
        logger.error(f"Ticker set '{ticker_set}' not found or empty")
        return {}
    
    # Get date range
    range_config = config.get_date_range(date_range)
    if not range_config:
        logger.error(f"Date range '{date_range}' not found in configuration")
        return {}
    
    start_date = range_config.get_start_date()
    end_date = range_config.get_end_date()
    
    return await collector.collect_multiple(tickers, start_date, end_date)