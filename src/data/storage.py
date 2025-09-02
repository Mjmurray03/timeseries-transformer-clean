"""
Data Storage Module for Time-Series Transformer

This module implements efficient storage management for raw and processed data
following the data pipeline standards with support for Parquet, HDF5, and metadata tracking.
"""

import hashlib
import logging
import os
import pickle
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Optimized data storage management for time-series data.

    Supports multiple storage formats:
    - Parquet for raw data (columnar, compressed)
    - HDF5 for processed data (large datasets, fast access)
    - SQLite for metadata tracking
    """

    def __init__(self, config=None, base_path: str = None):
        """
        Initialize DataStorage with configuration and optional base path.

        Args:
            config: Data configuration object with storage settings
            base_path: Base directory for data storage (optional)
        """
        # Extract base path from config if available
        if hasattr(config, "raw_config") and base_path is None:
            base_path = config.raw_config.get("storage", {}).get("base_path", "data")
        elif base_path is None:
            base_path = "data"

        self.base_path = Path(base_path)
        self.config = self._get_storage_config(config) or self._get_default_config()

        # Create directory structure
        self._create_directory_structure()

        # Initialize metadata database
        self._init_metadata_db()

        logger.info(f"DataStorage initialized with base path: {self.base_path}")

    def _get_storage_config(self, config) -> Optional[Dict]:
        """Extract storage configuration from data config object."""
        if hasattr(config, "storage"):
            return {
                "raw_data": config.storage.raw_data,
                "processed_data": config.storage.processed_data,
                "metadata": config.storage.metadata,
            }
        return None

    def _get_default_config(self) -> Dict:
        """Get default storage configuration."""
        return {
            "raw_data": {"format": "parquet", "compression": "snappy", "partition_by": "ticker"},
            "processed_data": {"format": "hdf5", "compression": True},
            "metadata": {"database_type": "sqlite", "backup_enabled": True},
        }

    def _create_directory_structure(self):
        """Create the required directory structure."""
        directories = [
            self.base_path / "raw",
            self.base_path / "processed",
            self.base_path / "metadata",
            self.base_path / "cache",
            self.base_path / "backups",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def _init_metadata_db(self):
        """Initialize SQLite metadata database."""
        self.metadata_db_path = self.base_path / "metadata" / "data_catalog.db"

        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.cursor()

            # Create tables for metadata tracking
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_data_catalog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date_range_start DATE NOT NULL,
                    date_range_end DATE NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    row_count INTEGER,
                    columns TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum TEXT,
                    UNIQUE(ticker, date_range_start, date_range_end)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_data_catalog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    feature_set TEXT NOT NULL,
                    date_range_start DATE NOT NULL,
                    date_range_end DATE NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    row_count INTEGER,
                    feature_count INTEGER,
                    processing_config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum TEXT,
                    UNIQUE(ticker, feature_set, date_range_start, date_range_end)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS data_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT UNIQUE NOT NULL,
                    data_type TEXT NOT NULL,
                    ticker TEXT,
                    description TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            logger.debug("Metadata database initialized")

    async def save_raw_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        format: str = "parquet",
        date_range: Optional[Tuple[date, date]] = None,
        compress: bool = True,
    ) -> str:
        """
        Async wrapper for saving raw data.

        Args:
            ticker: Stock ticker symbol
            data: DataFrame with OHLCV data
            format: Output format (parquet, csv, hdf5)
            date_range: Optional date range tuple (start, end)
            compress: Whether to compress the data

        Returns:
            Path to saved file
        """
        return self._save_raw_data_sync(data, ticker, date_range, compress, format)

    def _save_raw_data_sync(
        self,
        data: pd.DataFrame,
        ticker: str,
        date_range: Optional[Tuple[date, date]] = None,
        compress: bool = True,
        format: str = "parquet",
    ) -> str:
        """
        Save raw OHLCV data in Parquet format.

        Args:
            data: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            date_range: Optional date range tuple (start, end)
            compress: Whether to compress the data

        Returns:
            Path to saved file
        """
        if date_range is None:
            date_range = (data.index.min().date(), data.index.max().date())

        start_date, end_date = date_range

        # Create ticker directory
        ticker_dir = self.base_path / "raw" / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on format and date range
        if format == "parquet":
            filename = f"{start_date}_{end_date}.parquet"
            file_path = ticker_dir / filename
            compression = self.config["raw_data"]["compression"] if compress else None
            data.to_parquet(file_path, compression=compression)
        elif format == "csv":
            filename = f"{start_date}_{end_date}.csv"
            file_path = ticker_dir / filename
            compression = "gzip" if compress else None
            data.to_csv(file_path, compression=compression)
        elif format == "hdf5":
            filename = f"{start_date}_{end_date}.h5"
            file_path = ticker_dir / filename
            try:
                complevel = 9 if compress else 0
                data.to_hdf(file_path, key="data", mode="w", complevel=complevel)
            except ImportError:
                # Fallback to parquet if HDF5 not available
                logger.warning("HDF5 not available, falling back to parquet")
                filename = f"{start_date}_{end_date}.parquet"
                file_path = ticker_dir / filename
                compression = self.config["raw_data"]["compression"] if compress else None
                data.to_parquet(file_path, compression=compression)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save data using the determined method above

        # Calculate checksum
        checksum = self._calculate_file_checksum(file_path)

        # Update metadata
        self._update_raw_data_metadata(
            ticker=ticker,
            date_range_start=start_date,
            date_range_end=end_date,
            file_path=str(file_path.relative_to(self.base_path)),
            file_size=file_path.stat().st_size,
            row_count=len(data),
            columns=list(data.columns),
            checksum=checksum,
        )

        logger.info(f"Saved raw data for {ticker} ({start_date} to {end_date}): {file_path}")
        return str(file_path)

    def load_raw_data(
        self, ticker: str, date_range: Optional[Tuple[date, date]] = None
    ) -> pd.DataFrame:
        """
        Load raw OHLCV data from Parquet files.

        Args:
            ticker: Stock ticker symbol
            date_range: Optional date range tuple (start, end)

        Returns:
            DataFrame with OHLCV data
        """
        if date_range is None:
            # Load all available data for ticker
            return self._load_all_raw_data(ticker)

        start_date, end_date = date_range
        filename = f"{start_date}_{end_date}.parquet"
        file_path = self.base_path / "raw" / ticker / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        data = pd.read_parquet(file_path)
        logger.debug(f"Loaded raw data for {ticker} ({start_date} to {end_date}): {len(data)} rows")

        return data

    def save_processed_data(
        self,
        data: pd.DataFrame,
        ticker: str,
        feature_set: str = "default",
        processing_config: Optional[Dict] = None,
        compress: bool = True,
    ) -> str:
        """
        Save processed/engineered features in HDF5 format (fallback to Parquet if HDF5 unavailable).

        Args:
            data: DataFrame with engineered features
            ticker: Stock ticker symbol
            feature_set: Name of the feature set (e.g., "technical_indicators")
            processing_config: Configuration used for processing
            compress: Whether to compress the data

        Returns:
            Path to saved file
        """
        date_range = (data.index.min().date(), data.index.max().date())
        start_date, end_date = date_range

        # Create ticker directory
        ticker_dir = self.base_path / "processed" / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        # Try HDF5 first, fallback to Parquet if not available
        try:
            # Generate filename for HDF5
            filename = f"{feature_set}_{start_date}_{end_date}.h5"
            file_path = ticker_dir / filename

            # Save data in HDF5 format
            complevel = 9 if compress else 0
            data.to_hdf(file_path, key="data", mode="w", complevel=complevel)

        except ImportError:
            # Fallback to Parquet if HDF5 (pytables) is not available
            logger.warning("HDF5 support not available, using Parquet format for processed data")
            filename = f"{feature_set}_{start_date}_{end_date}.parquet"
            file_path = ticker_dir / filename

            # Save data in Parquet format
            compression = "snappy" if compress else None
            data.to_parquet(file_path, compression=compression)

        # Save processing config if provided
        if processing_config:
            config_path = ticker_dir / f"{feature_set}_{start_date}_{end_date}_config.pkl"
            with open(config_path, "wb") as f:
                pickle.dump(processing_config, f)

        # Calculate checksum
        checksum = self._calculate_file_checksum(file_path)

        # Update metadata
        self._update_processed_data_metadata(
            ticker=ticker,
            feature_set=feature_set,
            date_range_start=start_date,
            date_range_end=end_date,
            file_path=str(file_path.relative_to(self.base_path)),
            file_size=file_path.stat().st_size,
            row_count=len(data),
            feature_count=len(data.columns),
            processing_config=str(processing_config) if processing_config else None,
            checksum=checksum,
        )

        logger.info(f"Saved processed data for {ticker} ({feature_set}): {file_path}")
        return str(file_path)

    def load_processed_data(
        self,
        ticker: str,
        feature_set: str = "default",
        date_range: Optional[Tuple[date, date]] = None,
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Load processed/engineered features from HDF5 or Parquet files.

        Args:
            ticker: Stock ticker symbol
            feature_set: Name of the feature set
            date_range: Optional date range tuple (start, end)

        Returns:
            Tuple of (DataFrame with features, processing config)
        """
        if date_range is None:
            # Find the most recent processed data (try both HDF5 and Parquet)
            ticker_dir = self.base_path / "processed" / ticker
            h5_files = list(ticker_dir.glob(f"{feature_set}_*.h5"))
            parquet_files = list(ticker_dir.glob(f"{feature_set}_*.parquet"))

            all_files = h5_files + parquet_files
            if not all_files:
                raise FileNotFoundError(
                    f"No processed data found for {ticker} with feature set {feature_set}"
                )

            # Sort by modification time and get the most recent
            file_path = max(all_files, key=lambda p: p.stat().st_mtime)
        else:
            start_date, end_date = date_range
            ticker_dir = self.base_path / "processed" / ticker

            # Try HDF5 first, then Parquet
            h5_filename = f"{feature_set}_{start_date}_{end_date}.h5"
            parquet_filename = f"{feature_set}_{start_date}_{end_date}.parquet"

            h5_path = ticker_dir / h5_filename
            parquet_path = ticker_dir / parquet_filename

            if h5_path.exists():
                file_path = h5_path
            elif parquet_path.exists():
                file_path = parquet_path
            else:
                raise FileNotFoundError(
                    f"Processed data file not found for {ticker} with feature set {feature_set}"
                )

        # Load data based on file extension
        if file_path.suffix == ".h5":
            try:
                data = pd.read_hdf(file_path, key="data")
            except ImportError:
                raise ImportError(
                    "HDF5 support not available. Please install pytables: pip install tables"
                )
        elif file_path.suffix == ".parquet":
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Load processing config if available
        config_path = file_path.with_suffix(".pkl").with_name(file_path.stem + "_config.pkl")
        processing_config = None
        if config_path.exists():
            with open(config_path, "rb") as f:
                processing_config = pickle.load(f)

        logger.debug(
            f"Loaded processed data for {ticker} ({feature_set}): {len(data)} rows, {len(data.columns)} features"
        )

        return data, processing_config

    def save_timeseries(self, data: pd.DataFrame, path: str, compress: bool = True):
        """
        Save time-series data efficiently based on file extension.

        Args:
            data: DataFrame to save
            path: File path with extension
            compress: Whether to compress the data
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".parquet":
            # Parquet for columnar storage
            compression = "snappy" if compress else None
            data.to_parquet(path, compression=compression)

        elif path.suffix == ".h5":
            # HDF5 for large datasets
            try:
                complevel = 9 if compress else 0
                data.to_hdf(path, key="data", mode="w", complevel=complevel)
            except ImportError:
                logger.warning("HDF5 support not available, falling back to Parquet")
                # Change extension and save as Parquet
                parquet_path = path.with_suffix(".parquet")
                compression = "snappy" if compress else None
                data.to_parquet(parquet_path, compression=compression)
                # Update the path to reflect the actual saved file
                path = parquet_path

        elif path.suffix == ".feather":
            # Feather for fast I/O
            compression = "lz4" if compress else None
            data.to_feather(path, compression=compression)

        else:
            # CSV as fallback
            compression = "gzip" if compress else None
            data.to_csv(path, compression=compression)

        logger.debug(f"Saved time-series data: {path} ({len(data)} rows)")

    def load_timeseries(self, path: str) -> pd.DataFrame:
        """
        Load time-series data with appropriate method based on file extension.

        Args:
            path: File path

        Returns:
            DataFrame with time-series data
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".h5":
            try:
                return pd.read_hdf(path, key="data")
            except ImportError:
                raise ImportError(
                    "HDF5 support not available. Please install pytables: pip install tables"
                )
        elif path.suffix == ".feather":
            return pd.read_feather(path)
        else:
            # Try to read as CSV with date parsing
            try:
                return pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            except:
                return pd.read_csv(path, index_col=0, parse_dates=True)

    def create_data_version(
        self,
        data: pd.DataFrame,
        data_type: str,
        ticker: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a versioned snapshot of data.

        Args:
            data: DataFrame to version
            data_type: Type of data ('raw', 'processed', etc.)
            ticker: Optional ticker symbol
            description: Optional description
            metadata: Optional metadata dictionary

        Returns:
            Version ID
        """
        # Generate version hash
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(data).values).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}_{data_hash}"

        # Save versioned data
        version_dir = self.base_path / "versions"
        version_dir.mkdir(parents=True, exist_ok=True)
        version_path = version_dir / f"{version_id}.parquet"

        data.to_parquet(version_path, compression="snappy")

        # Store version metadata in database
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO data_versions 
                (version_id, data_type, ticker, description, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (version_id, data_type, ticker, description, str(metadata) if metadata else None),
            )
            conn.commit()

        logger.info(f"Created data version: {version_id}")
        return version_id

    def load_data_version(self, version_id: str) -> pd.DataFrame:
        """
        Load specific data version.

        Args:
            version_id: Version identifier

        Returns:
            DataFrame with versioned data
        """
        version_path = self.base_path / "versions" / f"{version_id}.parquet"

        if not version_path.exists():
            raise FileNotFoundError(f"Version not found: {version_id}")

        return pd.read_parquet(version_path)

    def get_data_catalog(self, data_type: str = "raw") -> pd.DataFrame:
        """
        Get catalog of available data.

        Args:
            data_type: Type of data ('raw' or 'processed')

        Returns:
            DataFrame with catalog information
        """
        table_name = f"{data_type}_data_catalog"

        with sqlite3.connect(self.metadata_db_path) as conn:
            query = f"SELECT * FROM {table_name} ORDER BY updated_at DESC"
            catalog = pd.read_sql_query(query, conn)

        return catalog

    def cleanup_old_data(self, days_old: int = 30, data_type: str = "processed"):
        """
        Clean up old data files.

        Args:
            days_old: Remove files older than this many days
            data_type: Type of data to clean ('raw', 'processed', 'versions')
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=days_old)

        data_dir = self.base_path / data_type
        if not data_dir.exists():
            return

        removed_count = 0
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    removed_count += 1

        logger.info(f"Cleaned up {removed_count} old {data_type} files")

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        stats = {}

        for data_type in ["raw", "processed", "metadata", "versions"]:
            data_dir = self.base_path / data_type
            if data_dir.exists():
                total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
                file_count = len(list(data_dir.rglob("*")))

                stats[data_type] = {
                    "total_size_mb": total_size / (1024 * 1024),
                    "file_count": file_count,
                }

        return stats

    def close(self):
        """Close any open database connections."""
        # SQLite connections are automatically closed when the context manager exits
        # This method is provided for explicit cleanup if needed
        pass

    def _load_all_raw_data(self, ticker: str) -> pd.DataFrame:
        """Load all available raw data for a ticker."""
        ticker_dir = self.base_path / "raw" / ticker

        if not ticker_dir.exists():
            raise FileNotFoundError(f"No raw data found for ticker: {ticker}")

        parquet_files = list(ticker_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for ticker: {ticker}")

        # Load and concatenate all files
        dataframes = []
        for file_path in sorted(parquet_files):
            df = pd.read_parquet(file_path)
            dataframes.append(df)

        combined_data = pd.concat(dataframes, axis=0)
        combined_data = combined_data.sort_index()

        # Remove duplicates if any
        combined_data = combined_data[~combined_data.index.duplicated(keep="first")]

        logger.debug(
            f"Loaded all raw data for {ticker}: {len(combined_data)} rows from {len(parquet_files)} files"
        )

        return combined_data

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _update_raw_data_metadata(self, **kwargs):
        """Update raw data metadata in database."""
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.cursor()

            # Try to update existing record first
            cursor.execute(
                """
                UPDATE raw_data_catalog 
                SET file_path=?, file_size=?, row_count=?, columns=?, 
                    updated_at=CURRENT_TIMESTAMP, checksum=?
                WHERE ticker=? AND date_range_start=? AND date_range_end=?
            """,
                (
                    kwargs["file_path"],
                    kwargs["file_size"],
                    kwargs["row_count"],
                    ",".join(kwargs["columns"]),
                    kwargs["checksum"],
                    kwargs["ticker"],
                    kwargs["date_range_start"],
                    kwargs["date_range_end"],
                ),
            )

            # If no rows were updated, insert new record
            if cursor.rowcount == 0:
                cursor.execute(
                    """
                    INSERT INTO raw_data_catalog 
                    (ticker, date_range_start, date_range_end, file_path, 
                     file_size, row_count, columns, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        kwargs["ticker"],
                        kwargs["date_range_start"],
                        kwargs["date_range_end"],
                        kwargs["file_path"],
                        kwargs["file_size"],
                        kwargs["row_count"],
                        ",".join(kwargs["columns"]),
                        kwargs["checksum"],
                    ),
                )

            conn.commit()

    def _update_processed_data_metadata(self, **kwargs):
        """Update processed data metadata in database."""
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.cursor()

            # Try to update existing record first
            cursor.execute(
                """
                UPDATE processed_data_catalog 
                SET file_path=?, file_size=?, row_count=?, feature_count=?,
                    processing_config=?, updated_at=CURRENT_TIMESTAMP, checksum=?
                WHERE ticker=? AND feature_set=? AND date_range_start=? AND date_range_end=?
            """,
                (
                    kwargs["file_path"],
                    kwargs["file_size"],
                    kwargs["row_count"],
                    kwargs["feature_count"],
                    kwargs["processing_config"],
                    kwargs["checksum"],
                    kwargs["ticker"],
                    kwargs["feature_set"],
                    kwargs["date_range_start"],
                    kwargs["date_range_end"],
                ),
            )

            # If no rows were updated, insert new record
            if cursor.rowcount == 0:
                cursor.execute(
                    """
                    INSERT INTO processed_data_catalog 
                    (ticker, feature_set, date_range_start, date_range_end, 
                     file_path, file_size, row_count, feature_count, 
                     processing_config, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        kwargs["ticker"],
                        kwargs["feature_set"],
                        kwargs["date_range_start"],
                        kwargs["date_range_end"],
                        kwargs["file_path"],
                        kwargs["file_size"],
                        kwargs["row_count"],
                        kwargs["feature_count"],
                        kwargs["processing_config"],
                        kwargs["checksum"],
                    ),
                )

            conn.commit()
