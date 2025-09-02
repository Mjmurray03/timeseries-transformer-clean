"""
Unit tests for DataStorage class

Tests all storage functionality including Parquet, HDF5, metadata tracking,
and data versioning following the testing standards.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from src.data.storage import DataStorage


class TestDataStorage:
    """Test suite for DataStorage"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_dir):
        """Create DataStorage instance for testing"""
        return DataStorage(base_path=temp_dir)
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 100),
            'High': np.random.uniform(100, 110, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(95, 105, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    @pytest.fixture
    def sample_processed_data(self, sample_ohlcv_data):
        """Generate sample processed data with features"""
        data = sample_ohlcv_data.copy()
        
        # Add some engineered features
        data['Returns'] = data['Close'].pct_change()
        data['RSI'] = np.random.uniform(20, 80, len(data))
        data['MACD'] = np.random.uniform(-2, 2, len(data))
        data['Volume_Ratio'] = np.random.uniform(0.5, 2.0, len(data))
        
        return data
    
    def test_initialization(self, temp_dir):
        """Test DataStorage initializes correctly"""
        storage = DataStorage(base_path=temp_dir)
        
        assert storage.base_path == Path(temp_dir)
        assert storage.config is not None
        
        # Check directory structure was created
        expected_dirs = ['raw', 'processed', 'metadata', 'cache', 'backups']
        for dir_name in expected_dirs:
            assert (Path(temp_dir) / dir_name).exists()
        
        # Check metadata database was created
        assert (Path(temp_dir) / "metadata" / "data_catalog.db").exists()
    
    def test_initialization_with_custom_config(self, temp_dir):
        """Test initialization with custom configuration"""
        custom_config = {
            'raw_data': {
                'format': 'parquet',
                'compression': 'gzip',
                'partition_by': 'date'
            },
            'processed_data': {
                'format': 'hdf5',
                'compression': False
            }
        }
        
        storage = DataStorage(base_path=temp_dir, config=custom_config)
        assert storage.config['raw_data']['compression'] == 'gzip'
        assert storage.config['processed_data']['compression'] is False
    
    def test_save_raw_data_parquet(self, storage, sample_ohlcv_data):
        """Test saving raw data in Parquet format"""
        ticker = "AAPL"
        file_path = storage.save_raw_data(sample_ohlcv_data, ticker)
        
        # Check file was created
        assert Path(file_path).exists()
        assert file_path.endswith('.parquet')
        
        # Check file can be loaded
        loaded_data = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(loaded_data, sample_ohlcv_data)
        
        # Check metadata was recorded
        catalog = storage.get_data_catalog('raw')
        assert len(catalog) == 1
        assert catalog.iloc[0]['ticker'] == ticker
        assert catalog.iloc[0]['row_count'] == len(sample_ohlcv_data)
    
    def test_load_raw_data(self, storage, sample_ohlcv_data):
        """Test loading raw data"""
        ticker = "AAPL"
        
        # Save data first
        storage.save_raw_data(sample_ohlcv_data, ticker)
        
        # Load data
        date_range = (sample_ohlcv_data.index.min().date(), sample_ohlcv_data.index.max().date())
        loaded_data = storage.load_raw_data(ticker, date_range)
        
        pd.testing.assert_frame_equal(loaded_data, sample_ohlcv_data)
    
    def test_load_raw_data_all(self, storage, sample_ohlcv_data):
        """Test loading all raw data for a ticker"""
        ticker = "AAPL"
        
        # Split data into two parts and save separately
        mid_point = len(sample_ohlcv_data) // 2
        data1 = sample_ohlcv_data.iloc[:mid_point]
        data2 = sample_ohlcv_data.iloc[mid_point:]
        
        storage.save_raw_data(data1, ticker)
        storage.save_raw_data(data2, ticker)
        
        # Load all data
        loaded_data = storage.load_raw_data(ticker)
        
        # Should get combined data
        expected_data = pd.concat([data1, data2]).sort_index()
        pd.testing.assert_frame_equal(loaded_data, expected_data)
    
    def test_save_processed_data_hdf5(self, storage, sample_processed_data):
        """Test saving processed data in HDF5 format"""
        ticker = "AAPL"
        feature_set = "technical_indicators"
        processing_config = {'rsi_period': 14, 'macd_fast': 12}
        
        file_path = storage.save_processed_data(
            sample_processed_data, 
            ticker, 
            feature_set,
            processing_config
        )
        
        # Check file was created
        assert Path(file_path).exists()
        assert file_path.endswith('.h5')
        
        # Check file can be loaded
        loaded_data = pd.read_hdf(file_path, key='data')
        pd.testing.assert_frame_equal(loaded_data, sample_processed_data)
        
        # Check metadata was recorded
        catalog = storage.get_data_catalog('processed')
        assert len(catalog) == 1
        assert catalog.iloc[0]['ticker'] == ticker
        assert catalog.iloc[0]['feature_set'] == feature_set
        assert catalog.iloc[0]['feature_count'] == len(sample_processed_data.columns)
    
    def test_load_processed_data(self, storage, sample_processed_data):
        """Test loading processed data"""
        ticker = "AAPL"
        feature_set = "technical_indicators"
        processing_config = {'rsi_period': 14}
        
        # Save data first
        storage.save_processed_data(
            sample_processed_data, 
            ticker, 
            feature_set,
            processing_config
        )
        
        # Load data
        date_range = (sample_processed_data.index.min().date(), sample_processed_data.index.max().date())
        loaded_data, loaded_config = storage.load_processed_data(ticker, feature_set, date_range)
        
        pd.testing.assert_frame_equal(loaded_data, sample_processed_data)
        assert loaded_config == processing_config
    
    def test_load_processed_data_most_recent(self, storage, sample_processed_data):
        """Test loading most recent processed data when no date range specified"""
        ticker = "AAPL"
        feature_set = "technical_indicators"
        
        # Save data
        storage.save_processed_data(sample_processed_data, ticker, feature_set)
        
        # Load without specifying date range (should get most recent)
        loaded_data, loaded_config = storage.load_processed_data(ticker, feature_set)
        
        pd.testing.assert_frame_equal(loaded_data, sample_processed_data)
    
    def test_save_timeseries_parquet(self, storage, sample_ohlcv_data):
        """Test generic timeseries saving with Parquet format"""
        file_path = "test_data.parquet"
        full_path = storage.base_path / file_path
        
        storage.save_timeseries(sample_ohlcv_data, str(full_path))
        
        assert full_path.exists()
        loaded_data = pd.read_parquet(full_path)
        pd.testing.assert_frame_equal(loaded_data, sample_ohlcv_data)
    
    def test_save_timeseries_hdf5(self, storage, sample_processed_data):
        """Test generic timeseries saving with HDF5 format"""
        file_path = "test_data.h5"
        full_path = storage.base_path / file_path
        
        storage.save_timeseries(sample_processed_data, str(full_path))
        
        assert full_path.exists()
        loaded_data = pd.read_hdf(full_path, key='data')
        pd.testing.assert_frame_equal(loaded_data, sample_processed_data)
    
    def test_save_timeseries_csv(self, storage, sample_ohlcv_data):
        """Test generic timeseries saving with CSV format"""
        file_path = "test_data.csv"
        full_path = storage.base_path / file_path
        
        storage.save_timeseries(sample_ohlcv_data, str(full_path))
        
        assert full_path.exists()
        # CSV loading might have slight differences due to precision
        loaded_data = storage.load_timeseries(str(full_path))
        assert len(loaded_data) == len(sample_ohlcv_data)
        assert list(loaded_data.columns) == list(sample_ohlcv_data.columns)
    
    def test_load_timeseries_file_not_found(self, storage):
        """Test loading non-existent file raises appropriate error"""
        with pytest.raises(FileNotFoundError):
            storage.load_timeseries("non_existent_file.parquet")
    
    def test_create_data_version(self, storage, sample_ohlcv_data):
        """Test data versioning functionality"""
        version_id = storage.create_data_version(
            sample_ohlcv_data,
            data_type="raw",
            ticker="AAPL",
            description="Test version",
            metadata={"source": "test"}
        )
        
        assert version_id.startswith("v_")
        
        # Check version file was created
        version_path = storage.base_path / "versions" / f"{version_id}.parquet"
        assert version_path.exists()
        
        # Check version can be loaded
        loaded_data = storage.load_data_version(version_id)
        pd.testing.assert_frame_equal(loaded_data, sample_ohlcv_data)
    
    def test_load_data_version_not_found(self, storage):
        """Test loading non-existent version raises appropriate error"""
        with pytest.raises(FileNotFoundError):
            storage.load_data_version("non_existent_version")
    
    def test_get_data_catalog_raw(self, storage, sample_ohlcv_data):
        """Test getting raw data catalog"""
        # Save some data
        storage.save_raw_data(sample_ohlcv_data, "AAPL")
        storage.save_raw_data(sample_ohlcv_data, "MSFT")
        
        catalog = storage.get_data_catalog("raw")
        
        assert len(catalog) == 2
        assert "AAPL" in catalog['ticker'].values
        assert "MSFT" in catalog['ticker'].values
        assert all(catalog['row_count'] == len(sample_ohlcv_data))
    
    def test_get_data_catalog_processed(self, storage, sample_processed_data):
        """Test getting processed data catalog"""
        # Save some data
        storage.save_processed_data(sample_processed_data, "AAPL", "features1")
        storage.save_processed_data(sample_processed_data, "AAPL", "features2")
        
        catalog = storage.get_data_catalog("processed")
        
        assert len(catalog) == 2
        assert all(catalog['ticker'] == "AAPL")
        assert "features1" in catalog['feature_set'].values
        assert "features2" in catalog['feature_set'].values
    
    def test_get_storage_stats(self, storage, sample_ohlcv_data, sample_processed_data):
        """Test getting storage statistics"""
        # Save some data
        storage.save_raw_data(sample_ohlcv_data, "AAPL")
        storage.save_processed_data(sample_processed_data, "AAPL", "features")
        
        stats = storage.get_storage_stats()
        
        assert 'raw' in stats
        assert 'processed' in stats
        assert 'metadata' in stats
        
        assert stats['raw']['file_count'] > 0
        assert stats['raw']['total_size_mb'] > 0
        assert stats['processed']['file_count'] > 0
        assert stats['processed']['total_size_mb'] > 0
    
    def test_cleanup_old_data(self, storage, sample_ohlcv_data):
        """Test cleanup of old data files"""
        # Save some data
        storage.save_processed_data(sample_ohlcv_data, "AAPL", "features")
        
        # Verify file exists
        processed_dir = storage.base_path / "processed"
        files_before = list(processed_dir.rglob("*"))
        assert len(files_before) > 0
        
        # Cleanup with 0 days (should remove everything)
        storage.cleanup_old_data(days_old=0, data_type="processed")
        
        # Check files were removed
        files_after = list(processed_dir.rglob("*"))
        # Should only have directories left
        files_after = [f for f in files_after if f.is_file()]
        assert len(files_after) == 0
    
    def test_file_checksum_calculation(self, storage, sample_ohlcv_data):
        """Test file checksum calculation"""
        # Save data
        file_path = storage.save_raw_data(sample_ohlcv_data, "AAPL")
        
        # Calculate checksum
        checksum1 = storage._calculate_file_checksum(Path(file_path))
        checksum2 = storage._calculate_file_checksum(Path(file_path))
        
        # Should be consistent
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex length
    
    def test_metadata_database_integrity(self, storage, sample_ohlcv_data, sample_processed_data):
        """Test metadata database integrity"""
        # Save data
        storage.save_raw_data(sample_ohlcv_data, "AAPL")
        storage.save_processed_data(sample_processed_data, "AAPL", "features")
        
        # Check database directly
        with sqlite3.connect(storage.metadata_db_path) as conn:
            cursor = conn.cursor()
            
            # Check raw data table
            cursor.execute("SELECT COUNT(*) FROM raw_data_catalog")
            raw_count = cursor.fetchone()[0]
            assert raw_count == 1
            
            # Check processed data table
            cursor.execute("SELECT COUNT(*) FROM processed_data_catalog")
            processed_count = cursor.fetchone()[0]
            assert processed_count == 1
            
            # Check data integrity
            cursor.execute("SELECT ticker, row_count FROM raw_data_catalog")
            raw_record = cursor.fetchone()
            assert raw_record[0] == "AAPL"
            assert raw_record[1] == len(sample_ohlcv_data)
    
    def test_duplicate_data_handling(self, storage, sample_ohlcv_data):
        """Test handling of duplicate data saves"""
        ticker = "AAPL"
        
        # Save same data twice
        file_path1 = storage.save_raw_data(sample_ohlcv_data, ticker)
        file_path2 = storage.save_raw_data(sample_ohlcv_data, ticker)
        
        # Should overwrite, not create duplicate
        assert file_path1 == file_path2
        
        # Should only have one record in catalog
        catalog = storage.get_data_catalog("raw")
        assert len(catalog) == 1
    
    def test_compression_options(self, storage, sample_ohlcv_data):
        """Test compression options for different formats"""
        # Test with compression
        file_path_compressed = storage.save_raw_data(sample_ohlcv_data, "AAPL", compress=True)
        
        # Test without compression
        file_path_uncompressed = storage.save_raw_data(sample_ohlcv_data, "MSFT", compress=False)
        
        # Both should work and be loadable
        data_compressed = storage.load_raw_data("AAPL")
        data_uncompressed = storage.load_raw_data("MSFT")
        
        pd.testing.assert_frame_equal(data_compressed, sample_ohlcv_data)
        pd.testing.assert_frame_equal(data_uncompressed, sample_ohlcv_data)
        
        # Compressed file should typically be smaller (though not guaranteed for small test data)
        compressed_size = Path(file_path_compressed).stat().st_size
        uncompressed_size = Path(file_path_uncompressed).stat().st_size
        
        # Both should be reasonable sizes
        assert compressed_size > 0
        assert uncompressed_size > 0
    
    def test_error_handling_invalid_ticker(self, storage):
        """Test error handling for invalid ticker"""
        with pytest.raises(FileNotFoundError):
            storage.load_raw_data("INVALID_TICKER")
    
    def test_error_handling_invalid_date_range(self, storage, sample_ohlcv_data):
        """Test error handling for invalid date range"""
        ticker = "AAPL"
        storage.save_raw_data(sample_ohlcv_data, ticker)
        
        # Try to load with non-existent date range
        invalid_date_range = (date(2020, 1, 1), date(2020, 1, 31))
        
        with pytest.raises(FileNotFoundError):
            storage.load_raw_data(ticker, invalid_date_range)
    
    @pytest.mark.parametrize("file_format", [".parquet", ".h5", ".feather", ".csv"])
    def test_save_load_timeseries_formats(self, storage, sample_ohlcv_data, file_format):
        """Test saving and loading with different file formats"""
        file_path = f"test_data{file_format}"
        full_path = storage.base_path / file_path
        
        # Save data
        storage.save_timeseries(sample_ohlcv_data, str(full_path))
        assert full_path.exists()
        
        # Load data
        loaded_data = storage.load_timeseries(str(full_path))
        
        # Check basic properties (exact equality might not hold for all formats)
        assert len(loaded_data) == len(sample_ohlcv_data)
        assert len(loaded_data.columns) == len(sample_ohlcv_data.columns)
    
    def test_memory_efficiency(self, storage):
        """Test that storage operations don't cause memory leaks"""
        import tracemalloc
        
        # Create large dataset
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=5000, freq='D')
        large_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 5000),
            'High': np.random.uniform(100, 110, 5000),
            'Low': np.random.uniform(90, 100, 5000),
            'Close': np.random.uniform(95, 105, 5000),
            'Volume': np.random.randint(1000000, 10000000, 5000)
        }, index=dates)
        
        tracemalloc.start()
        
        # Perform multiple storage operations
        for i in range(3):
            ticker = f"TEST{i}"
            storage.save_raw_data(large_data, ticker)
            loaded_data = storage.load_raw_data(ticker)
            del loaded_data
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 200MB)
        assert peak / (1024 * 1024) < 200, f"Memory usage too high: {peak / (1024 * 1024):.2f} MB"