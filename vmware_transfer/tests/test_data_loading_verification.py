"""
Test Suite for Data Loading Verification Script
Validates the verify_data_loading.py script works correctly with positive and negative test cases.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import json

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from verify_data_loading import DataLoadingVerifier


class TestDataLoadingVerification:
    """Test suite for data loading verification functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def valid_data_file(self, temp_data_dir):
        """Create valid OHLCV parquet file."""
        # Create 100 days of valid stock data
        start_date = datetime(2023, 1, 1)
        dates = pd.bdate_range(start=start_date, periods=100)
        
        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible test data
        base_price = 100.0
        
        data = []
        current_price = base_price
        
        for date in dates:
            # Generate daily price movements
            daily_change = np.random.normal(0, 0.02)  # 2% daily volatility
            open_price = current_price * (1 + daily_change)
            
            # Ensure realistic OHLC relationships
            high = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close = low + (high - low) * np.random.random()
            volume = int(np.random.uniform(1000000, 5000000))
            
            # Ensure price constraints
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume,
                'Ticker': 'TEST'
            })
            
            current_price = close
        
        df = pd.DataFrame(data, index=dates)
        file_path = Path(temp_data_dir) / 'valid_data.parquet'
        df.to_parquet(file_path)
        
        return file_path
    
    @pytest.fixture
    def invalid_data_file(self, temp_data_dir):
        """Create invalid parquet file with NaN values and wrong data types."""
        # Create 50 days of data with issues
        start_date = datetime(2023, 1, 1)
        dates = pd.bdate_range(start=start_date, periods=50)
        
        np.random.seed(123)
        base_price = 100.0
        
        data = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            daily_change = np.random.normal(0, 0.02)
            open_price = current_price * (1 + daily_change)
            high = open_price * 1.02
            low = open_price * 0.98
            close = open_price * (1 + daily_change * 0.5)
            volume = int(np.random.uniform(1000000, 5000000))
            
            # Introduce deliberate issues
            if i == 10:
                # NaN in Close column
                close = np.nan
            elif i == 20:
                # NaN in Volume column  
                volume = np.nan
            elif i == 30:
                # Invalid price relationship: High < Low
                high = 95.0
                low = 105.0
            elif i == 40:
                # String in numeric column (wrong data type)
                open_price = "invalid_price"
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume,
                'Ticker': 'INVALID'
            })
            
            current_price = close if not pd.isna(close) else current_price
        
        df = pd.DataFrame(data, index=dates)
        file_path = Path(temp_data_dir) / 'invalid_data.parquet'
        df.to_parquet(file_path)
        
        return file_path
    
    @pytest.fixture
    def malformed_data_file(self, temp_data_dir):
        """Create malformed parquet file with missing required columns."""
        # Create data with missing required columns
        start_date = datetime(2023, 1, 1)
        dates = pd.bdate_range(start=start_date, periods=30)
        
        np.random.seed(456)
        
        # Only include some columns, missing required ones
        data = []
        for date in dates:
            data.append({
                'Price': np.random.uniform(90, 110),  # Wrong column name
                'Vol': int(np.random.uniform(1000000, 5000000)),  # Wrong column name
                'Ticker': 'MALFORMED'
                # Missing: Open, High, Low, Close, Volume
            })
        
        df = pd.DataFrame(data, index=dates)
        file_path = Path(temp_data_dir) / 'malformed_data.parquet'
        df.to_parquet(file_path)
        
        return file_path
    
    @pytest.fixture
    def non_chronological_file(self, temp_data_dir):
        """Create file with non-chronological data."""
        dates = pd.to_datetime([
            '2023-01-01', '2023-01-03', '2023-01-02', '2023-01-05', '2023-01-04'  # Out of order
        ])
        
        np.random.seed(789)
        data = []
        
        for date in dates:
            data.append({
                'Open': round(np.random.uniform(95, 105), 2),
                'High': round(np.random.uniform(105, 115), 2), 
                'Low': round(np.random.uniform(85, 95), 2),
                'Close': round(np.random.uniform(95, 105), 2),
                'Volume': int(np.random.uniform(1000000, 5000000)),
                'Ticker': 'UNSORTED'
            })
        
        df = pd.DataFrame(data, index=dates)
        file_path = Path(temp_data_dir) / 'non_chronological.parquet'
        df.to_parquet(file_path)
        
        return file_path
    
    def test_valid_data_passes_all_checks(self, valid_data_file, temp_data_dir):
        """Test that valid data passes all verification checks."""
        verifier = DataLoadingVerifier(str(Path(valid_data_file).parent))
        report = verifier.verify_all()
        
        # Assert overall success
        assert report['status'] == 'PASS', f"Valid data should pass all checks. Failures: {report.get('failures', [])}"
        
        # Assert all individual checks passed
        expected_checks = [
            'FILE_LOADING', 'COLUMN_CHECK', 'DATA_TYPE_CHECK', 'NAN_CHECK',
            'DATETIME_INDEX_CHECK', 'CHRONOLOGICAL_ORDER_CHECK', 'TRADING_DAYS_CHECK',
            'PRICE_RELATIONSHIP_CHECK', 'PERFORMANCE_CHECK'
        ]
        
        for check in expected_checks:
            assert report['results'].get(check) == 'PASS', f"Check {check} should pass for valid data"
        
        # Verify data shape was captured
        assert 'DATA_SHAPE' in report['results']
        shape_info = report['results']['DATA_SHAPE']
        assert shape_info['rows'] == 100
        assert shape_info['columns'] == 6
        assert 'date_range' in shape_info
    
    def test_invalid_data_fails_with_specific_errors(self, invalid_data_file):
        """Test that invalid data fails with specific error messages."""
        verifier = DataLoadingVerifier(str(Path(invalid_data_file).parent))
        report = verifier.verify_all()
        
        # Assert overall failure
        assert report['status'] == 'FAIL', "Invalid data should fail verification"
        
        # Check for specific expected failures
        failed_checks = [result for result in report['results'].values() if result == 'FAIL']
        assert len(failed_checks) > 0, "Invalid data should have at least one failed check"
        
        # Verify failure details are captured
        assert len(report['failures']) > 0, "Should have detailed failure information"
        
        # Check for NaN detection
        nan_failures = [f for f in report['failures'] if 'NaN' in f['message'] or 'NAN_CHECK' in f['check']]
        assert len(nan_failures) > 0, "Should detect NaN values"
        
        # Check for price relationship violations  
        price_failures = [f for f in report['failures'] if 'PRICE_RELATIONSHIP_CHECK' in f['check']]
        assert len(price_failures) > 0, "Should detect invalid price relationships"
    
    def test_malformed_data_fails_column_check(self, malformed_data_file):
        """Test that malformed data fails with missing column errors."""
        verifier = DataLoadingVerifier(str(Path(malformed_data_file).parent))
        report = verifier.verify_all()
        
        # Assert overall failure
        assert report['status'] == 'FAIL', "Malformed data should fail verification"
        
        # Check that column check specifically failed
        assert report['results'].get('COLUMN_CHECK') == 'FAIL', "Should fail column check"
        
        # Verify specific error about missing columns
        column_failures = [f for f in report['failures'] if 'COLUMN_CHECK' in f['check']]
        assert len(column_failures) > 0, "Should have column check failure"
        
        failure = column_failures[0]
        assert 'Missing required columns' in failure['message'], "Should specify missing columns"
        
        # Check that required columns are identified
        expected_missing = {'Open', 'High', 'Low', 'Close', 'Volume'}
        assert any(col in failure['message'] for col in expected_missing), "Should identify specific missing columns"
    
    def test_non_chronological_data_fails_order_check(self, non_chronological_file):
        """Test that non-chronological data fails chronological order check.""" 
        verifier = DataLoadingVerifier(str(Path(non_chronological_file).parent))
        report = verifier.verify_all()
        
        # Should fail overall due to chronological order
        assert report['status'] == 'FAIL', "Non-chronological data should fail verification"
        
        # Check specific failure  
        assert report['results'].get('CHRONOLOGICAL_ORDER_CHECK') == 'FAIL', "Should fail chronological order check"
        
        # Verify failure details
        order_failures = [f for f in report['failures'] if 'CHRONOLOGICAL_ORDER_CHECK' in f['check']]
        assert len(order_failures) > 0, "Should have chronological order failure"
        
        failure = order_failures[0]
        assert 'not sorted chronologically' in failure['message'], "Should specify chronological order issue"
        assert 'first_unsorted_position' in str(failure.get('details', {})), "Should identify unsorted position"
    
    def test_performance_requirement_check(self, valid_data_file):
        """Test that performance requirements are verified."""
        verifier = DataLoadingVerifier(str(Path(valid_data_file).parent))
        report = verifier.verify_all()
        
        # Performance check should pass for small dataset
        assert report['results'].get('PERFORMANCE_CHECK') == 'PASS', "Performance check should pass"
        
        # Execution time should be recorded
        assert report['summary']['execution_time'] > 0, "Should record execution time"
        assert report['summary']['execution_time'] < 10.0, "Should complete within 10 seconds"
    
    def test_script_executable_directly(self, valid_data_file):
        """Test that the script can be executed directly from command line."""
        # Change to directory containing the valid file
        original_cwd = os.getcwd()
        try:
            os.chdir(Path(valid_data_file).parent.parent)  # Go up one level from temp dir
            
            # Update the temp dir name in the script call
            temp_dir_name = Path(valid_data_file).parent.name
            
            # Create a small test script to run verification on our temp data
            test_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from verify_data_loading import DataLoadingVerifier

verifier = DataLoadingVerifier('{temp_dir_name}')
report = verifier.verify_all()
sys.exit(0 if report['status'] == 'PASS' else 1)
"""
            
            with open('temp_test_verification.py', 'w') as f:
                f.write(test_script)
            
            # Run the test script
            result = subprocess.run([sys.executable, 'temp_test_verification.py'], 
                                    capture_output=True, text=True)
            
            # Clean up
            os.remove('temp_test_verification.py')
            
            # Verify it ran successfully
            assert result.returncode == 0, f"Script should execute successfully. Error: {result.stderr}"
            assert "VERIFICATION SUMMARY" in result.stdout, "Should produce verification report"
            
        finally:
            os.chdir(original_cwd)
    
    def test_detailed_failure_reporting(self, invalid_data_file):
        """Test that failures include exact row/column information."""
        verifier = DataLoadingVerifier(str(Path(invalid_data_file).parent))
        report = verifier.verify_all()
        
        # Should have detailed failure information
        assert len(report['failures']) > 0, "Should have failure details"
        
        # Check that failures include specific details
        for failure in report['failures']:
            assert 'check' in failure, "Failure should specify which check failed"
            assert 'message' in failure, "Failure should have descriptive message"
            
            # Some failures should include detailed location information
            if failure['check'] in ['NAN_CHECK', 'PRICE_RELATIONSHIP_CHECK']:
                assert failure.get('details') is not None, f"Check {failure['check']} should include details"
    
    def test_zero_tolerance_false_positives(self, valid_data_file):
        """Test zero tolerance for false positives - valid data must always pass."""
        # Run verification multiple times to ensure consistency
        verifier = DataLoadingVerifier(str(Path(valid_data_file).parent))
        
        for i in range(3):
            report = verifier.verify_all()
            assert report['status'] == 'PASS', f"Run {i+1}: Valid data must always pass, got failures: {report.get('failures', [])}"
            
            # All checks should consistently pass
            for check_name, result in report['results'].items():
                if check_name != 'DATA_SHAPE':  # DATA_SHAPE is informational
                    assert result == 'PASS', f"Run {i+1}: Check {check_name} should always pass for valid data"
    
    def test_zero_tolerance_false_negatives(self, invalid_data_file, malformed_data_file):
        """Test zero tolerance for false negatives - invalid data must always fail."""
        test_files = [invalid_data_file, malformed_data_file]
        
        for file_path in test_files:
            verifier = DataLoadingVerifier(str(Path(file_path).parent))
            
            # Run multiple times to ensure consistency
            for i in range(2):
                report = verifier.verify_all()
                assert report['status'] == 'FAIL', f"File {file_path.name}, Run {i+1}: Invalid data must always fail"
                assert len(report['failures']) > 0, f"File {file_path.name}, Run {i+1}: Must have specific failure details"
    
    def test_cleanup_and_isolation(self, temp_data_dir):
        """Test that tests properly clean up and don't affect each other."""
        # Verify temp directory exists and contains our test files
        temp_path = Path(temp_data_dir)
        assert temp_path.exists(), "Temp directory should exist during test"
        
        # The fixture cleanup will be tested by pytest's fixture system
        # This test verifies the directory is properly isolated
        assert str(temp_path).startswith(tempfile.gettempdir()), "Should use system temp directory"
    
    def test_comprehensive_validation_coverage(self, valid_data_file):
        """Test that all required validation scenarios are covered."""
        verifier = DataLoadingVerifier(str(Path(valid_data_file).parent))
        report = verifier.verify_all()
        
        # Verify all required checks are implemented
        required_validations = [
            'FILE_LOADING',           # File can be loaded
            'COLUMN_CHECK',           # Required columns present
            'DATA_TYPE_CHECK',        # Correct data types
            'NAN_CHECK',             # No NaN values in critical columns
            'DATETIME_INDEX_CHECK',   # Proper DatetimeIndex
            'CHRONOLOGICAL_ORDER_CHECK',  # Data sorted chronologically
            'TRADING_DAYS_CHECK',     # Weekend/holiday gap validation
            'PRICE_RELATIONSHIP_CHECK',   # Price constraint validation
            'PERFORMANCE_CHECK'       # Performance requirements
        ]
        
        for validation in required_validations:
            assert validation in report['results'], f"Required validation {validation} not implemented"
        
        # Verify data shape reporting
        assert 'DATA_SHAPE' in report['results'], "Must report data shape and range"


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__, "-v"])