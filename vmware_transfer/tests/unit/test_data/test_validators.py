"""
Unit tests for DataValidator implementation.
Tests schema, range, consistency, and completeness validation stages.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock

from src.data.validators import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    SchemaValidator,
    RangeValidator,
    ConsistencyValidator,
    CompletenessValidator,
    DataValidator,
    validate_stock_data,
    create_validation_report
)
from src.config.data_config import DataConfig


class TestValidationIssue:
    """Test ValidationIssue dataclass."""
    
    def test_issue_creation(self):
        """Test creating validation issue."""
        issue = ValidationIssue(
            validator="TestValidator",
            severity=ValidationSeverity.ERROR,
            message="Test error message",
            column="Close",
            value=100.0
        )
        
        assert issue.validator == "TestValidator"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error message"
        assert issue.column == "Close"
        assert issue.value == 100.0
    
    def test_issue_string_representation(self):
        """Test string representation of validation issue."""
        issue = ValidationIssue(
            validator="TestValidator",
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            column="Volume"
        )
        
        str_repr = str(issue)
        assert "[WARNING]" in str_repr
        assert "TestValidator" in str_repr
        assert "Test warning" in str_repr
        assert "(column: Volume)" in str_repr


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_result_creation(self):
        """Test creating validation result."""
        issues = [
            ValidationIssue("Test", ValidationSeverity.WARNING, "Warning message"),
            ValidationIssue("Test", ValidationSeverity.ERROR, "Error message")
        ]
        
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            metadata={"ticker": "AAPL"}
        )
        
        assert result.is_valid is False
        assert len(result.issues) == 2
        assert result.metadata["ticker"] == "AAPL"
    
    def test_get_issues_by_severity(self):
        """Test filtering issues by severity."""
        issues = [
            ValidationIssue("Test", ValidationSeverity.INFO, "Info message"),
            ValidationIssue("Test", ValidationSeverity.WARNING, "Warning message"),
            ValidationIssue("Test", ValidationSeverity.ERROR, "Error message"),
            ValidationIssue("Test", ValidationSeverity.CRITICAL, "Critical message")
        ]
        
        result = ValidationResult(True, issues, {})
        
        warnings = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert len(warnings) == 1
        assert warnings[0].message == "Warning message"
        
        errors = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(errors) == 1
        assert errors[0].message == "Error message"
    
    def test_has_critical_issues(self):
        """Test checking for critical issues."""
        # No critical issues
        result1 = ValidationResult(True, [
            ValidationIssue("Test", ValidationSeverity.WARNING, "Warning")
        ], {})
        assert result1.has_critical_issues() is False
        
        # Has critical issues
        result2 = ValidationResult(False, [
            ValidationIssue("Test", ValidationSeverity.CRITICAL, "Critical")
        ], {})
        assert result2.has_critical_issues() is True
    
    def test_has_errors(self):
        """Test checking for error-level issues."""
        # No errors
        result1 = ValidationResult(True, [
            ValidationIssue("Test", ValidationSeverity.INFO, "Info")
        ], {})
        assert result1.has_errors() is False
        
        # Has errors
        result2 = ValidationResult(False, [
            ValidationIssue("Test", ValidationSeverity.ERROR, "Error")
        ], {})
        assert result2.has_errors() is True
        
        # Has critical (which counts as error)
        result3 = ValidationResult(False, [
            ValidationIssue("Test", ValidationSeverity.CRITICAL, "Critical")
        ], {})
        assert result3.has_errors() is True
    
    def test_summary(self):
        """Test issue summary by severity."""
        issues = [
            ValidationIssue("Test", ValidationSeverity.INFO, "Info 1"),
            ValidationIssue("Test", ValidationSeverity.INFO, "Info 2"),
            ValidationIssue("Test", ValidationSeverity.WARNING, "Warning"),
            ValidationIssue("Test", ValidationSeverity.ERROR, "Error")
        ]
        
        result = ValidationResult(False, issues, {})
        summary = result.summary()
        
        assert summary["info"] == 2
        assert summary["warning"] == 1
        assert summary["error"] == 1
        assert summary["critical"] == 0


class TestSchemaValidator:
    """Test SchemaValidator functionality."""
    
    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
    
    def test_validator_initialization(self):
        """Test schema validator initializes correctly."""
        validator = SchemaValidator()
        
        assert validator.name == "SchemaValidator"
        assert 'Open' in validator.required_columns
        assert 'High' in validator.required_columns
        assert 'Low' in validator.required_columns
        assert 'Close' in validator.required_columns
        assert 'Volume' in validator.required_columns
    
    def test_valid_data_passes(self, valid_stock_data):
        """Test that valid data passes schema validation."""
        validator = SchemaValidator()
        
        issues = validator.validate(valid_stock_data, "AAPL")
        
        # Should have no critical or error issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        
        assert len(critical_issues) == 0
        assert len(error_issues) == 0
    
    def test_empty_data_fails(self):
        """Test that empty data fails validation."""
        validator = SchemaValidator()
        empty_data = pd.DataFrame()
        
        issues = validator.validate(empty_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        assert any("empty" in issue.message.lower() for issue in issues)
    
    def test_missing_required_columns(self, valid_stock_data):
        """Test detection of missing required columns."""
        validator = SchemaValidator()
        
        # Remove required column
        incomplete_data = valid_stock_data.drop('Volume', axis=1)
        
        issues = validator.validate(incomplete_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        assert any("missing required columns" in issue.message.lower() for issue in issues)
    
    def test_unexpected_columns_warning(self, valid_stock_data):
        """Test warning for unexpected columns."""
        validator = SchemaValidator()
        
        # Add unexpected column
        data_with_extra = valid_stock_data.copy()
        data_with_extra['UnexpectedColumn'] = range(len(data_with_extra))
        
        issues = validator.validate(data_with_extra, "AAPL")
        
        warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        assert len(warning_issues) > 0
        assert any("unexpected columns" in issue.message.lower() for issue in warning_issues)
    
    def test_wrong_index_type(self, valid_stock_data):
        """Test detection of wrong index type."""
        validator = SchemaValidator()
        
        # Reset index to integer
        data_wrong_index = valid_stock_data.reset_index(drop=True)
        
        issues = validator.validate(data_wrong_index, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("datetimeindex" in issue.message.lower() for issue in issues)
    
    def test_duplicate_index_detection(self, valid_stock_data):
        """Test detection of duplicate index values."""
        validator = SchemaValidator()
        
        # Create duplicate index
        data_with_duplicates = pd.concat([valid_stock_data, valid_stock_data.iloc[:2]])
        
        issues = validator.validate(data_with_duplicates, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("duplicate" in issue.message.lower() for issue in issues)


class TestRangeValidator:
    """Test RangeValidator functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config_dict = {
            "data_quality": {
                "min_price": 1.0,
                "max_price": 10000.0,
                "min_volume": 100000,
                "outlier_threshold": 10.0
            }
        }
        return DataConfig(config_dict)
    
    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
    
    def test_validator_initialization(self, mock_config):
        """Test range validator initializes correctly."""
        validator = RangeValidator(mock_config)
        
        assert validator.name == "RangeValidator"
        assert validator.validation_rules['min_price'] == 1.0
        assert validator.validation_rules['max_price'] == 10000.0
    
    def test_valid_data_passes(self, valid_stock_data, mock_config):
        """Test that valid data passes range validation."""
        validator = RangeValidator(mock_config)
        
        issues = validator.validate(valid_stock_data, "AAPL")
        
        # Should have no critical or error issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        
        assert len(critical_issues) == 0
        assert len(error_issues) == 0
    
    def test_negative_prices_detected(self, valid_stock_data, mock_config):
        """Test detection of negative prices."""
        validator = RangeValidator(mock_config)
        
        # Add negative price
        data_with_negative = valid_stock_data.copy()
        data_with_negative.loc[data_with_negative.index[0], 'Close'] = -10.0
        
        issues = validator.validate(data_with_negative, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        assert any("negative" in issue.message.lower() for issue in issues)
    
    def test_price_below_minimum(self, valid_stock_data, mock_config):
        """Test detection of prices below minimum."""
        validator = RangeValidator(mock_config)
        
        # Set price below minimum
        data_low_price = valid_stock_data.copy()
        data_low_price.loc[data_low_price.index[0], 'Close'] = 0.5  # Below min_price of 1.0
        
        issues = validator.validate(data_low_price, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("below minimum" in issue.message.lower() for issue in issues)
    
    def test_price_above_maximum(self, valid_stock_data, mock_config):
        """Test detection of prices above maximum."""
        validator = RangeValidator(mock_config)
        
        # Set price above maximum
        data_high_price = valid_stock_data.copy()
        data_high_price.loc[data_high_price.index[0], 'Close'] = 20000.0  # Above max_price of 10000.0
        
        issues = validator.validate(data_high_price, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.WARNING for issue in issues)
        assert any("above maximum" in issue.message.lower() for issue in issues)
    
    def test_negative_volume_detected(self, valid_stock_data, mock_config):
        """Test detection of negative volume."""
        validator = RangeValidator(mock_config)
        
        # Add negative volume
        data_with_negative_volume = valid_stock_data.copy()
        data_with_negative_volume.loc[data_with_negative_volume.index[0], 'Volume'] = -1000
        
        issues = validator.validate(data_with_negative_volume, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        assert any("negative volume" in issue.message.lower() for issue in issues)
    
    def test_infinite_values_detected(self, valid_stock_data, mock_config):
        """Test detection of infinite values."""
        validator = RangeValidator(mock_config)
        
        # Add infinite value
        data_with_inf = valid_stock_data.copy()
        data_with_inf.loc[data_with_inf.index[0], 'Close'] = np.inf
        
        issues = validator.validate(data_with_inf, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        assert any("infinite" in issue.message.lower() for issue in issues)


class TestConsistencyValidator:
    """Test ConsistencyValidator functionality."""
    
    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
    
    def test_validator_initialization(self):
        """Test consistency validator initializes correctly."""
        validator = ConsistencyValidator()
        
        assert validator.name == "ConsistencyValidator"
    
    def test_valid_ohlc_passes(self, valid_stock_data):
        """Test that valid OHLC data passes consistency validation."""
        validator = ConsistencyValidator()
        
        issues = validator.validate(valid_stock_data, "AAPL")
        
        # Should have no critical or error issues for OHLC
        ohlc_issues = [i for i in issues if "high" in i.message.lower() or "low" in i.message.lower()]
        critical_ohlc = [i for i in ohlc_issues if i.severity == ValidationSeverity.CRITICAL]
        error_ohlc = [i for i in ohlc_issues if i.severity == ValidationSeverity.ERROR]
        
        assert len(critical_ohlc) == 0
        assert len(error_ohlc) == 0
    
    def test_high_less_than_low_detected(self, valid_stock_data):
        """Test detection of High < Low inconsistency."""
        validator = ConsistencyValidator()
        
        # Create inconsistent data
        inconsistent_data = valid_stock_data.copy()
        inconsistent_data.loc[inconsistent_data.index[0], 'High'] = 95.0  # Less than Low (99.0)
        
        issues = validator.validate(inconsistent_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        assert any("high < low" in issue.message.lower() for issue in issues)
    
    def test_high_less_than_open_detected(self, valid_stock_data):
        """Test detection of High < Open inconsistency."""
        validator = ConsistencyValidator()
        
        # Create inconsistent data
        inconsistent_data = valid_stock_data.copy()
        inconsistent_data.loc[inconsistent_data.index[0], 'High'] = 95.0  # Less than Open (100.0)
        
        issues = validator.validate(inconsistent_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("high < open" in issue.message.lower() for issue in issues)
    
    def test_low_greater_than_close_detected(self, valid_stock_data):
        """Test detection of Low > Close inconsistency."""
        validator = ConsistencyValidator()
        
        # Create inconsistent data
        inconsistent_data = valid_stock_data.copy()
        inconsistent_data.loc[inconsistent_data.index[0], 'Low'] = 105.0  # Greater than Close (101.0)
        
        issues = validator.validate(inconsistent_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("low > close" in issue.message.lower() for issue in issues)
    
    def test_zero_price_range_detected(self, valid_stock_data):
        """Test detection of zero price range (High == Low)."""
        validator = ConsistencyValidator()
        
        # Create zero range data
        zero_range_data = valid_stock_data.copy()
        zero_range_data.loc[zero_range_data.index[0], 'High'] = 100.0
        zero_range_data.loc[zero_range_data.index[0], 'Low'] = 100.0
        
        issues = validator.validate(zero_range_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.WARNING for issue in issues)
        assert any("zero price range" in issue.message.lower() for issue in issues)
    
    def test_zero_volume_detected(self, valid_stock_data):
        """Test detection of zero volume."""
        validator = ConsistencyValidator()
        
        # Add zero volume
        zero_volume_data = valid_stock_data.copy()
        zero_volume_data.loc[zero_volume_data.index[0], 'Volume'] = 0
        
        issues = validator.validate(zero_volume_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.WARNING for issue in issues)
        assert any("zero volume" in issue.message.lower() for issue in issues)
    
    def test_weekend_dates_detected(self, valid_stock_data):
        """Test detection of weekend dates."""
        validator = ConsistencyValidator()
        
        # Create data with weekend date (Saturday)
        weekend_dates = pd.date_range('2023-01-07', periods=10, freq='D')  # Starts on Saturday
        weekend_data = valid_stock_data.copy()
        weekend_data.index = weekend_dates
        
        issues = validator.validate(weekend_data, "AAPL")
        
        # Should detect weekend dates
        weekend_issues = [i for i in issues if "weekend" in i.message.lower()]
        assert len(weekend_issues) > 0


class TestCompletenessValidator:
    """Test CompletenessValidator functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config_dict = {
            "data_quality": {
                "max_missing_ratio": 0.05,
                "min_trading_days": 5  # Reduced for testing
            }
        }
        return DataConfig(config_dict)
    
    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
    
    def test_validator_initialization(self, mock_config):
        """Test completeness validator initializes correctly."""
        validator = CompletenessValidator(mock_config)
        
        assert validator.name == "CompletenessValidator"
        assert validator.max_missing_ratio == 0.05
        assert validator.min_trading_days == 5
    
    def test_valid_data_passes(self, valid_stock_data, mock_config):
        """Test that valid complete data passes validation."""
        validator = CompletenessValidator(mock_config)
        
        issues = validator.validate(valid_stock_data, "AAPL")
        
        # Should have no critical or error issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        
        assert len(critical_issues) == 0
        assert len(error_issues) == 0
    
    def test_empty_data_detected(self, mock_config):
        """Test detection of empty data."""
        validator = CompletenessValidator(mock_config)
        empty_data = pd.DataFrame()
        
        issues = validator.validate(empty_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        assert any("no data" in issue.message.lower() for issue in issues)
    
    def test_insufficient_data_detected(self, valid_stock_data, mock_config):
        """Test detection of insufficient data."""
        validator = CompletenessValidator(mock_config)
        
        # Use only 3 rows (less than min_trading_days of 5)
        insufficient_data = valid_stock_data.head(3)
        
        issues = validator.validate(insufficient_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("insufficient data" in issue.message.lower() for issue in issues)
    
    def test_excessive_missing_data_detected(self, valid_stock_data, mock_config):
        """Test detection of excessive missing data."""
        validator = CompletenessValidator(mock_config)
        
        # Add lots of missing data (>5%)
        missing_data = valid_stock_data.copy()
        missing_data.loc[missing_data.index[:6], 'Close'] = np.nan  # 60% missing in Close column
        
        issues = validator.validate(missing_data, "AAPL")
        
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("too much missing data" in issue.message.lower() for issue in issues)
    
    def test_column_missing_data_detected(self, valid_stock_data, mock_config):
        """Test detection of missing data in specific columns."""
        validator = CompletenessValidator(mock_config)
        
        # Add some missing data to specific column
        missing_data = valid_stock_data.copy()
        missing_data.loc[missing_data.index[0], 'Volume'] = np.nan
        
        issues = validator.validate(missing_data, "AAPL")
        
        # Should detect missing data in Volume column
        volume_issues = [i for i in issues if i.column == 'Volume']
        assert len(volume_issues) > 0
    
    def test_consecutive_missing_data_detected(self, valid_stock_data, mock_config):
        """Test detection of consecutive missing data."""
        validator = CompletenessValidator(mock_config)
        
        # Add consecutive missing data (6 consecutive NaNs)
        consecutive_missing = valid_stock_data.copy()
        consecutive_missing.loc[consecutive_missing.index[:6], 'Close'] = np.nan
        
        issues = validator.validate(consecutive_missing, "AAPL")
        
        # Should detect long consecutive missing runs
        consecutive_issues = [i for i in issues if "consecutive" in i.message.lower()]
        assert len(consecutive_issues) > 0


class TestDataValidator:
    """Test main DataValidator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config_dict = {
            "data_quality": {
                "min_price": 1.0,
                "max_price": 10000.0,
                "min_volume": 100000,
                "max_missing_ratio": 0.05,
                "min_trading_days": 5,
                "outlier_threshold": 10.0
            }
        }
        return DataConfig(config_dict)
    
    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
    
    def test_validator_initialization(self, mock_config):
        """Test data validator initializes correctly."""
        validator = DataValidator(mock_config)
        
        assert len(validator.validators) == 4
        assert any(v.name == "SchemaValidator" for v in validator.validators)
        assert any(v.name == "RangeValidator" for v in validator.validators)
        assert any(v.name == "ConsistencyValidator" for v in validator.validators)
        assert any(v.name == "CompletenessValidator" for v in validator.validators)
    
    def test_valid_data_validation(self, valid_stock_data, mock_config):
        """Test validation of valid data."""
        validator = DataValidator(mock_config)
        
        result = validator.validate(valid_stock_data, "AAPL")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.metadata['ticker'] == "AAPL"
        assert result.metadata['data_shape'] == valid_stock_data.shape
        assert len(result.metadata['validators_run']) == 4
    
    def test_invalid_data_validation(self, valid_stock_data, mock_config):
        """Test validation of invalid data."""
        validator = DataValidator(mock_config)
        
        # Create invalid data (negative price)
        invalid_data = valid_stock_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Close'] = -10.0
        
        result = validator.validate(invalid_data, "AAPL")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert result.has_errors() is True
    
    def test_batch_validation(self, valid_stock_data, mock_config):
        """Test batch validation of multiple datasets."""
        validator = DataValidator(mock_config)
        
        # Create batch data
        data_dict = {
            "AAPL": valid_stock_data,
            "MSFT": valid_stock_data.copy(),
            "GOOGL": valid_stock_data.copy()
        }
        
        results = validator.validate_batch(data_dict)
        
        assert len(results) == 3
        assert all(isinstance(result, ValidationResult) for result in results.values())
        assert all(result.is_valid for result in results.values())
    
    def test_validator_statistics(self, valid_stock_data, mock_config):
        """Test validator statistics tracking."""
        validator = DataValidator(mock_config)
        
        # Perform validation
        validator.validate(valid_stock_data, "AAPL")
        
        stats = validator.get_statistics()
        
        assert 'overall' in stats
        assert 'validators' in stats
        assert stats['overall']['validations_performed'] == 1
        
        # Check individual validator stats
        for validator_name in ['SchemaValidator', 'RangeValidator', 'ConsistencyValidator', 'CompletenessValidator']:
            assert validator_name in stats['validators']
            assert stats['validators'][validator_name]['validations_performed'] == 1
    
    def test_statistics_reset(self, valid_stock_data, mock_config):
        """Test resetting validator statistics."""
        validator = DataValidator(mock_config)
        
        # Perform validation
        validator.validate(valid_stock_data, "AAPL")
        
        # Reset statistics
        validator.reset_statistics()
        
        stats = validator.get_statistics()
        assert stats['overall']['validations_performed'] == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
    
    def test_validate_stock_data_function(self, valid_stock_data):
        """Test validate_stock_data convenience function."""
        result = validate_stock_data(valid_stock_data, "AAPL")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.metadata['ticker'] == "AAPL"
    
    def test_create_validation_report(self, valid_stock_data):
        """Test create_validation_report function."""
        # Create some validation results
        results = {
            "AAPL": ValidationResult(True, [], {"ticker": "AAPL"}),
            "MSFT": ValidationResult(False, [
                ValidationIssue("Test", ValidationSeverity.ERROR, "Test error")
            ], {"ticker": "MSFT"})
        }
        
        report = create_validation_report(results)
        
        assert isinstance(report, str)
        assert "Data Validation Report" in report
        assert "Total tickers validated: 2" in report
        assert "Passed: 1" in report
        assert "Failed: 1" in report
        assert "MSFT" in report  # Failed ticker should be listed