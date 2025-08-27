"""
Data validation pipeline with multi-stage validation.
Implements schema, range, consistency, and completeness validation.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.config.data_config import DataConfig

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    validator: str
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_count: Optional[int] = None
    value: Optional[Any] = None
    
    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}]", f"{self.validator}:", self.message]
        if self.column:
            parts.append(f"(column: {self.column})")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    metadata: Dict[str, Any]
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def summary(self) -> Dict[str, int]:
        """Get summary of issues by severity."""
        summary = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            summary[issue.severity.value] += 1
        return summary


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str, config: Optional[DataConfig] = None):
        self.name = name
        self.config = config
        self.stats = {
            'validations_performed': 0,
            'issues_found': 0,
            'last_validation': None
        }
    
    @abstractmethod
    def validate(self, data: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        """
        Validate data and return list of issues.
        
        Args:
            data: DataFrame to validate
            ticker: Ticker symbol for context
            
        Returns:
            List of validation issues
        """
        pass
    
    def _create_issue(self, severity: ValidationSeverity, message: str, 
                     column: Optional[str] = None, **kwargs) -> ValidationIssue:
        """Create a validation issue."""
        return ValidationIssue(
            validator=self.name,
            severity=severity,
            message=message,
            column=column,
            **kwargs
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset validator statistics."""
        self.stats = {
            'validations_performed': 0,
            'issues_found': 0,
            'last_validation': None
        }


class SchemaValidator(BaseValidator):
    """Validates data schema and column presence."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        super().__init__("SchemaValidator", config)
        
        # Required columns for stock data
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.optional_columns = ['Adj Close', 'Dividends', 'Stock Splits']
        self.expected_dtypes = {
            'Open': [np.float64, np.float32],
            'High': [np.float64, np.float32],
            'Low': [np.float64, np.float32],
            'Close': [np.float64, np.float32],
            'Volume': [np.int64, np.int32, np.float64],  # Volume can be float in some sources
            'Adj Close': [np.float64, np.float32]
        }
    
    def validate(self, data: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        """Validate data schema."""
        issues = []
        
        self.stats['validations_performed'] += 1
        self.stats['last_validation'] = datetime.now()
        
        # Check if data is empty
        if data.empty:
            issues.append(self._create_issue(
                ValidationSeverity.CRITICAL,
                "DataFrame is empty"
            ))
            self.stats['issues_found'] += len(issues)
            return issues
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            issues.append(self._create_issue(
                ValidationSeverity.CRITICAL,
                f"Missing required columns: {missing_columns}"
            ))
        
        # Check for unexpected columns
        expected_all = set(self.required_columns + self.optional_columns + ['Ticker'])
        unexpected_columns = [col for col in data.columns if col not in expected_all]
        if unexpected_columns:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Unexpected columns found: {unexpected_columns}"
            ))
        
        # Check data types
        for column in data.columns:
            if column in self.expected_dtypes:
                expected_types = self.expected_dtypes[column]
                actual_type = data[column].dtype
                
                if actual_type not in expected_types:
                    issues.append(self._create_issue(
                        ValidationSeverity.WARNING,
                        f"Unexpected data type: expected {expected_types}, got {actual_type}",
                        column=column
                    ))
        
        # Check index type (should be datetime)
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append(self._create_issue(
                ValidationSeverity.ERROR,
                f"Index should be DatetimeIndex, got {type(data.index)}"
            ))
        
        # Check for duplicate index values
        if data.index.duplicated().any():
            duplicate_count = data.index.duplicated().sum()
            issues.append(self._create_issue(
                ValidationSeverity.ERROR,
                f"Found {duplicate_count} duplicate index values"
            ))
        
        self.stats['issues_found'] += len(issues)
        return issues


class RangeValidator(BaseValidator):
    """Validates data value ranges."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        super().__init__("RangeValidator", config)
        
        # Default validation rules (can be overridden by config)
        self.validation_rules = {
            'min_price': 0.01,  # Minimum stock price
            'max_price': 100000.0,  # Maximum stock price (sanity check)
            'min_volume': 0,  # Minimum volume
            'max_volume': 1e12,  # Maximum volume (sanity check)
            'max_price_change': 0.5,  # Maximum single-day price change (50%)
            'outlier_threshold': 10  # Standard deviations for outlier detection
        }
        
        # Update rules from config if available
        if config and hasattr(config, 'data_quality'):
            quality_config = config.data_quality
            self.validation_rules.update({
                'min_price': quality_config.min_price,
                'max_price': quality_config.max_price,
                'min_volume': quality_config.min_volume,
                'outlier_threshold': quality_config.outlier_threshold
            })
    
    def validate(self, data: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        """Validate data value ranges."""
        issues = []
        
        self.stats['validations_performed'] += 1
        self.stats['last_validation'] = datetime.now()
        
        if data.empty:
            return issues
        
        # Validate price columns
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for column in price_columns:
            if column in data.columns:
                issues.extend(self._validate_price_column(data[column], column))
        
        # Validate volume
        if 'Volume' in data.columns:
            issues.extend(self._validate_volume_column(data['Volume']))
        
        # Check for outliers in returns
        if 'Close' in data.columns and len(data) > 1:
            issues.extend(self._validate_returns_outliers(data['Close']))
        
        self.stats['issues_found'] += len(issues)
        return issues
    
    def _validate_price_column(self, series: pd.Series, column: str) -> List[ValidationIssue]:
        """Validate a price column."""
        issues = []
        
        # Check for negative prices
        negative_count = (series < 0).sum()
        if negative_count > 0:
            issues.append(self._create_issue(
                ValidationSeverity.CRITICAL,
                f"Found {negative_count} negative values",
                column=column
            ))
        
        # Check minimum price
        min_price = series.min()
        if pd.notna(min_price) and min_price < self.validation_rules['min_price']:
            issues.append(self._create_issue(
                ValidationSeverity.ERROR,
                f"Price below minimum: {min_price} < {self.validation_rules['min_price']}",
                column=column,
                value=min_price
            ))
        
        # Check maximum price
        max_price = series.max()
        if pd.notna(max_price) and max_price > self.validation_rules['max_price']:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Price above maximum: {max_price} > {self.validation_rules['max_price']}",
                column=column,
                value=max_price
            ))
        
        # Check for infinite values
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            issues.append(self._create_issue(
                ValidationSeverity.CRITICAL,
                f"Found {inf_count} infinite values",
                column=column
            ))
        
        return issues
    
    def _validate_volume_column(self, series: pd.Series) -> List[ValidationIssue]:
        """Validate volume column."""
        issues = []
        
        # Check for negative volume
        negative_count = (series < 0).sum()
        if negative_count > 0:
            issues.append(self._create_issue(
                ValidationSeverity.CRITICAL,
                f"Found {negative_count} negative volume values",
                column='Volume'
            ))
        
        # Check minimum volume
        min_volume = series.min()
        if pd.notna(min_volume) and min_volume < self.validation_rules['min_volume']:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Volume below minimum: {min_volume} < {self.validation_rules['min_volume']}",
                column='Volume',
                value=min_volume
            ))
        
        # Check maximum volume
        max_volume = series.max()
        if pd.notna(max_volume) and max_volume > self.validation_rules['max_volume']:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Volume above maximum: {max_volume} > {self.validation_rules['max_volume']}",
                column='Volume',
                value=max_volume
            ))
        
        return issues
    
    def _validate_returns_outliers(self, close_prices: pd.Series) -> List[ValidationIssue]:
        """Validate returns for outliers."""
        issues = []
        
        try:
            # Calculate daily returns
            returns = close_prices.pct_change().dropna()
            
            if len(returns) == 0:
                return issues
            
            # Check for extreme returns
            returns_std = returns.std()
            if pd.notna(returns_std) and returns_std > 0:
                threshold = self.validation_rules['outlier_threshold'] * returns_std
                outliers = np.abs(returns) > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    max_outlier = np.abs(returns[outliers]).max()
                    issues.append(self._create_issue(
                        ValidationSeverity.WARNING,
                        f"Found {outlier_count} return outliers (max: {max_outlier:.2%})",
                        column='Close'
                    ))
        
        except Exception as e:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Error calculating returns outliers: {e}",
                column='Close'
            ))
        
        return issues


class ConsistencyValidator(BaseValidator):
    """Validates OHLC consistency and logical relationships."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        super().__init__("ConsistencyValidator", config)
    
    def validate(self, data: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        """Validate data consistency."""
        issues = []
        
        self.stats['validations_performed'] += 1
        self.stats['last_validation'] = datetime.now()
        
        if data.empty:
            return issues
        
        # Check OHLC relationships
        required_ohlc = ['Open', 'High', 'Low', 'Close']
        if all(col in data.columns for col in required_ohlc):
            issues.extend(self._validate_ohlc_relationships(data))
        
        # Check volume consistency
        if 'Volume' in data.columns:
            issues.extend(self._validate_volume_consistency(data['Volume']))
        
        # Check temporal consistency
        if isinstance(data.index, pd.DatetimeIndex):
            issues.extend(self._validate_temporal_consistency(data.index))
        
        self.stats['issues_found'] += len(issues)
        return issues
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate OHLC price relationships."""
        issues = []
        
        try:
            # High should be >= Open, Close, Low
            high_vs_open = (data['High'] < data['Open']).sum()
            if high_vs_open > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.ERROR,
                    f"High < Open in {high_vs_open} rows"
                ))
            
            high_vs_close = (data['High'] < data['Close']).sum()
            if high_vs_close > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.ERROR,
                    f"High < Close in {high_vs_close} rows"
                ))
            
            high_vs_low = (data['High'] < data['Low']).sum()
            if high_vs_low > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.CRITICAL,
                    f"High < Low in {high_vs_low} rows"
                ))
            
            # Low should be <= Open, Close, High
            low_vs_open = (data['Low'] > data['Open']).sum()
            if low_vs_open > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.ERROR,
                    f"Low > Open in {low_vs_open} rows"
                ))
            
            low_vs_close = (data['Low'] > data['Close']).sum()
            if low_vs_close > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.ERROR,
                    f"Low > Close in {low_vs_close} rows"
                ))
            
            # Check for zero ranges (High == Low)
            zero_range = (data['High'] == data['Low']).sum()
            if zero_range > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.WARNING,
                    f"Zero price range (High == Low) in {zero_range} rows"
                ))
        
        except Exception as e:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Error validating OHLC relationships: {e}"
            ))
        
        return issues
    
    def _validate_volume_consistency(self, volume: pd.Series) -> List[ValidationIssue]:
        """Validate volume consistency."""
        issues = []
        
        try:
            # Check for zero volume
            zero_volume = (volume == 0).sum()
            if zero_volume > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.WARNING,
                    f"Zero volume in {zero_volume} rows",
                    column='Volume'
                ))
            
            # Check for volume spikes (more than 10x average)
            if len(volume) > 10:
                avg_volume = volume.rolling(window=10, min_periods=5).mean()
                volume_spikes = (volume > avg_volume * 10).sum()
                if volume_spikes > 0:
                    issues.append(self._create_issue(
                        ValidationSeverity.INFO,
                        f"Volume spikes (>10x average) in {volume_spikes} rows",
                        column='Volume'
                    ))
        
        except Exception as e:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Error validating volume consistency: {e}",
                column='Volume'
            ))
        
        return issues
    
    def _validate_temporal_consistency(self, index: pd.DatetimeIndex) -> List[ValidationIssue]:
        """Validate temporal consistency."""
        issues = []
        
        try:
            # Check for future dates
            now = pd.Timestamp.now()
            future_dates = (index > now).sum()
            if future_dates > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.WARNING,
                    f"Found {future_dates} future dates"
                ))
            
            # Check for weekend dates (assuming stock market data)
            weekend_dates = index[index.weekday >= 5]  # Saturday=5, Sunday=6
            if len(weekend_dates) > 0:
                issues.append(self._create_issue(
                    ValidationSeverity.INFO,
                    f"Found {len(weekend_dates)} weekend dates"
                ))
            
            # Check for large gaps in data
            if len(index) > 1:
                time_diffs = index.to_series().diff().dropna()
                # Assuming daily data, gaps > 7 days might be suspicious
                large_gaps = (time_diffs > pd.Timedelta(days=7)).sum()
                if large_gaps > 0:
                    issues.append(self._create_issue(
                        ValidationSeverity.INFO,
                        f"Found {large_gaps} large time gaps (>7 days)"
                    ))
        
        except Exception as e:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Error validating temporal consistency: {e}"
            ))
        
        return issues


class CompletenessValidator(BaseValidator):
    """Validates data completeness and missing data patterns."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        super().__init__("CompletenessValidator", config)
        
        # Default thresholds
        self.max_missing_ratio = 0.05  # 5% max missing data
        self.min_trading_days = 252  # 1 year minimum
        
        # Update from config if available
        if config and hasattr(config, 'data_quality'):
            quality_config = config.data_quality
            self.max_missing_ratio = quality_config.max_missing_ratio
            self.min_trading_days = quality_config.min_trading_days
    
    def validate(self, data: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        """Validate data completeness."""
        issues = []
        
        self.stats['validations_performed'] += 1
        self.stats['last_validation'] = datetime.now()
        
        if data.empty:
            issues.append(self._create_issue(
                ValidationSeverity.CRITICAL,
                "No data available"
            ))
            self.stats['issues_found'] += len(issues)
            return issues
        
        # Check minimum data length
        if len(data) < self.min_trading_days:
            issues.append(self._create_issue(
                ValidationSeverity.ERROR,
                f"Insufficient data: {len(data)} rows < {self.min_trading_days} required",
                row_count=len(data)
            ))
        
        # Check overall missing data ratio
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        if missing_ratio > self.max_missing_ratio:
            issues.append(self._create_issue(
                ValidationSeverity.ERROR,
                f"Too much missing data: {missing_ratio:.2%} > {self.max_missing_ratio:.2%}"
            ))
        
        # Check missing data by column
        for column in data.columns:
            column_missing = data[column].isnull().sum()
            if column_missing > 0:
                column_missing_ratio = column_missing / len(data)
                severity = (ValidationSeverity.ERROR if column_missing_ratio > self.max_missing_ratio 
                          else ValidationSeverity.WARNING)
                
                issues.append(self._create_issue(
                    severity,
                    f"Missing data: {column_missing} rows ({column_missing_ratio:.2%})",
                    column=column
                ))
        
        # Check for consecutive missing data
        issues.extend(self._check_consecutive_missing(data))
        
        # Check for missing data patterns
        issues.extend(self._check_missing_patterns(data))
        
        self.stats['issues_found'] += len(issues)
        return issues
    
    def _check_consecutive_missing(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Check for consecutive missing data."""
        issues = []
        
        try:
            for column in data.columns:
                if data[column].isnull().any():
                    # Find consecutive missing data runs
                    is_missing = data[column].isnull()
                    missing_runs = []
                    current_run = 0
                    
                    for missing in is_missing:
                        if missing:
                            current_run += 1
                        else:
                            if current_run > 0:
                                missing_runs.append(current_run)
                                current_run = 0
                    
                    # Add final run if it ends with missing data
                    if current_run > 0:
                        missing_runs.append(current_run)
                    
                    # Report long consecutive missing runs
                    max_consecutive = max(missing_runs) if missing_runs else 0
                    if max_consecutive > 5:  # More than 5 consecutive missing
                        issues.append(self._create_issue(
                            ValidationSeverity.WARNING,
                            f"Long consecutive missing data: {max_consecutive} rows",
                            column=column
                        ))
        
        except Exception as e:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Error checking consecutive missing data: {e}"
            ))
        
        return issues
    
    def _check_missing_patterns(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Check for suspicious missing data patterns."""
        issues = []
        
        try:
            # Check if all columns are missing for the same rows
            if len(data.columns) > 1:
                all_missing_rows = data.isnull().all(axis=1).sum()
                if all_missing_rows > 0:
                    issues.append(self._create_issue(
                        ValidationSeverity.WARNING,
                        f"Rows with all columns missing: {all_missing_rows}"
                    ))
            
            # Check for missing data at the beginning or end
            if not data.empty:
                # Check first few rows
                first_rows_missing = data.head(10).isnull().any(axis=1).sum()
                if first_rows_missing > 5:
                    issues.append(self._create_issue(
                        ValidationSeverity.INFO,
                        f"Missing data in first 10 rows: {first_rows_missing}"
                    ))
                
                # Check last few rows
                last_rows_missing = data.tail(10).isnull().any(axis=1).sum()
                if last_rows_missing > 5:
                    issues.append(self._create_issue(
                        ValidationSeverity.INFO,
                        f"Missing data in last 10 rows: {last_rows_missing}"
                    ))
        
        except Exception as e:
            issues.append(self._create_issue(
                ValidationSeverity.WARNING,
                f"Error checking missing data patterns: {e}"
            ))
        
        return issues


class DataValidator:
    """
    Multi-stage data validation pipeline.
    
    Coordinates multiple validators to provide comprehensive data validation
    following the design pattern from design.md.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize data validator with all validation stages.
        
        Args:
            config: Data configuration for validation rules
        """
        self.config = config
        
        # Initialize all validators
        self.validators = [
            SchemaValidator(config),      # Check column presence and types
            RangeValidator(config),       # Check value ranges
            ConsistencyValidator(config), # Check OHLC relationships
            CompletenessValidator(config) # Check missing data
        ]
        
        # Overall statistics
        self.stats = {
            'validations_performed': 0,
            'total_issues_found': 0,
            'last_validation': None
        }
        
        logger.info(f"DataValidator initialized with {len(self.validators)} validators")
    
    def validate(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """
        Run complete validation pipeline.
        
        Args:
            data: DataFrame to validate
            ticker: Ticker symbol for context
            
        Returns:
            ValidationResult with all issues found
        """
        all_issues = []
        metadata = {
            'ticker': ticker,
            'data_shape': data.shape,
            'validation_timestamp': datetime.now(),
            'validators_run': []
        }
        
        self.stats['validations_performed'] += 1
        self.stats['last_validation'] = datetime.now()
        
        logger.debug(f"Starting validation for {ticker} with shape {data.shape}")
        
        # Run each validator
        for validator in self.validators:
            try:
                validator_issues = validator.validate(data, ticker)
                all_issues.extend(validator_issues)
                
                metadata['validators_run'].append({
                    'name': validator.name,
                    'issues_found': len(validator_issues),
                    'statistics': validator.get_statistics()
                })
                
                logger.debug(f"{validator.name} found {len(validator_issues)} issues")
                
            except Exception as e:
                error_issue = ValidationIssue(
                    validator=validator.name,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validator failed with error: {e}"
                )
                all_issues.append(error_issue)
                logger.error(f"{validator.name} failed: {e}")
        
        # Determine overall validity
        is_valid = not any(issue.has_errors() for issue in [ValidationResult(True, all_issues, {})])
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in all_issues)
        
        self.stats['total_issues_found'] += len(all_issues)
        
        result = ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            metadata=metadata
        )
        
        logger.info(f"Validation completed for {ticker}: "
                   f"{'PASSED' if is_valid else 'FAILED'} "
                   f"({len(all_issues)} issues)")
        
        return result
    
    def validate_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, ValidationResult]:
        """
        Validate multiple datasets.
        
        Args:
            data_dict: Dictionary mapping ticker to DataFrame
            
        Returns:
            Dictionary mapping ticker to ValidationResult
        """
        results = {}
        
        logger.info(f"Starting batch validation for {len(data_dict)} tickers")
        
        for ticker, data in data_dict.items():
            try:
                results[ticker] = self.validate(data, ticker)
            except Exception as e:
                logger.error(f"Batch validation failed for {ticker}: {e}")
                results[ticker] = ValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        validator="DataValidator",
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validation failed: {e}"
                    )],
                    metadata={'ticker': ticker, 'error': str(e)}
                )
        
        logger.info(f"Batch validation completed for {len(results)} tickers")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall validation statistics."""
        validator_stats = {}
        for validator in self.validators:
            validator_stats[validator.name] = validator.get_statistics()
        
        return {
            'overall': self.stats.copy(),
            'validators': validator_stats
        }
    
    def reset_statistics(self):
        """Reset all validation statistics."""
        self.stats = {
            'validations_performed': 0,
            'total_issues_found': 0,
            'last_validation': None
        }
        
        for validator in self.validators:
            validator.reset_statistics()
        
        logger.info("All validation statistics reset")


# Convenience functions
def validate_stock_data(data: pd.DataFrame, ticker: str, 
                       config: Optional[DataConfig] = None) -> ValidationResult:
    """
    Convenience function to validate stock data.
    
    Args:
        data: Stock data DataFrame
        ticker: Ticker symbol
        config: Optional configuration
        
    Returns:
        ValidationResult
    """
    validator = DataValidator(config)
    return validator.validate(data, ticker)


def create_validation_report(results: Dict[str, ValidationResult]) -> str:
    """
    Create a human-readable validation report.
    
    Args:
        results: Dictionary of validation results
        
    Returns:
        Formatted report string
    """
    report_lines = ["Data Validation Report", "=" * 50, ""]
    
    total_tickers = len(results)
    passed_tickers = sum(1 for result in results.values() if result.is_valid)
    failed_tickers = total_tickers - passed_tickers
    
    report_lines.extend([
        f"Total tickers validated: {total_tickers}",
        f"Passed: {passed_tickers}",
        f"Failed: {failed_tickers}",
        ""
    ])
    
    # Summary by severity
    all_issues = []
    for result in results.values():
        all_issues.extend(result.issues)
    
    if all_issues:
        severity_counts = {}
        for issue in all_issues:
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
        
        report_lines.extend(["Issue Summary:", "-" * 20])
        for severity, count in severity_counts.items():
            report_lines.append(f"{severity.upper()}: {count}")
        report_lines.append("")
    
    # Details for failed tickers
    if failed_tickers > 0:
        report_lines.extend(["Failed Tickers:", "-" * 20])
        for ticker, result in results.items():
            if not result.is_valid:
                report_lines.append(f"\n{ticker}:")
                for issue in result.issues:
                    if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        report_lines.append(f"  - {issue}")
    
    return "\n".join(report_lines)