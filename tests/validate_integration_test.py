#!/usr/bin/env python3
"""
Integration Test Validation Suite

This test validates that the complete pipeline test (test_complete_pipeline.py) actually
tests everything properly and catches all failure modes.

Requirements:
1. Verify test coverage - confirm all components are tested
2. Test failure detection by intentionally breaking each component
3. Validate performance and memory usage
4. Check reproducibility with multiple test runs
5. Validate JSON report completeness and format
"""

import sys
import os
import json
import time
import subprocess
import tempfile
import shutil
import psutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_complete_pipeline import EndToEndPipelineTest, CompleteTestResult


class IntegrationTestValidator:
    """Validates the integration test comprehensively."""
    
    def __init__(self):
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_success': False,
            'coverage_validation': {},
            'failure_detection': {},
            'performance_validation': {},
            'reproducibility_check': {},
            'report_validation': {},
            'errors': []
        }
        self.test_script_path = Path(__file__).parent.parent / "scripts" / "test_complete_pipeline.py"
        
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation of the integration test."""
        print("="*80)
        print("INTEGRATION TEST VALIDATION SUITE")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 1. Verify test coverage
            print("\n[VALIDATION 1] Test Coverage Verification")
            print("-" * 50)
            self._validate_test_coverage()
            
            # 2. Test failure detection
            print("\n[VALIDATION 2] Failure Detection Testing")
            print("-" * 50)
            self._test_failure_detection()
            
            # 3. Performance validation
            print("\n[VALIDATION 3] Performance Validation")
            print("-" * 50)
            self._validate_performance()
            
            # 4. Reproducibility check
            print("\n[VALIDATION 4] Reproducibility Check")
            print("-" * 50)
            self._check_reproducibility()
            
            # 5. Report validation
            print("\n[VALIDATION 5] Report Validation")
            print("-" * 50)
            self._validate_report()
            
            # Overall validation success
            all_validations_passed = (
                self.validation_results['coverage_validation'].get('passed', False) and
                self.validation_results['failure_detection'].get('passed', False) and
                self.validation_results['performance_validation'].get('passed', False) and
                self.validation_results['reproducibility_check'].get('passed', False) and
                self.validation_results['report_validation'].get('passed', False)
            )
            
            self.validation_results['validation_success'] = all_validations_passed
            
        except Exception as e:
            error_msg = f"Validation failed with error: {str(e)}\n{traceback.format_exc()}"
            self.validation_results['errors'].append(error_msg)
            print(f"\nVALIDATION ERROR: {error_msg}")
        
        finally:
            total_time = time.time() - start_time
            self.validation_results['total_validation_time'] = total_time
            
            print(f"\n{'='*80}")
            print(f"VALIDATION COMPLETED")
            print(f"Success: {self.validation_results['validation_success']}")
            print(f"Total Time: {total_time:.2f} seconds")
            print(f"{'='*80}")
        
        return self.validation_results
    
    def _validate_test_coverage(self):
        """Verify that the integration test covers all required components."""
        coverage_results = {
            'components_tested': [],
            'missing_components': [],
            'assertions_verified': [],
            'passed': False
        }
        
        try:
            # Read the integration test script
            with open(self.test_script_path, 'r') as f:
                test_script_content = f.read()
            
            # Check for required components
            required_components = [
                'data loading',
                'feature engineering', 
                'dataset creation',
                'model initialization',
                'training configuration',
                'training orchestrator',
                'verification'
            ]
            
            # Check each component is tested
            for component in required_components:
                component_patterns = {
                    'data loading': ['_test_data_loading_stage', 'pd.read_parquet', 'raw_data'],
                    'feature engineering': ['_test_feature_engineering_stage', 'FeatureEngineer', 'engineer_features'],
                    'dataset creation': ['_test_dataset_creation_stage', 'StockSequenceDataset', 'sequences'],
                    'model initialization': ['_test_model_initialization_stage', 'TimeSeriesTransformer', 'model'],
                    'training configuration': ['_test_training_configuration_stage', 'TrainingConfig', 'from_args'],
                    'training orchestrator': ['_test_training_orchestrator_stage', 'TrainingOrchestrator', 'train'],
                    'verification': ['_test_verification_stage', 'verification', 'data_flow']
                }
                
                patterns = component_patterns.get(component, [])
                found = any(pattern in test_script_content for pattern in patterns)
                
                if found:
                    coverage_results['components_tested'].append(component)
                    print(f"+ {component}: TESTED")
                else:
                    coverage_results['missing_components'].append(component)
                    print(f"X {component}: MISSING")
            
            # Check for meaningful assertions
            assertion_patterns = [
                'assert',
                'raise ValueError',
                'raise FileNotFoundError', 
                'shape',
                'dtype',
                'isfinite',
                'validate'
            ]
            
            found_assertions = []
            for pattern in assertion_patterns:
                if pattern in test_script_content:
                    count = test_script_content.count(pattern)
                    found_assertions.append(f"{pattern}({count})")
                    coverage_results['assertions_verified'].append(pattern)
            
            print(f"+ Assertions found: {', '.join(found_assertions)}")
            
            # Check for stage results tracking
            stage_tracking = [
                'add_stage_result',
                'success',
                'duration',
                'metrics',
                'error'
            ]
            
            tracking_found = sum(1 for pattern in stage_tracking if pattern in test_script_content)
            print(f"+ Stage tracking patterns: {tracking_found}/{len(stage_tracking)}")
            
            # Determine if coverage is adequate
            coverage_score = (
                len(coverage_results['components_tested']) / len(required_components) * 0.6 +
                min(len(coverage_results['assertions_verified']) / len(assertion_patterns), 1.0) * 0.3 +
                min(tracking_found / len(stage_tracking), 1.0) * 0.1
            )
            
            coverage_results['coverage_score'] = coverage_score
            coverage_results['passed'] = coverage_score >= 0.8
            
            print(f"+ Coverage Score: {coverage_score:.2f}/1.0 ({'PASS' if coverage_results['passed'] else 'FAIL'})")
            
        except Exception as e:
            coverage_results['error'] = str(e)
            print(f"X Coverage validation failed: {e}")
        
        self.validation_results['coverage_validation'] = coverage_results
    
    def _test_failure_detection(self):
        """Test that the integration test catches failures by breaking components."""
        failure_results = {
            'broken_components_tested': [],
            'failures_detected': [],
            'failures_missed': [],
            'passed': False
        }
        
        # Test scenarios to validate failure detection
        test_scenarios = [
            {
                'name': 'missing_parquet_file',
                'description': 'Test with non-existent data file',
                'setup': self._create_missing_file_scenario,
                'expected_stage': '1_data_loading'
            },
            {
                'name': 'corrupted_data',
                'description': 'Test with invalid data format',
                'setup': self._create_corrupted_data_scenario,
                'expected_stage': '1_data_loading'
            },
            {
                'name': 'feature_engineering_failure',
                'description': 'Test feature engineering with invalid data',
                'setup': self._create_feature_failure_scenario,
                'expected_stage': '2_feature_engineering'
            },
            {
                'name': 'insufficient_data_for_sequences',
                'description': 'Test with insufficient data for sequence creation',
                'setup': self._create_insufficient_data_scenario,
                'expected_stage': '3_dataset_creation'
            },
            {
                'name': 'model_invalid_params',
                'description': 'Test with invalid model parameters',
                'setup': self._create_model_param_failure_scenario,
                'expected_stage': '4_model_initialization'
            },
            {
                'name': 'training_config_invalid',
                'description': 'Test with invalid training configuration',
                'setup': self._create_training_config_failure_scenario,
                'expected_stage': '5_training_configuration'
            }
        ]
        
        try:
            for scenario in test_scenarios:
                print(f"\nTesting: {scenario['description']}")
                
                # Create test environment
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # Setup failure scenario
                        scenario['setup'](temp_dir)
                        
                        # Run integration test and expect failure
                        result = self._run_integration_test_isolated(temp_dir)
                        
                        # Check if failure was detected
                        if result and not result.overall_success:
                            # Check if failure occurred in expected stage
                            expected_stage = scenario['expected_stage']
                            failed_stages = [stage for stage, data in result.stages.items() 
                                           if not data['success']]
                            
                            if expected_stage in failed_stages:
                                failure_results['failures_detected'].append(scenario['name'])
                                print(f"  + {scenario['name']}: Correctly detected failure in {expected_stage}")
                            else:
                                failure_results['failures_missed'].append(f"{scenario['name']}_wrong_stage")
                                print(f"  X {scenario['name']}: Failed in wrong stage {failed_stages}")
                        else:
                            failure_results['failures_missed'].append(scenario['name'])
                            print(f"  X {scenario['name']}: Failed to detect failure")
                        
                        failure_results['broken_components_tested'].append(scenario['name'])
                        
                    except Exception as e:
                        failure_results['failures_missed'].append(f"{scenario['name']}_exception")
                        print(f"  X {scenario['name']}: Exception during test: {e}")
            
            # Calculate success rate
            total_tests = len(test_scenarios)
            detected_failures = len(failure_results['failures_detected'])
            detection_rate = detected_failures / total_tests if total_tests > 0 else 0
            
            failure_results['detection_rate'] = detection_rate
            failure_results['passed'] = detection_rate >= 0.75  # At least 75% detection rate
            
            print(f"\n+ Failure Detection Rate: {detected_failures}/{total_tests} ({detection_rate:.1%})")
            print(f"+ Detection Validation: {'PASS' if failure_results['passed'] else 'FAIL'}")
            
        except Exception as e:
            failure_results['error'] = str(e)
            print(f"X Failure detection testing failed: {e}")
        
        self.validation_results['failure_detection'] = failure_results
    
    def _validate_performance(self):
        """Validate test performance and memory usage."""
        perf_results = {
            'execution_times': [],
            'memory_usage': [],
            'performance_metrics': {},
            'passed': False
        }
        
        try:
            print("Running performance tests...")
            
            # Run test multiple times to measure performance
            num_runs = 3
            
            for run in range(num_runs):
                print(f"  Performance run {run + 1}/{num_runs}")
                
                # Measure memory before test
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run integration test with timing
                start_time = time.time()
                
                # Use subprocess to isolate memory measurement
                result = subprocess.run([
                    sys.executable, str(self.test_script_path)
                ], capture_output=True, text=True, cwd=str(self.test_script_path.parent.parent))
                
                execution_time = time.time() - start_time
                
                # Measure memory after test
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                perf_results['execution_times'].append(execution_time)
                perf_results['memory_usage'].append(memory_used)
                
                print(f"    Execution time: {execution_time:.2f}s")
                print(f"    Memory usage: {memory_used:.2f}MB")
                
                # Small delay between runs
                time.sleep(0.5)
            
            # Calculate performance metrics
            avg_time = sum(perf_results['execution_times']) / len(perf_results['execution_times'])
            max_time = max(perf_results['execution_times'])
            avg_memory = sum(perf_results['memory_usage']) / len(perf_results['memory_usage'])
            max_memory = max(perf_results['memory_usage'])
            
            perf_results['performance_metrics'] = {
                'average_execution_time': avg_time,
                'max_execution_time': max_time,
                'average_memory_usage': avg_memory,
                'max_memory_usage': max_memory,
                'execution_time_std': np.std(perf_results['execution_times']),
                'memory_usage_std': np.std(perf_results['memory_usage'])
            }
            
            # Performance requirements validation
            time_acceptable = max_time < 30.0  # Should complete in under 30 seconds
            memory_acceptable = max_memory < 500.0  # Should use less than 500MB extra
            consistency_acceptable = np.std(perf_results['execution_times']) < 2.0  # Low variance
            
            perf_results['requirements_met'] = {
                'execution_time_acceptable': time_acceptable,
                'memory_usage_acceptable': memory_acceptable,
                'consistency_acceptable': consistency_acceptable
            }
            
            perf_results['passed'] = all(perf_results['requirements_met'].values())
            
            print(f"\n+ Performance Metrics:")
            print(f"  Average execution time: {avg_time:.2f}s")
            print(f"  Maximum execution time: {max_time:.2f}s")
            print(f"  Average memory usage: {avg_memory:.2f}MB")
            print(f"  Maximum memory usage: {max_memory:.2f}MB")
            print(f"+ Requirements: Time<30s: {time_acceptable}, Memory<500MB: {memory_acceptable}")
            print(f"+ Performance Validation: {'PASS' if perf_results['passed'] else 'FAIL'}")
            
        except Exception as e:
            perf_results['error'] = str(e)
            print(f"X Performance validation failed: {e}")
        
        self.validation_results['performance_validation'] = perf_results
    
    def _check_reproducibility(self):
        """Check that the integration test produces consistent results."""
        repro_results = {
            'test_runs': [],
            'consistency_metrics': {},
            'seed_handling': {},
            'passed': False
        }
        
        try:
            print("Testing reproducibility...")
            
            num_runs = 3
            results = []
            
            for run in range(num_runs):
                print(f"  Reproducibility run {run + 1}/{num_runs}")
                
                # Run the integration test
                test_instance = EndToEndPipelineTest(test_ticker="AAPL")
                result = test_instance.run_complete_test()
                
                if result.overall_success:
                    # Extract key metrics for comparison
                    run_data = {
                        'overall_success': result.overall_success,
                        'total_duration': result.total_duration,
                        'stages_success': {stage: data['success'] for stage, data in result.stages.items()},
                        'data_shape': result.stages.get('1_data_loading', {}).get('metrics', {}).get('data_shape', []),
                        'feature_count': result.stages.get('2_feature_engineering', {}).get('metrics', {}).get('feature_columns', 0),
                        'sequence_count': result.stages.get('3_dataset_creation', {}).get('metrics', {}).get('total_sequences', 0),
                        'model_params': result.stages.get('4_model_initialization', {}).get('metrics', {}).get('total_parameters', 0)
                    }
                    results.append(run_data)
                    repro_results['test_runs'].append(run_data)
                    
                    print(f"    Success: {result.overall_success}")
                    print(f"    Duration: {result.total_duration:.2f}s")
                else:
                    print(f"    Run {run + 1} failed - cannot test reproducibility")
                    repro_results['test_runs'].append({'failed': True, 'errors': result.errors})
            
            # Analyze consistency
            if len(results) >= 2:
                # Check deterministic values (should be identical)
                deterministic_keys = ['data_shape', 'feature_count', 'sequence_count', 'model_params']
                
                consistency_check = {}
                for key in deterministic_keys:
                    values = [run[key] for run in results if key in run]
                    if len(values) > 1:
                        all_same = all(v == values[0] for v in values)
                        consistency_check[key] = {
                            'consistent': all_same,
                            'values': values
                        }
                        print(f"    {key}: {'CONSISTENT' if all_same else 'INCONSISTENT'} - {values}")
                
                # Check timing variation (should be reasonable)
                durations = [run['total_duration'] for run in results]
                duration_std = np.std(durations) if len(durations) > 1 else 0
                duration_cv = duration_std / np.mean(durations) if np.mean(durations) > 0 else 0
                
                consistency_check['timing_variation'] = {
                    'durations': durations,
                    'std_deviation': duration_std,
                    'coefficient_of_variation': duration_cv,
                    'acceptable': duration_cv < 0.5  # Less than 50% variation (relaxed for integration test)
                }
                
                repro_results['consistency_metrics'] = consistency_check
                
                # Overall consistency score
                deterministic_consistent = all(check['consistent'] for check in consistency_check.values() 
                                             if isinstance(check, dict) and 'consistent' in check)
                timing_acceptable = consistency_check['timing_variation']['acceptable']
                
                repro_results['passed'] = deterministic_consistent and timing_acceptable
                
                print(f"+ Deterministic values consistent: {deterministic_consistent}")
                print(f"+ Timing variation acceptable: {timing_acceptable} (CV: {duration_cv:.3f})")
                print(f"+ Reproducibility: {'PASS' if repro_results['passed'] else 'FAIL'}")
                
            else:
                repro_results['error'] = "Insufficient successful runs for reproducibility check"
                print("X Insufficient successful runs for reproducibility analysis")
            
        except Exception as e:
            repro_results['error'] = str(e)
            print(f"X Reproducibility check failed: {e}")
        
        self.validation_results['reproducibility_check'] = repro_results
    
    def _validate_report(self):
        """Validate that the JSON report contains all required metrics."""
        report_results = {
            'required_fields_present': [],
            'missing_fields': [],
            'data_quality': {},
            'human_readability': {},
            'passed': False
        }
        
        try:
            print("Validating report format and content...")
            
            # Run integration test to generate report
            test_instance = EndToEndPipelineTest(test_ticker="AAPL")
            result = test_instance.run_complete_test()
            
            # Save and load JSON to validate format
            temp_file = "temp_validation_results.json"
            result.save_to_file(temp_file)
            
            with open(temp_file, 'r') as f:
                report_data = json.load(f)
            
            os.remove(temp_file)  # Clean up
            
            # Check required top-level fields
            required_fields = [
                'timestamp',
                'overall_success', 
                'total_duration_seconds',
                'stages',
                'errors',
                'final_metrics'
            ]
            
            for field in required_fields:
                if field in report_data:
                    report_results['required_fields_present'].append(field)
                    print(f"  + {field}: Present")
                else:
                    report_results['missing_fields'].append(field)
                    print(f"  X {field}: Missing")
            
            # Check stage structure
            if 'stages' in report_data:
                expected_stages = [
                    '1_data_loading',
                    '2_feature_engineering', 
                    '3_dataset_creation',
                    '4_model_initialization',
                    '5_training_configuration',
                    '6_training_orchestrator',
                    '7_verification'
                ]
                
                present_stages = []
                for stage in expected_stages:
                    if stage in report_data['stages']:
                        present_stages.append(stage)
                        # Check stage structure
                        stage_data = report_data['stages'][stage]
                        required_stage_fields = ['success', 'duration_seconds', 'metrics']
                        
                        stage_complete = all(field in stage_data for field in required_stage_fields)
                        print(f"    {stage}: {'Complete' if stage_complete else 'Incomplete'}")
                    else:
                        print(f"    X {stage}: Missing")
                
                report_results['stages_present'] = present_stages
            
            # Validate numerical data quality
            numerical_checks = {
                'durations_positive': True,
                'metrics_present': True,
                'no_null_values': True
            }
            
            # Check for reasonable durations
            if 'stages' in report_data:
                for stage_name, stage_data in report_data['stages'].items():
                    duration = stage_data.get('duration_seconds', -1)
                    if duration < 0 or duration > 60:  # Reasonable bounds
                        numerical_checks['durations_positive'] = False
                        print(f"    X {stage_name}: Unreasonable duration {duration}s")
            
            # Check for metrics presence
            metrics_count = 0
            if 'stages' in report_data:
                for stage_data in report_data['stages'].values():
                    if 'metrics' in stage_data and stage_data['metrics']:
                        metrics_count += len(stage_data['metrics'])
            
            numerical_checks['metrics_present'] = metrics_count > 20  # Expect substantial metrics
            print(f"    Metrics count: {metrics_count}")
            
            report_results['data_quality'] = numerical_checks
            
            # Human readability check
            readability_checks = {
                'has_timestamp': 'timestamp' in report_data,
                'has_error_messages': len(report_data.get('errors', [])) == 0 or all(
                    isinstance(error, str) and len(error) > 10 for error in report_data.get('errors', [])
                ),
                'has_final_summary': 'final_metrics' in report_data,
                'json_valid': True  # Already validated by successful parsing
            }
            
            report_results['human_readability'] = readability_checks
            
            # Overall report validation
            fields_complete = len(report_results['missing_fields']) == 0
            data_quality_good = all(numerical_checks.values())
            human_readable = all(readability_checks.values())
            
            report_results['passed'] = fields_complete and data_quality_good and human_readable
            
            print(f"+ Required fields complete: {fields_complete}")
            print(f"+ Data quality good: {data_quality_good}")
            print(f"+ Human readable: {human_readable}")
            print(f"+ Report Validation: {'PASS' if report_results['passed'] else 'FAIL'}")
            
        except Exception as e:
            report_results['error'] = str(e)
            print(f"X Report validation failed: {e}")
        
        self.validation_results['report_validation'] = report_results
    
    # Helper methods for failure scenarios
    
    def _create_missing_file_scenario(self, temp_dir):
        """Create scenario with missing parquet file."""
        # Don't create any files - this will cause file not found error
        pass
    
    def _create_corrupted_data_scenario(self, temp_dir):
        """Create scenario with corrupted data file."""
        data_dir = Path(temp_dir) / "data" / "raw" / "AAPL"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create corrupted parquet file (empty file)
        corrupted_file = data_dir / "corrupted.parquet"
        corrupted_file.write_text("not a parquet file")
    
    def _create_feature_failure_scenario(self, temp_dir):
        """Create scenario that causes feature engineering to fail."""
        data_dir = Path(temp_dir) / "data" / "raw" / "AAPL"  
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data that will pass loading but fail feature engineering
        # Use data with all infinite values which will break calculations
        dates = pd.date_range('2023-01-01', periods=100)
        bad_data = pd.DataFrame({
            'Open': [float('inf')] * 100,  # Infinite values will break feature calculations
            'High': [float('inf')] * 100,
            'Low': [float('inf')] * 100,
            'Close': [float('inf')] * 100,
            'Volume': [float('inf')] * 100,
            'Ticker': ['AAPL'] * 100
        }, index=dates)
        
        parquet_file = data_dir / "feature_fail.parquet"
        bad_data.to_parquet(parquet_file)
    
    def _create_insufficient_data_scenario(self, temp_dir):
        """Create scenario with insufficient data for sequence creation."""
        data_dir = Path(temp_dir) / "data" / "raw" / "AAPL"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data with only 10 rows - insufficient for 60-day sequences
        dates = pd.date_range('2023-01-01', periods=10)
        insufficient_data = pd.DataFrame({
            'Open': range(10),
            'High': range(1, 11),
            'Low': range(10), 
            'Close': range(10),
            'Volume': range(100, 110),
            'Ticker': ['AAPL'] * 10
        }, index=dates)
        
        parquet_file = data_dir / "insufficient_data.parquet"
        insufficient_data.to_parquet(parquet_file)
    
    def _create_model_param_failure_scenario(self, temp_dir):
        """Create scenario that will cause model initialization to fail."""
        data_dir = Path(temp_dir) / "data" / "raw" / "AAPL"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sufficient valid data (need at least 100+ rows after cleaning for 60-day sequences)
        # Use realistic stock data values to pass feature engineering
        dates = pd.date_range('2023-01-01', periods=300)
        np.random.seed(42)  # For reproducible test data
        
        # Generate realistic stock price data
        base_price = 100.0
        price_changes = np.random.randn(300) * 2  # 2% daily volatility
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change / 100))
        
        valid_data = pd.DataFrame({
            'Open': [p * (1 + np.random.randn() * 0.01) for p in prices],
            'High': [p * (1 + abs(np.random.randn()) * 0.02) for p in prices], 
            'Low': [p * (1 - abs(np.random.randn()) * 0.02) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 300),
            'Ticker': ['AAPL'] * 300
        }, index=dates)
        
        # Ensure High >= Close >= Low
        for i in range(len(valid_data)):
            close = valid_data.iloc[i]['Close']
            valid_data.iloc[i, valid_data.columns.get_loc('High')] = max(valid_data.iloc[i]['High'], close)
            valid_data.iloc[i, valid_data.columns.get_loc('Low')] = min(valid_data.iloc[i]['Low'], close)
        
        parquet_file = data_dir / "model_param_fail.parquet"
        valid_data.to_parquet(parquet_file)
        
        # This will be caught by modifying model parameters in the isolated test
        # We'll patch the model creation to use invalid parameters
    
    def _create_training_config_failure_scenario(self, temp_dir):
        """Create scenario that will cause training configuration to fail."""
        data_dir = Path(temp_dir) / "data" / "raw" / "AAPL"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sufficient valid data that will pass through all stages until training config
        dates = pd.date_range('2023-01-01', periods=300)
        np.random.seed(43)  # Different seed than model scenario
        
        # Generate realistic stock price data
        base_price = 150.0
        price_changes = np.random.randn(300) * 1.5
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change / 100))
        
        valid_data = pd.DataFrame({
            'Open': [p * (1 + np.random.randn() * 0.01) for p in prices],
            'High': [p * (1 + abs(np.random.randn()) * 0.02) for p in prices], 
            'Low': [p * (1 - abs(np.random.randn()) * 0.02) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 300),
            'Ticker': ['AAPL'] * 300
        }, index=dates)
        
        # Ensure High >= Close >= Low
        for i in range(len(valid_data)):
            close = valid_data.iloc[i]['Close']
            valid_data.iloc[i, valid_data.columns.get_loc('High')] = max(valid_data.iloc[i]['High'], close)
            valid_data.iloc[i, valid_data.columns.get_loc('Low')] = min(valid_data.iloc[i]['Low'], close)
        
        parquet_file = data_dir / "training_config_fail.parquet"
        valid_data.to_parquet(parquet_file)
    
    def _run_integration_test_isolated(self, temp_dir):
        """Run integration test in isolated environment."""
        # Copy the original test data files to temp directory if they don't exist
        original_data_dir = Path.cwd() / "data" / "raw" / "AAPL"
        temp_data_dir = Path(temp_dir) / "data" / "raw" / "AAPL"
        
        # If temp dir doesn't have the original data file, copy it
        if original_data_dir.exists() and temp_data_dir.exists():
            original_files = list(original_data_dir.glob("*.parquet"))
            temp_files = list(temp_data_dir.glob("*.parquet"))
            
            # Only proceed with failure injection if we have temp files
            if temp_files:
                filename = temp_files[0].name
                
                # Import the test class without changing directories
                test_instance = EndToEndPipelineTest(test_ticker="AAPL")
                
                # Apply monkey patching based on scenario
                if "feature_fail" in filename:
                    # Patch FeatureEngineer to fail
                    import src.data.processors.feature_engineering as fe
                    original_engineer_features = fe.FeatureEngineer.engineer_features
                    
                    def failing_engineer_features(self, data):
                        raise ValueError("Feature engineering forced to fail for testing")
                    
                    fe.FeatureEngineer.engineer_features = failing_engineer_features
                    
                    try:
                        # Change directory only for the test execution
                        original_cwd = os.getcwd()
                        os.chdir(temp_dir)
                        result = test_instance.run_complete_test()
                        os.chdir(original_cwd)
                        return result
                    finally:
                        fe.FeatureEngineer.engineer_features = original_engineer_features
                        
                elif "model_param_fail" in filename:
                    # Patch TimeSeriesTransformer constructor to fail
                    from src.models.timeseries_transformer import TimeSeriesTransformer
                    original_init = TimeSeriesTransformer.__init__
                    
                    def failing_init(self, *args, **kwargs):
                        raise ValueError("Model initialization forced to fail for testing")
                    
                    TimeSeriesTransformer.__init__ = failing_init
                    
                    try:
                        original_cwd = os.getcwd()
                        os.chdir(temp_dir)
                        result = test_instance.run_complete_test()
                        os.chdir(original_cwd)
                        return result
                    finally:
                        TimeSeriesTransformer.__init__ = original_init
                        
                elif "training_config_fail" in filename:
                    # Patch TrainingConfig.from_args to fail
                    from src.config.training_config import TrainingConfig
                    original_from_args = TrainingConfig.from_args
                    
                    @classmethod
                    def failing_from_args(cls, args_dict):
                        raise ValueError("TrainingConfig validation forced to fail for testing")
                    
                    TrainingConfig.from_args = failing_from_args
                    
                    try:
                        original_cwd = os.getcwd()
                        os.chdir(temp_dir)
                        result = test_instance.run_complete_test()
                        os.chdir(original_cwd)
                        return result
                    finally:
                        TrainingConfig.from_args = original_from_args
                        
                elif "insufficient_data" in filename:
                    # This scenario should naturally fail - just run the test
                    original_cwd = os.getcwd()
                    os.chdir(temp_dir)
                    result = test_instance.run_complete_test()
                    os.chdir(original_cwd)
                    return result
        
        # Handle scenarios that just need corrupted/missing data (no monkey patching needed)
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        test_instance = EndToEndPipelineTest(test_ticker="AAPL")
        result = test_instance.run_complete_test()
        os.chdir(original_cwd)
        return result
    
    def save_validation_report(self, filepath: str):
        """Save detailed validation report."""
        with open(filepath, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)


def main():
    """Run the integration test validation."""
    print("Starting Integration Test Validation...")
    
    validator = IntegrationTestValidator()
    results = validator.run_validation()
    
    # Save validation report
    report_file = "integration_test_validation_report.json"
    validator.save_validation_report(report_file)
    
    # Summary
    print(f"\nValidation report saved to: {report_file}")
    
    if results['validation_success']:
        print("\nINTEGRATION TEST VALIDATION: SUCCESS!")
        print("The integration test properly validates all components and catches failures.")
    else:
        print(f"\nINTEGRATION TEST VALIDATION: FAILED")
        print(f"Issues found ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    return 0 if results['validation_success'] else 1


if __name__ == "__main__":
    sys.exit(main())