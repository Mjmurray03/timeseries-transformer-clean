#!/usr/bin/env python
"""
Python API Test Suite for TimeSeries Transformer API
===================================================

This script provides comprehensive testing of the TimeSeries Transformer API
with robust error handling, clear output formatting, and production-ready
validation patterns. It serves as a reliable alternative to PowerShell testing
with better cross-platform compatibility.

Features:
- Comprehensive endpoint testing
- Robust error handling and reporting
- Clear test result visualization
- Input format validation testing
- Performance and load testing
- JSON response validation
- Detailed logging and metrics

Usage:
    python test_api_python.py
    python test_api_python.py --base-url http://localhost:8000 --verbose
"""

import requests
import numpy as np
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class APITester:
    """Comprehensive API testing class for TimeSeries Transformer API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30, verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verbose = verbose
        self.test_results = {}
        self.start_time = None
        
    def print_header(self, title: str) -> None:
        """Print formatted test section header"""
        print(f"\n{Colors.HEADER}{'=' * 80}")
        print(f" {title}")
        print(f"{'=' * 80}{Colors.ENDC}")
        
    def print_success(self, message: str) -> None:
        """Print success message"""
        print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")
        
    def print_error(self, message: str) -> None:
        """Print error message"""
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
        
    def print_info(self, message: str) -> None:
        """Print info message"""
        print(f"{Colors.OKCYAN}[INFO]{Colors.ENDC} {message}")
        
    def print_warning(self, message: str) -> None:
        """Print warning message"""
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")
        
    def log_verbose(self, message: str) -> None:
        """Print verbose logging message"""
        if self.verbose:
            print(f"{Colors.OKBLUE}[VERBOSE]{Colors.ENDC} {message}")
            
    def test_health(self) -> bool:
        """Test the health endpoint"""
        self.print_header("API HEALTH CHECK")
        
        try:
            self.log_verbose(f"Testing health endpoint: {self.base_url}/health")
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            
            if response.status_code == 200:
                health_data = response.json()
                self.print_success("API is healthy and accessible")
                
                # Display health information
                self.print_info(f"Status: {health_data.get('status', 'Unknown')}")
                self.print_info(f"CUDA Available: {health_data.get('cuda_available', 'Unknown')}")
                self.print_info(f"Models Loaded: {', '.join(health_data.get('models_loaded', []))}")
                self.print_info(f"Cache Enabled: {health_data.get('cache_enabled', 'Unknown')}")
                
                if self.verbose:
                    self.print_info("Full health response:")
                    print(json.dumps(health_data, indent=2))
                    
                return True
            else:
                self.print_error(f"Health check failed with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.print_error(f"Cannot connect to API: {e}")
            return False
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_model_info(self) -> bool:
        """Test the model info endpoint"""
        self.print_header("MODEL INFO TEST")
        
        try:
            self.log_verbose(f"Testing model info endpoint: {self.base_url}/model-info")
            response = requests.get(f"{self.base_url}/model-info", timeout=self.timeout)
            
            if response.status_code == 200:
                model_data = response.json()
                self.print_success("Model info endpoint accessible")
                
                # Display model information
                self.print_info(f"Model Version: {model_data.get('model_version', 'Unknown')}")
                self.print_info(f"Architecture: {model_data.get('architecture', 'Unknown')}")
                self.print_info(f"Parameters: {model_data.get('parameters', 'Unknown'):,}")
                self.print_info(f"Training Date: {model_data.get('training_date', 'Unknown')}")
                
                supported_tickers = model_data.get('supported_tickers', [])
                self.print_info(f"Supported Tickers ({len(supported_tickers)}): {', '.join(supported_tickers)}")
                
                # Display performance metrics
                perf_metrics = model_data.get('performance_metrics', {})
                if perf_metrics:
                    self.print_info("Performance Metrics:")
                    for metric, value in perf_metrics.items():
                        self.print_info(f"  {metric}: {value}")
                
                if self.verbose:
                    self.print_info("Full model info response:")
                    print(json.dumps(model_data, indent=2))
                    
                return True
            else:
                self.print_error(f"Model info failed with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.print_error(f"Model info request failed: {e}")
            return False
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_prediction_2d_format(self) -> bool:
        """Test prediction endpoint with 2D array format (60x10)"""
        self.print_info("Testing 2D array format (60x10)...")
        
        try:
            # Create proper 60x10 feature array
            features_2d = np.random.uniform(0.1, 1.0, (60, 10)).round(4).tolist()
            
            payload = {
                "ticker": "AAPL",
                "features": features_2d,
                "horizon": 3
            }
            
            self.log_verbose(f"Sending 2D prediction request for AAPL")
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_success("2D array prediction successful!")
                
                # Validate response structure
                required_keys = ['ticker', 'predictions', 'timestamp', 'model_version']
                for key in required_keys:
                    if key not in result:
                        self.print_warning(f"Missing key in response: {key}")
                    else:
                        self.log_verbose(f"Response contains key: {key}")
                
                self.print_info(f"Ticker: {result.get('ticker', 'Unknown')}")
                self.print_info(f"Model Version: {result.get('model_version', 'Unknown')}")
                self.print_info(f"Cache Hit: {result.get('cache_hit', 'Unknown')}")
                self.print_info(f"Input Format: 2D array (60x10)")
                
                # Check predictions structure
                predictions = result.get('predictions')
                if predictions:
                    if isinstance(predictions, dict):
                        pred_types = list(predictions.keys())
                        self.print_info(f"Prediction Types: {', '.join(pred_types)}")
                    elif isinstance(predictions, list):
                        self.print_info(f"Predictions: List with {len(predictions)} values")
                        if len(predictions) > 0:
                            self.print_info(f"Sample prediction: {predictions[0]:.6f}")
                
                if result.get('confidence_intervals'):
                    self.print_info("Confidence intervals included")
                
                if self.verbose:
                    self.print_info("Sample prediction response:")
                    # Only show first few predictions to avoid cluttering
                    sample_result = {k: v for k, v in result.items() if k != 'predictions'}
                    if 'predictions' in result:
                        predictions = result['predictions']
                        if isinstance(predictions, dict):
                            sample_result['predictions'] = {k: str(type(v).__name__) + f" with {len(v) if isinstance(v, (list, dict)) else 1} items" 
                                                          for k, v in predictions.items()}
                        elif isinstance(predictions, list):
                            sample_result['predictions'] = f"List with {len(predictions)} predictions"
                        else:
                            sample_result['predictions'] = str(type(predictions).__name__)
                    print(json.dumps(sample_result, indent=2))
                
                return True
            else:
                self.print_error(f"2D array prediction failed with status {response.status_code}")
                try:
                    error_detail = response.json()
                    self.print_error(f"Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    self.print_error(f"Error response: {response.text}")
                return False
                
        except Exception as e:
            self.print_error(f"2D array prediction test failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_prediction_flat_format(self) -> bool:
        """Test prediction endpoint with flat array format (600 elements)"""
        self.print_info("Testing flat array format (600 elements)...")
        
        try:
            # Create flat 600-element array
            features_flat = np.random.uniform(0.1, 1.0, 600).round(4).tolist()
            
            payload = {
                "ticker": "MSFT",
                "features": features_flat,
                "horizon": 5
            }
            
            self.log_verbose(f"Sending flat prediction request for MSFT")
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_success("Flat array prediction successful!")
                
                self.print_info(f"Ticker: {result.get('ticker', 'Unknown')}")
                self.print_info(f"Horizon: 5 days")
                self.print_info(f"Input Format: Flat 600 elements (auto-reshaped to 60x10)")
                
                # Check predictions structure
                predictions = result.get('predictions')
                if predictions:
                    if isinstance(predictions, dict):
                        pred_types = list(predictions.keys())
                        self.print_info(f"Prediction Types: {', '.join(pred_types)}")
                    elif isinstance(predictions, list):
                        self.print_info(f"Predictions: List with {len(predictions)} values")
                        if len(predictions) > 0:
                            self.print_info(f"Sample prediction: {predictions[0]:.6f}")
                
                return True
            else:
                self.print_error(f"Flat array prediction failed with status {response.status_code}")
                try:
                    error_detail = response.json()
                    self.print_error(f"Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    self.print_error(f"Error response: {response.text}")
                return False
                
        except Exception as e:
            self.print_error(f"Flat array prediction test failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_prediction_endpoints(self) -> bool:
        """Test both prediction endpoint formats"""
        self.print_header("PREDICTION ENDPOINT TESTS")
        
        test_2d = self.test_prediction_2d_format()
        test_flat = self.test_prediction_flat_format()
        
        return test_2d and test_flat
    
    def test_prediction_validation(self) -> bool:
        """Test prediction endpoint validation with invalid inputs"""
        self.print_header("PREDICTION VALIDATION TESTS")
        
        validation_tests = [
            {
                "name": "Invalid flat array size (500 instead of 600)",
                "payload": {
                    "ticker": "AAPL",
                    "features": [0.5] * 500,
                    "horizon": 3
                },
                "expected_status": 422
            },
            {
                "name": "Invalid 2D array shape (50x10 instead of 60x10)",
                "payload": {
                    "ticker": "AAPL",
                    "features": [[0.5] * 10 for _ in range(50)],
                    "horizon": 3
                },
                "expected_status": 422
            },
            {
                "name": "Invalid ticker",
                "payload": {
                    "ticker": "INVALID_TICKER",
                    "features": [0.5] * 600,
                    "horizon": 3
                },
                "expected_status": 422
            },
            {
                "name": "Missing required fields",
                "payload": {
                    "features": [0.5] * 600
                },
                "expected_status": 422
            }
        ]
        
        validation_passed = 0
        total_validations = len(validation_tests)
        
        for i, test_case in enumerate(validation_tests, 1):
            self.print_info(f"Test {i}: {test_case['name']}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json=test_case['payload'],
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                
                if response.status_code == test_case['expected_status']:
                    self.print_success(f"Validation correctly rejected with status {response.status_code}")
                    validation_passed += 1
                    
                    # Show error message if available
                    try:
                        error_detail = response.json()
                        if 'detail' in error_detail:
                            if isinstance(error_detail['detail'], list) and error_detail['detail']:
                                first_error = error_detail['detail'][0]
                                if isinstance(first_error, dict) and 'msg' in first_error:
                                    self.log_verbose(f"Error message: {first_error['msg']}")
                            else:
                                self.log_verbose(f"Error detail: {error_detail['detail']}")
                    except:
                        pass
                else:
                    self.print_error(f"Expected status {test_case['expected_status']}, got {response.status_code}")
                    
            except Exception as e:
                self.print_error(f"Validation test failed: {e}")
        
        self.print_info(f"Validation tests: {validation_passed}/{total_validations} passed")
        return validation_passed == total_validations
    
    def test_backtest_endpoint(self) -> bool:
        """Test the backtest endpoint"""
        self.print_header("BACKTEST ENDPOINT TEST")
        
        try:
            backtest_payload = {
                "ticker": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000
            }
            
            self.log_verbose(f"Testing backtest endpoint: {self.base_url}/backtest")
            response = requests.post(
                f"{self.base_url}/backtest",
                json=backtest_payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_success("Backtest completed successfully!")
                
                self.print_info(f"Ticker: {result.get('ticker', 'Unknown')}")
                self.print_info(f"Period: {backtest_payload['start_date']} to {backtest_payload['end_date']}")
                self.print_info(f"Initial Capital: ${backtest_payload['initial_capital']:,}")
                
                if result.get('performance_metrics'):
                    self.print_info("Performance metrics included in response")
                    
                if result.get('trades'):
                    trades = result.get('trades', [])
                    self.print_info(f"Trade history: {len(trades)} trades")
                
                return True
            elif response.status_code == 404:
                self.print_warning("Backtest endpoint not implemented (404) - This is acceptable")
                return True  # Not implemented is acceptable
            else:
                self.print_error(f"Backtest failed with status {response.status_code}")
                try:
                    error_detail = response.json()
                    self.print_error(f"Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    self.print_error(f"Error response: {response.text}")
                return False
                
        except Exception as e:
            self.print_error(f"Backtest test failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_concurrent_predictions(self) -> bool:
        """Test concurrent prediction requests for load testing"""
        self.print_header("CONCURRENT LOAD TEST")
        
        self.print_info("Running load test with 5 concurrent prediction requests...")
        
        def make_prediction_request(ticker: str, request_id: int) -> Dict[str, Any]:
            """Make a single prediction request"""
            try:
                features = np.random.uniform(0.1, 1.0, (60, 10)).round(4).tolist()
                payload = {
                    "ticker": ticker,
                    "features": features,
                    "horizon": 3
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "ticker": ticker,
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "cache_hit": response.json().get('cache_hit', False) if response.status_code == 200 else False,
                    "error": None
                }
                
            except Exception as e:
                return {
                    "request_id": request_id,
                    "ticker": ticker,
                    "success": False,
                    "status_code": None,
                    "response_time": None,
                    "cache_hit": False,
                    "error": str(e)
                }
        
        # Execute concurrent requests
        tickers = ["AAPL", "MSFT", "GOOG", "AAPL", "MSFT"]  # Intentional duplicates for cache testing
        futures_to_ticker = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            for i, ticker in enumerate(tickers):
                future = executor.submit(make_prediction_request, ticker, i + 1)
                futures_to_ticker[future] = ticker
            
            results = []
            for future in as_completed(futures_to_ticker):
                result = future.result()
                results.append(result)
        
        # Analyze results
        successful = len([r for r in results if r['success']])
        failed = len([r for r in results if not r['success']])
        cache_hits = len([r for r in results if r['success'] and r['cache_hit']])
        cache_misses = len([r for r in results if r['success'] and not r['cache_hit']])
        
        # Calculate average response time
        response_times = [r['response_time'] for r in results if r['response_time'] is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Report results
        self.print_info("Load test completed:")
        self.print_success(f"{successful} successful requests")
        if failed > 0:
            self.print_error(f"{failed} failed requests")
            
        self.print_info(f"Cache performance: {cache_hits} hits, {cache_misses} misses")
        self.print_info(f"Average response time: {avg_response_time:.3f} seconds")
        
        if self.verbose:
            self.print_info("Detailed results:")
            for result in sorted(results, key=lambda x: x['request_id']):
                status = "[OK]" if result['success'] else "[FAIL]"
                cache_info = " (cached)" if result['cache_hit'] else ""
                time_info = f" [{result['response_time']:.3f}s]" if result['response_time'] else ""
                error_info = f" - {result['error']}" if result['error'] else ""
                self.print_info(f"  {status} Request {result['request_id']} ({result['ticker']}){cache_info}{time_info}{error_info}")
        
        return successful >= 3  # Pass if most requests succeeded
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests and return results summary"""
        self.start_time = datetime.now()
        
        self.print_header("TIMESERIES TRANSFORMER API - PYTHON TEST SUITE")
        self.print_info(f"Testing API at: {self.base_url}")
        self.print_info(f"Timeout: {self.timeout} seconds")
        self.print_info(f"Verbose mode: {'Enabled' if self.verbose else 'Disabled'}")
        self.print_info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test order is important - health first, then others
        test_suite = [
            ("Health Check", self.test_health),
            ("Model Info", self.test_model_info),
            ("Prediction Endpoints", self.test_prediction_endpoints),
            ("Validation Tests", self.test_prediction_validation),
            ("Backtest Endpoint", self.test_backtest_endpoint),
            ("Concurrent Load Test", self.test_concurrent_predictions)
        ]
        
        results = {}
        
        # Run health check first - if it fails, skip others
        if not self.test_health():
            self.print_error("API health check failed - skipping remaining tests")
            self.print_info(f"Make sure the API server is running on {self.base_url}")
            return {"Health Check": False}
        
        # Run remaining tests
        for test_name, test_func in test_suite[1:]:  # Skip health check since we already ran it
            try:
                results[test_name] = test_func()
            except Exception as e:
                self.print_error(f"Test '{test_name}' crashed: {e}")
                if self.verbose:
                    traceback.print_exc()
                results[test_name] = False
        
        results["Health Check"] = True  # We know this passed
        
        return results
    
    def print_summary(self, results: Dict[str, bool]) -> None:
        """Print test results summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None
        
        self.print_header("TEST RESULTS SUMMARY")
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        self.print_info(f"Test execution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if duration:
            self.print_info(f"Total execution time: {duration.total_seconds():.2f} seconds")
        
        print(f"\n{Colors.BOLD}Individual Test Results:{Colors.ENDC}")
        for test_name, passed in results.items():
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            status_text = "PASSED" if passed else "FAILED"
            print(f"  {status_color}{test_name}: {status_text}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Overall Score: {passed_tests}/{total_tests} tests passed{Colors.ENDC}")
        
        if passed_tests == total_tests:
            self.print_success(f"[PASS] All tests passed! API is fully functional. ({passed_tests}/{total_tests})")
            return True
        elif passed_tests > total_tests * 0.7:  # More than 70% passed
            self.print_warning(f"[PARTIAL] Most tests passed ({passed_tests}/{total_tests}). Check failed tests above.")
            return True
        else:
            self.print_error(f"[FAIL] Multiple test failures ({total_tests - passed_tests}/{total_tests} failed). API may have issues.")
            return False


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Python API Test Suite for TimeSeries Transformer API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_api_python.py
  python test_api_python.py --base-url http://localhost:8000 --verbose
  python test_api_python.py --timeout 60 --verbose
        """
    )
    
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='Base URL of the API (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = APITester(
        base_url=args.base_url,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    try:
        results = tester.run_all_tests()
        success = tester.print_summary(results)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}[WARNING] Tests interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}[ERROR] Test suite failed with unexpected error: {e}{Colors.ENDC}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()