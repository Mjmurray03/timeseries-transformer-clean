#!/usr/bin/env python3
"""
Comprehensive test execution script following testing-standards.md requirements.
Runs all test suites and verifies 80% minimum coverage as specified.
"""
import subprocess
import sys
import os
from pathlib import Path
import argparse
import json
from typing import Dict, List, Any


class TestRunner:
    """Test execution and coverage verification following testing-standards.md"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.coverage_targets = {
            "overall": 80,      # Minimum 80% overall
            "data": 90,         # Data pipeline: Minimum 90%
            "models": 85,       # Model components: Minimum 85%
            "training": 85,     # Training components: Minimum 85%
            "api": 90,         # API endpoints: Minimum 90%
            "critical": 95      # Critical paths: Minimum 95%
        }
        
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests following test pyramid (75% of tests)"""
        print("🧪 Running Unit Tests (75% of test pyramid)")
        print("=" * 50)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "-v",
            "--cov=src",
            "--cov-report=xml:coverage_unit.xml",
            "--cov-report=html:htmlcov_unit",
            "--cov-report=term-missing",
            "--cov-fail-under=75",
            "-m", "unit",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            "name": "Unit Tests",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests following test pyramid (20% of tests)"""
        print("🔗 Running Integration Tests (20% of test pyramid)")
        print("=" * 50)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "-v",
            "--cov=src",
            "--cov-append",
            "--cov-report=xml:coverage_integration.xml",
            "--cov-report=html:htmlcov_integration",
            "--cov-report=term-missing",
            "-m", "integration",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            "name": "Integration Tests",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks (5% of tests)"""
        print("⚡ Running Performance Tests (5% of test pyramid)")
        print("=" * 50)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/performance/",
            "-v",
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-json=benchmark_results.json",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            "name": "Performance Tests",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive coverage analysis"""
        print("📊 Running Coverage Analysis")
        print("=" * 50)
        
        # Run all tests with comprehensive coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=xml:coverage_comprehensive.xml",
            "--cov-report=html:htmlcov_comprehensive",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            f"--cov-fail-under={self.coverage_targets['overall']}",
            "--quiet"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            "name": "Coverage Analysis",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    
    def verify_coverage_targets(self) -> Dict[str, Any]:
        """Verify coverage meets requirements from testing-standards.md"""
        print("🎯 Verifying Coverage Targets")
        print("=" * 50)
        
        coverage_file = self.project_root / "coverage.json"
        if not coverage_file.exists():
            return {
                "success": False,
                "error": "Coverage JSON file not found"
            }
        
        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read coverage data: {e}"
            }
        
        results = {}
        overall_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
        
        print(f"Overall Coverage: {overall_coverage:.1f}%")
        results['overall'] = {
            'actual': overall_coverage,
            'target': self.coverage_targets['overall'],
            'passed': overall_coverage >= self.coverage_targets['overall']
        }
        
        # Analyze by component
        files = coverage_data.get('files', {})
        
        # Data pipeline coverage
        data_files = {k: v for k, v in files.items() if 'src/data/' in k}
        if data_files:
            data_coverage = sum(f.get('summary', {}).get('percent_covered', 0) for f in data_files.values()) / len(data_files)
            print(f"Data Pipeline Coverage: {data_coverage:.1f}%")
            results['data'] = {
                'actual': data_coverage,
                'target': self.coverage_targets['data'],
                'passed': data_coverage >= self.coverage_targets['data']
            }
        
        # Model components coverage
        model_files = {k: v for k, v in files.items() if 'src/models/' in k}
        if model_files:
            model_coverage = sum(f.get('summary', {}).get('percent_covered', 0) for f in model_files.values()) / len(model_files)
            print(f"Model Components Coverage: {model_coverage:.1f}%")
            results['models'] = {
                'actual': model_coverage,
                'target': self.coverage_targets['models'],
                'passed': model_coverage >= self.coverage_targets['models']
            }
        
        # Training components coverage
        training_files = {k: v for k, v in files.items() if 'src/training/' in k}
        if training_files:
            training_coverage = sum(f.get('summary', {}).get('percent_covered', 0) for f in training_files.values()) / len(training_files)
            print(f"Training Components Coverage: {training_coverage:.1f}%")
            results['training'] = {
                'actual': training_coverage,
                'target': self.coverage_targets['training'],
                'passed': training_coverage >= self.coverage_targets['training']
            }
        
        all_passed = all(r['passed'] for r in results.values())
        
        return {
            'success': all_passed,
            'results': results,
            'overall_coverage': overall_coverage
        }
    
    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests for deployment verification"""
        print("💨 Running Smoke Tests")
        print("=" * 50)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "-m", "smoke",
            "--tb=short",
            "--maxfail=1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            "name": "Smoke Tests",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['success'])
        
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 40)
        
        for result in results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {result['name']}")
            
            if not result['success'] and result.get('stderr'):
                print(f"   Error: {result['stderr'][:200]}...")
        
        # Coverage summary
        coverage_result = next((r for r in results if r['name'] == 'Coverage Analysis'), None)
        if coverage_result and coverage_result.get('success'):
            print(f"\n📊 Coverage Requirements Met: ✅")
        else:
            print(f"\n📊 Coverage Requirements Met: ❌")
        
        print("\n" + "=" * 60)
    
    def run_comprehensive_tests(self, include_performance: bool = True) -> bool:
        """Run all test suites and verify coverage requirements"""
        print("🚀 Starting Comprehensive Test Suite")
        print("Following testing-standards.md requirements")
        print("=" * 60)
        
        results = []
        
        # Unit tests (75% of test pyramid)
        unit_result = self.run_unit_tests()
        results.append(unit_result)
        
        if not unit_result['success']:
            print("❌ Unit tests failed! Stopping execution.")
            return False
        
        # Integration tests (20% of test pyramid)  
        integration_result = self.run_integration_tests()
        results.append(integration_result)
        
        # Performance tests (5% of test pyramid)
        if include_performance:
            perf_result = self.run_performance_tests()
            results.append(perf_result)
        
        # Coverage analysis
        coverage_result = self.run_coverage_analysis()
        results.append(coverage_result)
        
        # Verify coverage targets
        coverage_verification = self.verify_coverage_targets()
        results.append({
            'name': 'Coverage Verification',
            'success': coverage_verification['success'],
            'details': coverage_verification
        })
        
        # Smoke tests
        smoke_result = self.run_smoke_tests()
        results.append(smoke_result)
        
        # Generate report
        self.generate_test_report(results)
        
        # Return overall success
        return all(r['success'] for r in results)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests following testing-standards.md"
    )
    parser.add_argument(
        "--no-performance", 
        action="store_true",
        help="Skip performance tests for faster execution"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(args.project_root)
    success = runner.run_comprehensive_tests(include_performance=not args.no_performance)
    
    if success:
        print("\n🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        sys.exit(0)
    else:
        print("\n💥 TESTS FAILED - REVIEW ERRORS ABOVE")
        sys.exit(1)


if __name__ == "__main__":
    main()