#!/usr/bin/env python3
"""
Pre-deployment validation script for time-series transformer
Validates model performance, compatibility, and resource requirements
"""

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import yaml
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


class DeploymentValidator:
    """Comprehensive deployment validation for ML models"""
    
    def __init__(self, model_path: str, requirements_path: str):
        self.model_path = Path(model_path)
        self.requirements_path = Path(requirements_path)
        self.validation_results = {}
        self.failed_checks = []
        
        # Load requirements
        with open(requirements_path, 'r') as f:
            self.requirements = yaml.safe_load(f)
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        print("🚀 Starting pre-deployment validation...")
        
        checks = [
            ("Model file existence", self.check_model_exists),
            ("Model size validation", self.check_model_size),
            ("Model loading", self.check_model_loading),
            ("Inference speed", self.check_inference_speed),
            ("Memory usage", self.check_memory_usage),
            ("Prediction quality", self.check_prediction_quality),
            ("API compatibility", self.check_api_compatibility),
            ("Resource requirements", self.check_resource_requirements),
            ("Security scan", self.check_security),
            ("Configuration validation", self.check_configuration)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                result = check_func()
                self.validation_results[check_name] = result
                if result:
                    print(f"✅ {check_name}: PASSED")
                else:
                    print(f"❌ {check_name}: FAILED")
                    self.failed_checks.append(check_name)
                    all_passed = False
            except Exception as e:
                print(f"❌ {check_name}: ERROR - {str(e)}")
                self.failed_checks.append(check_name)
                all_passed = False
                traceback.print_exc()
        
        self._generate_report()
        return all_passed
    
    def check_model_exists(self) -> bool:
        """Check if model file exists"""
        if not self.model_path.exists():
            print(f"Model file not found: {self.model_path}")
            return False
        
        # Check for required artifacts
        required_files = ['model.pt', 'config.json', 'scalers.pkl']
        missing_files = []
        
        for file_name in required_files:
            file_path = self.model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Missing required files: {missing_files}")
            return False
        
        return True
    
    def check_model_size(self) -> bool:
        """Validate model size meets deployment requirements"""
        model_file = self.model_path / 'model.pt'
        size_mb = model_file.stat().st_size / (1024 * 1024)
        max_size_mb = self.requirements.get('max_model_size_mb', 500)
        
        print(f"Model size: {size_mb:.2f} MB (limit: {max_size_mb} MB)")
        
        if size_mb > max_size_mb:
            print(f"Model size {size_mb:.2f} MB exceeds limit of {max_size_mb} MB")
            return False
        
        return True
    
    def check_model_loading(self) -> bool:
        """Test model loading"""
        try:
            model_file = self.model_path / 'model.pt'
            model = torch.jit.load(model_file, map_location='cpu')
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def check_inference_speed(self) -> bool:
        """Validate inference speed meets SLA requirements"""
        try:
            model_file = self.model_path / 'model.pt'
            model = torch.jit.load(model_file, map_location='cpu')
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 60, 7)
            
            # Warmup runs
            print("Warming up model...")
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Measure inference time
            print("Measuring inference latency...")
            times = []
            for _ in range(100):
                start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                times.append(time.time() - start)
            
            # Calculate statistics
            mean_latency = np.mean(times) * 1000  # ms
            p50_latency = np.percentile(times, 50) * 1000
            p95_latency = np.percentile(times, 95) * 1000
            p99_latency = np.percentile(times, 99) * 1000
            
            max_p99_latency = self.requirements.get('max_p99_latency_ms', 100)
            max_p95_latency = self.requirements.get('max_p95_latency_ms', 50)
            
            print(f"Inference latency - Mean: {mean_latency:.2f}ms, "
                  f"P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms, "
                  f"P99: {p99_latency:.2f}ms")
            
            if p99_latency > max_p99_latency:
                print(f"P99 latency {p99_latency:.2f}ms exceeds limit of {max_p99_latency}ms")
                return False
            
            if p95_latency > max_p95_latency:
                print(f"P95 latency {p95_latency:.2f}ms exceeds limit of {max_p95_latency}ms")
                return False
            
            return True
            
        except Exception as e:
            print(f"Inference speed check failed: {e}")
            return False
    
    def check_memory_usage(self) -> bool:
        """Validate memory usage during inference"""
        try:
            tracemalloc.start()
            
            model_file = self.model_path / 'model.pt'
            model = torch.jit.load(model_file, map_location='cpu')
            model.eval()
            
            # Test with batch inference
            batch_input = torch.randn(32, 60, 7)
            
            with torch.no_grad():
                _ = model(batch_input)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            peak_mb = peak / (1024 * 1024)
            max_memory_mb = self.requirements.get('max_memory_mb', 2000)
            
            print(f"Peak memory usage: {peak_mb:.2f} MB (limit: {max_memory_mb} MB)")
            
            if peak_mb > max_memory_mb:
                print(f"Memory usage {peak_mb:.2f} MB exceeds limit of {max_memory_mb} MB")
                return False
            
            return True
            
        except Exception as e:
            print(f"Memory usage check failed: {e}")
            return False
    
    def check_prediction_quality(self) -> bool:
        """Validate prediction quality and ranges"""
        try:
            model_file = self.model_path / 'model.pt'
            model = torch.jit.load(model_file, map_location='cpu')
            model.eval()
            
            # Generate test data with realistic ranges
            test_input = torch.randn(100, 60, 7) * 2  # Scaled random data
            
            with torch.no_grad():
                predictions = model(test_input)
            
            # Check predictions are reasonable
            pred_mean = predictions.mean().item()
            pred_std = predictions.std().item()
            pred_min = predictions.min().item()
            pred_max = predictions.max().item()
            
            print(f"Predictions - Mean: {pred_mean:.4f}, Std: {pred_std:.4f}, "
                  f"Min: {pred_min:.4f}, Max: {pred_max:.4f}")
            
            # Basic sanity checks
            if torch.isnan(predictions).any():
                print("NaN values detected in predictions")
                return False
            
            if torch.isinf(predictions).any():
                print("Infinite values detected in predictions")
                return False
            
            if pred_std == 0:
                print("Model predictions have zero variance")
                return False
            
            # Check reasonable ranges for stock predictions
            if abs(pred_mean) > 10:  # Extreme mean prediction
                print(f"Extreme mean prediction: {pred_mean}")
                return False
            
            if pred_std > 20:  # Extremely high variance
                print(f"Extremely high prediction variance: {pred_std}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Prediction quality check failed: {e}")
            return False
    
    def check_api_compatibility(self) -> bool:
        """Test API schema compatibility"""
        try:
            # Mock API schema validation
            config_file = self.model_path / 'config.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            required_fields = ['model_version', 'input_shape', 'output_shape', 'feature_names']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                print(f"Missing config fields: {missing_fields}")
                return False
            
            # Check input/output shapes
            expected_input_shape = [60, 7]  # sequence_length, n_features
            expected_output_shape = [5]     # prediction_horizon
            
            if config['input_shape'] != expected_input_shape:
                print(f"Input shape mismatch: {config['input_shape']} != {expected_input_shape}")
                return False
            
            if config['output_shape'] != expected_output_shape:
                print(f"Output shape mismatch: {config['output_shape']} != {expected_output_shape}")
                return False
            
            print("API schema compatibility verified")
            return True
            
        except Exception as e:
            print(f"API compatibility check failed: {e}")
            return False
    
    def check_resource_requirements(self) -> bool:
        """Validate Kubernetes resource requirements"""
        requirements = self.requirements.get('kubernetes_resources', {})
        
        # Check if resource limits are defined
        required_resources = ['cpu_request', 'memory_request', 'cpu_limit', 'memory_limit']
        missing_resources = [res for res in required_resources if res not in requirements]
        
        if missing_resources:
            print(f"Missing resource requirements: {missing_resources}")
            return False
        
        # Validate resource values
        cpu_request = requirements.get('cpu_request', '0')
        memory_request = requirements.get('memory_request', '0')
        
        if not cpu_request.endswith('m') or int(cpu_request[:-1]) < 500:
            print(f"CPU request too low: {cpu_request}")
            return False
        
        if not memory_request.endswith('Gi') or int(memory_request[:-2]) < 1:
            print(f"Memory request too low: {memory_request}")
            return False
        
        print("Resource requirements validated")
        return True
    
    def check_security(self) -> bool:
        """Basic security validation"""
        try:
            # Check for common security issues
            model_file = self.model_path / 'model.pt'
            
            # Check file permissions
            stat = model_file.stat()
            if stat.st_mode & 0o077:  # Check if group/others have write access
                print("Model file has overly permissive permissions")
                return False
            
            # Basic pickle safety check (simplified)
            try:
                torch.jit.load(model_file, map_location='cpu')
                print("Security checks passed")
                return True
            except Exception:
                print("Model file appears to be corrupted or unsafe")
                return False
            
        except Exception as e:
            print(f"Security check failed: {e}")
            return False
    
    def check_configuration(self) -> bool:
        """Validate deployment configuration"""
        try:
            config_file = self.model_path / 'config.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validate critical configuration values
            required_configs = {
                'model_version': str,
                'architecture': str,
                'sequence_length': int,
                'prediction_horizon': int,
                'feature_names': list
            }
            
            for key, expected_type in required_configs.items():
                if key not in config:
                    print(f"Missing configuration: {key}")
                    return False
                
                if not isinstance(config[key], expected_type):
                    print(f"Invalid type for {key}: expected {expected_type}")
                    return False
            
            # Validate specific values
            if config['sequence_length'] != 60:
                print(f"Invalid sequence length: {config['sequence_length']}")
                return False
            
            if config['prediction_horizon'] != 5:
                print(f"Invalid prediction horizon: {config['prediction_horizon']}")
                return False
            
            if len(config['feature_names']) != 7:
                print(f"Invalid number of features: {len(config['feature_names'])}")
                return False
            
            print("Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"Configuration check failed: {e}")
            return False
    
    def _generate_report(self):
        """Generate validation report"""
        print("\n" + "="*60)
        print("📊 DEPLOYMENT VALIDATION REPORT")
        print("="*60)
        
        passed_checks = len(self.validation_results) - len(self.failed_checks)
        total_checks = len(self.validation_results)
        
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {len(self.failed_checks)}")
        print(f"Success rate: {passed_checks/total_checks*100:.1f}%")
        
        if self.failed_checks:
            print(f"\n❌ Failed checks:")
            for check in self.failed_checks:
                print(f"  - {check}")
        
        # Write report to file
        report = {
            'timestamp': time.time(),
            'model_path': str(self.model_path),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': self.failed_checks,
            'success_rate': passed_checks/total_checks,
            'results': self.validation_results
        }
        
        report_file = Path('deployment-validation-report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate model deployment')
    parser.add_argument('--model-path', required=True, help='Path to model directory')
    parser.add_argument('--requirements', required=True, help='Path to requirements YAML file')
    parser.add_argument('--exit-on-failure', action='store_true', 
                       help='Exit with error code if validation fails')
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.requirements).exists():
        print(f"❌ Requirements file does not exist: {args.requirements}")
        sys.exit(1)
    
    validator = DeploymentValidator(args.model_path, args.requirements)
    
    try:
        success = validator.run_all_validations()
        
        if success:
            print("\n🎉 All validation checks PASSED! Deployment ready.")
            sys.exit(0)
        else:
            print("\n💥 Some validation checks FAILED! Deployment NOT ready.")
            if args.exit_on_failure:
                sys.exit(1)
            else:
                sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()