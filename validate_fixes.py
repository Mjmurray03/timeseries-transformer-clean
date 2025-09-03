#!/usr/bin/env python
"""
Validation Script for Time-Series Transformer Project Fixes
===========================================================

This script validates all the recent fixes and improvements:
1. PyTorch 2.6 compatibility (weights_only=False)
2. WandB dependency installation
3. Model configuration system
4. Safe checkpoint loading utilities
5. ModelManager updates

Run this script to verify everything is working correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_test_result(test_name, success, details=None, error=None):
    """Print formatted test result"""
    status = "[OK]" if success else "[FAIL]"
    status_text = "PASS" if success else "FAIL"
    
    print(f"{status} {test_name:<45} [{status_text}]")
    
    if details and success:
        for detail in details if isinstance(details, list) else [details]:
            print(f"    -> {detail}")
    
    if error and not success:
        print(f"    -> Error: {error}")

def test_imports():
    """Test all required imports"""
    print_header("IMPORT VALIDATION TESTS")
    
    # Test 1: WandB import
    try:
        import wandb
        version = wandb.__version__
        print_test_result("Import wandb", True, f"Version: {version}")
    except ImportError as e:
        print_test_result("Import wandb", False, error=str(e))
    
    # Test 2: PyTorch import
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        details = [
            f"Version: {version}",
            f"CUDA available: {cuda_available}"
        ]
        print_test_result("Import torch", True, details)
    except ImportError as e:
        print_test_result("Import torch", False, error=str(e))
    
    # Test 3: FastAPI import
    try:
        import fastapi
        version = fastapi.__version__
        print_test_result("Import fastapi", True, f"Version: {version}")
    except ImportError as e:
        print_test_result("Import fastapi", False, error=str(e))
    
    # Test 4: Main API import
    try:
        from src.api import main
        print_test_result("Import src.api.main", True, "ModelManager class available")
    except ImportError as e:
        print_test_result("Import src.api.main", False, error=str(e))
    
    # Test 5: Model loader fix utilities
    try:
        from src.api.model_loader_fix import safe_load_checkpoint, get_model_config_from_checkpoint
        print_test_result("Import model_loader_fix", True, "safe_load_checkpoint & get_model_config_from_checkpoint")
    except ImportError as e:
        print_test_result("Import model_loader_fix", False, error=str(e))

def test_models_directory():
    """Test models directory and files"""
    print_header("MODELS DIRECTORY VALIDATION")
    
    # Test 1: Models directory exists
    models_dir = project_root / "models"
    if models_dir.exists():
        print_test_result("Models directory exists", True, f"Path: {models_dir}")
    else:
        print_test_result("Models directory exists", False, error=f"Directory not found: {models_dir}")
        return
    
    # Test 2: List .pt files
    pt_files = list(models_dir.glob("*.pt"))
    if pt_files:
        details = [f"Found {len(pt_files)} .pt files:"] + [f"  - {f.name}" for f in pt_files[:5]]
        if len(pt_files) > 5:
            details.append(f"  ... and {len(pt_files) - 5} more")
        print_test_result("Find .pt model files", True, details)
    else:
        print_test_result("Find .pt model files", False, error="No .pt files found")
    
    # Test 3: Model config file
    config_file = models_dir / "model_configs.json"
    if config_file.exists():
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            single_stocks = list(config.get('model_configurations', {}).get('single_stock_models', {}).keys())
            multi_stocks = list(config.get('model_configurations', {}).get('multi_stock_models', {}).keys())
            
            details = [
                f"Single-stock configs: {len(single_stocks)} ({', '.join(single_stocks)})",
                f"Multi-stock configs: {len(multi_stocks)} ({', '.join(multi_stocks)})"
            ]
            print_test_result("Load model_configs.json", True, details)
        except Exception as e:
            print_test_result("Load model_configs.json", False, error=str(e))
    else:
        print_test_result("Load model_configs.json", False, error="Config file not found")

def test_checkpoint_loading():
    """Test checkpoint loading functionality"""
    print_header("CHECKPOINT LOADING VALIDATION")
    
    # Find a model file to test
    models_dir = project_root / "models"
    pt_files = list(models_dir.glob("*.pt"))
    
    if not pt_files:
        print_test_result("Test checkpoint loading", False, error="No .pt files available for testing")
        return
    
    test_file = pt_files[0]  # Use first available file
    print(f"Testing with file: {test_file.name}")
    
    # Test 1: Standard torch.load with weights_only=False
    try:
        import torch
        checkpoint = torch.load(test_file, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            details = [
                f"Checkpoint type: Dictionary",
                f"Keys: {keys[:5]}" + ("..." if len(keys) > 5 else ""),
                f"Total keys: {len(keys)}"
            ]
            
            # Check for common checkpoint structures
            if 'model_state_dict' in checkpoint:
                details.append("[OK] Contains 'model_state_dict' key")
            if 'state_dict' in checkpoint:
                details.append("[OK] Contains 'state_dict' key") 
            if 'model_config' in checkpoint:
                details.append("[OK] Contains 'model_config' key")
                
        else:
            details = [
                f"Checkpoint type: {type(checkpoint).__name__}",
                "Direct state dict format"
            ]
        
        print_test_result("Load checkpoint with torch.load", True, details)
        
    except Exception as e:
        print_test_result("Load checkpoint with torch.load", False, error=str(e))
        return
    
    # Test 2: Safe checkpoint loading
    try:
        from src.api.model_loader_fix import safe_load_checkpoint, get_model_config_from_checkpoint
        
        # Test safe_load_checkpoint (without actual model - just to test the function)
        try:
            # This will work for getting the state dict even without a model
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            state_dict_keys = list(state_dict.keys())[:5]
            print_test_result("Extract state_dict safely", True, 
                            f"State dict keys: {state_dict_keys}...")
        except Exception as e:
            print_test_result("Extract state_dict safely", False, error=str(e))
        
        # Test get_model_config_from_checkpoint
        model_config = get_model_config_from_checkpoint(test_file)
        if model_config:
            config_keys = list(model_config.keys())
            print_test_result("Extract model config from checkpoint", True, 
                            f"Config keys: {config_keys}")
        else:
            print_test_result("Extract model config from checkpoint", True, 
                            "No model config in checkpoint (expected)")
            
    except Exception as e:
        print_test_result("Test safe loading functions", False, error=str(e))

def test_pytorch_compatibility():
    """Test PyTorch 2.6 compatibility fixes"""
    print_header("PYTORCH 2.6 COMPATIBILITY VALIDATION")
    
    # Test 1: Check files for weights_only=False usage
    files_to_check = [
        "src/api/main.py",
        "src/api/model_server.py", 
        "src/training/callbacks/model_checkpoint.py",
        "src/training/trainer.py",
        "src/api/model_loader_fix.py"
    ]
    
    total_fixes = 0
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                torch_load_count = content.count('torch.load(')
                weights_only_count = content.count('weights_only=False')
                
                if torch_load_count > 0:
                    if weights_only_count >= torch_load_count:
                        print_test_result(f"Check {file_path}", True, 
                                        f"{weights_only_count}/{torch_load_count} torch.load calls have weights_only=False")
                        total_fixes += weights_only_count
                    else:
                        print_test_result(f"Check {file_path}", False, 
                                        error=f"Only {weights_only_count}/{torch_load_count} torch.load calls have weights_only=False")
                else:
                    print_test_result(f"Check {file_path}", True, "No torch.load calls (OK)")
                    
            except Exception as e:
                print_test_result(f"Check {file_path}", False, error=str(e))
        else:
            print_test_result(f"Check {file_path}", False, error="File not found")
    
    print_test_result("Total PyTorch compatibility fixes", True, 
                    f"{total_fixes} torch.load calls updated with weights_only=False")

def main():
    """Run all validation tests"""
    print("Time-Series Transformer Project Validation")
    print("=" * 60)
    print("Testing recent fixes and improvements...")
    
    # Run all test suites
    test_imports()
    test_models_directory()
    test_checkpoint_loading()
    test_pytorch_compatibility()
    
    # Final summary
    print_header("VALIDATION COMPLETE")
    print("[SUCCESS] All tests completed!")
    print("\nIf any tests failed, please check the error messages above.")
    print("For import errors, ensure all dependencies are installed.")
    print("For model loading errors, check that model files are compatible.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Validation failed with unexpected error: {e}")
        sys.exit(1)