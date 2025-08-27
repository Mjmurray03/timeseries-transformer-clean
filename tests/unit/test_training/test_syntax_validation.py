"""Syntax and structure validation tests for training components."""

import pytest
import ast
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestSyntaxValidation:
    """Test that all training files have valid Python syntax."""
    
    def test_trainer_syntax(self):
        """Test trainer.py has valid syntax."""
        trainer_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'trainer.py'
        
        with open(trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should parse without syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in trainer.py: {e}")
    
    def test_experiment_tracker_syntax(self):
        """Test experiment_tracker.py has valid syntax."""
        tracker_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'experiment_tracker.py'
        
        with open(tracker_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should parse without syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in experiment_tracker.py: {e}")
    
    def test_callbacks_syntax(self):
        """Test callback files have valid syntax."""
        callbacks_dir = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'callbacks'
        
        for callback_file in callbacks_dir.glob('*.py'):
            if callback_file.name == '__init__.py':
                continue
                
            with open(callback_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {callback_file.name}: {e}")
    
    def test_dataset_syntax(self):
        """Test dataset files have valid syntax."""
        dataset_path = Path(__file__).parent.parent.parent.parent / 'src' / 'data' / 'datasets' / 'stock_dataset.py'
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in stock_dataset.py: {e}")


class TestStructureValidation:
    """Test that training components have expected structure."""
    
    def test_trainer_class_structure(self):
        """Test TrainingOrchestrator class has expected methods."""
        trainer_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'trainer.py'
        
        with open(trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find TrainingOrchestrator class
        orchestrator_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'TrainingOrchestrator':
                orchestrator_class = node
                break
        
        assert orchestrator_class is not None, "TrainingOrchestrator class not found"
        
        # Check for expected methods
        method_names = [node.name for node in orchestrator_class.body if isinstance(node, ast.FunctionDef)]
        
        expected_methods = [
            '__init__',
            'train',
            'train_epoch',
            'validate',
            'evaluate',
            '_build_optimizer',
            '_build_scheduler',
            '_build_criterion'
        ]
        
        for method in expected_methods:
            assert method in method_names, f"Method {method} not found in TrainingOrchestrator"
    
    def test_experiment_tracker_class_structure(self):
        """Test ExperimentTracker class has expected methods."""
        tracker_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'experiment_tracker.py'
        
        with open(tracker_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find ExperimentTracker class
        tracker_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ExperimentTracker':
                tracker_class = node
                break
        
        assert tracker_class is not None, "ExperimentTracker class not found"
        
        # Check for expected methods
        method_names = [node.name for node in tracker_class.body if isinstance(node, ast.FunctionDef)]
        
        expected_methods = [
            '__init__',
            'log_metrics',
            'log_model',
            'log_config',
            'finish',
            '__enter__',
            '__exit__'
        ]
        
        for method in expected_methods:
            assert method in method_names, f"Method {method} not found in ExperimentTracker"
    
    def test_early_stopping_class_structure(self):
        """Test EarlyStopping class has expected methods."""
        callback_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'callbacks' / 'early_stopping.py'
        
        with open(callback_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find EarlyStopping class
        early_stopping_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'EarlyStopping':
                early_stopping_class = node
                break
        
        assert early_stopping_class is not None, "EarlyStopping class not found"
        
        # Check for expected methods
        method_names = [node.name for node in early_stopping_class.body if isinstance(node, ast.FunctionDef)]
        
        expected_methods = [
            '__init__',
            'should_stop',
            'get_best_score',
            'reset'
        ]
        
        for method in expected_methods:
            assert method in method_names, f"Method {method} not found in EarlyStopping"
    
    def test_model_checkpoint_class_structure(self):
        """Test ModelCheckpoint class has expected methods."""
        callback_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'callbacks' / 'model_checkpoint.py'
        
        with open(callback_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find ModelCheckpoint class
        checkpoint_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ModelCheckpoint':
                checkpoint_class = node
                break
        
        assert checkpoint_class is not None, "ModelCheckpoint class not found"
        
        # Check for expected methods
        method_names = [node.name for node in checkpoint_class.body if isinstance(node, ast.FunctionDef)]
        
        expected_methods = [
            '__init__',
            'on_epoch_end',
            'load_best_checkpoint',
            'get_best_score'
        ]
        
        for method in expected_methods:
            assert method in method_names, f"Method {method} not found in ModelCheckpoint"
    
    def test_dataset_classes_structure(self):
        """Test dataset classes have expected methods."""
        dataset_path = Path(__file__).parent.parent.parent.parent / 'src' / 'data' / 'datasets' / 'stock_dataset.py'
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find dataset classes
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
        
        # Test StockSequenceDataset
        assert 'StockSequenceDataset' in classes, "StockSequenceDataset class not found"
        stock_dataset = classes['StockSequenceDataset']
        method_names = [node.name for node in stock_dataset.body if isinstance(node, ast.FunctionDef)]
        
        expected_methods = ['__init__', '__len__', '__getitem__', 'get_feature_names', 'get_stats']
        for method in expected_methods:
            assert method in method_names, f"Method {method} not found in StockSequenceDataset"
        
        # Test MultiStockDataset
        assert 'MultiStockDataset' in classes, "MultiStockDataset class not found"
        multi_dataset = classes['MultiStockDataset']
        method_names = [node.name for node in multi_dataset.body if isinstance(node, ast.FunctionDef)]
        
        expected_methods = ['__init__', '__len__', '__getitem__', 'get_tickers', 'get_ticker_stats']
        for method in expected_methods:
            assert method in method_names, f"Method {method} not found in MultiStockDataset"


class TestImportStructure:
    """Test import structure and dependencies."""
    
    def test_training_init_imports(self):
        """Test training __init__.py has correct imports."""
        init_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / '__init__.py'
        
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for expected imports
        expected_imports = [
            'TrainingOrchestrator',
            'ExperimentTracker',
            'MetricsLogger',
            'EarlyStopping',
            'ModelCheckpoint'
        ]
        
        for import_name in expected_imports:
            assert import_name in content, f"Import {import_name} not found in training/__init__.py"
    
    def test_callbacks_init_imports(self):
        """Test callbacks __init__.py has correct imports."""
        init_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'callbacks' / '__init__.py'
        
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for expected imports
        expected_imports = [
            'EarlyStopping',
            'ModelCheckpoint'
        ]
        
        for import_name in expected_imports:
            assert import_name in content, f"Import {import_name} not found in callbacks/__init__.py"
    
    def test_datasets_init_imports(self):
        """Test datasets __init__.py has correct imports."""
        init_path = Path(__file__).parent.parent.parent.parent / 'src' / 'data' / 'datasets' / '__init__.py'
        
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for expected imports
        expected_imports = [
            'StockSequenceDataset',
            'MultiStockDataset',
            'DataAugmentation',
            'create_data_loaders',
            'split_sequences',
            'SequenceCollator'
        ]
        
        for import_name in expected_imports:
            assert import_name in content, f"Import {import_name} not found in datasets/__init__.py"


class TestDocstrings:
    """Test that classes and methods have proper docstrings."""
    
    def test_trainer_docstrings(self):
        """Test TrainingOrchestrator has docstrings."""
        trainer_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'trainer.py'
        
        with open(trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find TrainingOrchestrator class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'TrainingOrchestrator':
                # Check class docstring
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    assert isinstance(node.body[0].value.value, str), "TrainingOrchestrator should have a docstring"
                else:
                    pytest.fail("TrainingOrchestrator class should have a docstring")
                
                # Check method docstrings for key methods
                key_methods = ['__init__', 'train', 'train_epoch', 'validate']
                for method_node in node.body:
                    if isinstance(method_node, ast.FunctionDef) and method_node.name in key_methods:
                        if (method_node.body and 
                            isinstance(method_node.body[0], ast.Expr) and 
                            isinstance(method_node.body[0].value, ast.Constant)):
                            assert isinstance(method_node.body[0].value.value, str), f"Method {method_node.name} should have a docstring"
                        else:
                            pytest.fail(f"Method {method_node.name} should have a docstring")
                break
        else:
            pytest.fail("TrainingOrchestrator class not found")
    
    def test_experiment_tracker_docstrings(self):
        """Test ExperimentTracker has docstrings."""
        tracker_path = Path(__file__).parent.parent.parent.parent / 'src' / 'training' / 'experiment_tracker.py'
        
        with open(tracker_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find ExperimentTracker class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ExperimentTracker':
                # Check class docstring
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    assert isinstance(node.body[0].value.value, str), "ExperimentTracker should have a docstring"
                else:
                    pytest.fail("ExperimentTracker class should have a docstring")
                break
        else:
            pytest.fail("ExperimentTracker class not found")