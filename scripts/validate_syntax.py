#!/usr/bin/env python3
"""
Syntax validation script for model components.
Validates Python syntax without importing dependencies.
"""

import ast
import os
from pathlib import Path


def validate_python_syntax(file_path: str) -> bool:
    """
    Validate Python syntax of a file.

    Args:
        file_path: Path to Python file

    Returns:
        bool: True if syntax is valid
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Parse the AST
        ast.parse(source)
        return True

    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def main():
    """Validate syntax of all model component files."""

    # Files to validate
    files_to_check = [
        "src/models/components/attention_pooling.py",
        "src/models/components/interpretable_attention.py",
        "src/models/components/temporal_masking.py",
        "tests/unit/test_models/test_attention_pooling.py",
        "tests/unit/test_models/test_interpretable_attention.py",
        "tests/unit/test_models/test_temporal_masking.py",
    ]

    all_valid = True

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"Validating {file_path}...")
            if validate_python_syntax(file_path):
                print(f"✓ {file_path} - Syntax OK")
            else:
                print(f"✗ {file_path} - Syntax Error")
                all_valid = False
        else:
            print(f"⚠ {file_path} - File not found")
            all_valid = False

    if all_valid:
        print("\n✓ All files have valid Python syntax!")
        return 0
    else:
        print("\n✗ Some files have syntax errors!")
        return 1


if __name__ == "__main__":
    exit(main())
