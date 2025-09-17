#!/usr/bin/env python3
"""
Cross-Platform Environment Setup Script

Automatically detects the operating system and sets up the TimeSeries Transformer
environment with appropriate platform-specific configurations and dependencies.

Features:
- OS detection (Windows, Linux, macOS)
- Platform-specific dependency installation
- Environment validation
- GPU/CUDA detection and setup
- Virtual environment recommendations
- Path and encoding configuration

Usage:
    python scripts/setup_environment.py
    python scripts/setup_environment.py --check-only
    python scripts/setup_environment.py --verbose
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EnvironmentSetup:
    """Cross-platform environment setup and validation."""

    def __init__(self, verbose: bool = False, check_only: bool = False):
        self.verbose = verbose
        self.check_only = check_only
        self.os_type = platform.system()
        self.python_version = sys.version_info
        self.errors = []
        self.warnings = []

        # Project root directory
        self.project_root = Path(__file__).parent.parent

    def log(self, message: str, level: str = "INFO"):
        """Log messages with optional verbosity control."""
        prefix = f"[{level}]" if level != "INFO" else ""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"{prefix} {message}")

    def run_command(self, cmd: List[str], description: str = "", capture_output: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        self.log(f"Running: {' '.join(cmd)}", "DEBUG")

        if self.check_only and any(keyword in ' '.join(cmd) for keyword in ['install', 'pip', 'conda']):
            self.log(f"CHECK-ONLY: Would run: {description or ' '.join(cmd)}")
            return True, "Skipped (check-only mode)"

        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True, result.stdout.strip()
            else:
                result = subprocess.run(cmd, check=True)
                return True, "Success"
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed: {' '.join(cmd)}"
            if hasattr(e, 'stderr') and e.stderr:
                error_msg += f"\nError: {e.stderr.strip()}"
            self.log(error_msg, "ERROR")
            return False, error_msg
        except FileNotFoundError:
            error_msg = f"Command not found: {cmd[0]}"
            self.log(error_msg, "ERROR")
            return False, error_msg

    def detect_system_info(self) -> Dict[str, str]:
        """Detect and return comprehensive system information."""
        info = {
            "os": self.os_type,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "python_version": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            "python_executable": sys.executable,
        }

        # Add OS-specific information
        if self.os_type == "Windows":
            info["windows_version"] = platform.win32_ver()[0]
            info["windows_edition"] = platform.win32_edition()
        elif self.os_type == "Linux":
            try:
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME="):
                            info["linux_distro"] = line.split("=")[1].strip().strip('"')
                            break
            except FileNotFoundError:
                info["linux_distro"] = "Unknown"
        elif self.os_type == "Darwin":
            info["macos_version"] = platform.mac_ver()[0]

        return info

    def check_python_compatibility(self) -> bool:
        """Check if Python version is compatible."""
        min_version = (3, 8)
        current = self.python_version[:2]

        if current < min_version:
            self.errors.append(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"but {current[0]}.{current[1]} found"
            )
            return False

        self.log(f"[OK] Python {current[0]}.{current[1]} is compatible")
        return True

    def check_gpu_support(self) -> Dict[str, Optional[str]]:
        """Check for GPU and CUDA support."""
        gpu_info = {
            "cuda_available": None,
            "cuda_version": None,
            "gpu_count": None,
            "gpu_names": []
        }

        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()
            if gpu_info["cuda_available"]:
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_info["gpu_names"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
                self.log(f"[OK] CUDA {gpu_info['cuda_version']} detected with {gpu_info['gpu_count']} GPU(s)")
                for i, name in enumerate(gpu_info["gpu_names"]):
                    self.log(f"  GPU {i}: {name}")
            else:
                self.log("[WARNING] CUDA not available - will use CPU", "WARNING")
        except ImportError:
            self.log("[WARNING] PyTorch not installed - cannot check GPU support", "WARNING")

        return gpu_info

    def setup_windows(self) -> bool:
        """Windows-specific setup."""
        self.log("Setting up for Windows...")
        success = True

        # Set encoding for Windows
        if 'PYTHONIOENCODING' not in os.environ:
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            self.log("Set PYTHONIOENCODING=utf-8 for Windows compatibility")

        # Install Windows-specific packages
        windows_packages = [
            "pywin32",  # Windows API access
            "colorama", # Colored terminal output on Windows
        ]

        for package in windows_packages:
            if not self.check_only:
                self.log(f"Installing Windows package: {package}")
                cmd_success, _ = self.run_command(
                    [sys.executable, "-m", "pip", "install", package],
                    f"Install {package}",
                    capture_output=False
                )
                if not cmd_success:
                    self.warnings.append(f"Failed to install {package}")
                    success = False

        # Check for Visual Studio Build Tools (needed for some packages)
        self.log("Checking for Visual Studio Build Tools...")
        vswhere_path = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"

        if vswhere_path.exists():
            cmd_success, output = self.run_command([str(vswhere_path), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"])
            if cmd_success and output.strip():
                self.log("[OK] Visual Studio Build Tools detected")
            else:
                self.warnings.append(
                    "Visual Studio Build Tools not found. "
                    "Some packages may fail to install. "
                    "Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
                )
        else:
            self.warnings.append("Cannot check for Visual Studio Build Tools")

        return success

    def setup_linux(self) -> bool:
        """Linux-specific setup."""
        self.log("Setting up for Linux...")
        success = True

        # Check for common system dependencies
        system_deps = {
            "git": "git --version",
            "gcc": "gcc --version",
            "make": "make --version",
            "python3-dev": "python3-config --exists",
        }

        missing_deps = []
        for dep, check_cmd in system_deps.items():
            cmd_success, _ = self.run_command(check_cmd.split())
            if not cmd_success:
                missing_deps.append(dep)

        if missing_deps:
            self.warnings.append(
                f"Missing system dependencies: {', '.join(missing_deps)}. "
                f"Install with: sudo apt-get install {' '.join(missing_deps)}"
            )

        return success

    def setup_macos(self) -> bool:
        """macOS-specific setup."""
        self.log("Setting up for macOS...")
        success = True

        # Check for Xcode Command Line Tools
        cmd_success, _ = self.run_command(["xcode-select", "--print-path"])
        if not cmd_success:
            self.warnings.append(
                "Xcode Command Line Tools not found. "
                "Install with: xcode-select --install"
            )
        else:
            self.log("[OK] Xcode Command Line Tools detected")

        # Check for Homebrew (optional but recommended)
        cmd_success, _ = self.run_command(["brew", "--version"])
        if cmd_success:
            self.log("[OK] Homebrew detected")
        else:
            self.log("[WARNING] Homebrew not found (optional but recommended)", "WARNING")

        return success

    def install_requirements(self) -> bool:
        """Install Python requirements."""
        requirements_file = self.project_root / "requirements.txt"

        if not requirements_file.exists():
            self.errors.append(f"Requirements file not found: {requirements_file}")
            return False

        self.log(f"Installing requirements from {requirements_file}")

        # Upgrade pip first
        self.log("Upgrading pip...")
        cmd_success, _ = self.run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            "Upgrade pip",
            capture_output=False
        )

        if not cmd_success:
            self.warnings.append("Failed to upgrade pip")

        # Install requirements
        cmd_success, _ = self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            "Install requirements",
            capture_output=False
        )

        if not cmd_success:
            self.errors.append("Failed to install requirements")
            return False

        self.log("[OK] Requirements installed successfully")
        return True

    def validate_installation(self) -> bool:
        """Validate that key packages are properly installed."""
        key_packages = [
            "torch",
            "numpy",
            "pandas",
            "yaml",
            "fastapi",
            "uvicorn"
        ]

        self.log("Validating installation...")
        failed_imports = []

        for package in key_packages:
            try:
                __import__(package)
                self.log(f"[OK] {package}")
            except ImportError:
                failed_imports.append(package)
                self.log(f"[FAIL] {package}", "ERROR")

        if failed_imports:
            self.errors.append(f"Failed to import: {', '.join(failed_imports)}")
            return False

        return True

    def create_sample_env_file(self) -> bool:
        """Create a .env file from .env.example if it doesn't exist."""
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"

        if env_file.exists():
            self.log("[OK] .env file already exists")
            return True

        if not env_example.exists():
            self.warnings.append(".env.example not found - cannot create .env file")
            return False

        if not self.check_only:
            try:
                shutil.copy2(env_example, env_file)
                self.log(f"[OK] Created .env file from .env.example")
                self.log("Please edit .env file with your specific configuration")
                return True
            except Exception as e:
                self.warnings.append(f"Failed to create .env file: {e}")
                return False
        else:
            self.log("CHECK-ONLY: Would create .env file from .env.example")
            return True

    def print_summary(self):
        """Print setup summary."""
        print("\n" + "="*60)
        print("ENVIRONMENT SETUP SUMMARY")
        print("="*60)

        if self.errors:
            print("\n[ERROR] ERRORS:")
            for error in self.errors:
                print(f"  * {error}")

        if self.warnings:
            print("\n[WARNING] WARNINGS:")
            for warning in self.warnings:
                print(f"  * {warning}")

        if not self.errors and not self.warnings:
            print("\n[SUCCESS] All checks passed! Environment setup completed successfully.")
        elif not self.errors:
            print("\n[SUCCESS] Setup completed with warnings (review above).")
        else:
            print("\n[FAILED] Setup failed (see errors above).")

        print("\n[NEXT STEPS]:")
        print("1. Review and edit .env file with your configuration")
        print("2. Test the training script: python scripts/training/train_ultra_simple.py --ticker AAPL")
        print("3. Check the README.md for additional setup instructions")
        print("="*60)

    def setup_environment(self) -> bool:
        """Main setup function."""
        print("TimeSeries Transformer Environment Setup")
        print("="*60)

        # Detect system
        system_info = self.detect_system_info()
        print(f"Detected OS: {system_info['os']} ({system_info['platform']})")
        print(f"Python: {system_info['python_version']}")
        print(f"Architecture: {system_info['architecture']}")

        # Check Python compatibility
        if not self.check_python_compatibility():
            return False

        # Platform-specific setup
        setup_success = True
        if self.os_type == "Windows":
            setup_success = self.setup_windows()
        elif self.os_type == "Linux":
            setup_success = self.setup_linux()
        elif self.os_type == "Darwin":
            setup_success = self.setup_macos()
        else:
            self.warnings.append(f"Unsupported OS: {self.os_type}")

        # Common setup steps
        if setup_success and not self.check_only:
            # Install requirements
            if not self.install_requirements():
                setup_success = False

            # Validate installation
            if setup_success and not self.validate_installation():
                setup_success = False

            # Create .env file
            self.create_sample_env_file()

        # Check GPU support (after package installation)
        if not self.check_only:
            gpu_info = self.check_gpu_support()

        # Print summary
        self.print_summary()

        return setup_success and not self.errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-platform environment setup for TimeSeries Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/setup_environment.py              # Full setup
    python scripts/setup_environment.py --check-only # Check environment only
    python scripts/setup_environment.py --verbose    # Verbose output
        """
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment, don't install packages"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Create setup instance and run
    setup = EnvironmentSetup(verbose=args.verbose, check_only=args.check_only)
    success = setup.setup_environment()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()