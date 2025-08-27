#!/usr/bin/env python3
"""
Setup script for the time-series transformer project.
Initializes the project structure and database.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.init_database import create_metadata_database

def setup_project():
    """Initialize the project structure and database."""
    print("Setting up time-series transformer project...")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize database
    print("Initializing metadata database...")
    create_metadata_database()
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file from template...")
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
        else:
            print("Warning: .env.example not found")
    
    print("Project setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run tests: pytest tests/")

if __name__ == "__main__":
    setup_project()