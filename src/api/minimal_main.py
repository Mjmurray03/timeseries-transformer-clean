"""
Minimal FastAPI Application for Time-Series Transformer Project
===============================================================

This is a lightweight fallback API that can be used to test:
1. FastAPI framework functionality
2. Basic system health and device detection
3. Model file discovery without loading
4. API routing and JSON response formatting

Use this to isolate issues between API framework and model loading logic.

Usage:
    python -m uvicorn src.api.minimal_main:app --reload --host 0.0.0.0 --port 8001
    or
    python src/api/minimal_main.py
"""

from pathlib import Path
from typing import Dict, List, Any, Union
import torch
from fastapi import FastAPI
from datetime import datetime
import platform
import sys

# Initialize FastAPI application
app = FastAPI(
    title="Minimal TimeSeries Transformer API",
    version="0.1.0",
    description="Lightweight API for system validation and basic model file discovery",
    docs_url="/docs",  # Enable Swagger UI
    redoc_url="/redoc"  # Enable ReDoc
)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root status endpoint - confirms API is running.
    
    Returns basic information about the API service status.
    """
    return {
        "status": "running",
        "service": "Minimal TimeSeries Transformer API",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "message": "API is operational - use /docs for interactive documentation"
    }


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """
    List available model files without loading them.
    
    Scans the models directory and returns information about available
    model files (.pt) and configuration files (.json).
    """
    try:
        if not MODELS_DIR.exists():
            return {
                "status": "error",
                "message": f"Models directory not found: {MODELS_DIR}",
                "model_files": [],
                "config_files": [],
                "total_files": 0
            }
        
        # Find model files
        model_files = []
        for pt_file in MODELS_DIR.glob("*.pt"):
            file_info = {
                "filename": pt_file.name,
                "size_mb": round(pt_file.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(pt_file.stat().st_mtime).isoformat()
            }
            model_files.append(file_info)
        
        # Find configuration files
        config_files = []
        for config_file in MODELS_DIR.glob("*.json"):
            file_info = {
                "filename": config_file.name,
                "size_kb": round(config_file.stat().st_size / 1024, 2),
                "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
            }
            config_files.append(file_info)
        
        return {
            "status": "success",
            "models_directory": str(MODELS_DIR),
            "model_files": sorted(model_files, key=lambda x: x["filename"]),
            "config_files": sorted(config_files, key=lambda x: x["filename"]),
            "summary": {
                "total_model_files": len(model_files),
                "total_config_files": len(config_files),
                "total_model_size_mb": sum(f["size_mb"] for f in model_files)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error scanning models directory: {str(e)}",
            "model_files": [],
            "config_files": [],
            "total_files": 0
        }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    System health check endpoint.
    
    Returns comprehensive system information including:
    - PyTorch installation and CUDA availability
    - Device information
    - Python and platform details
    - Memory information (if available)
    """
    try:
        # PyTorch and device information
        device_info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        }
        
        # Add CUDA device details if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_info["cuda_devices"] = []
            for i in range(torch.cuda.device_count()):
                device_properties = {
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                }
                device_info["cuda_devices"].append(device_properties)
        
        # System information
        system_info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0]
        }
        
        # Try to get memory information
        try:
            import psutil
            memory_info = {
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "memory_usage_percent": psutil.virtual_memory().percent
            }
        except ImportError:
            memory_info = {
                "message": "psutil not available - memory info unavailable"
            }
        
        # Directory checks
        directory_checks = {
            "project_root_exists": PROJECT_ROOT.exists(),
            "models_dir_exists": MODELS_DIR.exists(),
            "models_dir_path": str(MODELS_DIR)
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Minimal TimeSeries Transformer API",
            "device_info": device_info,
            "system_info": system_info,
            "memory_info": memory_info,
            "directory_checks": directory_checks,
            "health_score": "100%" if all([
                device_info["pytorch_version"],
                system_info["python_version"],
                directory_checks["project_root_exists"]
            ]) else "Partial"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Health check failed - see error details"
        }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print("=" * 60)
    print("🚀 Minimal TimeSeries Transformer API Starting...")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"🤖 Models Directory: {MODELS_DIR}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    print(f"⚡ CUDA Available: {torch.cuda.is_available()}")
    print(f"🖥️  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    if MODELS_DIR.exists():
        pt_files = len(list(MODELS_DIR.glob("*.pt")))
        json_files = len(list(MODELS_DIR.glob("*.json")))
        print(f"📊 Found {pt_files} model files and {json_files} config files")
    else:
        print("⚠️  Models directory not found")
    
    print("=" * 60)
    print("✅ API Ready! Visit http://localhost:8001/docs for interactive documentation")
    print("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information"""
    print("\n" + "=" * 60)
    print("🛑 Minimal TimeSeries Transformer API Shutting Down...")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Minimal TimeSeries Transformer API...")
    print("This API provides basic system validation without model loading.")
    print("Use this to test FastAPI functionality and system health.")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )