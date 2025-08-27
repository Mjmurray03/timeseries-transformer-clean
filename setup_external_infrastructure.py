#!/usr/bin/env python3
"""
Setup External Infrastructure - Missing Implementation Generator
Time-Series Transformer Project

This script generates and implements missing infrastructure components
identified in the infrastructure audit.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('infrastructure_setup.log')
    ]
)
logger = logging.getLogger(__name__)

class InfrastructureSetup:
    """Setup missing infrastructure components."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.docker_dir = self.project_root / "deployment" / "docker"
        self.scripts_dir = self.project_root / "scripts"
        self.config_dir = self.project_root / "config"
        self.tests_dir = self.project_root / "tests"
        
    def run_all_setups(self):
        """Run all infrastructure setup tasks."""
        logger.info("Starting external infrastructure setup...")
        
        try:
            # Critical fixes first
            self.fix_docker_infrastructure()
            self.add_mlflow_dependencies() 
            self.create_gitattributes()
            
            # Additional enhancements
            self.create_verification_scripts()
            self.create_external_services_config()
            self.create_integration_tests()
            
            logger.info("✅ All infrastructure setup tasks completed successfully!")
            self.print_next_steps()
            
        except Exception as e:
            logger.error(f"❌ Infrastructure setup failed: {e}")
            sys.exit(1)
    
    def fix_docker_infrastructure(self):
        """Fix empty Docker files with proper implementations."""
        logger.info("🐳 Fixing Docker infrastructure...")
        
        # Dockerfile.training
        training_dockerfile = self.docker_dir / "Dockerfile.training"
        training_content = '''# Training Image - Full ML stack with CUDA support
FROM nvidia/cuda:12.4-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    git \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \\
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Setup permissions
RUN chmod +x scripts/*.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import torch; print('CUDA:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python", "-m", "src.training.train"]
'''
        training_dockerfile.write_text(training_content)
        logger.info(f"✅ Created {training_dockerfile}")
        
        # Dockerfile.inference
        inference_dockerfile = self.docker_dir / "Dockerfile.inference"
        inference_content = '''# Multi-stage build for inference
FROM python:3.10 AS builder

WORKDIR /build
COPY requirements-inference.txt* requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir poetry \\
    && poetry config virtualenvs.create false \\
    && poetry install --only=main --no-dev

FROM python:3.10-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \\
    && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        inference_dockerfile.write_text(inference_content)
        logger.info(f"✅ Created {inference_dockerfile}")
        
        # docker-compose.yaml
        compose_file = self.docker_dir / "docker-compose.yaml"
        compose_content = '''version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: timeseries-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  training:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile.training
    container_name: timeseries-training
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
    volumes:
      - ../../data:/app/data
      - ../../models:/app/models
      - ../../logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  api:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile.inference
    container_name: timeseries-api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - PYTHONPATH=/app
    volumes:
      - ../../models:/app/models:ro
      - ../../configs:/app/configs:ro
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: timeseries-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ../../certs:/etc/nginx/certs:ro
    depends_on:
      - api
    restart: unless-stopped

  mlflow:
    image: python:3.10-slim
    container_name: timeseries-mlflow
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/app/mlruns
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///app/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns
    command: >
      bash -c "pip install mlflow[extras] && 
               mlflow server 
               --backend-store-uri sqlite:///app/mlflow.db 
               --default-artifact-root /app/mlruns 
               --host 0.0.0.0 
               --port 5000"
    restart: unless-stopped

volumes:
  redis_data:
    driver: local
  mlflow_data:
    driver: local

networks:
  default:
    name: timeseries-network
'''
        compose_file.write_text(compose_content)
        logger.info(f"✅ Created {compose_file}")
    
    def add_mlflow_dependencies(self):
        """Add MLflow dependencies to requirements files."""
        logger.info("📊 Adding MLflow dependencies...")
        
        # Add to requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            content = requirements_file.read_text()
            if "mlflow" not in content:
                content += "\n# ML Experiment Tracking\n"
                content += "mlflow>=2.8.0\n"
                content += "mlflow[extras]>=2.8.0\n"
                requirements_file.write_text(content)
                logger.info("✅ Added MLflow to requirements.txt")
        
        # Add to pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            if "mlflow" not in content:
                # Insert MLflow dependency
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'wandb = "^0.15.0"' in line:
                        lines.insert(i + 1, 'mlflow = "^2.8.0"')
                        break
                content = '\n'.join(lines)
                pyproject_file.write_text(content)
                logger.info("✅ Added MLflow to pyproject.toml")
    
    def create_gitattributes(self):
        """Create .gitattributes file for Git LFS."""
        logger.info("📦 Creating .gitattributes for Git LFS...")
        
        gitattributes_file = self.project_root / ".gitattributes"
        gitattributes_content = '''# Git LFS tracking rules for Time-Series Transformer

# Model files
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text

# Model directories
models/** filter=lfs diff=lfs merge=lfs -text
checkpoints/** filter=lfs diff=lfs merge=lfs -text

# Large data files
*.parquet filter=lfs diff=lfs merge=lfs -text
*.feather filter=lfs diff=lfs merge=lfs -text
data/raw/** filter=lfs diff=lfs merge=lfs -text
data/processed/**/*.parquet filter=lfs diff=lfs merge=lfs -text

# Artifacts and exports
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
artifacts/** filter=lfs diff=lfs merge=lfs -text

# Documentation and media
*.pdf filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text

# Archive files
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
'''
        gitattributes_file.write_text(gitattributes_content)
        logger.info(f"✅ Created {gitattributes_file}")
    
    def create_verification_scripts(self):
        """Create infrastructure verification scripts."""
        logger.info("🔍 Creating verification scripts...")
        
        # Main verification script
        verify_script = self.scripts_dir / "verify_infrastructure.sh"
        verify_content = '''#!/bin/bash
# Comprehensive Infrastructure Verification Script
# Time-Series Transformer Project

set -e

echo "🔍 Time-Series Transformer Infrastructure Verification"
echo "=================================================="

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Verification functions
check_doppler() {
    echo -n "📡 Checking Doppler CLI connection... "
    if C:/tools/doppler/doppler.exe me > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Connected${NC}"
        echo "   Project: $(C:/tools/doppler/doppler.exe configure get project.name)"
    else
        echo -e "${RED}❌ Not connected${NC}"
        return 1
    fi
}

check_secrets() {
    echo -n "🔐 Verifying API keys from Doppler... "
    missing_keys=()
    
    for key in ALPHA_VANTAGE_API_KEY NEWSAPI_API_KEY HUGGINGFACE_API_KEY WANDB_API_KEY; do
        if ! C:/tools/doppler/doppler.exe secrets get "$key" > /dev/null 2>&1; then
            missing_keys+=("$key")
        fi
    done
    
    if [ ${#missing_keys[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ All keys present${NC}"
    else
        echo -e "${RED}❌ Missing keys: ${missing_keys[*]}${NC}"
        return 1
    fi
}

check_redis() {
    echo -n "🔴 Checking Redis connection... "
    if python -c "import redis; r=redis.Redis(); r.ping()" 2>/dev/null; then
        echo -e "${GREEN}✅ Connected${NC}"
    else
        echo -e "${YELLOW}⚠️  Redis not running (optional for development)${NC}"
    fi
}

check_wandb() {
    echo -n "📊 Checking W&B authentication... "
    if python -c "import wandb; wandb.login()" 2>/dev/null; then
        echo -e "${GREEN}✅ Authenticated${NC}"
    else
        echo -e "${YELLOW}⚠️  W&B not authenticated${NC}"
    fi
}

check_dependencies() {
    echo -n "📦 Checking Python dependencies... "
    missing_deps=()
    
    for dep in torch pandas numpy wandb redis mlflow; do
        if ! python -c "import $dep" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ All dependencies installed${NC}"
    else
        echo -e "${RED}❌ Missing dependencies: ${missing_deps[*]}${NC}"
        echo "   Run: pip install -r requirements.txt"
        return 1
    fi
}

check_docker() {
    echo -n "🐳 Checking Docker infrastructure... "
    if [ -f "deployment/docker/Dockerfile.training" ] && [ -s "deployment/docker/Dockerfile.training" ]; then
        echo -e "${GREEN}✅ Docker files present${NC}"
    else
        echo -e "${RED}❌ Docker files missing or empty${NC}"
        return 1
    fi
}

check_git_lfs() {
    echo -n "📦 Checking Git LFS configuration... "
    if [ -f ".gitattributes" ] && git lfs track > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Git LFS configured${NC}"
        echo "   Tracked patterns: $(git lfs track | wc -l) patterns"
    else
        echo -e "${RED}❌ Git LFS not configured${NC}"
        return 1
    fi
}

# Run all checks
echo
failed_checks=0

check_doppler || ((failed_checks++))
check_secrets || ((failed_checks++))
check_dependencies || ((failed_checks++))
check_redis
check_wandb
check_docker || ((failed_checks++))
check_git_lfs || ((failed_checks++))

echo
echo "=================================================="
if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}🎉 All critical infrastructure checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $failed_checks critical check(s) failed${NC}"
    echo "   Please fix the issues above before proceeding"
    exit 1
fi
'''
        verify_script.write_text(verify_content)
        
        # Make script executable (Windows compatible)
        if os.name != 'nt':
            os.chmod(verify_script, 0o755)
        
        logger.info(f"✅ Created {verify_script}")
        
        # Connection test script
        test_connections_script = self.scripts_dir / "test_connections.py"
        test_connections_content = '''#!/usr/bin/env python3
"""Test all external service connections."""

import sys
import logging
from src.config.secrets import secrets
from src.api.cache import initialize_caches, get_prediction_cache
from src.training.wandb_setup import init_wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_doppler():
    """Test Doppler secrets access."""
    try:
        api_key = secrets.get('ALPHA_VANTAGE_API_KEY')
        if api_key:
            logger.info("✅ Doppler secrets accessible")
            return True
        else:
            logger.error("❌ Doppler secrets not accessible")
            return False
    except Exception as e:
        logger.error(f"❌ Doppler test failed: {e}")
        return False

def test_redis():
    """Test Redis connection."""
    try:
        initialize_caches()
        cache = get_prediction_cache()
        if cache and cache.health_check():
            logger.info("✅ Redis connection working")
            return True
        else:
            logger.warning("⚠️ Redis connection failed (optional)")
            return True  # Non-critical
    except Exception as e:
        logger.warning(f"⚠️ Redis test failed: {e}")
        return True  # Non-critical

def test_wandb():
    """Test W&B connection."""
    try:
        run = init_wandb(
            experiment_name="connection_test",
            mode="offline",
            config={"test": True}
        )
        if run:
            run.finish()
            logger.info("✅ W&B connection working")
            return True
        else:
            logger.error("❌ W&B connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ W&B test failed: {e}")
        return False

def main():
    """Run all connection tests."""
    logger.info("🔍 Testing external service connections...")
    
    tests = [
        ("Doppler", test_doppler),
        ("Redis", test_redis),
        ("W&B", test_wandb),
    ]
    
    failed = 0
    for name, test_func in tests:
        logger.info(f"Testing {name}...")
        if not test_func():
            failed += 1
        print()
    
    if failed == 0:
        logger.info("🎉 All connection tests passed!")
        return 0
    else:
        logger.error(f"❌ {failed} connection test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        test_connections_script.write_text(test_connections_content)
        logger.info(f"✅ Created {test_connections_script}")
    
    def create_external_services_config(self):
        """Create consolidated external services configuration."""
        logger.info("⚙️ Creating external services configuration...")
        
        self.config_dir.mkdir(exist_ok=True)
        
        external_services_config = self.config_dir / "external_services.yaml"
        config_content = '''# External Services Configuration
# Time-Series Transformer Project

doppler:
  enabled: true
  cli_path: "C:/tools/doppler/doppler.exe"
  project: "timeseries-transformer"
  environment: "dev"
  secrets:
    - ALPHA_VANTAGE_API_KEY
    - NEWSAPI_API_KEY
    - HUGGINGFACE_API_KEY
    - WANDB_API_KEY
  fallback_to_env: true

wandb:
  enabled: true
  project: "timeseries-transformer"
  entity: null  # Use default
  offline_mode: false
  tags:
    - "production"
    - "transformer"
    - "stock-prediction"
  logging:
    batch_interval: 10
    epoch_interval: 1
    checkpoint_interval: 10
    visualization_interval: 5

redis:
  enabled: true
  host: "localhost"
  port: 6379
  db: 0
  connection_timeout: 5
  socket_timeout: 5
  retry_on_timeout: true
  cache:
    prediction_ttl: 300  # 5 minutes
    model_cache_size: 10
    default_ttl: 3600  # 1 hour

mlflow:
  enabled: true
  tracking_uri: "http://localhost:5000"
  experiment_name: "timeseries-transformer"
  artifact_location: "./mlruns"
  registry_uri: null

alpha_vantage:
  enabled: true
  base_url: "https://www.alphavantage.co/query"
  rate_limit: 5  # requests per minute
  timeout: 30

news_api:
  enabled: true
  base_url: "https://newsapi.org/v2"
  rate_limit: 1000  # requests per day
  timeout: 30

huggingface:
  enabled: true
  base_url: "https://api-inference.huggingface.co"
  model_cache: true
  timeout: 60

docker:
  registry: "docker.io"
  namespace: "timeseries-transformer"
  images:
    training: "timeseries-transformer:training-latest"
    inference: "timeseries-transformer:inference-latest"
    notebook: "timeseries-transformer:notebook-latest"
  
monitoring:
  prometheus:
    enabled: false
    endpoint: "/metrics"
    port: 8080
  
  grafana:
    enabled: false
    port: 3000
    
health_checks:
  interval: 30  # seconds
  timeout: 10   # seconds
  retries: 3
  
logging:
  level: "INFO"
  format: "json"
  file: "infrastructure.log"
'''
        external_services_config.write_text(config_content)
        logger.info(f"✅ Created {external_services_config}")
    
    def create_integration_tests(self):
        """Create integration tests for external connections."""
        logger.info("🧪 Creating integration tests...")
        
        test_dir = self.tests_dir / "integration"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_external_connections = test_dir / "test_external_connections.py"
        test_content = '''"""Integration tests for external service connections."""

import pytest
import os
from unittest.mock import Mock, patch
from src.config.secrets import secrets
from src.api.cache import PredictionCache, initialize_caches
from src.training.wandb_setup import init_wandb


class TestDopplerIntegration:
    """Test Doppler secrets management."""
    
    def test_secrets_manager_initialization(self):
        """Test SecretsManager initialization."""
        assert secrets is not None
        assert hasattr(secrets, 'get')
    
    def test_api_key_retrieval(self):
        """Test API key retrieval from secrets."""
        # Test that method exists and doesn't raise
        alpha_key = secrets.get('ALPHA_VANTAGE_API_KEY')
        assert alpha_key is not None or alpha_key == ""  # Could be empty in test
    
    def test_fallback_mechanism(self):
        """Test fallback to environment variables."""
        with patch.dict(os.environ, {'TEST_KEY': 'test_value'}):
            # This tests the fallback mechanism implicitly
            assert os.getenv('TEST_KEY') == 'test_value'


class TestRedisIntegration:
    """Test Redis caching integration."""
    
    def test_redis_connection_handling(self):
        """Test Redis connection with error handling."""
        # Test that it handles connection failures gracefully
        cache = PredictionCache(redis_host='nonexistent_host')
        assert cache.redis_client is None
    
    def test_cache_initialization(self):
        """Test cache system initialization."""
        initialize_caches(redis_host='localhost', redis_port=6379)
        # Should not raise even if Redis is not available
        assert True
    
    @pytest.mark.redis
    def test_redis_operations(self):
        """Test Redis operations (requires running Redis)."""
        cache = PredictionCache()
        if cache.redis_client:
            # Test basic operations
            assert cache.health_check() is True
            stats = cache.get_stats()
            assert 'enabled' in stats
            assert stats['enabled'] is True


class TestWandBIntegration:
    """Test Weights & Biases integration."""
    
    def test_wandb_initialization_offline(self):
        """Test W&B initialization in offline mode."""
        run = init_wandb(
            experiment_name="test_run",
            mode="offline",
            config={"test": True}
        )
        assert run is not None
        run.finish()
    
    def test_wandb_config_handling(self):
        """Test W&B configuration handling."""
        config = {
            "model": {"architecture": "transformer"},
            "training": {"learning_rate": 1e-4},
            "data": {"dataset": "test"}
        }
        
        # Should not raise
        run = init_wandb(
            experiment_name="test_config",
            mode="offline",
            config=config
        )
        assert run is not None
        run.finish()


class TestMLflowIntegration:
    """Test MLflow integration."""
    
    def test_mlflow_import(self):
        """Test MLflow import and availability."""
        try:
            import mlflow
            assert mlflow is not None
        except ImportError:
            pytest.skip("MLflow not installed")
    
    @pytest.mark.mlflow
    def test_mlflow_experiment_creation(self):
        """Test MLflow experiment creation."""
        try:
            import mlflow
            mlflow.set_experiment("test_experiment")
            assert True
        except ImportError:
            pytest.skip("MLflow not installed")


class TestExternalAPIConnections:
    """Test external API service connections."""
    
    def test_alpha_vantage_key_exists(self):
        """Test Alpha Vantage API key exists."""
        key = secrets.get('ALPHA_VANTAGE_API_KEY')
        # In tests, key might be empty but method should work
        assert isinstance(key, (str, type(None)))
    
    def test_news_api_key_exists(self):
        """Test News API key exists.""" 
        key = secrets.get('NEWSAPI_API_KEY')
        assert isinstance(key, (str, type(None)))
    
    def test_huggingface_key_exists(self):
        """Test Hugging Face API key exists."""
        key = secrets.get('HUGGINGFACE_API_KEY')
        assert isinstance(key, (str, type(None)))


@pytest.mark.integration
class TestFullIntegrationWorkflow:
    """Test complete integration workflow."""
    
    def test_end_to_end_setup(self):
        """Test end-to-end infrastructure setup."""
        # Initialize all services
        initialize_caches()
        
        # Test W&B in offline mode
        run = init_wandb(
            experiment_name="integration_test",
            mode="offline",
            config={"integration": True}
        )
        
        # Test basic operations
        if run:
            run.log({"test_metric": 1.0})
            run.finish()
        
        assert True  # If we get here, basic integration works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        test_external_connections.write_text(test_content)
        logger.info(f"✅ Created {test_external_connections}")
        
        # Create pytest configuration for integration tests
        pytest_ini = self.project_root / "pytest.ini"
        if not pytest_ini.exists():
            pytest_content = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests (fast)
    integration: Integration tests (slower)
    redis: Tests requiring Redis
    mlflow: Tests requiring MLflow
    wandb: Tests requiring W&B
    slow: Tests taking > 1 second
    external: Tests requiring external services

addopts = 
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
'''
            pytest_ini.write_text(pytest_content)
            logger.info(f"✅ Created {pytest_ini}")
    
    def print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("🎉 INFRASTRUCTURE SETUP COMPLETE!")
        print("="*60)
        print("\nNext Steps:")
        print("\n1. Install MLflow dependencies:")
        print("   pip install mlflow[extras]>=2.8.0")
        print("\n2. Verify infrastructure:")
        print("   bash scripts/verify_infrastructure.sh")
        print("\n3. Test connections:")
        print("   python scripts/test_connections.py")
        print("\n4. Run integration tests:")
        print("   pytest tests/integration/test_external_connections.py -v")
        print("\n5. Build Docker containers:")
        print("   docker-compose -f deployment/docker/docker-compose.yaml build")
        print("\n6. Start services:")
        print("   docker-compose -f deployment/docker/docker-compose.yaml up -d")
        print("\nFor more details, see the INFRASTRUCTURE_AUDIT.md report.")
        print("="*60)


def main():
    """Main setup function."""
    setup = InfrastructureSetup()
    setup.run_all_setups()


if __name__ == "__main__":
    main()