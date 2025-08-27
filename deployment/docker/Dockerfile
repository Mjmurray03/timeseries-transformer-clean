# Multi-stage Dockerfile for timeseries-transformer
# Following deployment-standards.md specifications

# Stage 1: Builder
FROM python:3.10 AS builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install poetry and build wheel
RUN pip install --no-cache-dir poetry==1.7.1 && \
    poetry config virtualenvs.create false && \
    poetry build

# Stage 2: Testing
FROM builder AS tester
WORKDIR /test

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./

# Install dev dependencies
RUN poetry install --with dev

# Run tests and generate coverage
RUN pytest tests/ --cov=src --cov-report=xml --cov-report=term

# Stage 3: Training Image - Full ML stack
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04 AS training
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create requirements file for training
COPY requirements.txt ./requirements-train.txt
RUN echo "torch==2.1.0+cu121" >> requirements-train.txt && \
    echo "wandb==0.16.0" >> requirements-train.txt && \
    echo "optuna==3.4.0" >> requirements-train.txt

# Install training dependencies
RUN pip3 install --no-cache-dir -r requirements-train.txt

# Copy application code
COPY --from=builder /build/dist/*.whl ./
RUN pip3 install *.whl && rm *.whl

# Copy scripts and configs
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Set environment variables for GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Training entrypoint
CMD ["python3", "-m", "scripts.train"]

# Stage 4: Inference Image - Minimal footprint
FROM python:3.10-slim AS inference
WORKDIR /app

# Install only inference dependencies
COPY requirements.txt ./requirements-inference.txt
RUN sed -i '/wandb/d; /optuna/d; /pytest/d; /black/d; /mypy/d' requirements-inference.txt && \
    pip install --no-cache-dir -r requirements-inference.txt

# Copy application wheel from builder
COPY --from=builder /build/dist/*.whl ./
RUN pip install *.whl && rm *.whl

# Create model directory
RUN mkdir -p /app/models /app/logs

# Copy model artifacts (will be mounted in production)
# COPY models/ ./models/

# Health check script
COPY deployment/docker/health_check.py ./

# Set non-root user for security
RUN useradd -m -u 1000 modelserver && \
    chown -R modelserver:modelserver /app

USER modelserver

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python health_check.py || exit 1

# Production entrypoint
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 5: Development Image
FROM python:3.10 AS development
WORKDIR /app

# Install all dependencies including dev tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy and install all requirements
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code (mounted in dev)
COPY . .

# Install in editable mode
RUN pip install -e .

# Development tools
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    jupyterlab

# Expose ports for API and Jupyter
EXPOSE 8000 8888

# Development entrypoint with hot reload
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]