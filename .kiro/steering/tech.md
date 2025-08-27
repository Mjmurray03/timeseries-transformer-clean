# Technical Stack - Time-Series Transformer
---
inclusion: always
priority: 1
---

## Core ML Framework
- **PyTorch 2.4.0**: Primary deep learning framework
- **CUDA 12.4**: GPU acceleration
- **cuDNN 8.9**: Optimized neural network primitives
- **Mixed Precision (fp16)**: Memory optimization with torch.amp

## Model Architecture Components
```python
# Transformer Configuration
MODEL_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.1,
    "max_seq_length": 60,
    "forecast_horizon": 5
}
```

## Data Pipeline Stack
- **yfinance**: Primary market data source
- **pandas 2.0+**: Data manipulation with PyArrow backend
- **numpy**: Numerical operations
- **ta-lib**: Technical indicators (RSI, MACD, Bollinger Bands)
- **scikit-learn**: Preprocessing (StandardScaler, train_test_split)

## Training Infrastructure
- **Hardware**: NVIDIA L4 GPU (24GB VRAM)
- **Distributed Training**: torch.nn.DataParallel (single-node multi-GPU ready)
- **Gradient Accumulation**: Effective batch size 128 with accumulation
- **Optimizer**: AdamW with cosine annealing schedule
- **Early Stopping**: Patience=10 epochs on validation loss

## Model Serving
- **FastAPI**: REST API framework
- **Pydantic**: Request/response validation
- **Redis**: Prediction caching (5-minute TTL)
- **uvicorn**: ASGI server with multiple workers
- **Docker**: Containerization with multi-stage builds

## Monitoring & Logging
- **Weights & Biases**: Experiment tracking, hyperparameter sweeps
- **TensorBoard**: Real-time training visualization
- **Prometheus**: Metrics collection
- **Grafana**: Dashboard visualization
- **structlog**: Structured JSON logging

## Testing Framework
- **pytest**: Unit and integration testing
- **pytest-cov**: Coverage reporting (target >80%)
- **hypothesis**: Property-based testing for data pipelines
- **pytest-benchmark**: Performance regression testing

## Development Tools
- **black**: Code formatting (line-length=100)
- **isort**: Import sorting
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality
- **poetry**: Dependency management

## Deployment Pipeline
- **GitHub Actions**: CI/CD automation
- **Docker Hub**: Container registry
- **AWS ECR**: Private container storage (alternative)
- **Kubernetes**: Orchestration (future scaling)

## Data Storage
- **Local Development**: SQLite for metadata, Parquet for time-series
- **Production**: PostgreSQL + TimescaleDB extension
- **Model Registry**: MLflow for versioning
- **Feature Store**: Feast (future enhancement)

## Security & Compliance
- **python-dotenv**: Environment variable management
- **cryptography**: API key encryption
- **requests-ratelimiter**: API rate limiting
- **audit-log**: Trading decision logging

## Performance Optimization
- **torch.jit.script**: Model compilation for inference
- **ONNX Runtime**: Cross-platform deployment
- **TensorRT**: GPU inference optimization (optional)
- **Ray Tune**: Hyperparameter optimization

## Version Requirements
```toml
[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.4.0"
pandas = "^2.0.0"
numpy = "^1.24.0"
fastapi = "^0.100.0"
yfinance = "^0.2.28"
wandb = "^0.15.0"
```

## Environment Setup Commands
```bash
# GPU environment
conda create -n timeseries-transformer python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

# CPU-only development
pip install torch --index-url https://download.pytorch.org/whl/cpu
```