# Codebase Structure - Time-Series Transformer
---
inclusion: always
priority: 1
---

## Directory Organization

```
timeseries-transformer/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── yahoo_finance.py      # YFinance data collection
│   │   │   ├── alpha_vantage.py      # Alternative data source
│   │   │   └── news_sentiment.py     # Sentiment data collector
│   │   ├── processors/
│   │   │   ├── feature_engineering.py # Technical indicators, returns
│   │   │   ├── normalization.py      # Per-stock scaling
│   │   │   └── sequence_builder.py   # Sliding window creation
│   │   └── datasets/
│   │       ├── stock_dataset.py      # PyTorch Dataset class
│   │       └── data_loader.py        # Custom DataLoader with caching
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── attention.py          # Multi-head attention implementation
│   │   │   ├── positional_encoding.py # Learned/sinusoidal encoding
│   │   │   ├── feed_forward.py       # FFN with GELU activation
│   │   │   └── quantile_head.py      # Quantile regression layers
│   │   ├── architectures/
│   │   │   ├── transformer.py        # Main model class
│   │   │   ├── lstm_baseline.py      # LSTM for comparison
│   │   │   └── ensemble.py           # Model ensemble wrapper
│   │   └── losses/
│   │       ├── composite_loss.py     # Multi-objective loss
│   │       ├── directional_loss.py   # Classification component
│   │       └── quantile_loss.py      # Pinball loss for intervals
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   ├── validators.py            # Validation metrics
│   │   ├── callbacks/
│   │   │   ├── early_stopping.py    # Patience-based stopping
│   │   │   ├── model_checkpoint.py  # Best model saving
│   │   │   └── wandb_logger.py      # W&B integration
│   │   └── optimizers/
│   │       ├── scheduler.py         # Learning rate scheduling
│   │       └── gradient_clipper.py  # Gradient norm clipping
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── financial_metrics.py # Sharpe, Sortino, Calmar ratios
│   │   │   ├── ml_metrics.py        # RMSE, MAE, directional accuracy
│   │   │   └── calibration.py       # Confidence interval coverage
│   │   ├── backtesting/
│   │   │   ├── strategy.py          # Trading strategy implementation
│   │   │   ├── portfolio.py         # Portfolio simulation
│   │   │   └── costs.py             # Transaction costs, slippage
│   │   └── visualization/
│   │       ├── attention_maps.py    # Attention weight heatmaps
│   │       ├── predictions.py       # Prediction vs actual plots
│   │       └── performance.py       # Cumulative returns, drawdown
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py            # Model inference wrapper
│   │   ├── ensemble_predictor.py   # Multi-model predictions
│   │   └── explainer.py            # Attention-based explanations
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py                  # FastAPI application
│       ├── routes/
│       │   ├── predictions.py      # Prediction endpoints
│       │   ├── analysis.py         # Analysis endpoints
│       │   └── health.py           # Health checks
│       ├── schemas/
│       │   ├── requests.py         # Pydantic request models
│       │   └── responses.py        # Pydantic response models
│       └── middleware/
│           ├── rate_limiter.py     # API rate limiting
│           └── cache.py            # Redis caching layer
│
├── configs/
│   ├── model/
│   │   ├── transformer_base.yaml   # Base transformer config
│   │   ├── transformer_large.yaml  # Scaled up version
│   │   └── lstm_baseline.yaml      # LSTM config
│   ├── training/
│   │   ├── quick_test.yaml         # Fast debugging config
│   │   ├── full_training.yaml      # Production training
│   │   └── hyperparameter_sweep.yaml # HPO configuration
│   └── deployment/
│       ├── docker-compose.yaml     # Local deployment
│       ├── kubernetes.yaml         # K8s manifests
│       └── nginx.conf              # Reverse proxy config
│
├── data/
│   ├── raw/                        # Original downloaded data
│   │   └── {ticker}/
│   │       └── {date}.parquet
│   ├── processed/                  # Feature-engineered data
│   │   └── sequences/
│   │       └── {ticker}_{window}_{horizon}.pt
│   ├── cache/                      # Temporary computation cache
│   └── metadata/
│       ├── scalers/                # Saved StandardScaler objects
│       └── feature_stats.json      # Feature statistics
│
├── models/
│   ├── checkpoints/                # Training checkpoints
│   │   └── {experiment_id}/
│   │       ├── epoch_{n}.pt
│   │       └── best_model.pt
│   ├── production/                 # Production-ready models
│   │   ├── v1.0.0/
│   │   │   ├── model.pt
│   │   │   ├── config.json
│   │   │   └── metrics.json
│   │   └── latest/                # Symlink to current version
│   └── exports/                    # ONNX, TorchScript exports
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA and data analysis
│   ├── 02_feature_engineering.ipynb # Feature development
│   ├── 03_model_development.ipynb  # Architecture experiments
│   ├── 04_hyperparameter_tuning.ipynb # HPO analysis
│   └── 05_results_analysis.ipynb   # Final results and plots
│
├── tests/
│   ├── unit/
│   │   ├── test_data/              # Data pipeline tests
│   │   ├── test_models/            # Model component tests
│   │   └── test_training/          # Training logic tests
│   ├── integration/
│   │   ├── test_pipeline.py        # End-to-end pipeline
│   │   └── test_api.py             # API integration tests
│   └── performance/
│       ├── test_inference_speed.py # Latency benchmarks
│       └── test_memory_usage.py    # Memory profiling
│
├── scripts/
│   ├── data/
│   │   ├── download_historical.py  # Bulk data download
│   │   └── update_daily.py         # Daily data update
│   ├── training/
│   │   ├── train_single_gpu.py     # Single GPU training
│   │   ├── train_distributed.py    # Multi-GPU training
│   │   └── hyperparameter_search.py # HPO script
│   ├── evaluation/
│   │   ├── run_backtests.py        # Backtesting suite
│   │   └── generate_reports.py     # Performance reports
│   └── deployment/
│       ├── build_docker.sh         # Docker build script
│       ├── deploy_api.sh           # Deployment script
│       └── monitor_health.py       # Health monitoring
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.training     # Training environment
│   │   ├── Dockerfile.inference    # Lean inference image
│   │   └── Dockerfile.notebook     # Development environment
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├inia ├── service.yaml
│   │   └── ingress.yaml
│   └── monitoring/
│       ├── prometheus/
│       │   └── config.yaml
│       └── grafana/
│           └── dashboards/
│
├── docs/
│   ├── architecture/
│   │   ├── model_design.md        # Model architecture details
│   │   └── system_design.md       # System architecture
│   ├── api/
│   │   └── openapi.json           # API documentation
│   └── guides/
│       ├── setup.md               # Setup instructions
│       ├── training.md            # Training guide
│       └── deployment.md          # Deployment guide
│
└── .kiro/                         # Kiro IDE configuration

## Import Structure

```python
# Absolute imports from src/
from src.data.collectors import yahoo_finance
from src.models.architectures import transformer
from src.training import trainer
from src.evaluation.metrics import financial_metrics

# Relative imports within modules
from .components import attention
from ..losses import composite_loss
```

## Naming Conventions

### Files
- **Snake_case**: `feature_engineering.py`
- **Descriptive names**: `quantile_regression_head.py` not `qrh.py`
- **Test prefix**: `test_<module_name>.py`

### Classes
- **PascalCase**: `TemporalTransformer`, `StockDataset`
- **Descriptive**: `MultiHeadSelfAttention` not `MHSA`

### Functions
- **Snake_case**: `calculate_sharpe_ratio()`, `build_sequences()`
- **Verb prefixes**: `get_`, `create_`, `calculate_`, `validate_`

### Constants
- **UPPER_SNAKE_CASE**: `MAX_SEQUENCE_LENGTH`, `DEFAULT_BATCH_SIZE`

### Configuration Files
- **Descriptive hyphenated**: `transformer-base.yaml`
- **Environment prefix**: `prod-`, `dev-`, `test-`

## Module Responsibilities

### data/
- Single responsibility: Data acquisition and preprocessing
- No model logic
- Outputs: Clean, normalized PyTorch tensors

### models/
- Pure model definitions
- No training logic
- Stateless components

### training/
- Training orchestration
- Metric calculation during training
- Checkpoint management

### evaluation/
- Post-training analysis
- Backtesting logic
- Visualization generation

### api/
- HTTP interface only
- Thin wrapper around inference/
- No business logic

## Development Workflow

1. **Feature branches**: `feature/attention-visualization`
2. **Commit messages**: Conventional commits (feat:, fix:, docs:)
3. **PR requirements**: Tests pass, coverage >80%, approved review
4. **Version tags**: Semantic versioning `v1.2.3`