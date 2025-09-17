# Time-Series Transformer for Stock Market Prediction

## A Complete ML Pipeline Case Study: From Architecture to Production

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Recent Updates (September 2025)

### ✨ Major System Overhaul Completed!
This project has been completely updated with production-ready fixes and enhancements:

- **🐛 All Critical Bugs Fixed**: PyTorch 2.6+ compatibility, model loading, API endpoints
- **🎨 Professional UI**: Custom dark-themed Swagger UI with educational features
- **🔄 Robust Input Validation**: Support for multiple input formats with automatic reshaping
- **🧪 Comprehensive Testing**: Full test suite with 100% success rate
- **📚 Educational Content**: Built-in documentation perfect for students learning AI/ML
- **🏗️ Production Ready**: Proper error handling, caching, and monitoring

### 🎯 What's New
- **Custom Swagger UI**: Beautiful dark-themed documentation at `/docs`
- **Flexible API**: Accepts both flat arrays (600 elements) and 2D arrays (60×10)
- **Enhanced Models**: Dynamic model loading with fallback configurations
- **Student-Friendly**: Educational explanations for transformer concepts and AI workflows
- **Comprehensive Tests**: Python and PowerShell test suites for cross-platform validation

> **Key Insight**: This project demonstrates a complete end-to-end ML pipeline for financial time series prediction. While the transformer model achieved impressive technical metrics (RMSE: $0.268), it failed to generate profitable trading signals - providing valuable lessons about the gap between ML metrics and real-world performance.

## 📊 Project Overview

This repository contains a production-ready implementation of a transformer-based model for stock price prediction, complete with:
- GPU-optimized training pipeline
- Real-time data ingestion from multiple sources  
- Comprehensive backtesting framework with transaction costs
- RESTful API for model serving
- Complete MLOps infrastructure (Docker, MLflow, W&B)

### The Reality Check

**Technical Success:**
- ✅ 2.3M parameter transformer model successfully trained
- ✅ RMSE of $0.268 (~0.12% error) on price predictions
- ✅ Complete pipeline from data to deployment
- ✅ Professional infrastructure and monitoring

**Trading Performance:**
- ❌ 0% return (model made zero trades)
- ❌ -35% underperformance vs buy-and-hold
- ❌ Uniform predictions across all stocks
- ❌ No actionable trading signals generated

This documentation shares both the successes and failures transparently, as the lessons learned are more valuable than hiding the shortcomings.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, for training)
- 16GB RAM minimum
- 50GB disk space for data and models

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mjmurray03/timeseries-transformer-clean.git
cd timeseries-transformer-clean
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys (optional - works without them)
```

## Quick Start - Complete Demo

Run the entire pipeline with one command:
```bash
python scripts/run_demo.py
```

This will:
- ✅ **Download AAPL stock data** (2 years of historical data)
- ✅ **Train a model** (auto-detects GPU/CPU, ~5 minutes)
- ✅ **Run predictions** (generates future price forecasts)
- ✅ **Start API server** (interactive Swagger UI at http://localhost:8000/docs)
- ✅ **Show you how to extend the project** (configuration examples and next steps)

The demo is completely automated and works without any user intervention!

### Detailed Usage

#### Training a Model
```bash
# Basic training (auto-detects GPU)
python scripts/training/train_ultra_simple.py --ticker AAPL --epochs 50

# With config file
python scripts/training/train_ultra_simple.py --ticker AAPL --config config_example.yaml

# Multiple stocks
python scripts/training/train_multi_stock.py --tickers AAPL MSFT GOOG --epochs 50

# Force CPU usage
python scripts/training/train_ultra_simple.py --ticker AAPL --device cpu
```

#### Running Backtests
```bash
python scripts/backtesting/run_backtest.py \
    --predictions-path models/predictions.csv \
    --market-data-path data/processed/AAPL.csv \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

#### Using the API
```bash
# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "features": [/* 600 features */],
    "horizon": 3
  }'
```

#### Extending the Project

**Adding More Data Sources:**
Edit the `.env` file to add new data providers:
```bash
# Add to .env
ALPHA_VANTAGE_API_KEY=your_key_here
NEWSAPI_API_KEY=your_key_here
```

**Modifying Model Architecture:**
Edit `config_example.yaml` to change model parameters:
```yaml
# Model architecture
hidden_dim: 256     # Increase for more complexity
num_layers: 8       # Deeper networks
num_heads: 16       # More attention heads
dropout: 0.15       # Regularization
```

**Automating Daily Reports:**
Use `scripts/setup_environment.py` to set up platform-specific automation:
```bash
# Windows Task Scheduler
python scripts/setup_environment.py --check-only

# Linux/Mac cron
python scripts/daily_report.py --schedule daily
```

### Running the Complete Pipeline

1. **Download Historical Data:**
```bash
python scripts/data/download_data.py --tickers AAPL,MSFT,GOOGL,NVDA --years 5
```

2. **Train Model (GPU recommended):**
```bash
python scripts/training/train_ultra_simple.py --config configs/transformer_base.yaml
```

3. **Run Backtesting:**
```bash
python scripts/backtesting/run_backtest.py --model models/best_model.pt
```

4. **Start API Server:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

## 🚀 Quick Start (Under 5 Minutes!)

### Automated Installation (Linux/Mac)
```bash
git clone https://github.com/Mjmurray03/timeseries-transformer-clean
cd timeseries-transformer-clean
bash install.sh
source venv/bin/activate
```

### Test the Installation
```bash
# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

# Download sample data (2 stocks, 2 years)
python scripts/download_stock_data.py --tickers AAPL,MSFT --years 2

# Quick training test (should take ~3.5 minutes)
python scripts/training/train_ultra_simple.py --epochs 10 --ticker AAPL

# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API documentation available at: http://localhost:8000/docs

## 🔥 Production-Ready API Features

### 🎨 Custom Professional Swagger UI
Experience our beautiful, educational documentation interface:
- **Dark Theme**: Professional gradient styling with modern design
- **Educational Content**: Built-in explanations of AI concepts for students
- **Interactive Testing**: Try all endpoints directly in the browser
- **Quick Start Guide**: Step-by-step instructions for using the API

```bash
# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Access the stunning documentation
# http://localhost:8000/docs
```

### 🚀 API Endpoints

#### 1. Health Check - `GET /health`
Monitor system status and loaded models:
```bash
curl http://localhost:8000/health
```
**Returns**: Server status, CUDA availability, loaded models, cache status

#### 2. Model Information - `GET /model-info`
Get detailed AI model architecture information:
```bash
curl http://localhost:8000/model-info
```
**Returns**: Model parameters, architecture details, supported tickers, performance metrics

#### 3. Stock Prediction - `POST /predict`
The main prediction endpoint with **flexible input support**:

**Option A: 2D Array (60×10)**
```python
import requests
import numpy as np

# Preferred format: 60 days × 10 features
features = np.random.random((60, 10)).tolist()

response = requests.post("http://localhost:8000/predict", json={
    "ticker": "AAPL",
    "features": features,
    "horizon": 3  # Predict 3 days ahead
})

print(response.json()["predictions"])
```

**Option B: Flat Array (600 elements)**
```python
# Alternative format: flat array (automatically reshaped)
features_flat = np.random.random(600).tolist()

response = requests.post("http://localhost:8000/predict", json={
    "ticker": "AAPL", 
    "features": features_flat
})
```

**Returns**: Price predictions, confidence intervals, model metadata, caching info

#### 4. Backtesting - `POST /backtest`
Historical performance analysis:
```python
response = requests.post("http://localhost:8000/backtest", json={
    "ticker": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-08-01",
    "initial_capital": 100000
})
```

### 🧪 Comprehensive Testing Suite

Run our extensive test suites to verify everything works:

```bash
# Python test suite (recommended)
python test_api_python.py

# PowerShell test suite (Windows)
powershell -ExecutionPolicy Bypass -File test_api_powershell.ps1

# Custom UI verification
python test_custom_ui.py

# System validation
python validate_fixes.py
```

**Test Coverage**: 
- ✅ All API endpoints functional
- ✅ Input validation working
- ✅ Error handling robust
- ✅ Custom UI serving properly
- ✅ Model loading successful

### 🔧 Recent Bug Fixes

#### Issues Resolved:
1. **PyTorch 2.6+ Compatibility** - Fixed all `torch.load()` calls with `weights_only=False`
2. **Model Loading Architecture** - Dynamic configuration system for different model architectures  
3. **Input Validation** - Robust handling of multiple input formats with clear error messages
4. **Custom Swagger UI** - Proper HTMLResponse implementation with educational features
5. **API Endpoints** - All endpoints now functional with comprehensive error handling

#### New Files Added:
- `src/api/validators.py` - Robust input validation functions
- `src/api/custom_docs.html` - Professional dark-themed documentation
- `models/model_configs.json` - Dynamic model configuration system
- `test_api_python.py` - Comprehensive Python test suite
- `test_custom_ui.py` - Custom UI verification script

#### Enhanced Features:
- **Flexible Input Formats**: API now accepts both flat arrays (600 elements) and 2D arrays (60×10)
- **Educational Interface**: Custom Swagger UI with explanations of transformer concepts
- **Debug Endpoints**: Built-in troubleshooting tools at `/debug/paths`
- **Graceful Fallbacks**: Robust error handling with informative error messages

### 🎓 Perfect for Learning
This API is specifically designed for **students and educators**:
- **Interactive Documentation** with step-by-step guides
- **Educational Explanations** of AI/ML concepts throughout the interface
- **Live Examples** you can run directly in the browser
- **Comprehensive Error Messages** that help you learn from mistakes
- **Professional Styling** that demonstrates production-quality interfaces

### 🌐 Supported Tickers
Currently supports: **AAPL**, **GOOG**, **MSFT**, **NVDA**, **TSLA**, **NFLX**, **AMZN**, **META**

## 📈 Results and Performance

### Model Metrics
| Metric | Value | Note |
|--------|-------|------|
| Parameters | 2.3M | 4 layers, 8 attention heads |
| Training Time | 3.5 minutes | NVIDIA L4 GPU |
| Validation RMSE | $0.268 | ~0.12% error |
| Directional Accuracy | 55.9% | Barely better than random |
| Inference Speed | <100ms | Single prediction |

### Backtesting Results (Jan-Aug 2024)
| Strategy | Return | Sharpe | Max Drawdown | Trades |
|----------|--------|--------|--------------|--------|
| **Model** | 0.00% | 0.00 | 0.00% | 0 |
| **Buy & Hold** | +35.08% | 2.44 | -12.14% | 8 |
| **Random** | -15.56% | -1.36 | -24.64% | 291 |

![Backtesting Results](results/figures/backtest_results.png)

## 🏗️ Architecture

### Model Architecture
- **Type**: Transformer with temporal attention
- **Input**: 60-day windows of OHLCV + technical indicators
- **Output**: 5-day ahead price predictions
- **Features**: 10 (OHLCV + 5 technical indicators)

### System Architecture
```
Data Sources → Feature Engineering → Model Training → Backtesting → API Serving
     ↓              ↓                    ↓              ↓            ↓
Yahoo Finance   Technical Indicators  GPU Training  Risk Analysis  FastAPI
Alpha Vantage   Normalization        W&B Tracking  Portfolio Sim   Redis Cache
                Data Validation       Checkpointing Cost Modeling  Load Balancing
```

### Infrastructure Components
- **Secrets Management**: Doppler
- **Experiment Tracking**: Weights & Biases + MLflow
- **Caching**: Redis
- **Containerization**: Docker
- **API Framework**: FastAPI
- **Monitoring**: Prometheus + Grafana

## 🔍 Why It Failed: Technical Analysis

### 1. Mode Collapse
The transformer converged to predicting nearly identical values for all stocks (~0.95% return), indicating:
- Insufficient training data diversity
- Loss function didn't penalize uniform predictions
- No cross-sectional ranking component

### 2. Data Limitations
- Only 5 years of daily data (~1,250 points per stock)
- Transformers typically need 10,000+ samples
- Limited to price/volume features

### 3. Architecture Mismatch
- Transformers excel at sequence-to-sequence tasks
- Stock prediction is more suited to gradient boosting
- Attention mechanism found no meaningful patterns

## 💡 Lessons Learned

### What Worked
1. **Infrastructure**: Complete MLOps pipeline functions perfectly
2. **Engineering**: Clean code, proper testing, documentation
3. **Backtesting**: Realistic simulation with transaction costs
4. **GPU Training**: Successfully utilized CUDA acceleration

### What Didn't Work
1. **Model Choice**: Transformers need more data than available
2. **Loss Function**: MSE encouraged trivial solutions
3. **Feature Engineering**: Needed market-relative features
4. **Validation**: Good RMSE didn't translate to profits

### Key Insights
> "In quantitative finance, a model that makes no trades is often better than one that trades randomly, but both are worthless compared to buy-and-hold in a bull market."

- Transaction costs destroy marginal strategies
- Market efficiency makes alpha extremely difficult
- Backtesting with realistic constraints is essential
- Technical metrics (RMSE) != trading performance

## 🛠️ Technical Stack (Updated)

### Core Technologies
- **Python 3.8+**: Primary language (backwards compatible)
- **PyTorch 2.6+**: Deep learning framework with latest compatibility fixes
- **FastAPI**: Modern API framework with custom Swagger UI
- **Redis**: Caching layer (optional)
- **HTML/CSS/JavaScript**: Custom professional documentation interface

### Key Libraries
- `yfinance`: Market data collection
- `pandas/numpy`: Data manipulation and processing
- `scikit-learn`: Preprocessing and feature engineering
- `wandb>=0.15.0`: Experiment tracking (newly added)
- `requests`: HTTP client for API testing
- `uvicorn`: ASGI server for production deployment

### New Dependencies Added
- **wandb**: For experiment tracking and model monitoring
- **HTMLResponse**: For serving custom Swagger UI documentation
- **validators**: Robust input validation and error handling
- **pathlib**: Enhanced file path management

### Testing Framework
- **test_api_python.py**: Comprehensive Python test suite
- **test_custom_ui.py**: Custom UI verification
- **validate_fixes.py**: System validation script
- Cross-platform PowerShell and Python testing support

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Port Already in Use
```powershell
# Windows - Find and kill process
netstat -ano | findstr :8000
taskkill /PID [PID] /F

# Alternative - Use different port
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

#### 2. PyTorch Compatibility Issues
```bash
# Ensure you have the latest compatible version
pip install torch>=2.4.0

# If you see pickle warnings, the fix is already applied
# All torch.load() calls now use weights_only=False
```

#### 3. Model Loading Errors
```bash
# Check if models directory exists and contains .pt files
ls models/
# Expected: *.pt files and model_configs.json

# Verify model configurations
python -c "
import json
from pathlib import Path
config_file = Path('models/model_configs.json')
if config_file.exists():
    print('✅ Model config exists')
    with open(config_file) as f:
        config = json.load(f)
        print(f'Models configured: {list(config.get(\"model_configurations\", {}).keys())}')
else:
    print('❌ Model config missing - API will use fallback configurations')
"
```

#### 4. API Endpoint Not Working
```bash
# Test health endpoint first
curl http://localhost:8000/health

# Check if custom docs are loading
curl http://localhost:8000/debug/paths

# Verify all endpoints
python test_api_python.py
```

#### 5. Custom Swagger UI Not Loading
```bash
# Check if custom HTML file exists
ls src/api/custom_docs.html

# Test debug endpoint
curl http://localhost:8000/debug/paths

# Restart server to reload changes
# Ctrl+C to stop, then restart:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. Input Validation Errors
The API now accepts **two input formats**:

```python
# Option 1: 2D Array (60 days × 10 features) - PREFERRED
features = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]] * 60

# Option 2: Flat Array (600 elements) - AUTOMATICALLY RESHAPED  
features = [0.1, 0.2, 0.3] * 200  # 600 total elements

# Both formats work! The API automatically handles conversion.
```

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Disk Space**: 1GB for models and data
- **GPU**: Optional (CPU-only mode works fine for inference)
- **OS**: Windows, Linux, macOS (all supported)

## 📚 Project Structure

```
timeseries-transformer/
├── README.md                 # This file
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies  
├── setup.py                  # Package setup
├── .gitignore               # Git ignore rules
├── .env.example             # Example environment variables
├── LICENSE                  # MIT License
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── config/             # Configuration management
│   ├── data/               # Data collection and processing
│   ├── models/             # Model architecture
│   ├── training/           # Training logic
│   ├── evaluation/         # Evaluation and metrics
│   ├── backtesting/        # Backtesting framework
│   └── api/                # FastAPI implementation
│
├── scripts/                 # Executable scripts
│   ├── data/               # Data collection scripts
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation scripts
│   └── backtesting/        # Backtesting execution
│
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_backtest_results.ipynb
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
│
├── docs/                   # Documentation
│   ├── architecture.md    # System architecture
│   ├── lessons_learned.md # Detailed failure analysis
│   └── future_work.md     # Potential improvements
│
├── results/                # Results and visualizations
│   ├── figures/           # Charts and plots
│   ├── metrics/           # Performance metrics
│   └── backtest/          # Backtesting results
│
├── deployment/            # Deployment configuration
│   ├── docker/           # Docker files
│   └── kubernetes/       # K8s manifests
│
├── configs/               # Configuration files
├── data/                  # Data storage (gitignored)
├── models/                # Model storage (gitignored)
└── logs/                  # Log files (gitignored)
```

See [docs/architecture.md](docs/architecture.md) for detailed documentation.

## 🔮 Future Improvements

### Immediate Fixes
1. Replace transformer with XGBoost/LightGBM
2. Add ranking loss for relative performance
3. Include market regime features
4. Implement portfolio optimization layer

### Long-term Enhancements
1. Integrate alternative data (news sentiment)
2. Multi-timeframe analysis
3. Reinforcement learning for portfolio management
4. Options pricing models

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [Lessons Learned](docs/lessons_learned.md)
- [API Documentation](http://localhost:8000/docs)
- [Future Work](docs/future_work.md)

## 🤝 Contributing

This project is primarily educational, but contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use this code for research or education, please cite:
```bibtex
@misc{timeseries-transformer-2024,
  title={Time-Series Transformer for Stock Prediction: A Case Study in ML Pipeline Development},
  author={Your Name},
  year={2024},
  url={https://github.com/Mjmurray03/timeseries-transformer-clean}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.


## 📧 Contact

Michael Murray - mjm085@uark.edu

Project Link: [https://github.com/Mjmurray03/timeseries-transformer](https://github.com/Mjmurray03/timeseries-transformer)

---

**Final Thought**: This project proves that building the infrastructure is often more valuable than achieving marginal model improvements. The complete MLOps pipeline, professional documentation, and evaluation/analysis make this repository a great learning resource.

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.
