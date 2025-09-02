# Time-Series Transformer for Stock Market Prediction

## A Complete ML Pipeline Case Study: From Architecture to Production

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
git clone https://github.com/yourusername/timeseries-transformer.git
cd timeseries-transformer
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

## 📈 Results and Performance

### Model Metrics
| Metric | Value | Note |
|--------|-------|------|
| Parameters | 2.3M | 4 layers, 8 attention heads |
| Training Time | 3.5 hours | NVIDIA L4 GPU |
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

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.10**: Primary language
- **PyTorch 2.4**: Deep learning framework
- **FastAPI**: API framework
- **Redis**: Caching layer
- **Docker**: Containerization
- **PostgreSQL**: Metadata storage

### Key Libraries
- `yfinance`: Market data
- `pandas/numpy`: Data manipulation
- `scikit-learn`: Preprocessing
- `wandb`: Experiment tracking
- `pytest`: Testing framework

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
  url={https://github.com/yourusername/timeseries-transformer}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- University of Arkansas Walton College for GPU resources
- Professor Zach Steelman for guidance
- Open source community for foundational libraries

## 📧 Contact

Your Name - [your.email@uark.edu](mailto:your.email@uark.edu)

Project Link: [https://github.com/yourusername/timeseries-transformer](https://github.com/yourusername/timeseries-transformer)

---

**Final Thought**: This project proves that building the infrastructure is often more valuable than achieving marginal model improvements. The complete MLOps pipeline, professional documentation, and honest failure analysis make this repository a learning resource worth more than a slightly profitable trading algorithm.

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.