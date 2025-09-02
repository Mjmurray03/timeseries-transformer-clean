# System Architecture

## Overview

This document describes the complete architecture of the time-series transformer project, from data ingestion to model deployment.

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Engine  │    │  Model Training │
│                 │───▶│                  │───▶│                 │
│ Yahoo Finance   │    │ Technical Indic. │    │ Transformer     │
│ Alpha Vantage   │    │ Normalization    │    │ GPU Training    │
│ News API        │    │ Validation       │    │ W&B Tracking    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Storage  │    │   Backtesting    │    │  Model Serving  │
│                 │    │                  │    │                 │
│ Parquet Files   │    │ Risk Analysis    │    │ FastAPI         │
│ HDF5 Format     │    │ Portfolio Sim    │    │ Redis Cache     │
│ Metadata DB     │    │ Cost Modeling    │    │ Load Balancer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Component Details

### 1. Data Collection Layer

**Purpose**: Collect, validate, and store financial data from multiple sources.

**Components**:
- **Yahoo Finance Collector** (`src/data/collectors/yahoo.py`)
  - Primary data source for OHLCV data
  - Rate limiting: 2000 requests/hour
  - Automatic retry logic with exponential backoff
  
- **Alpha Vantage Collector** (`src/data/collectors/alpha_vantage.py`)
  - Secondary source for validation/backup
  - Premium features: fundamentals, news sentiment
  - Rate limiting: 5 requests/minute (free tier)

- **Data Validator** (`src/data/validation/`)
  - Schema validation using Pydantic
  - Data quality checks (missing values, outliers)
  - Cross-source consistency verification

**Data Flow**:
```python
Raw Market Data → Schema Validation → Quality Checks → Storage
```

### 2. Feature Engineering Pipeline

**Purpose**: Transform raw market data into ML-ready features.

**Components**:
- **Technical Indicators** (`src/data/features/technical.py`)
  - Moving averages (SMA, EMA)
  - Momentum indicators (RSI, MACD)
  - Volatility measures (Bollinger Bands, ATR)
  
- **Price Features** (`src/data/features/price.py`)
  - Returns (log, simple)
  - Price ratios
  - Volume weighted prices
  
- **Normalization** (`src/data/preprocessing/`)
  - Min-Max scaling per stock
  - Rolling z-score normalization
  - Robust scaling (median/IQR)

**Feature Set**:
```python
Features = [
    'open', 'high', 'low', 'close', 'volume',  # OHLCV
    'sma_20', 'ema_12', 'rsi_14',             # Technical
    'log_return', 'volatility'                 # Derived
]  # Total: 10 features per timestep
```

### 3. Model Architecture

**Transformer Model** (`src/models/transformer.py`):

```python
class TransformerPredictor(nn.Module):
    def __init__(self, 
                 input_size: int = 10,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 seq_length: int = 60,
                 prediction_horizon: int = 5):
        
        # Components:
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(...)
        self.output_projection = nn.Linear(d_model, prediction_horizon)
```

**Key Architecture Decisions**:
- **Input**: 60-day sequences of 10 features
- **Output**: 5-day ahead price predictions
- **Attention**: 8 heads for multi-scale pattern recognition
- **Depth**: 4 layers (sweet spot for this data size)
- **Model Size**: 2.3M parameters

### 4. Training Pipeline

**Training Loop** (`src/training/trainer.py`):

```python
Training Process:
1. Data Loading (DataLoader with custom sampler)
2. Forward Pass (Transformer inference)  
3. Loss Calculation (MSE + custom penalties)
4. Backward Pass (Mixed precision training)
5. Gradient Accumulation (Effective batch size: 256)
6. Optimizer Step (AdamW with scheduling)
7. Validation (Every 100 steps)
8. Checkpointing (Best model saving)
```

**Training Configuration**:
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine annealing with warm restarts
- **Batch Size**: 32 (accumulated to 256)
- **Mixed Precision**: FP16 for faster training
- **Early Stopping**: Patience = 10 epochs

### 5. Backtesting Framework

**Backtesting Engine** (`src/backtesting/engine.py`):

```python
class BacktestEngine:
    def __init__(self):
        self.portfolio = Portfolio(initial_capital=100000)
        self.risk_manager = RiskManager(max_position=0.1)
        self.cost_model = TransactionCostModel()
    
    def run_backtest(self, predictions, prices, dates):
        for date in dates:
            signals = self.generate_signals(predictions[date])
            orders = self.risk_manager.process_signals(signals)
            self.portfolio.execute_orders(orders, prices[date])
```

**Features**:
- **Realistic Costs**: 0.1% per trade + 0.01% slippage
- **Risk Management**: Max 10% position size per stock
- **Portfolio Tracking**: Daily P&L, drawdown, Sharpe ratio
- **Benchmark Comparison**: Buy-and-hold, random strategies

### 6. API Serving Layer

**FastAPI Application** (`src/api/main.py`):

```python
app = FastAPI(title="Time-Series Transformer API")

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Load model from cache or disk
    model = await model_cache.get_model()
    
    # Preprocess input data
    features = await preprocess_data(request.data)
    
    # Generate prediction
    prediction = model.predict(features)
    
    return PredictionResponse(
        prices=prediction.tolist(),
        confidence=model.uncertainty,
        timestamp=datetime.now()
    )
```

**API Features**:
- **Model Caching**: Redis-based model caching
- **Input Validation**: Pydantic schemas
- **Rate Limiting**: 100 requests/minute per user
- **Monitoring**: Prometheus metrics
- **Documentation**: Auto-generated OpenAPI docs

### 7. Infrastructure Components

**Secrets Management**:
- **Doppler**: Centralized secrets management
- **Environment Variables**: Local development overrides
- **Kubernetes Secrets**: Production deployment

**Monitoring Stack**:
- **Weights & Biases**: Experiment tracking, model monitoring
- **MLflow**: Model registry, experiment comparison  
- **Prometheus**: Application metrics collection
- **Grafana**: Visualization dashboards
- **Redis**: Caching layer for models and data

**Containerization**:
```dockerfile
# Multi-stage builds for efficiency
FROM python:3.10-slim as base
FROM base as training    # GPU-enabled training image
FROM base as inference   # Lightweight serving image
```

## Performance Characteristics

### Training Performance
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Training Time**: 3.5 hours for 100 epochs
- **Memory Usage**: ~18GB during training
- **Throughput**: ~500 samples/second

### Inference Performance  
- **Latency**: <100ms per prediction (CPU)
- **Throughput**: 50 predictions/second (single core)
- **Memory**: <2GB for loaded model
- **Scalability**: Horizontal scaling with Redis cache

### Data Storage
- **Raw Data**: ~2GB per year of daily data (8 stocks)
- **Processed Features**: ~500MB per year
- **Model Checkpoints**: 45MB per model
- **Database**: PostgreSQL for metadata (~100MB)

## Failure Analysis

### Why the Architecture Failed for Trading

1. **Data Insufficiency**:
   - Only 1,250 daily samples per stock
   - Transformers need 10,000+ samples typically
   - Limited feature diversity

2. **Architecture Mismatch**:
   - Transformers excel at language tasks
   - Stock prediction needs different inductive biases
   - Attention didn't find meaningful patterns

3. **Loss Function Issues**:
   - MSE encouraged uniform predictions
   - No penalty for non-actionable signals
   - Missing ranking/relative performance component

4. **Missing Components**:
   - No portfolio optimization layer
   - No regime detection
   - No risk-adjusted loss functions

## Lessons for Future Architecture

### What to Keep
- ✅ Data pipeline (robust, well-tested)
- ✅ Backtesting framework (realistic, comprehensive)
- ✅ API infrastructure (scalable, monitored)
- ✅ MLOps components (tracking, deployment)

### What to Change
- ❌ Replace Transformer with XGBoost/LightGBM
- ❌ Add portfolio optimization layer
- ❌ Implement ranking-based loss functions
- ❌ Include regime detection features
- ❌ Add multi-timeframe analysis

## Scalability Considerations

### Current Limits
- **Stocks**: ~50 concurrent (memory bound)
- **Features**: ~20 per stock (model capacity)
- **Prediction Horizon**: 5 days (validation constraint)
- **Update Frequency**: Daily (data availability)

### Scaling Solutions
- **Horizontal**: Multiple model instances with load balancing
- **Data Sharding**: Separate models per market sector
- **Feature Selection**: Automatic feature importance ranking
- **Model Ensemble**: Combine multiple architectures

This architecture successfully demonstrates a complete ML pipeline but reveals the challenges of applying transformers to financial time series prediction.