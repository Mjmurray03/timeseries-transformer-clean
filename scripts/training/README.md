# Multi-Stock Training Pipeline

This directory contains production-ready implementations for training transformer models on multiple stocks using two complementary approaches.

## Overview

**Baseline**: Single-stock AAPL model achieving $0.268 RMSE with 464,571 parameters
**Goal**: Extend to multi-stock training while maintaining or improving performance

## Training Approaches

### Approach A: Per-Ticker Models
Individual models trained separately for each stock ticker.

**Files**: 
- `train_ultra_simple.py` - Production single-ticker training
- `train_all_tickers.ps1` - PowerShell automation for batch training

**Usage**:
```bash
# Train single ticker
python scripts/training/train_ultra_simple.py --ticker AAPL --epochs 20 --use-wandb

# Train all available tickers (PowerShell)
.\scripts\training\train_all_tickers.ps1 -UseWandB -Epochs 20
```

### Approach B: Unified Multi-Stock Model
Single model trained on all stocks simultaneously with ticker embeddings.

**Files**:
- `train_multi_stock.py` - Unified multi-stock training
- Extends base transformer with `MultiStockTransformer`

**Usage**:
```bash
# Train multi-stock model
python scripts/training/train_multi_stock.py --tickers AAPL MSFT NVDA --epochs 50 --use-wandb
```

## Production Features

### Data Pipeline
- **Independent scaling per ticker** (critical for different price ranges)
- **Percentage returns as targets** for scale normalization
- **NaN/Inf handling** with graceful degradation
- **Missing data validation** with comprehensive error handling

### Model Architecture
- **Ticker embeddings** (Approach B) for stock differentiation
- **Attention pooling** for sequence aggregation
- **Gradient clipping** to prevent explosion
- **Proper weight initialization** using Xavier uniform

### Training Protocol
- **Volatility-weighted loss** (inverse volatility weighting)
- **Stratified sampling** to balance ticker representation
- **Per-ticker validation metrics** for granular monitoring
- **Early stopping** with patience-based convergence

### Monitoring & Logging
- **Comprehensive W&B integration** with per-ticker tags
- **Per-ticker loss tracking** during training
- **Gradient norm monitoring** for stability
- **Model checkpoint management** with metadata

## Quick Start

1. **Verify data availability**:
```bash
python scripts/training/test_training_approaches.py
```

2. **Train single ticker** (recommended first):
```bash
python scripts/training/train_ultra_simple.py --ticker AAPL --epochs 10 --use-wandb
```

3. **Train all tickers** (batch):
```bash
.\scripts\training\train_all_tickers.ps1 -Epochs 10 -UseWandB
```

4. **Train multi-stock model**:
```bash
python scripts/training/train_multi_stock.py --tickers AAPL MSFT --epochs 20 --use-wandb
```