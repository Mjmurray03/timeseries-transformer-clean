#!/bin/bash
# Time-Series Transformer Installation Script

echo "Setting up Time-Series Transformer..."
echo "================================"

# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed models logs results/figures results/metrics results/backtest

# Download sample data (optional)
echo ""
echo "Setup complete!"
echo "To download sample data, run: python scripts/download_stock_data.py --tickers AAPL,MSFT --years 2"
echo "To start API server, run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "API documentation will be available at: http://localhost:8000/docs"